// Modèle byte-level, backward écrit à la main.
//
//   tokens → tok_emb + pos_emb
//          → [ RMSNorm → attention causale → +résidu → RMSNorm → MLP(SiLU) → +résidu ] × L
//          → RMSNorm → logits = x · tok_embᵀ          (poids liés)
//
// Les activations sont des matrices (N, d), N = batch × T, rangées séquence par séquence.

use crate::rng::Rng;
use crate::Real;
use ndarray::{s, Array2, Zip};

const NORM_EPS: Real = 1e-6;

// ───────────────────────────── paramètres ─────────────────────────────

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_len: usize,
}

/// Tenseur entraînable. Toujours 2D : un vecteur est une matrice 1×n.
pub struct Param {
    pub name: String,
    pub w: Array2<Real>, // valeurs
    pub g: Array2<Real>, // gradient accumulé
    pub m: Array2<Real>, // moment 1 (AdamW)
    pub v: Array2<Real>, // moment 2 (AdamW)
    pub decay: bool,     // weight decay appliqué ?
}

impl Param {
    fn new(name: &str, rows: usize, cols: usize, decay: bool) -> Self {
        Param {
            name: name.to_string(),
            w: Array2::zeros((rows, cols)),
            g: Array2::zeros((rows, cols)),
            m: Array2::zeros((rows, cols)),
            v: Array2::zeros((rows, cols)),
            decay,
        }
    }

    fn randn(mut self, rng: &mut Rng, sd: Real) -> Self {
        for x in self.w.iter_mut() {
            *x = rng.gauss() as Real * sd;
        }
        self
    }

    fn ones(mut self) -> Self {
        self.w.fill(1.0);
        self
    }
}

pub struct Layer {
    pub ln1: Param, // (1, d)
    pub wq: Param,  // (d, d)
    pub wk: Param,
    pub wv: Param,
    pub wo: Param,
    pub ln2: Param, // (1, d)
    pub w1: Param,  // (d, d_ff)
    pub w2: Param,  // (d_ff, d)
}

pub struct Model {
    pub cfg: Config,
    pub tok_emb: Param, // (vocab, d), aussi utilisé comme tête de sortie
    pub pos_emb: Param, // (max_len, d)
    pub layers: Vec<Layer>,
    pub lnf: Param, // (1, d)
}

// ───────────────────────────── caches du forward ─────────────────────────────

struct LayerCache {
    x_in: Array2<Real>,
    inv1: Vec<Real>,
    h1: Array2<Real>,
    q: Array2<Real>,
    k: Array2<Real>,
    v: Array2<Real>,
    p: Vec<Array2<Real>>, // probas d'attention, indexées par bi * n_heads + hi
    o: Array2<Real>,
    x_mid: Array2<Real>,
    inv2: Vec<Real>,
    h2: Array2<Real>,
    u: Array2<Real>, // pré-activation SiLU
    r: Array2<Real>, // SiLU(u)
}

pub struct Cache {
    tokens: Vec<usize>,
    b: usize,
    t: usize,
    layers: Vec<LayerCache>,
    x_last: Array2<Real>, // entrée du RMSNorm final
    inv_f: Vec<Real>,
    xf: Array2<Real>, // sortie du RMSNorm final
}

// ───────────────────────────── modèle ─────────────────────────────

impl Model {
    pub fn new(cfg: Config, seed: u64) -> Self {
        assert!(cfg.d_model % cfg.n_heads == 0, "d_model doit être divisible par n_heads");
        let mut rng = Rng::new(seed);
        let d = cfg.d_model;
        let sd: Real = 0.02;
        let sd_out = sd / ((2 * cfg.n_layers) as Real).sqrt();

        let tok_emb = Param::new("tok_emb", cfg.vocab, d, false).randn(&mut rng, sd);
        let pos_emb = Param::new("pos_emb", cfg.max_len, d, false).randn(&mut rng, sd);

        let mut layers = Vec::new();
        for l in 0..cfg.n_layers {
            let nm = |n: &str| format!("l{l}.{n}");
            layers.push(Layer {
                ln1: Param::new(&nm("ln1"), 1, d, false).ones(),
                wq: Param::new(&nm("wq"), d, d, true).randn(&mut rng, sd),
                wk: Param::new(&nm("wk"), d, d, true).randn(&mut rng, sd),
                wv: Param::new(&nm("wv"), d, d, true).randn(&mut rng, sd),
                wo: Param::new(&nm("wo"), d, d, true).randn(&mut rng, sd_out),
                ln2: Param::new(&nm("ln2"), 1, d, false).ones(),
                w1: Param::new(&nm("w1"), d, cfg.d_ff, true).randn(&mut rng, sd),
                w2: Param::new(&nm("w2"), cfg.d_ff, d, true).randn(&mut rng, sd_out),
            });
        }
        let lnf = Param::new("lnf", 1, d, false).ones();
        Model { cfg, tok_emb, pos_emb, layers, lnf }
    }

    pub fn params(&self) -> Vec<&Param> {
        let mut v: Vec<&Param> = vec![&self.tok_emb, &self.pos_emb];
        for l in &self.layers {
            v.extend([&l.ln1, &l.wq, &l.wk, &l.wv, &l.wo, &l.ln2, &l.w1, &l.w2]);
        }
        v.push(&self.lnf);
        v
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut v: Vec<&mut Param> = vec![&mut self.tok_emb, &mut self.pos_emb];
        for l in self.layers.iter_mut() {
            v.extend([
                &mut l.ln1, &mut l.wq, &mut l.wk, &mut l.wv,
                &mut l.wo, &mut l.ln2, &mut l.w1, &mut l.w2,
            ]);
        }
        v.push(&mut self.lnf);
        v
    }

    pub fn n_params(&self) -> usize {
        self.params().iter().map(|p| p.w.len()).sum()
    }

    pub fn zero_grad(&mut self) {
        for p in self.params_mut() {
            p.g.fill(0.0);
        }
    }

    /// Clippe la norme globale du gradient ; renvoie la norme avant clipping.
    pub fn clip_grad_norm(&mut self, max_norm: Real) -> Real {
        let mut sq: Real = 0.0;
        for p in self.params() {
            for &g in p.g.iter() {
                sq += g * g;
            }
        }
        let norm = sq.sqrt();
        if norm > max_norm {
            let k = max_norm / (norm + 1e-12);
            for p in self.params_mut() {
                p.g *= k;
            }
        }
        norm
    }

    /// tokens : b*t indices, séquence par séquence. Renvoie les logits (b*t, vocab).
    pub fn forward(&self, tokens: &[usize], b: usize, t: usize) -> (Array2<Real>, Cache) {
        let cfg = &self.cfg;
        let (d, h) = (cfg.d_model, cfg.n_heads);
        let dh = d / h;
        let n = b * t;
        assert_eq!(tokens.len(), n, "tokens doit contenir b*t éléments");
        assert!(t <= cfg.max_len, "séquence plus longue que max_len");
        let scale = 1.0 / (dh as Real).sqrt();

        // embeddings
        let mut x = Array2::<Real>::zeros((n, d));
        for bi in 0..b {
            for ti in 0..t {
                let row = bi * t + ti;
                let tok = tokens[row];
                for j in 0..d {
                    x[[row, j]] = self.tok_emb.w[[tok, j]] + self.pos_emb.w[[ti, j]];
                }
            }
        }

        let mut caches = Vec::with_capacity(cfg.n_layers);
        for layer in &self.layers {
            let x_in = x;
            let (h1, inv1) = rmsnorm_fwd(&x_in, &layer.ln1.w);
            let q = h1.dot(&layer.wq.w);
            let k = h1.dot(&layer.wk.w);
            let v = h1.dot(&layer.wv.w);

            let mut o = Array2::<Real>::zeros((n, d));
            let mut ps: Vec<Array2<Real>> = Vec::with_capacity(b * h);
            for bi in 0..b {
                for hi in 0..h {
                    let (r0, r1, c0, c1) = (bi * t, (bi + 1) * t, hi * dh, (hi + 1) * dh);
                    let qh = q.slice(s![r0..r1, c0..c1]);
                    let kh = k.slice(s![r0..r1, c0..c1]);
                    let vh = v.slice(s![r0..r1, c0..c1]);
                    let p = causal_softmax(&qh.dot(&kh.t()), scale);
                    o.slice_mut(s![r0..r1, c0..c1]).assign(&p.dot(&vh));
                    ps.push(p);
                }
            }
            let a = o.dot(&layer.wo.w);
            let x_mid = &x_in + &a;

            let (h2, inv2) = rmsnorm_fwd(&x_mid, &layer.ln2.w);
            let u = h2.dot(&layer.w1.w);
            let r = u.mapv(silu);
            let m = r.dot(&layer.w2.w);
            x = &x_mid + &m;

            caches.push(LayerCache { x_in, inv1, h1, q, k, v, p: ps, o, x_mid, inv2, h2, u, r });
        }

        let (xf, inv_f) = rmsnorm_fwd(&x, &self.lnf.w);
        let logits = xf.dot(&self.tok_emb.w.t());
        let cache = Cache { tokens: tokens.to_vec(), b, t, layers: caches, x_last: x, inv_f, xf };
        (logits, cache)
    }

    /// Rétropropage dlogits (N, vocab) et ACCUMULE dans les `.g` (appeler zero_grad avant).
    pub fn backward(&mut self, cache: &Cache, dlogits: &Array2<Real>) {
        let cfg = self.cfg.clone();
        let (d, h) = (cfg.d_model, cfg.n_heads);
        let dh = d / h;
        let (b, t) = (cache.b, cache.t);
        let n = b * t;
        let scale = 1.0 / (dh as Real).sqrt();

        // logits = xf · Eᵀ  →  dxf = dlogits · E ;  dE += dlogitsᵀ · xf
        let dxf = dlogits.dot(&self.tok_emb.w);
        self.tok_emb.g += &dlogits.t().dot(&cache.xf);
        let mut dx = rmsnorm_bwd(&cache.x_last, &self.lnf.w, &cache.inv_f, &dxf, &mut self.lnf.g);

        for li in (0..cfg.n_layers).rev() {
            let c = &cache.layers[li];
            let layer = &mut self.layers[li];

            // ── MLP : x_out = x_mid + SiLU(h2·W1)·W2 ──
            layer.w2.g += &c.r.t().dot(&dx);
            let mut du = dx.dot(&layer.w2.w.t());
            Zip::from(&mut du).and(&c.u).for_each(|g, &u| *g *= silu_grad(u));
            layer.w1.g += &c.h2.t().dot(&du);
            let dh2 = du.dot(&layer.w1.w.t());
            let dxm = rmsnorm_bwd(&c.x_mid, &layer.ln2.w, &c.inv2, &dh2, &mut layer.ln2.g);
            let dx_mid = &dx + &dxm; // chemin résiduel + chemin par la norme

            // ── Attention : x_mid = x_in + concat(softmax(QKᵀ·scale)·V)·Wo ──
            layer.wo.g += &c.o.t().dot(&dx_mid);
            let d_o = dx_mid.dot(&layer.wo.w.t());

            let mut dq = Array2::<Real>::zeros((n, d));
            let mut dk = Array2::<Real>::zeros((n, d));
            let mut dv = Array2::<Real>::zeros((n, d));
            for bi in 0..b {
                for hi in 0..h {
                    let (r0, r1, c0, c1) = (bi * t, (bi + 1) * t, hi * dh, (hi + 1) * dh);
                    let p = &c.p[bi * h + hi];
                    let qh = c.q.slice(s![r0..r1, c0..c1]);
                    let kh = c.k.slice(s![r0..r1, c0..c1]);
                    let vh = c.v.slice(s![r0..r1, c0..c1]);
                    let doh = d_o.slice(s![r0..r1, c0..c1]);

                    let dp = doh.dot(&vh.t()); // dP = dO · Vᵀ
                    let dvh = p.t().dot(&doh); // dV = Pᵀ · dO
                    let mut ds = Array2::<Real>::zeros((t, t));
                    for i in 0..t {
                        let mut sum: Real = 0.0;
                        for j in 0..=i {
                            sum += dp[[i, j]] * p[[i, j]];
                        }
                        for j in 0..=i {
                            // softmax : dS = P ∘ (dP − Σ dP∘P), puis facteur d'échelle
                            ds[[i, j]] = p[[i, j]] * (dp[[i, j]] - sum) * scale;
                        }
                    }
                    let dqh = ds.dot(&kh);
                    let dkh = ds.t().dot(&qh);
                    dq.slice_mut(s![r0..r1, c0..c1]).assign(&dqh);
                    dk.slice_mut(s![r0..r1, c0..c1]).assign(&dkh);
                    dv.slice_mut(s![r0..r1, c0..c1]).assign(&dvh);
                }
            }
            layer.wq.g += &c.h1.t().dot(&dq);
            layer.wk.g += &c.h1.t().dot(&dk);
            layer.wv.g += &c.h1.t().dot(&dv);
            let mut dh1 = dq.dot(&layer.wq.w.t());
            dh1 += &dk.dot(&layer.wk.w.t());
            dh1 += &dv.dot(&layer.wv.w.t());
            let dxi = rmsnorm_bwd(&c.x_in, &layer.ln1.w, &c.inv1, &dh1, &mut layer.ln1.g);
            dx = &dx_mid + &dxi;
        }

        // embeddings (tok_emb reçoit déjà le gradient de la tête de sortie)
        for bi in 0..b {
            for ti in 0..t {
                let row = bi * t + ti;
                let tok = cache.tokens[row];
                for j in 0..d {
                    let g = dx[[row, j]];
                    self.tok_emb.g[[tok, j]] += g;
                    self.pos_emb.g[[ti, j]] += g;
                }
            }
        }
    }

    /// forward + perte + backward (accumule dans `.g`). Renvoie la perte.
    pub fn loss_and_grad(
        &mut self,
        tokens: &[usize],
        targets: &[usize],
        weights: &[Real],
        b: usize,
        t: usize,
    ) -> Real {
        let (logits, cache) = self.forward(tokens, b, t);
        let (loss, dlogits) = cross_entropy(&logits, targets, weights);
        self.backward(&cache, &dlogits);
        loss
    }

    /// Perte seule (sans gradient).
    pub fn loss(&self, tokens: &[usize], targets: &[usize], weights: &[Real], b: usize, t: usize) -> Real {
        let (logits, _cache) = self.forward(tokens, b, t);
        cross_entropy(&logits, targets, weights).0
    }
}

// ───────────────────────────── briques ─────────────────────────────

fn silu(x: Real) -> Real {
    x / (1.0 + (-x).exp())
}

fn silu_grad(x: Real) -> Real {
    let sg = 1.0 / (1.0 + (-x).exp());
    sg * (1.0 + x * (1.0 - sg))
}

/// y = g ∘ x / rms(x). Renvoie (y, 1/rms par ligne).
fn rmsnorm_fwd(x: &Array2<Real>, g: &Array2<Real>) -> (Array2<Real>, Vec<Real>) {
    let (n, d) = x.dim();
    let mut y = Array2::<Real>::zeros((n, d));
    let mut inv = vec![0.0 as Real; n];
    for i in 0..n {
        let mut ss: Real = 0.0;
        for j in 0..d {
            ss += x[[i, j]] * x[[i, j]];
        }
        let r = 1.0 / (ss / d as Real + NORM_EPS).sqrt();
        inv[i] = r;
        for j in 0..d {
            y[[i, j]] = x[[i, j]] * r * g[[0, j]];
        }
    }
    (y, inv)
}

/// dx_k = r·g_k·dy_k − (r³·x_k/d)·Σ_i dy_i·g_i·x_i ;  dg_j += Σ_lignes dy_j·x_j·r
fn rmsnorm_bwd(
    x: &Array2<Real>,
    g: &Array2<Real>,
    inv: &[Real],
    dy: &Array2<Real>,
    dg: &mut Array2<Real>,
) -> Array2<Real> {
    let (n, d) = x.dim();
    let mut dx = Array2::<Real>::zeros((n, d));
    for i in 0..n {
        let r = inv[i];
        let mut dot: Real = 0.0;
        for j in 0..d {
            dot += dy[[i, j]] * g[[0, j]] * x[[i, j]];
            dg[[0, j]] += dy[[i, j]] * x[[i, j]] * r;
        }
        let c = r * r * r * dot / d as Real;
        for j in 0..d {
            dx[[i, j]] = r * g[[0, j]] * dy[[i, j]] - c * x[[i, j]];
        }
    }
    dx
}

/// softmax causale ligne par ligne sur scale·sc ; les positions j > i valent 0.
fn causal_softmax(sc: &Array2<Real>, scale: Real) -> Array2<Real> {
    let t = sc.nrows();
    let mut p = Array2::<Real>::zeros((t, t));
    for i in 0..t {
        let mut mx = Real::NEG_INFINITY;
        for j in 0..=i {
            let v = sc[[i, j]] * scale;
            if v > mx {
                mx = v;
            }
        }
        let mut z: Real = 0.0;
        for j in 0..=i {
            let e = (sc[[i, j]] * scale - mx).exp();
            p[[i, j]] = e;
            z += e;
        }
        for j in 0..=i {
            p[[i, j]] /= z;
        }
    }
    p
}

/// Entropie croisée moyenne pondérée (poids 0 = position ignorée). Renvoie (perte, dlogits).
pub fn cross_entropy(logits: &Array2<Real>, targets: &[usize], weights: &[Real]) -> (Real, Array2<Real>) {
    let (n, v) = logits.dim();
    assert_eq!(targets.len(), n);
    assert_eq!(weights.len(), n);
    let wsum: Real = weights.iter().sum();
    assert!(wsum > 0.0, "somme des poids nulle");
    let mut dl = Array2::<Real>::zeros((n, v));
    let mut loss: Real = 0.0;
    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let mut mx = Real::NEG_INFINITY;
        for j in 0..v {
            if logits[[i, j]] > mx {
                mx = logits[[i, j]];
            }
        }
        let mut z: Real = 0.0;
        for j in 0..v {
            z += (logits[[i, j]] - mx).exp();
        }
        loss += w * (mx + z.ln() - logits[[i, targets[i]]]);
        let c = w / wsum;
        for j in 0..v {
            dl[[i, j]] = c * (logits[[i, j]] - mx).exp() / z;
        }
        dl[[i, targets[i]]] -= c;
    }
    (loss / wsum, dl)
}