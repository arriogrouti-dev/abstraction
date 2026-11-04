#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n { s += a[i] * b[i]; }
    s
}

pub fn l2_normalize(v: &mut [f32]) {
    let mut n2 = 0.0;
    for &x in v.iter() { n2 += x * x; }
    let n = n2.sqrt().max(1e-8);
    for x in v.iter_mut() { *x /= n; }
}

pub fn mean_vec(vs: &[Vec<f32>]) -> Vec<f32> {
    if vs.is_empty() { return Vec::new(); }
    let d = vs[0].len();
    let mut out = vec![0.0f32; d];
    for v in vs {
        for i in 0..d { out[i] += v[i]; }
    }
    for i in 0..d { out[i] /= vs.len() as f32; }
    out
}

/// InfoNCE batch (les représentations doivent être L2-normalisées).
pub fn batch_info_nce(anchors: &[Vec<f32>], positives: &[Vec<f32>], tau: f32) -> f32 {
    let n = anchors.len();
    if n == 0 { return 0.0; }
    let mut total = 0.0f32;
    for i in 0..n {
        let pos_sim = dot(&anchors[i], &positives[i]) / tau;
        let mut denom = pos_sim.exp();
        for j in 0..n {
            if i == j { continue; }
            denom += (dot(&anchors[i], &positives[j]) / tau).exp();
        }
        total += -(pos_sim - denom.ln());
    }
    total / n as f32
}

/// Marge contrastive causale : cos(anchor, noncritical_int) - cos(anchor, critical_int) ≥ margin.
pub fn batch_causal_margin(anchors: &[Vec<f32>], noncrit: &[Vec<f32>], crit: &[Vec<f32>], margin: f32) -> f32 {
    let n = anchors.len();
    if n == 0 { return 0.0; }
    let mut total = 0.0f32;
    for i in 0..n {
        let sp = dot(&anchors[i], &noncrit[i]);
        let sn = dot(&anchors[i], &crit[i]);
        total += (sn - sp + margin).max(0.0);
    }
    total / n as f32
}