use crate::config::Config;
use crate::encoder::encode_bytes;
use crate::losses::{batch_causal_margin, batch_info_nce, dot, l2_normalize, mean_vec};
use crate::network::Network;
use crate::world::FactorWorld;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

pub struct Batch {
    pub anchors_inp: Vec<Vec<Vec<f32>>>,
    pub positives_inp: Vec<Vec<Vec<f32>>>,
    pub noncrit_inp: Vec<Vec<Vec<f32>>>,
    pub crit_inp: Vec<Vec<Vec<f32>>>,
    pub noise: Vec<Vec<Vec<f32>>>,
}

pub struct Trainer {
    pub cfg: Config,
    pub net: Network,
    pub world: FactorWorld,
}

impl Trainer {
    pub fn new(cfg: Config) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let net = Network::build(&cfg, &mut rng);
        let world = FactorWorld::new(cfg.vocab, cfg.seq_len, cfg.n_a, cfg.n_b, cfg.seed + 1);
        Self { cfg, net, world }
    }

    pub fn sample_batch(&self, rng: &mut impl Rng, sigma: f32) -> Batch {
        let bs = self.cfg.batch_size;
        let mut b = Batch {
            anchors_inp: Vec::with_capacity(bs),
            positives_inp: Vec::with_capacity(bs),
            noncrit_inp: Vec::with_capacity(bs),
            crit_inp: Vec::with_capacity(bs),
            noise: Vec::with_capacity(bs),
        };
        for _ in 0..bs {
            let idx = rng.random_range(0..self.world.train_combos.len());
            let (a, bb) = self.world.train_combos[idx];
            let x = self.world.sample(a, bb, rng);
            let x_aug = self.world.augment(&x, rng);
            let x_nc = self.world.intervene_noncritical(&x, rng);
            let x_c = self.world.intervene_critical(&x, a, bb, rng);

            b.anchors_inp.push(encode_bytes(&x, self.cfg.n_input, self.cfg.d));
            b.positives_inp.push(encode_bytes(&x_aug, self.cfg.n_input, self.cfg.d));
            b.noncrit_inp.push(encode_bytes(&x_nc, self.cfg.n_input, self.cfg.d));
            b.crit_inp.push(encode_bytes(&x_c, self.cfg.n_input, self.cfg.d));
            b.noise.push(self.net.sample_noise(sigma, rng));
        }
        b
    }

    pub fn loss(&self, params: &[f32], batch: &Batch) -> f32 {
        let bs = batch.anchors_inp.len();
        let mut anchors = Vec::with_capacity(bs);
        let mut positives = Vec::with_capacity(bs);
        let mut ncs = Vec::with_capacity(bs);
        let mut cs = Vec::with_capacity(bs);
        let mut mus_norm_sq = 0.0f32;

        for i in 0..bs {
            let noise = &batch.noise[i];
            let fa = self.net.forward(params, &batch.anchors_inp[i], Some(noise), None);
            let fp = self.net.forward(params, &batch.positives_inp[i], Some(noise), None);
            let fnc = self.net.forward(params, &batch.noncrit_inp[i], Some(noise), None);
            let fc = self.net.forward(params, &batch.crit_inp[i], Some(noise), None);
            let fmu = self.net.forward(params, &batch.anchors_inp[i], None, None);

            let mu = self.net.read_representation(&fmu.final_states);
            mus_norm_sq += mu.iter().map(|x| x * x).sum::<f32>();

            let mut ra = self.net.read_representation(&fa.final_states);
            let mut rp = self.net.read_representation(&fp.final_states);
            let mut rnc = self.net.read_representation(&fnc.final_states);
            let mut rc = self.net.read_representation(&fc.final_states);
            l2_normalize(&mut ra);
            l2_normalize(&mut rp);
            l2_normalize(&mut rnc);
            l2_normalize(&mut rc);
            anchors.push(ra);
            positives.push(rp);
            ncs.push(rnc);
            cs.push(rc);
        }

        let l_contrast = batch_info_nce(&anchors, &positives, self.cfg.tau);
        let l_causal = batch_causal_margin(&anchors, &ncs, &cs, self.cfg.margin_causal);

        let rep_dim = (self.net.hidden_nodes.len() + self.net.output_nodes.len()) * self.net.d;
        let log_sigma2 = params[self.net.layout.log_sigma2_off];
        let sigma2 = self.net.sigma(params).powi(2);
        let mu_mean_sq = mus_norm_sq / (bs * rep_dim).max(1) as f32;
        let l_ib = 0.5 * (sigma2 + mu_mean_sq - 1.0 - log_sigma2);

        let mut l_sparse = 0.0f32;
        let mut l_l2 = 0.0f32;
        let mut n = 0usize;
        for (i, &p) in params.iter().enumerate() {
            if i == self.net.layout.log_sigma2_off { continue; }
            l_sparse += p.abs();
            l_l2 += p * p;
            n += 1;
        }
        l_sparse /= n as f32;
        l_l2 /= n as f32;

        self.cfg.lambda_contrast * l_contrast
            + self.cfg.lambda_causal * l_causal
            + self.cfg.beta_ib * l_ib
            + self.cfg.lambda_sparsity * l_sparse
            + self.cfg.lambda_l2 * l_l2
    }

    pub fn train(&mut self) {
        let mut rng = StdRng::seed_from_u64(self.cfg.seed + 2);
        let mut params = self.net.init_params(&mut rng);
        let n = params.len();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let normal = Normal::new(0.0, 1.0).unwrap();

        println!("params: {} | pop: {} | batch: {} | steps: {}",
                 n, self.cfg.es_pop, self.cfg.batch_size, self.cfg.train_steps);

        for step in 0..self.cfg.train_steps {
            let sigma_now = self.net.sigma(&params);
            let batch = self.sample_batch(&mut rng, sigma_now);

            // ES antithetic
            let pop = self.cfg.es_pop;
            let mut grad = vec![0.0f32; n];
            for _ in 0..pop {
                let eps: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
                let pp: Vec<f32> = params.iter().zip(&eps)
                    .map(|(&a, &e)| a + self.cfg.es_sigma * e).collect();
                let pm: Vec<f32> = params.iter().zip(&eps)
                    .map(|(&a, &e)| a - self.cfg.es_sigma * e).collect();
                let lp = self.loss(&pp, &batch);
                let lm = self.loss(&pm, &batch);
                let coeff = (lp - lm) / (2.0 * self.cfg.es_sigma);
                for i in 0..n { grad[i] += coeff * eps[i] / pop as f32; }
            }

            // Adam
            for i in 0..n {
                m[i] = self.cfg.es_beta1 * m[i] + (1.0 - self.cfg.es_beta1) * grad[i];
                v[i] = self.cfg.es_beta2 * v[i] + (1.0 - self.cfg.es_beta2) * grad[i] * grad[i];
                let mh = m[i] / (1.0 - self.cfg.es_beta1.powi(step as i32 + 1));
                let vh = v[i] / (1.0 - self.cfg.es_beta2.powi(step as i32 + 1));
                params[i] -= self.cfg.es_lr * mh / (vh.sqrt() + self.cfg.es_eps);
            }

            if step % self.cfg.eval_every == 0 {
                let l = self.loss(&params, &batch);
                println!("step {:5} | loss {:8.4} | sigma {:6.4}",
                         step, l, self.net.sigma(&params));
            }
            if let Some(dir) = &self.cfg.checkpoint_dir {
                if step > 0 && step % (self.cfg.eval_every * 5) == 0 {
                    self.save(dir, &params);
                }
            }
        }

        self.evaluate(&params, &mut rng);

        if let Some(dir) = &self.cfg.checkpoint_dir {
            self.save(dir, &params);
        }
    }

    fn save(&self, dir: &str, params: &[f32]) {
        let _ = std::fs::create_dir_all(dir);
        let path = format!("{}/model.bin", dir);
        if let Ok(bytes) = bincode::serialize(&(params.to_vec(), self.cfg.clone())) {
            let _ = std::fs::write(&path, bytes);
            println!("saved model to {}", path);
        }
    }

    /// Test de généralisation compositionnelle sur combos tenus hors-jeu.
    /// Mesure si la représentation d'un combo test (a,b) est plus proche des
    /// combos entraînés partageant un facteur (a,b') ou (a',b) que des combos
    /// partageant aucun facteur. Score > 0 => structure compositionnelle acquise.
    pub fn evaluate(&self, params: &[f32], rng: &mut impl Rng) {
        let mut train_reps: Vec<((usize, usize), Vec<f32>)> = Vec::new();
        for &(a, b) in &self.world.train_combos {
            let mut reps = Vec::new();
            for _ in 0..3 {
                let x = self.world.sample(a, b, rng);
                let inp = encode_bytes(&x, self.cfg.n_input, self.cfg.d);
                let f = self.net.forward(params, &inp, None, None);
                reps.push(self.net.read_representation(&f.final_states));
            }
            let mut mv = mean_vec(&reps);
            l2_normalize(&mut mv);
            train_reps.push(((a, b), mv));
        }

        let mut total_score = 0.0f32;
        let mut n_pairs = 0usize;
        for &(a, b) in &self.world.test_combos {
            let mut reps = Vec::new();
            for _ in 0..3 {
                let x = self.world.sample(a, b, rng);
                let inp = encode_bytes(&x, self.cfg.n_input, self.cfg.d);
                let f = self.net.forward(params, &inp, None, None);
                reps.push(self.net.read_representation(&f.final_states));
            }
            let mut test_rep = mean_vec(&reps);
            l2_normalize(&mut test_rep);

            let mut sim_rel = 0.0f32; let mut n_rel = 0usize;
            let mut sim_unrel = 0.0f32; let mut n_unrel = 0usize;
            for ((a2, b2), r2) in &train_reps {
                let sim = dot(&test_rep, r2);
                if *a2 == a || *b2 == b {
                    sim_rel += sim; n_rel += 1;
                } else {
                    sim_unrel += sim; n_unrel += 1;
                }
            }
            if n_rel > 0 && n_unrel > 0 {
                total_score += sim_rel / n_rel as f32 - sim_unrel / n_unrel as f32;
                n_pairs += 1;
            }
        }
        if n_pairs > 0 {
            let score = total_score / n_pairs as f32;
            println!("=== Compositional generalization score: {:+.4} ===", score);
            println!("    (train combos: {}, test combos: {})",
                     self.world.train_combos.len(), self.world.test_combos.len());
        }
    }

    /// Compréhension d'un flux d'octets arbitraire en production.
    pub fn comprehend(&self, params: &[f32], bytes: &[u8]) -> Vec<f32> {
        let inp = encode_bytes(bytes, self.cfg.n_input, self.cfg.d);
        let f = self.net.forward(params, &inp, None, None);
        self.net.read_representation(&f.final_states)
    }
}