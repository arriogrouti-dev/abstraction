use crate::model::Param;
use crate::Real;
use ndarray::Zip;

/// AdamW (weight decay découplé, appliqué seulement aux Param avec `decay = true`).
pub struct AdamW {
    pub lr: Real,
    pub beta1: Real,
    pub beta2: Real,
    pub eps: Real,
    pub weight_decay: Real,
    t: i32,
}

impl AdamW {
    pub fn new(lr: Real) -> Self {
        AdamW { lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, weight_decay: 0.1, t: 0 }
    }

    pub fn step(&mut self, params: Vec<&mut Param>) {
        self.t += 1;
        let (lr, b1, b2, eps) = (self.lr, self.beta1, self.beta2, self.eps);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        for p in params {
            let wd = if p.decay { self.weight_decay } else { 0.0 };
            Zip::from(&mut p.w)
                .and(&p.g)
                .and(&mut p.m)
                .and(&mut p.v)
                .for_each(|w, &g, m, v| {
                    *m = b1 * *m + (1.0 - b1) * g;
                    *v = b2 * *v + (1.0 - b2) * g * g;
                    let m_hat = *m / bc1;
                    let v_hat = *v / bc2;
                    *w -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * *w);
                });
        }
    }
}