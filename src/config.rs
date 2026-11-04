use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    // Architecture
    pub d: usize,
    pub n_nodes: usize,
    pub n_input: usize,
    pub n_output: usize,
    pub edge_prob: f32,
    pub local_edge_frac: f32,
    pub n_rounds: usize,

    // IB
    pub beta_ib: f32,
    pub init_log_sigma2: f32,

    // Contrastive
    pub tau: f32,
    pub lambda_contrast: f32,
    pub lambda_causal: f32,
    pub margin_causal: f32,

    // Regularization
    pub lambda_sparsity: f32,
    pub lambda_l2: f32,

    // ES (Evolution Strategies optimizer)
    pub es_sigma: f32,
    pub es_lr: f32,
    pub es_pop: usize,
    pub es_beta1: f32,
    pub es_beta2: f32,
    pub es_eps: f32,

    // Training
    pub batch_size: usize,
    pub seq_len: usize,
    pub vocab: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub train_steps: usize,
    pub eval_every: usize,
    pub seed: u64,
    pub checkpoint_dir: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            d: 8,
            n_nodes: 32,
            n_input: 4,
            n_output: 4,
            edge_prob: 0.10,
            local_edge_frac: 1.5,
            n_rounds: 3,

            beta_ib: 0.005,
            init_log_sigma2: -4.6,

            tau: 0.1,
            lambda_contrast: 1.0,
            lambda_causal: 0.5,
            margin_causal: 0.2,

            lambda_sparsity: 0.001,
            lambda_l2: 0.00001,

            es_sigma: 0.02,
            es_lr: 0.005,
            es_pop: 4,
            es_beta1: 0.9,
            es_beta2: 0.999,
            es_eps: 1e-8,

            batch_size: 16,
            seq_len: 32,
            vocab: 256,
            n_a: 8,
            n_b: 8,
            train_steps: 5000,
            eval_every: 200,
            seed: 42,
            checkpoint_dir: None,
        }
    }
}