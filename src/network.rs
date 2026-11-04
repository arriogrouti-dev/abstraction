use crate::config::Config;
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug)]
pub enum ActFn { Tanh, Relu, Gelu, Sigmoid }

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

pub struct Node {
    pub bias_off: usize,
    pub act: ActFn,
    pub is_input: bool,
    pub is_output: bool,
}

pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub w_off: usize,
}

pub struct Layout {
    pub bias_offsets: Vec<usize>,
    pub edge_offsets: Vec<usize>,
    pub log_sigma2_off: usize,
    pub total: usize,
}

pub struct Network {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub incoming: Vec<Vec<usize>>,
    pub d: usize,
    pub n_rounds: usize,
    pub layout: Layout,
    pub input_nodes: Vec<usize>,
    pub output_nodes: Vec<usize>,
    pub hidden_nodes: Vec<usize>,
    pub n_input_cfg: usize,
}

pub struct ForwardResult {
    pub final_states: Vec<Vec<f32>>,
}

impl Network {
    pub fn build(cfg: &Config, rng: &mut impl Rng) -> Self {
        let mut nodes = Vec::with_capacity(cfg.n_nodes);
        let mut bias_offsets = Vec::with_capacity(cfg.n_nodes);
        let mut cursor = 0usize;

        for i in 0..cfg.n_nodes {
            bias_offsets.push(cursor);
            cursor += cfg.d;
            let act = match i % 4 {
                0 => ActFn::Tanh,
                1 => ActFn::Gelu,
                2 => ActFn::Sigmoid,
                _ => ActFn::Relu,
            };
            let is_input = i < cfg.n_input;
            let is_output = i >= cfg.n_nodes - cfg.n_output;
            nodes.push(Node { bias_off: bias_offsets[i], act, is_input, is_output });
        }

        let mut edges = Vec::new();
        let mut edge_offsets = Vec::new();
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); cfg.n_nodes];

        for to in 0..cfg.n_nodes {
            if nodes[to].is_input { continue; }
            for from in 0..cfg.n_nodes {
                if from == to || nodes[from].is_output { continue; }
                let dist = (from as i64 - to as i64).unsigned_abs() as usize;
                let p = if dist <= 3 {
                    cfg.edge_prob * (1.0 + cfg.local_edge_frac)
                } else {
                    cfg.edge_prob
                };
                if rng.random::<f32>() < p {
                    let eidx = edges.len();
                    edge_offsets.push(cursor);
                    cursor += cfg.d * cfg.d;
                    edges.push(Edge { from, to, w_off: edge_offsets[eidx] });
                    incoming[to].push(eidx);
                }
            }
        }

        let log_sigma2_off = cursor;
        cursor += 1;

        let layout = Layout { bias_offsets, edge_offsets, log_sigma2_off, total: cursor };
        let input_nodes: Vec<usize> = (0..cfg.n_nodes).filter(|&i| nodes[i].is_input).collect();
        let output_nodes: Vec<usize> = (0..cfg.n_nodes).filter(|&i| nodes[i].is_output).collect();
        let hidden_nodes: Vec<usize> = (0..cfg.n_nodes)
            .filter(|&i| !nodes[i].is_input && !nodes[i].is_output).collect();

        Self { nodes, edges, incoming, d: cfg.d, n_rounds: cfg.n_rounds, layout,
               input_nodes, output_nodes, hidden_nodes, n_input_cfg: cfg.n_input }
    }

    pub fn layout_input_capacity(&self) -> usize { self.n_input_cfg * self.d }

    pub fn init_params(&self, rng: &mut impl Rng) -> Vec<f32> {
        let normal = Normal::new(0.0, 0.05).unwrap();
        let mut p = vec![0.0f32; self.layout.total];
        for &off in &self.layout.bias_offsets {
            for i in 0..self.d { p[off + i] = 0.0; }
        }
        for &off in &self.layout.edge_offsets {
            for i in 0..self.d * self.d {
                p[off + i] = normal.sample(rng);
            }
        }
        p[self.layout.log_sigma2_off] = -4.6;
        p
    }

    pub fn sigma(&self, params: &[f32]) -> f32 {
        let ls = params[self.layout.log_sigma2_off].clamp(-20.0, 5.0);
        (0.5 * ls).exp()
    }

    pub fn sample_noise(&self, sigma: f32, rng: &mut impl Rng) -> Vec<Vec<f32>> {
        let normal = Normal::new(0.0, sigma.max(1e-6)).unwrap();
        let mut v = vec![vec![0.0; self.d]; self.nodes.len()];
        for i in 0..self.nodes.len() {
            if self.nodes[i].is_input { continue; }
            for k in 0..self.d { v[i][k] = normal.sample(rng); }
        }
        v
    }

    pub fn forward(
        &self,
        params: &[f32],
        input: &[Vec<f32>],
        noise: Option<&[Vec<f32>]>,
        interventions: Option<&[(usize, Vec<f32>)]>,
    ) -> ForwardResult {
        let d = self.d;
        let n = self.nodes.len();
        let mut states: Vec<Vec<f32>> = vec![vec![0.0; d]; n];

        for (k, &i) in self.input_nodes.iter().enumerate() {
            if k < input.len() { states[i] = input[k].clone(); }
        }

        for _r in 0..self.n_rounds {
            let mut new_states = vec![vec![0.0f32; d]; n];
            for to in 0..n {
                if self.nodes[to].is_input { continue; }
                let mut acc = vec![0.0f32; d];
                let boff = self.nodes[to].bias_off;
                for i in 0..d { acc[i] = params[boff + i]; }
                for &eidx in &self.incoming[to] {
                    let e = &self.edges[eidx];
                    let src = &states[e.from];
                    let w = &params[e.w_off..e.w_off + d * d];
                    for i in 0..d {
                        let row = &w[i * d..(i + 1) * d];
                        let mut s = 0.0f32;
                        for j in 0..d { s += row[j] * src[j]; }
                        acc[i] += s;
                    }
                }
                for i in 0..d {
                    new_states[to][i] = match self.nodes[to].act {
                        ActFn::Tanh => acc[i].tanh(),
                        ActFn::Relu => acc[i].max(0.0),
                        ActFn::Gelu => gelu(acc[i]),
                        ActFn::Sigmoid => 1.0 / (1.0 + (-acc[i]).exp()),
                    };
                }
            }
            for (k, &i) in self.input_nodes.iter().enumerate() {
                if k < input.len() { new_states[i] = input[k].clone(); }
            }
            if let Some(nz) = noise {
                for i in 0..n {
                    if self.nodes[i].is_input { continue; }
                    for k in 0..d { new_states[i][k] += nz[i][k]; }
                }
            }
            if let Some(ivs) = interventions {
                for (node, val) in ivs { new_states[*node] = val.clone(); }
            }
            states = new_states;
        }

        ForwardResult { final_states: states }
    }

    pub fn read_representation(&self, states: &[Vec<f32>]) -> Vec<f32> {
        let mut rep = Vec::with_capacity((self.hidden_nodes.len() + self.output_nodes.len()) * self.d);
        for &i in &self.hidden_nodes { rep.extend_from_slice(&states[i]); }
        for &i in &self.output_nodes { rep.extend_from_slice(&states[i]); }
        rep
    }
}