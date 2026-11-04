use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Monde synthétique à structure factorielle : concept = (a, b).
/// Les positions critiques pour a et b sont disjointes des positions de bruit.
/// L'entraînement utilise `train_combos`, l'évaluation compositionnelle utilise
/// `test_combos` (combinaisons tenues hors-jeu).
pub struct FactorWorld {
    pub vocab: usize,
    pub seq_len: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub pat_a: Vec<Vec<u8>>,
    pub pat_b: Vec<Vec<u8>>,
    pub critical_a: Vec<usize>,
    pub critical_b: Vec<usize>,
    pub noise_positions: Vec<usize>,
    pub train_combos: Vec<(usize, usize)>,
    pub test_combos: Vec<(usize, usize)>,
}

impl FactorWorld {
    pub fn new(vocab: usize, seq_len: usize, n_a: usize, n_b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // permutation aléatoire des positions
        let mut perm: Vec<usize> = (0..seq_len).collect();
        for i in (1..seq_len).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        let n_crit_a = (seq_len / 4).max(1);
        let n_crit_b = (seq_len / 4).max(1);
        let critical_a: Vec<usize> = perm[0..n_crit_a].to_vec();
        let critical_b: Vec<usize> = perm[n_crit_a..n_crit_a + n_crit_b].to_vec();
        let noise_positions: Vec<usize> = perm[n_crit_a + n_crit_b..].to_vec();

        let mut pat_a = Vec::new();
        for _ in 0..n_a {
            let mut p = vec![0u8; seq_len];
            for &pos in &critical_a { p[pos] = rng.random_range(0..vocab as u8); }
            pat_a.push(p);
        }
        let mut pat_b = Vec::new();
        for _ in 0..n_b {
            let mut p = vec![0u8; seq_len];
            for &pos in &critical_b { p[pos] = rng.random_range(0..vocab as u8); }
            pat_b.push(p);
        }

        let mut all: Vec<(usize, usize)> = Vec::new();
        for a in 0..n_a { for b in 0..n_b { all.push((a, b)); } }
        for i in (1..all.len()).rev() {
            let j = rng.random_range(0..=i);
            all.swap(i, j);
        }
        let n_test = (all.len() / 5).max(1);
        let test_combos = all[..n_test].to_vec();
        let train_combos = all[n_test..].to_vec();

        Self { vocab, seq_len, n_a, n_b, pat_a, pat_b, critical_a, critical_b,
               noise_positions, train_combos, test_combos }
    }

    pub fn sample(&self, a: usize, b: usize, rng: &mut impl Rng) -> Vec<u8> {
        let mut s = vec![0u8; self.seq_len];
        for &pos in &self.critical_a { s[pos] = self.pat_a[a][pos]; }
        for &pos in &self.critical_b { s[pos] = self.pat_b[b][pos]; }
        for &pos in &self.noise_positions { s[pos] = rng.random_range(0..self.vocab as u8); }
        s
    }

    /// Même concept (a,b), seule la composante de bruit change.
    pub fn augment(&self, stream: &[u8], rng: &mut impl Rng) -> Vec<u8> {
        let mut s = stream.to_vec();
        for &pos in &self.noise_positions {
            if rng.random::<f32>() < 0.5 {
                s[pos] = rng.random_range(0..self.vocab as u8);
            }
        }
        s
    }

    /// Intervention sur une position de bruit — le concept (a,b) doit être préservé.
    pub fn intervene_noncritical(&self, stream: &[u8], rng: &mut impl Rng) -> Vec<u8> {
        let mut s = stream.to_vec();
        if self.noise_positions.is_empty() { return s; }
        let pos = self.noise_positions[rng.random_range(0..self.noise_positions.len())];
        s[pos] = rng.random_range(0..self.vocab as u8);
        s
    }

    /// Intervention critique — change le concept en (a', b) ou (a, b').
    pub fn intervene_critical(&self, stream: &[u8], a: usize, b: usize, rng: &mut impl Rng) -> Vec<u8> {
        let mut s = stream.to_vec();
        let flip_a = rng.random::<bool>();
        if flip_a && self.n_a > 1 {
            let mut a2 = rng.random_range(0..self.n_a);
            while a2 == a { a2 = rng.random_range(0..self.n_a); }
            for &pos in &self.critical_a { s[pos] = self.pat_a[a2][pos]; }
        } else if self.n_b > 1 {
            let mut b2 = rng.random_range(0..self.n_b);
            while b2 == b { b2 = rng.random_range(0..self.n_b); }
            for &pos in &self.critical_b { s[pos] = self.pat_b[b2][pos]; }
        }
        s
    }
}