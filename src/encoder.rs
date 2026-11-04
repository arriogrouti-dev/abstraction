/// Convertit un flux d'octets en états d'entrée. Aucune connaissance du domaine :
/// simple réduction numérique. Les octets au-delà de la capacité sont tronqués.
pub fn encode_bytes(stream: &[u8], n_input: usize, d: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; d]; n_input];
    for i in 0..n_input {
        for k in 0..d {
            let idx = i * d + k;
            let b = if idx < stream.len() { stream[idx] } else { 0 };
            out[i][k] = (b as f32 / 255.0) * 2.0 - 1.0;
        }
    }
    out
}