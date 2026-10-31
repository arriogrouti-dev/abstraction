use ndarray::Array2;
use std::time::Instant;

macro_rules! bench {
    ($t:ty, $name:expr) => {{
        let (n, k, m) = (512usize, 128usize, 128usize);
        let a = Array2::<$t>::from_elem((n, k), 0.5);
        let b = Array2::<$t>::from_elem((k, m), 0.25);
        let reps = 300;
        let t0 = Instant::now();
        let mut sink: $t = 0.0;
        for _ in 0..reps {
            let c = a.dot(&b);
            sink += c[[0, 0]];
        }
        let dt = t0.elapsed().as_secs_f64();
        let gflops = 2.0 * (n * k * m) as f64 * reps as f64 / dt / 1e9;
        println!("{}: {:.1} GFLOP/s (matmul {}x{} · {}x{}) [{}]", $name, gflops, n, k, k, m, sink);
    }};
}

fn main() {
    bench!(f32, "f32");
    bench!(f64, "f64");
}