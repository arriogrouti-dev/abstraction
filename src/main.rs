mod config;
mod network;
mod encoder;
mod world;
mod losses;
mod trainer;

use config::Config;
use trainer::Trainer;
use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            let mut cfg = Config::default();
            if let Some(p) = args.get(2) { cfg.checkpoint_dir = Some(p.clone()); }
            let mut t = Trainer::new(cfg);
            t.train();
        }
        Some("comprehend") => {
            let model = args.get(2).ok_or("need model path")?;
            let file = args.get(3).ok_or("need input file")?;
            let bytes = std::fs::read(file)?;
            let cfg = Config::default();
            let mut net = network::Network::build(&cfg, &mut rand::rngs::StdRng::seed_from_u64(cfg.seed));
            let loaded: (Vec<f32>, Config) = bincode::deserialize(&std::fs::read(model)?)?;
            let (params, _cfg) = loaded;
            let inp = encoder::encode_bytes(&bytes, net.layout_input_capacity(), cfg.d);
            let f = net.forward(&params, &inp, None, None);
            let rep = net.read_representation(&f.final_states);
            println!("representation (len={}):", rep.len());
            for (i, v) in rep.iter().enumerate() {
                write!(std::io::stdout(), "{:.4} ", v)?;
                if (i + 1) % 8 == 0 { writeln!(std::io::stdout())?; }
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  compnet train [checkpoint_dir]");
            eprintln!("  compnet comprehend <model.bin> <input_file>");
        }
    }
    Ok(())
}