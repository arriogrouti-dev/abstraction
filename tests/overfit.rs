use abstraction::model::{Config, Model};
use abstraction::optim::AdamW;
use abstraction::rng::Rng;
use abstraction::Real;

#[test]
fn initial_loss_is_close_to_ln_vocab() {
    let cfg = Config { vocab: 256, d_model: 64, n_heads: 4, n_layers: 2, d_ff: 256, max_len: 32 };
    let model = Model::new(cfg, 0);
    let mut rng = Rng::new(1);
    let (b, t) = (4usize, 32usize);
    let n = b * t;
    let tokens: Vec<usize> = (0..n).map(|_| rng.below(256)).collect();
    let targets: Vec<usize> = (0..n).map(|_| rng.below(256)).collect();
    let weights: Vec<Real> = vec![1.0; n];

    let loss = model.loss(&tokens, &targets, &weights, b, t);
    let expected = (256f64).ln();
    println!("{} paramètres ; perte initiale {loss:.4} (ln 256 = {expected:.4})", model.n_params());
    assert!((loss as f64 - expected).abs() < 0.1, "perte initiale {loss:.4} loin de ln(256)");
}

#[test]
fn overfit_one_minibatch() {
    let cfg = Config { vocab: 32, d_model: 32, n_heads: 2, n_layers: 2, d_ff: 64, max_len: 12 };
    let mut model = Model::new(cfg.clone(), 1);
    let mut rng = Rng::new(2);

    let (b, t) = (4usize, 12usize);
    let tokens: Vec<usize> = (0..b * t).map(|_| rng.below(cfg.vocab)).collect();
    // cible = token suivant ; poids 0 à la dernière position de chaque séquence
    let mut targets = vec![0usize; b * t];
    let mut weights: Vec<Real> = vec![0.0; b * t];
    for bi in 0..b {
        for ti in 0..t - 1 {
            targets[bi * t + ti] = tokens[bi * t + ti + 1];
            weights[bi * t + ti] = 1.0;
        }
    }

    let mut opt = AdamW::new(5e-3);
    opt.weight_decay = 0.0;

    let (mut first, mut last): (Real, Real) = (0.0, 0.0);
    for step in 0..2000 {
        model.zero_grad();
        let loss = model.loss_and_grad(&tokens, &targets, &weights, b, t);
        model.clip_grad_norm(1.0);
        opt.step(model.params_mut());
        if step == 0 {
            first = loss;
        }
        last = loss;
        if step % 250 == 0 {
            println!("pas {step:4}  perte {loss:.4}");
        }
    }
    println!("perte : {first:.4} -> {last:.4}");
    assert!(last < 0.05, "le modèle n'a pas sur-appris le mini-batch (perte finale {last:.4})");
}