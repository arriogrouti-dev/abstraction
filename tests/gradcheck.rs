use abstraction::model::{Config, Model};
use abstraction::rng::Rng;
use abstraction::Real;

#[cfg(feature = "f32")]
compile_error!("Les tests de gradient doivent tourner en f64 : lance `cargo test` sans `--features f32`.");

fn get_w(model: &Model, pi: usize, i: usize, j: usize) -> Real {
    model.params()[pi].w[[i, j]]
}

fn set_w(model: &mut Model, pi: usize, i: usize, j: usize, val: Real) {
    let mut ps = model.params_mut();
    ps[pi].w[[i, j]] = val;
}

#[test]
fn gradient_check_all_parameters() {
    let cfg = Config { vocab: 11, d_model: 8, n_heads: 2, n_layers: 2, d_ff: 16, max_len: 6 };
    let mut model = Model::new(cfg.clone(), 42);
    let mut rng = Rng::new(7);

    // Poids « non triviaux » : sinon l'attention est quasi uniforme et beaucoup de gradients valent ~0.
    for p in model.params_mut() {
        let is_gain = p.name.ends_with("ln1") || p.name.ends_with("ln2") || p.name == "lnf";
        for x in p.w.iter_mut() {
            let noise = rng.gauss() as Real * 0.3;
            *x = if is_gain { 1.0 + noise } else { noise };
        }
    }

    let (b, t) = (2usize, 5usize);
    let n = b * t;
    let tokens: Vec<usize> = (0..n).map(|_| rng.below(cfg.vocab)).collect();
    let targets: Vec<usize> = (0..n).map(|_| rng.below(cfg.vocab)).collect();
    // Quelques poids nuls pour tester aussi le masquage de la perte.
    let weights: Vec<Real> = (0..n)
        .map(|i| if i % 4 == 3 { 0.0 } else { 0.5 + rng.uniform() as Real })
        .collect();

    // Gradient analytique
    model.zero_grad();
    let loss = model.loss_and_grad(&tokens, &targets, &weights, b, t);
    println!("perte = {loss:.6}  (ln vocab = {:.6})", (cfg.vocab as f64).ln());
    let grads: Vec<_> = model.params().iter().map(|p| p.g.clone()).collect();
    let names: Vec<String> = model.params().iter().map(|p| p.name.clone()).collect();

    // Gradient numérique (différences centrées) sur TOUTES les coordonnées.
    let h: Real = 1e-5;
    let mut worst: Real = 0.0;
    let mut gmax: Real = 0.0;
    for pi in 0..grads.len() {
        let (rows, cols) = grads[pi].dim();
        let mut p_worst: Real = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let orig = get_w(&model, pi, i, j);
                set_w(&mut model, pi, i, j, orig + h);
                let lp = model.loss(&tokens, &targets, &weights, b, t);
                set_w(&mut model, pi, i, j, orig - h);
                let lm = model.loss(&tokens, &targets, &weights, b, t);
                set_w(&mut model, pi, i, j, orig);

                let numeric = (lp - lm) / (2.0 * h);
                let analytic = grads[pi][[i, j]];
                let err = (analytic - numeric).abs() / (analytic.abs() + numeric.abs()).max(1e-3);
                assert!(err.is_finite(), "NaN/inf dans {} [{i},{j}]", names[pi]);
                p_worst = p_worst.max(err);
                gmax = gmax.max(analytic.abs());
            }
        }
        println!("{:<10} erreur relative max = {:.2e}", names[pi], p_worst);
        worst = worst.max(p_worst);
    }
    assert!(gmax > 1e-3, "gradients trop petits, test peu informatif (gmax = {gmax:.2e})");
    assert!(worst < 1e-5, "GRADIENT CHECK ÉCHOUÉ : erreur relative max = {worst:.2e}");
}