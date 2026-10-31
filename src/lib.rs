//! Moteur d'entraînement byte-level, backward écrit à la main (ndarray).

pub mod model;
pub mod optim;
pub mod rng;

/// Type flottant du moteur : f64 par défaut, f32 avec `--features f32`.
#[cfg(feature = "f32")]
pub type Real = f32;
#[cfg(not(feature = "f32"))]
pub type Real = f64;