#![forbid(unsafe_code)]

mod error;
mod matrix;
mod ops;
mod quant;

pub use error::TensorError;
pub use matrix::{Matrix, bf16_to_f32};
pub use ops::{add_in_place, dot, ensure_finite, rms_norm_rows, rope, silu_mul, softmax};
pub use quant::{DEFAULT_GROUP, QuantBits, QuantScheme, Quantized, quantize};
