#![forbid(unsafe_code)]

mod attention;
mod attention_one;
mod attention_simd;
mod dispatch;
mod error;
mod gemm;
mod gemv;
mod kernel;
mod matrix;
mod norm;
mod ops;
mod quant;
mod silu;
mod threads;

pub use attention::{AttentionShape, attention};
pub use error::TensorError;
pub use matrix::{Matrix, bf16_to_f32};
pub use norm::rms_norm_rows;
pub use ops::{add_in_place, dot, ensure_finite, rope_tokens, softmax};
pub use quant::{DEFAULT_GROUP, QuantBits, QuantScheme, Quantized, quantize};
pub use silu::silu_mul;
pub use threads::{physical_cores, set_threads, threads};
