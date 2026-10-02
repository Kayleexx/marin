use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("{what}: expected {expected} elements, got {actual}")]
    Shape {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{what}: size arithmetic overflowed")]
    Overflow { what: &'static str },
    #[error("{what}: contains NaN or infinity")]
    NotFinite { what: &'static str },
    #[error("{what}: head size {size} must be even and non-zero")]
    OddHeadSize { what: &'static str, size: usize },
}
