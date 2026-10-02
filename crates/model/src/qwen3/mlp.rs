use tensor::{Matrix, silu_mul};

use crate::ModelError;

/// `down(silu(gate(x)) * up(x))` for `tokens` rows of `x`.
pub(crate) fn mlp(
    x: &[f32],
    tokens: usize,
    gate: &Matrix,
    up: &Matrix,
    down: &Matrix,
) -> Result<Vec<f32>, ModelError> {
    let inner = tokens
        .checked_mul(gate.rows())
        .ok_or_else(|| ModelError::InvalidConfig("mlp size overflows".into()))?;
    let outer = tokens
        .checked_mul(down.rows())
        .ok_or_else(|| ModelError::InvalidConfig("mlp size overflows".into()))?;
    let mut gated = vec![0.0; inner];
    let mut upped = vec![0.0; inner];
    gate.matmul(x, &mut gated)?;
    up.matmul(x, &mut upped)?;
    silu_mul(&mut gated, &upped)?;
    let mut out = vec![0.0; outer];
    down.matmul(&gated, &mut out)?;
    Ok(out)
}
