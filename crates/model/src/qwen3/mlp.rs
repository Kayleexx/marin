use tensor::{Matrix, silu_mul};

use crate::ModelError;

/// The MLP's intermediate results, reused by every layer of a pass.
#[derive(Default)]
pub(crate) struct MlpBuffers {
    gated: Vec<f32>,
    upped: Vec<f32>,
    pub(crate) out: Vec<f32>,
}

fn sized(buffer: &mut Vec<f32>, tokens: usize, width: usize) -> Result<(), ModelError> {
    let size = tokens
        .checked_mul(width)
        .ok_or_else(|| ModelError::InvalidConfig("mlp size overflows".into()))?;
    buffer.resize(size, 0.0);
    Ok(())
}

/// `down(silu(gate(x)) * up(x))` for `tokens` rows of `x`, left in `buffers.out`.
pub(crate) fn mlp(
    x: &[f32],
    tokens: usize,
    (gate, up, down): (&Matrix, &Matrix, &Matrix),
    buffers: &mut MlpBuffers,
) -> Result<(), ModelError> {
    let MlpBuffers { gated, upped, out } = buffers;
    sized(gated, tokens, gate.rows())?;
    sized(upped, tokens, gate.rows())?;
    sized(out, tokens, down.rows())?;
    gate.matmul(x, gated)?;
    up.matmul(x, upped)?;
    silu_mul(gated, upped)?;
    down.matmul(gated, out)?;
    Ok(())
}
