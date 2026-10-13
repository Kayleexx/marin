use crate::ModelError;

pub(super) fn invalid(what: impl Into<String>) -> ModelError {
    ModelError::InvalidConfig(what.into())
}
pub(super) fn size(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("Qwen3.5 size overflow"))
}
pub(super) fn zeros(n: usize) -> Result<Vec<f32>, ModelError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|e| invalid(format!("Qwen3.5 allocation: {e}")))?;
    out.resize(n, 0.0);
    Ok(out)
}
pub(super) fn copy(x: &[f32]) -> Result<Vec<f32>, ModelError> {
    let mut out = zeros(x.len())?;
    out.copy_from_slice(x);
    Ok(out)
}
pub(super) fn slice(x: &[f32], start: usize, len: usize) -> Result<&[f32], ModelError> {
    x.get(
        start
            ..start
                .checked_add(len)
                .ok_or_else(|| invalid("slice overflow"))?,
    )
    .ok_or_else(|| invalid("Qwen3.5 tensor shape"))
}
pub(super) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
pub(super) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
pub(super) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// FP32 reference weights: never round the fixture's FP32 values to BF16.
pub(super) struct Linear {
    data: LinearData,
    cols: usize,
}
enum LinearData {
    F32(Vec<f32>),
    Bf16(tensor::Matrix),
}
impl Linear {
    pub(super) fn new(data: Vec<f32>, cols: usize) -> Self {
        Self {
            data: LinearData::F32(data),
            cols,
        }
    }
    pub(super) fn bf16(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, ModelError> {
        Ok(Self {
            data: LinearData::Bf16(tensor::Matrix::new(rows, cols, data)?),
            cols,
        })
    }
    pub(super) fn is_bf16(&self) -> bool {
        matches!(self.data, LinearData::Bf16(_))
    }
    pub(super) fn apply(&self, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        if x.is_empty() || self.cols == 0 || !x.len().is_multiple_of(self.cols) {
            return Err(invalid("projection input shape"));
        }
        let tokens = x.len().checked_div(self.cols).unwrap_or(0);
        let data = match &self.data {
            LinearData::Bf16(matrix) => {
                let mut out = zeros(size(tokens, matrix.rows())?)?;
                matrix.matmul(x, &mut out)?;
                tensor::ensure_finite(&out, "Qwen3.5 projection")?;
                return Ok(out);
            }
            LinearData::F32(data) => data,
        };
        let rows = data.len().checked_div(self.cols).unwrap_or(0);
        let mut out = zeros(size(tokens, rows)?)?;
        for (input, output) in x.chunks_exact(self.cols).zip(out.chunks_exact_mut(rows)) {
            for (dst, row) in output.iter_mut().zip(data.chunks_exact(self.cols)) {
                *dst = tensor::dot(row, input);
            }
        }
        tensor::ensure_finite(&out, "Qwen3.5 projection")?;
        Ok(out)
    }
}
