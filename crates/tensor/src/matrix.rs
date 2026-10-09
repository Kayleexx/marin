use pulp::Arch;

use crate::TensorError;
use crate::gemm::Gemm;
use crate::gemv::Gemv;
use crate::kernel::Operands;

/// Row-major weight matrix of bf16 values, `rows` outputs by `cols` inputs. Each row holds `cols.div_ceil(2)`
/// pairs of bf16 packed in a `u32` (see `kernel`); an odd row ends with a zero.
#[derive(Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u32>,
}

const PACK_BLOCK_BYTES: usize = 8 << 20;

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, TensorError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(TensorError::Overflow { what: "matrix" })?;
        if expected != data.len() {
            return Err(TensorError::Shape {
                what: "matrix data",
                expected,
                actual: data.len(),
            });
        }
        let packed = pack(rows, cols, data)?;
        Ok(Self {
            rows,
            cols,
            data: packed,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    fn pairs(&self) -> usize {
        self.cols.div_ceil(2)
    }

    pub fn row_f32(&self, row: usize, out: &mut [f32]) -> Result<(), TensorError> {
        if out.len() != self.cols {
            return Err(TensorError::Shape {
                what: "row output",
                expected: self.cols,
                actual: out.len(),
            });
        }
        let source =
            self.data
                .chunks_exact(self.pairs().max(1))
                .nth(row)
                .ok_or(TensorError::Shape {
                    what: "row index",
                    expected: self.rows,
                    actual: row,
                })?;
        let halves = source
            .iter()
            .flat_map(|pair| [bf16_to_f32(*pair as u16), bf16_to_f32((*pair >> 16) as u16)]);
        for (dst, value) in out.iter_mut().zip(halves) {
            *dst = value;
        }
        Ok(())
    }

    /// `out[t][r] = dot(self[r], x[t])` for every token row `t` of `x`: a GEMV for one token, a register-blocked
    /// GEMM for more.
    pub fn matmul(&self, x: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
        let tokens = x.len().checked_div(self.cols).unwrap_or(0);
        check_len("matmul input", tokens.checked_mul(self.cols), x.len())?;
        check_len("matmul output", tokens.checked_mul(self.rows), out.len())?;
        if self.rows == 0 || tokens == 0 || self.cols == 0 {
            out.fill(0.0);
            return Ok(());
        }
        let (even, odd) = split_pairs(x, self.cols);
        let operands = Operands {
            weights: &self.data,
            even: &even,
            odd: &odd,
            out,
            rows: self.rows,
            pairs: self.pairs(),
        };
        if tokens == 1 {
            Arch::new().dispatch(Gemv(operands));
        } else {
            Arch::new().dispatch(Gemm(operands));
        }
        Ok(())
    }
}

/// Packs from the last rows to the first and gives the consumed source back as it goes, so a large tensor is
/// never held twice (the embedding alone is 300 to 600 MB).
fn pack(rows: usize, cols: usize, mut data: Vec<u16>) -> Result<Vec<u32>, TensorError> {
    let pairs = cols.div_ceil(2);
    let len = rows
        .checked_mul(pairs)
        .ok_or(TensorError::Overflow { what: "matrix" })?;
    let mut packed = vec![0_u32; len];
    if cols == 0 {
        return Ok(packed);
    }
    let block_rows = PACK_BLOCK_BYTES
        .checked_div(cols.saturating_mul(size_of::<u16>()))
        .unwrap_or(1)
        .max(1);
    for block in packed.rchunks_mut(block_rows.saturating_mul(pairs)) {
        let start = data.len().saturating_sub(block_rows.saturating_mul(cols));
        let source = data.get(start..).unwrap_or_default();
        for (dst, row) in block.chunks_exact_mut(pairs).zip(source.chunks_exact(cols)) {
            for (slot, pair) in dst.iter_mut().zip(row.chunks(2)) {
                let low = pair.first().copied().map_or(0, u32::from);
                let high = pair.get(1).copied().map_or(0, u32::from);
                *slot = low | high << 16;
            }
        }
        data.truncate(start);
        data.shrink_to_fit();
    }
    Ok(packed)
}

/// The even and odd elements of every `cols`-long token row; an odd row gets a zero for its missing last odd.
fn split_pairs(x: &[f32], cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut even = Vec::with_capacity(x.len().div_ceil(2));
    let mut odd = Vec::with_capacity(x.len().div_ceil(2));
    for token in x.chunks_exact(cols) {
        for pair in token.chunks(2) {
            even.push(pair.first().copied().unwrap_or(0.0));
            odd.push(pair.get(1).copied().unwrap_or(0.0));
        }
    }
    (even, odd)
}

fn check_len(
    what: &'static str,
    expected: Option<usize>,
    actual: usize,
) -> Result<(), TensorError> {
    let expected = expected.ok_or(TensorError::Overflow { what })?;
    if expected == actual {
        Ok(())
    } else {
        Err(TensorError::Shape {
            what,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
#[path = "accuracy_tests.rs"]
mod accuracy;

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn matrix(rows: usize, cols: usize, values: &[f32]) -> Matrix {
        Matrix::new(rows, cols, values.iter().map(|v| bf16(*v)).collect()).unwrap()
    }

    #[test]
    fn matmul_covers_every_token_and_row() {
        let w = matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = [1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
        let mut out = [0.0; 4];
        w.matmul(&x, &mut out).unwrap();
        assert_eq!(out, [4.0, 10.0, 4.0, 13.0]);
    }

    #[test]
    fn matmul_rejects_wrong_output_size() {
        let w = matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(w.matmul(&[1.0, 2.0], &mut [0.0; 3]).is_err());
    }

    #[test]
    fn matmul_rejects_ragged_input() {
        let w = matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(w.matmul(&[1.0, 2.0, 3.0], &mut [0.0; 2]).is_err());
    }

    #[test]
    fn new_rejects_wrong_data_length() {
        assert_eq!(
            Matrix::new(2, 2, vec![0; 3]).unwrap_err(),
            TensorError::Shape {
                what: "matrix data",
                expected: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn new_rejects_overflowing_shape() {
        assert!(Matrix::new(usize::MAX, 2, Vec::new()).is_err());
    }

    #[test]
    fn row_f32_converts_one_row() {
        let w = matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 2];
        w.row_f32(1, &mut out).unwrap();
        assert_eq!(out, [3.0, 4.0]);
        assert!(w.row_f32(2, &mut out).is_err());
    }
}
