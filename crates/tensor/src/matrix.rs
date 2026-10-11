use std::ops::Range;

use pulp::bytemuck;

use crate::TensorError;
use crate::dispatch::accelerated;
use crate::gemm::{Gemm, MR};
use crate::gemv::Gemv;
use crate::kernel::Operands;
use crate::threads;

/// Row-major weight matrix of bf16 values, `rows` outputs by `cols` inputs. The kernels read two bf16 as one `u32`
/// (see `kernel`), so a row is `cols.div_ceil(2)` such pairs. The loader's `Vec<u16>` is viewed that way in place
/// when it can be (even `cols`, little-endian, 4-byte aligned); otherwise it is repacked, an odd row ending in a zero.
#[derive(Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Storage,
}

#[derive(Debug)]
enum Storage {
    Bits(Vec<u16>),
    Pairs(Vec<u32>),
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
        let viewable = cfg!(target_endian = "little")
            && cols.is_multiple_of(2)
            && bytemuck::try_cast_slice::<u16, u32>(&data).is_ok();
        let data = if viewable {
            Storage::Bits(data)
        } else {
            Storage::Pairs(pack(rows, cols, data)?)
        };
        Ok(Self { rows, cols, data })
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

    fn words(&self) -> Result<&[u32], TensorError> {
        match &self.data {
            Storage::Pairs(words) => Ok(words),
            Storage::Bits(bits) => bytemuck::try_cast_slice(bits).map_err(|_| TensorError::Shape {
                what: "matrix alignment",
                expected: size_of::<u32>(),
                actual: size_of::<u16>(),
            }),
        }
    }

    pub fn row_f32(&self, row: usize, out: &mut [f32]) -> Result<(), TensorError> {
        if out.len() != self.cols {
            return Err(TensorError::Shape {
                what: "row output",
                expected: self.cols,
                actual: out.len(),
            });
        }
        let source = self
            .words()?
            .chunks_exact(self.pairs().max(1))
            .nth(row)
            .ok_or(TensorError::Shape {
                what: "row index",
                expected: self.rows,
                actual: row,
            })?;
        convert_row(source, out);
        Ok(())
    }

    /// `out[t][r] = dot(self[r], x[t])` for every token row `t` of `x`: a GEMV for one token and a
    /// register-blocked GEMM for more on AVX2 with FMA, the plain row-by-row loop elsewhere. Prompts big enough
    /// to pay for it split the weight rows over threads; the result does not depend on the thread count.
    pub fn matmul(&self, x: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
        let tokens = x.len().checked_div(self.cols).unwrap_or(0);
        check_len("matmul input", tokens.checked_mul(self.cols), x.len())?;
        check_len("matmul output", tokens.checked_mul(self.rows), out.len())?;
        if self.rows == 0 || tokens == 0 || self.cols == 0 {
            out.fill(0.0);
            return Ok(());
        }
        let (words, pairs) = (self.words()?, self.pairs());
        let work = tokens.saturating_mul(self.rows).saturating_mul(self.cols);
        let workers = if tokens > 1 {
            threads::workers(work, self.rows)
        } else {
            1
        };
        let ranges = threads::ranges(self.rows, workers, MR);
        let lens: Vec<usize> = ranges.iter().map(Range::len).collect();
        let parts = threads::split_columns(out, self.rows, &lens);
        threads::run_jobs(
            ranges.into_iter().zip(parts).collect(),
            |(range, mut outs)| {
                let from = range.start.saturating_mul(pairs);
                let weights = words
                    .get(from..range.end.saturating_mul(pairs))
                    .unwrap_or_default();
                multiply(weights, x, &mut outs, self.cols);
            },
        );
        Ok(())
    }
}

/// `weights` (some whole rows) times every token of `x`, into one output slice per token.
fn multiply(weights: &[u32], x: &[f32], out: &mut [&mut [f32]], cols: usize) {
    let operands = Operands {
        weights,
        x,
        out,
        cols,
        pairs: cols.div_ceil(2),
    };
    let unused = if x.len() == cols {
        accelerated(Gemv(operands)).map_err(|op| op.0)
    } else {
        accelerated(Gemm(operands)).map_err(|op| op.0)
    };
    if let Err(operands) = unused {
        multiply_scalar(operands.weights, operands.x, operands.out, cols);
    }
}

/// Each weight row is converted to f32 once and reused across all tokens.
fn multiply_scalar(weights: &[u32], x: &[f32], out: &mut [&mut [f32]], cols: usize) {
    let mut scratch = vec![0.0_f32; cols];
    for (r, row) in weights.chunks_exact(cols.div_ceil(2).max(1)).enumerate() {
        convert_row(row, &mut scratch);
        for (input, output) in x.chunks_exact(cols.max(1)).zip(out.iter_mut()) {
            if let Some(slot) = output.get_mut(r) {
                *slot = crate::dot(&scratch, input);
            }
        }
    }
}

/// Unpacks bf16 pairs into `out`, which may be one short for an odd row.
fn convert_row(pairs: &[u32], out: &mut [f32]) {
    let last = out.len() / 2;
    let mut chunks = out.chunks_exact_mut(2);
    for (dst, pair) in chunks.by_ref().zip(pairs) {
        if let [low, high] = dst {
            *low = bf16_to_f32(*pair as u16);
            *high = bf16_to_f32((*pair >> 16) as u16);
        }
    }
    let rest = chunks.into_remainder();
    if let (Some(slot), Some(pair)) = (rest.first_mut(), pairs.get(last)) {
        *slot = bf16_to_f32(*pair as u16);
    }
}

/// Repacks from the last rows to the first and gives the consumed source back as it goes, so a large tensor is
/// not held twice.
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
