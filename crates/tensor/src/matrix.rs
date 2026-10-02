use crate::TensorError;

/// Row-major weight matrix stored as raw bf16 bits, `rows` outputs by `cols` inputs.
#[derive(Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u16>,
}

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
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
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
            .data
            .chunks_exact(self.cols)
            .nth(row)
            .ok_or(TensorError::Shape {
                what: "row index",
                expected: self.rows,
                actual: row,
            })?;
        convert_row(source, out);
        Ok(())
    }

    /// `out[t][r] = dot(self[r], x[t])` for every token row `t` of `x`.
    /// Each weight row is converted to f32 once and reused across all tokens.
    pub fn matmul(&self, x: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
        let tokens = x.len().checked_div(self.cols).unwrap_or(0);
        check_len("matmul input", tokens.checked_mul(self.cols), x.len())?;
        check_len("matmul output", tokens.checked_mul(self.rows), out.len())?;
        let mut scratch = vec![0.0_f32; self.cols];
        for (r, weights) in self.data.chunks_exact(self.cols).enumerate() {
            convert_row(weights, &mut scratch);
            let inputs = x.chunks_exact(self.cols);
            let outputs = out.chunks_exact_mut(self.rows);
            for (input, output) in inputs.zip(outputs) {
                if let Some(slot) = output.get_mut(r) {
                    *slot = crate::dot(&scratch, input);
                }
            }
        }
        Ok(())
    }
}

fn convert_row(source: &[u16], out: &mut [f32]) {
    for (dst, src) in out.iter_mut().zip(source) {
        *dst = bf16_to_f32(*src);
    }
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
