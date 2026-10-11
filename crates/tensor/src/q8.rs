use std::ops::Range;

use pulp::bytemuck;

use crate::{QuantBits, QuantScheme, TensorError, quantize};
use crate::{dispatch::accelerated, q8_gemv::Q8Gemv, threads};

#[derive(Debug)]
pub(crate) struct Q8 {
    rows: usize,
    cols: usize,
    group: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8 {
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        bits: &[u16],
        scheme: QuantScheme,
    ) -> Result<Self, TensorError> {
        let scheme = QuantScheme::new(scheme.bits, scheme.group)?;
        if scheme.bits != QuantBits::Int8 {
            return Err(TensorError::Quantization(
                "CPU matrix supports INT8 only".into(),
            ));
        }
        let q = quantize(bits, rows, cols, scheme)?;
        Ok(Self {
            rows,
            cols,
            group: scheme.group,
            codes: q.codes,
            scales: q.scales,
        })
    }

    pub(crate) fn row_f32(&self, row: usize, out: &mut [f32]) -> Result<(), TensorError> {
        let source = self
            .codes
            .chunks_exact(self.cols)
            .nth(row)
            .ok_or(TensorError::Shape {
                what: "row index",
                expected: self.rows,
                actual: row,
            })?;
        let groups = self.cols.checked_div(self.group).unwrap_or(1);
        let scales = self
            .scales
            .chunks_exact(groups)
            .nth(row)
            .unwrap_or_default();
        decode(source, scales, self.group, out);
        Ok(())
    }

    pub(crate) fn matmul(&self, x: &[f32], out: &mut [f32]) -> Result<(), TensorError> {
        let tokens = x.len().checked_div(self.cols).unwrap_or(0);
        let workers = if tokens > 1 {
            threads::workers(
                tokens.saturating_mul(self.rows).saturating_mul(self.cols),
                self.rows,
            )
        } else {
            1
        };
        let ranges = threads::ranges(self.rows, workers, 4);
        let lens: Vec<_> = ranges.iter().map(Range::len).collect();
        let parts = threads::split_columns(out, self.rows, &lens);
        let groups = self.cols.checked_div(self.group).unwrap_or(1);
        threads::run_jobs(
            ranges.into_iter().zip(parts).collect(),
            |(range, mut outs)| {
                let codes = self
                    .codes
                    .get(range.start.saturating_mul(self.cols)..range.end.saturating_mul(self.cols))
                    .unwrap_or_default();
                let scales = self
                    .scales
                    .get(range.start.saturating_mul(groups)..range.end.saturating_mul(groups))
                    .unwrap_or_default();
                if tokens == 1 && cfg!(target_endian = "little") {
                    if let Ok(words) = bytemuck::try_cast_slice::<u8, u32>(codes) {
                        let op = Q8Gemv {
                            words,
                            scales,
                            x,
                            out: &mut outs,
                            cols: self.cols,
                            group: self.group,
                        };
                        if accelerated(op).is_ok() {
                            return;
                        }
                    }
                } else if accelerated(crate::q8_gemm::Q8Gemm {
                    codes,
                    scales,
                    x,
                    out: &mut outs,
                    cols: self.cols,
                    group: self.group,
                })
                .is_ok()
                {
                    return;
                }
                let mut row = vec![0.0; self.cols];
                for (r, (codes, scales)) in codes
                    .chunks_exact(self.cols)
                    .zip(scales.chunks_exact(groups))
                    .enumerate()
                {
                    decode(codes, scales, self.group, &mut row);
                    for (input, output) in x.chunks_exact(self.cols).zip(&mut outs) {
                        if let Some(slot) = output.get_mut(r) {
                            *slot = crate::dot(&row, input);
                        }
                    }
                }
            },
        );
        Ok(())
    }
}

pub(crate) fn decode(codes: &[u8], scales: &[f32], group: usize, out: &mut [f32]) {
    for ((codes, values), scale) in codes.chunks(group).zip(out.chunks_mut(group)).zip(scales) {
        for (value, code) in values.iter_mut().zip(codes) {
            *value = f32::from(*code as i8) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_is_bit_identical_across_thread_counts() {
        let scheme = QuantScheme::new(QuantBits::Int8, 32).unwrap();
        let bits: Vec<_> = (0..13 * 96)
            .map(|i| (((i % 101) as f32 - 50.0).to_bits() >> 16) as u16)
            .collect();
        let matrix = crate::Matrix::quantized(13, 96, bits, scheme).unwrap();
        let x: Vec<_> = (0..65 * 96)
            .map(|i| ((i % 71) as f32 - 35.0) / 17.0)
            .collect();
        threads::set_threads(1);
        let mut reference = vec![0.0; 65 * 13];
        matrix.matmul(&x, &mut reference).unwrap();
        for n in [2, 8, 16] {
            threads::set_threads(n);
            let mut got = vec![f32::NAN; reference.len()];
            matrix.matmul(&x, &mut got).unwrap();
            assert_eq!(got, reference);
        }
        threads::set_threads(0);
    }
}
