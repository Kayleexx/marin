use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::ops::{blocks, expect_len, to_u32};
use crate::{Buffer, CudaError, DeviceMatrix, Gpu};

pub(crate) const WARPS_PER_BLOCK: usize = 4;
pub(crate) const TOKEN_TILE: usize = 8;
pub(crate) const MMA_ROWS: usize = 64;
pub(crate) const MMA_TOKENS: usize = 64;
pub(crate) const MMA_BLOCK: u32 = 128;
/// Fewest tokens for which the tensor-core matmul is used; smaller passes keep the one-warp-per-row kernel.
pub(crate) const MMA_MIN_TOKENS: usize = 16;

impl Gpu {
    /// `out[t][r] = dot(w[r], x[t])` for `tokens` input rows.
    pub fn matmul(
        &self,
        w: &DeviceMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        self.matmul_bf16(w, x, tokens, out, true)
    }

    /// FP32 accumulation without tensor-core rounding for strict hybrid-reference checks.
    pub fn matmul_precise(
        &self,
        w: &DeviceMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        self.matmul_bf16(w, x, tokens, out, false)
    }

    fn matmul_bf16(
        &self,
        w: &DeviceMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
        tensor_cores: bool,
    ) -> Result<(), CudaError> {
        expect_len("matmul input", tokens.checked_mul(w.cols), x)?;
        expect_len("matmul output", tokens.checked_mul(w.rows), out)?;
        if tokens == 0 || w.rows == 0 || w.cols == 0 {
            return Ok(());
        }
        if tensor_cores && tokens >= MMA_MIN_TOKENS && w.cols.checked_rem(16) == Some(0) {
            return self.matmul_mma(w, x, tokens, out);
        }
        let config = LaunchConfig {
            grid_dim: (
                blocks(w.rows, WARPS_PER_BLOCK)?,
                blocks(tokens, TOKEN_TILE)?,
                1,
            ),
            block_dim: (
                to_u32(WARPS_PER_BLOCK.saturating_mul(32), "block size")?,
                1,
                1,
            ),
            shared_mem_bytes: 0,
        };
        let (rows, cols, tokens) = (
            to_u32(w.rows, "rows")?,
            to_u32(w.cols, "cols")?,
            to_u32(tokens, "tokens")?,
        );
        self.launch(&self.kernels.matmul, config, |a| {
            a.arg(&w.data).arg(&x.data).arg(&mut out.data);
            a.arg(&rows).arg(&cols).arg(&tokens);
        })
    }

    /// `matmul` on the tensor cores. The caller has checked the buffers and that `w.cols` is a multiple of 16.
    fn matmul_mma(
        &self,
        w: &DeviceMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        let config = LaunchConfig {
            grid_dim: (blocks(w.rows, MMA_ROWS)?, blocks(tokens, MMA_TOKENS)?, 1),
            block_dim: (MMA_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows, cols, tokens) = (
            to_u32(w.rows, "rows")?,
            to_u32(w.cols, "cols")?,
            to_u32(tokens, "tokens")?,
        );
        self.launch(&self.kernels.matmul_mma, config, |a| {
            a.arg(&w.data).arg(&x.data).arg(&mut out.data);
            a.arg(&rows).arg(&cols).arg(&tokens);
        })
    }

    /// `matmul` of two matrices that share the input: `y0 = w0 * x`, `y1 = w1 * x`.
    pub fn matmul_pair(
        &self,
        (w0, w1): (&DeviceMatrix, &DeviceMatrix),
        x: &Buffer,
        tokens: usize,
        (y0, y1): (&mut Buffer, &mut Buffer),
    ) -> Result<(), CudaError> {
        let plan = grouped_plan(&[w0, w1], &[&*y0, &*y1], x, tokens)?;
        match plan {
            Plan::Nothing => Ok(()),
            Plan::Each => {
                self.matmul(w0, x, tokens, y0)?;
                self.matmul(w1, x, tokens, y1)
            }
            Plan::Fused(config) => {
                let (r0, r1) = (to_u32(w0.rows, "rows")?, to_u32(w1.rows, "rows")?);
                let (cols, tokens) = (to_u32(w0.cols, "cols")?, to_u32(tokens, "tokens")?);
                self.launch(&self.kernels.matmul_x2, config, |a| {
                    a.arg(&w0.data).arg(&w1.data).arg(&x.data);
                    a.arg(&mut y0.data).arg(&mut y1.data);
                    a.arg(&r0).arg(&r1).arg(&cols).arg(&tokens);
                })
            }
        }
    }

    /// `matmul` of three matrices that share the input: `y0 = w0 * x`, `y1 = w1 * x`, `y2 = w2 * x`.
    pub fn matmul_triple(
        &self,
        (w0, w1, w2): (&DeviceMatrix, &DeviceMatrix, &DeviceMatrix),
        x: &Buffer,
        tokens: usize,
        (y0, y1, y2): (&mut Buffer, &mut Buffer, &mut Buffer),
    ) -> Result<(), CudaError> {
        let plan = grouped_plan(&[w0, w1, w2], &[&*y0, &*y1, &*y2], x, tokens)?;
        match plan {
            Plan::Nothing => Ok(()),
            Plan::Each => {
                self.matmul(w0, x, tokens, y0)?;
                self.matmul(w1, x, tokens, y1)?;
                self.matmul(w2, x, tokens, y2)
            }
            Plan::Fused(config) => {
                let rows = (
                    to_u32(w0.rows, "rows")?,
                    to_u32(w1.rows, "rows")?,
                    to_u32(w2.rows, "rows")?,
                );
                let (cols, tokens) = (to_u32(w0.cols, "cols")?, to_u32(tokens, "tokens")?);
                self.launch(&self.kernels.matmul_x3, config, |a| {
                    a.arg(&w0.data).arg(&w1.data).arg(&w2.data).arg(&x.data);
                    a.arg(&mut y0.data).arg(&mut y1.data).arg(&mut y2.data);
                    a.arg(&rows.0)
                        .arg(&rows.1)
                        .arg(&rows.2)
                        .arg(&cols)
                        .arg(&tokens);
                })
            }
        }
    }
}

/// How a group of matrices sharing one input is run.
enum Plan {
    Nothing,
    /// One matmul after another: the tensor-core kernel takes these passes.
    Each,
    Fused(LaunchConfig),
}

fn grouped_plan(
    ws: &[&DeviceMatrix],
    outs: &[&Buffer],
    x: &Buffer,
    tokens: usize,
) -> Result<Plan, CudaError> {
    let cols = ws.first().map_or(0, |w| w.cols);
    let mut total_rows = 0usize;
    for (w, out) in ws.iter().zip(outs) {
        if w.cols != cols {
            return Err(CudaError::Shape {
                what: "grouped matmul columns",
                expected: cols,
                actual: w.cols,
            });
        }
        expect_len("matmul output", tokens.checked_mul(w.rows), out)?;
        total_rows = total_rows.saturating_add(w.rows);
    }
    expect_len("matmul input", tokens.checked_mul(cols), x)?;
    if tokens == 0 || total_rows == 0 || cols == 0 {
        return Ok(Plan::Nothing);
    }
    if tokens >= MMA_MIN_TOKENS && cols.checked_rem(16) == Some(0) {
        return Ok(Plan::Each);
    }
    Ok(Plan::Fused(LaunchConfig {
        grid_dim: (
            blocks(total_rows, WARPS_PER_BLOCK)?,
            blocks(tokens, TOKEN_TILE)?,
            1,
        ),
        block_dim: (
            to_u32(WARPS_PER_BLOCK.saturating_mul(32), "block size")?,
            1,
            1,
        ),
        shared_mem_bytes: 0,
    }))
}
