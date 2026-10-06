use std::time::Instant;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use tensor::{QuantBits, QuantScheme, Quantized};

use crate::matmul::{MMA_BLOCK, MMA_MIN_TOKENS, MMA_ROWS, MMA_TOKENS, TOKEN_TILE, WARPS_PER_BLOCK};
use crate::ops::{blocks, embed_config, expect_len, to_u32};
use crate::{Buffer, CudaError, Gpu, Tokens};

/// Rows each warp takes in the one-token kernels; matches `ROWS_PER_WARP` in quant.cu.
const ROWS_PER_WARP: usize = 4;

/// A row-major weight matrix in GPU memory as int8 or int4 codes with one f32 scale per group of columns.
pub struct DeviceQuantMatrix {
    rows: usize,
    cols: usize,
    scheme: QuantScheme,
    codes: CudaSlice<u8>,
    scales: CudaSlice<f32>,
}

impl DeviceQuantMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }
}

fn shape(what: &'static str, expected: usize, actual: usize) -> CudaError {
    CudaError::Shape {
        what,
        expected,
        actual,
    }
}

impl Gpu {
    pub fn upload_quantized(&self, q: &Quantized) -> Result<DeviceQuantMatrix, CudaError> {
        let overflow = CudaError::Overflow {
            what: "quantized matrix",
        };
        let group = q.scheme.group;
        if group == 0 || group.checked_rem(32) != Some(0) || q.cols.checked_rem(group) != Some(0) {
            return Err(shape("quantization group", group, q.cols));
        }
        let codes = q
            .rows
            .checked_mul(q.scheme.row_bytes(q.cols))
            .ok_or(CudaError::Overflow {
                what: "quantized codes",
            })?;
        let scales = q
            .rows
            .checked_mul(q.cols.checked_div(group).ok_or(overflow)?)
            .ok_or(CudaError::Overflow {
                what: "quantization scales",
            })?;
        if q.codes.len() != codes {
            return Err(shape("quantized codes", codes, q.codes.len()));
        }
        if q.scales.len() != scales {
            return Err(shape("quantization scales", scales, q.scales.len()));
        }
        let started = Instant::now();
        let device_codes = self.stream.clone_htod(&q.codes)?;
        let device_scales = self.stream.clone_htod(&q.scales)?;
        let bytes = q
            .codes
            .len()
            .saturating_add(q.scales.len().saturating_mul(size_of::<f32>()));
        self.counters.upload(bytes, started);
        Ok(DeviceQuantMatrix {
            rows: q.rows,
            cols: q.cols,
            scheme: q.scheme,
            codes: device_codes,
            scales: device_scales,
        })
    }

    /// `out[t] = table[ids[t]]` dequantized to f32, for an int8 table. `ids` must be below `table.rows()`.
    pub fn embed_quant(
        &self,
        table: &DeviceQuantMatrix,
        ids: &[u32],
        scratch: &mut Tokens,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        if table.scheme.bits != QuantBits::Int8 {
            return Err(CudaError::Unsupported {
                what: "quantized embedding width in bits",
                max: 8,
                actual: 4,
            });
        }
        if !self.stage_ids((table.rows, table.cols), ids, scratch, out)? {
            return Ok(());
        }
        let (hidden, group) = (
            to_u32(table.cols, "hidden size")?,
            to_u32(table.scheme.group, "group size")?,
        );
        self.launch(&self.kernels.embed_q8, embed_config(ids.len())?, |a| {
            a.arg(&table.codes)
                .arg(&table.scales)
                .arg(&scratch.data)
                .arg(&mut out.data);
            a.arg(&hidden).arg(&group);
        })
    }

    /// `out[t][r] = dot(w[r], x[t])` for quantized `w`; passes of 16 or more tokens use the tensor cores.
    pub fn matmul_quant(
        &self,
        w: &DeviceQuantMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        expect_len("matmul input", tokens.checked_mul(w.cols), x)?;
        expect_len("matmul output", tokens.checked_mul(w.rows), out)?;
        if tokens == 0 || w.rows == 0 || w.cols == 0 {
            return Ok(());
        }
        if tokens == 1 {
            return self.matvec_quant(w, x, out);
        }
        let tensor_cores = tokens >= MMA_MIN_TOKENS;
        let k = &self.kernels;
        let function = match (w.scheme.bits, tensor_cores) {
            (QuantBits::Int8, false) => &k.matmul_q8,
            (QuantBits::Int4, false) => &k.matmul_q4,
            (QuantBits::Int8, true) => &k.matmul_q8_mma,
            (QuantBits::Int4, true) => &k.matmul_q4_mma,
        };
        let (rows_per_block, tokens_per_block, threads) = if tensor_cores {
            (MMA_ROWS, MMA_TOKENS, MMA_BLOCK)
        } else {
            (
                WARPS_PER_BLOCK,
                TOKEN_TILE,
                to_u32(WARPS_PER_BLOCK.saturating_mul(32), "block size")?,
            )
        };
        let config = LaunchConfig {
            grid_dim: (
                blocks(w.rows, rows_per_block)?,
                blocks(tokens, tokens_per_block)?,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rows, cols, tokens, group) = (
            to_u32(w.rows, "rows")?,
            to_u32(w.cols, "cols")?,
            to_u32(tokens, "tokens")?,
            to_u32(w.scheme.group, "group size")?,
        );
        self.launch(function, config, |a| {
            a.arg(&w.codes)
                .arg(&w.scales)
                .arg(&x.data)
                .arg(&mut out.data);
            a.arg(&rows).arg(&cols).arg(&tokens).arg(&group);
        })
    }

    /// One token: each warp takes `ROWS_PER_WARP` rows. The caller has checked the buffer sizes.
    fn matvec_quant(
        &self,
        w: &DeviceQuantMatrix,
        x: &Buffer,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        let function = match w.scheme.bits {
            QuantBits::Int8 => &self.kernels.matvec_q8,
            QuantBits::Int4 => &self.kernels.matvec_q4,
        };
        let rows_per_block = WARPS_PER_BLOCK.saturating_mul(ROWS_PER_WARP);
        let config = LaunchConfig {
            grid_dim: (blocks(w.rows, rows_per_block)?, 1, 1),
            block_dim: (
                to_u32(WARPS_PER_BLOCK.saturating_mul(32), "block size")?,
                1,
                1,
            ),
            shared_mem_bytes: 0,
        };
        let (rows, cols, group) = (
            to_u32(w.rows, "rows")?,
            to_u32(w.cols, "cols")?,
            to_u32(w.scheme.group, "group size")?,
        );
        self.launch(function, config, |a| {
            a.arg(&w.codes)
                .arg(&w.scales)
                .arg(&x.data)
                .arg(&mut out.data);
            a.arg(&rows).arg(&cols).arg(&group);
        })
    }
}
