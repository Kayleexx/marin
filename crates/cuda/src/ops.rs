use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{Buffer, CudaError, DeviceMatrix, Gpu, Tokens};

const ELEMENTWISE_BLOCK: u32 = 256;
const NORM_BLOCK: u32 = 256;

pub(crate) fn to_u32(value: usize, what: &'static str) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::Overflow { what })
}

pub(crate) fn expect_len(
    what: &'static str,
    expected: Option<usize>,
    buffer: &Buffer,
) -> Result<(), CudaError> {
    let expected = expected.ok_or(CudaError::Overflow { what })?;
    if buffer.len == expected {
        Ok(())
    } else {
        Err(CudaError::Shape {
            what,
            expected,
            actual: buffer.len,
        })
    }
}

fn rows_of(len: usize, width: usize, what: &'static str) -> Result<usize, CudaError> {
    match (width, len.checked_rem(width)) {
        (w, Some(0)) if w > 0 => len.checked_div(w).ok_or(CudaError::Overflow { what }),
        _ => Err(CudaError::Shape {
            what,
            expected: width,
            actual: len,
        }),
    }
}

/// One block per token.
pub(crate) fn embed_config(tokens: usize) -> Result<LaunchConfig, CudaError> {
    Ok(LaunchConfig {
        grid_dim: (to_u32(tokens, "token count")?, 1, 1),
        block_dim: (NORM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(crate) fn blocks(count: usize, per_block: usize) -> Result<u32, CudaError> {
    to_u32(count.div_ceil(per_block.max(1)), "grid size")
}

impl Gpu {
    /// `out[t] = table[ids[t]]` converted to f32. `ids` must be below `table.rows()`; `scratch` holds them on the GPU.
    pub fn embed(
        &self,
        table: &DeviceMatrix,
        ids: &[u32],
        scratch: &mut Tokens,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        if !self.stage_ids((table.rows, table.cols), ids, scratch, out)? {
            return Ok(());
        }
        let hidden = to_u32(table.cols, "hidden size")?;
        self.launch(&self.kernels.embed, embed_config(ids.len())?, |a| {
            a.arg(&table.data)
                .arg(&scratch.data)
                .arg(&mut out.data)
                .arg(&hidden);
        })
    }

    /// Checks `ids` against a table of `rows` by `cols` and the size of `out`, and copies them into `scratch`. False
    /// when there is nothing to embed.
    pub(crate) fn stage_ids(
        &self,
        (rows, cols): (usize, usize),
        ids: &[u32],
        scratch: &mut Tokens,
        out: &Buffer,
    ) -> Result<bool, CudaError> {
        expect_len("embed output", ids.len().checked_mul(cols), out)?;
        if let Some(bad) = ids
            .iter()
            .find(|id| usize::try_from(**id).map_or(true, |i| i >= rows))
        {
            return Err(CudaError::Shape {
                what: "token id",
                expected: rows,
                actual: *bad as usize,
            });
        }
        if ids.is_empty() {
            return Ok(false);
        }
        if ids.len() > scratch.capacity {
            return Err(CudaError::Shape {
                what: "embed token capacity",
                expected: scratch.capacity,
                actual: ids.len(),
            });
        }
        self.stream
            .memcpy_htod(ids, &mut scratch.data.slice_mut(..ids.len()))?;
        Ok(true)
    }

    /// RMSNorm in place over each `weight.len()`-sized row of `x`.
    pub fn rms_norm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) -> Result<(), CudaError> {
        let dim = weight.len;
        let rows = rows_of(x.len, dim, "rms_norm input")?;
        if rows == 0 {
            return Ok(());
        }
        let config = LaunchConfig {
            grid_dim: (to_u32(rows, "row count")?, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim = to_u32(dim, "norm size")?;
        self.launch(&self.kernels.rms_norm, config, |a| {
            a.arg(&mut x.data).arg(&weight.data).arg(&dim).arg(&eps);
        })
    }

    /// RMSNorm of each `weight.len()`-sized row of `x` into `out`, which must be as long as `x`.
    pub fn rms_norm_into(
        &self,
        x: &Buffer,
        weight: &Buffer,
        out: &mut Buffer,
        eps: f32,
    ) -> Result<(), CudaError> {
        let dim = weight.len;
        let rows = rows_of(x.len, dim, "rms_norm input")?;
        expect_len("rms_norm output", Some(x.len), out)?;
        if rows == 0 {
            return Ok(());
        }
        let config = LaunchConfig {
            grid_dim: (to_u32(rows, "row count")?, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim = to_u32(dim, "norm size")?;
        self.launch(&self.kernels.rms_norm_to, config, |a| {
            a.arg(&x.data)
                .arg(&mut out.data)
                .arg(&weight.data)
                .arg(&dim)
                .arg(&eps);
        })
    }

    /// Rotates every `head_dim` slice of `x` (`[token][head][head_dim]`) by its position.
    pub fn rope(
        &self,
        x: &mut Buffer,
        heads: usize,
        head_dim: usize,
        start: usize,
        theta: f32,
    ) -> Result<(), CudaError> {
        if head_dim == 0 || head_dim.checked_rem(2) != Some(0) || heads == 0 {
            return Err(CudaError::Shape {
                what: "rope head layout",
                expected: 2,
                actual: head_dim,
            });
        }
        let per_token = heads
            .checked_mul(head_dim)
            .ok_or(CudaError::Overflow { what: "rope" })?;
        let head_count = rows_of(x.len, head_dim, "rope input")?;
        rows_of(x.len, per_token, "rope input")?;
        if head_count == 0 {
            return Ok(());
        }
        let config = LaunchConfig {
            grid_dim: (to_u32(head_count, "head count")?, 1, 1),
            block_dim: (
                to_u32(
                    head_dim.checked_div(2).unwrap_or(1).clamp(1, 1024),
                    "block size",
                )?,
                1,
                1,
            ),
            shared_mem_bytes: 0,
        };
        let (heads, head_dim, start) = (
            to_u32(heads, "heads")?,
            to_u32(head_dim, "head size")?,
            to_u32(start, "start position")?,
        );
        self.launch(&self.kernels.rope, config, |a| {
            a.arg(&mut x.data)
                .arg(&heads)
                .arg(&head_dim)
                .arg(&start)
                .arg(&theta);
        })
    }

    /// `gate = silu(gate) * up`.
    pub fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) -> Result<(), CudaError> {
        expect_len("silu_mul operands", Some(gate.len), up)?;
        self.elementwise(&self.kernels.silu_mul, gate, up)
    }

    pub fn add(&self, a: &mut Buffer, b: &Buffer) -> Result<(), CudaError> {
        expect_len("add operands", Some(a.len), b)?;
        self.elementwise(&self.kernels.add, a, b)
    }

    fn elementwise(
        &self,
        function: &cudarc::driver::CudaFunction,
        a: &mut Buffer,
        b: &Buffer,
    ) -> Result<(), CudaError> {
        if a.len == 0 {
            return Ok(());
        }
        let config = LaunchConfig {
            grid_dim: (blocks(a.len, ELEMENTWISE_BLOCK as usize)?, 1, 1),
            block_dim: (ELEMENTWISE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = to_u32(a.len, "element count")?;
        self.launch(function, config, |args| {
            args.arg(&mut a.data).arg(&b.data).arg(&n);
        })
    }
}
