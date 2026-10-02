use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{Buffer, CudaError, DeviceKv, DeviceMatrix, Gpu};

const WARPS_PER_BLOCK: usize = 4;
const TOKEN_TILE: usize = 8;
const ELEMENTWISE_BLOCK: u32 = 256;
const NORM_BLOCK: u32 = 256;
const MAX_HEAD_DIM: usize = 256;

fn to_u32(value: usize, what: &'static str) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::Overflow { what })
}

fn expect_len(
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

fn blocks(count: usize, per_block: usize) -> Result<u32, CudaError> {
    to_u32(count.div_ceil(per_block.max(1)), "grid size")
}

impl Gpu {
    /// `out[t] = table[ids[t]]` converted to f32. `ids` must be below `table.rows()`.
    pub fn embed(
        &self,
        table: &DeviceMatrix,
        ids: &[u32],
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        expect_len("embed output", ids.len().checked_mul(table.cols), out)?;
        if let Some(bad) = ids
            .iter()
            .find(|id| usize::try_from(**id).map_or(true, |i| i >= table.rows))
        {
            return Err(CudaError::Shape {
                what: "token id",
                expected: table.rows,
                actual: *bad as usize,
            });
        }
        if ids.is_empty() {
            return Ok(());
        }
        let device_ids = self.stream.clone_htod(ids)?;
        let config = LaunchConfig {
            grid_dim: (to_u32(ids.len(), "token count")?, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden = to_u32(table.cols, "hidden size")?;
        self.launch(&self.kernels.embed, config, |a| {
            a.arg(&table.data)
                .arg(&device_ids)
                .arg(&mut out.data)
                .arg(&hidden);
        })
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

    /// `out[t][r] = dot(w[r], x[t])` for `tokens` input rows.
    pub fn matmul(
        &self,
        w: &DeviceMatrix,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        expect_len("matmul input", tokens.checked_mul(w.cols), x)?;
        expect_len("matmul output", tokens.checked_mul(w.rows), out)?;
        if tokens == 0 || w.rows == 0 || w.cols == 0 {
            return Ok(());
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

    /// Causal grouped-query attention for the `tokens` new tokens at `start..`, reading
    /// cached keys and values of `layer` (which must already include these tokens).
    pub fn attention(
        &self,
        q: &Buffer,
        kv: &DeviceKv,
        layer: usize,
        heads: usize,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        let (head_dim, kv_heads, start) = (kv.head_dim, kv.kv_heads, kv.len);
        if head_dim == 0
            || head_dim > MAX_HEAD_DIM
            || kv_heads == 0
            || heads == 0
            || heads.checked_rem(kv_heads) != Some(0)
        {
            return Err(CudaError::Unsupported {
                what: "head layout",
                max: MAX_HEAD_DIM,
                actual: head_dim,
            });
        }
        let q_len = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(head_dim));
        expect_len("attention queries", q_len, q)?;
        expect_len("attention output", q_len, out)?;
        let visible = start.checked_add(tokens).filter(|n| *n <= kv.capacity);
        let Some(visible) = visible else {
            return Err(CudaError::Shape {
                what: "attention positions",
                expected: kv.capacity,
                actual: start.saturating_add(tokens),
            });
        };
        let layers = kv.keys.len();
        let (Some(keys), Some(values)) = (kv.keys.get(layer), kv.values.get(layer)) else {
            return Err(CudaError::Shape {
                what: "attention layer",
                expected: layers,
                actual: layer,
            });
        };
        if tokens == 0 || visible == 0 {
            return Ok(());
        }
        let threads = head_dim.next_multiple_of(32);
        let shared_floats = head_dim.saturating_add(threads).saturating_add(32);
        let config = LaunchConfig {
            grid_dim: (
                to_u32(tokens.saturating_mul(heads), "attention blocks")?,
                1,
                1,
            ),
            block_dim: (to_u32(threads, "block size")?, 1, 1),
            shared_mem_bytes: to_u32(
                shared_floats.saturating_mul(size_of::<f32>()),
                "shared memory",
            )?,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (heads, kv_heads, head_dim, start) = (
            to_u32(heads, "heads")?,
            to_u32(kv_heads, "kv heads")?,
            to_u32(head_dim, "head size")?,
            to_u32(start, "start position")?,
        );
        self.launch(&self.kernels.attention, config, |a| {
            a.arg(&q.data).arg(keys).arg(values).arg(&mut out.data);
            a.arg(&heads)
                .arg(&kv_heads)
                .arg(&head_dim)
                .arg(&start)
                .arg(&scale);
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
