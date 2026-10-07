use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use crate::ops::{expect_len, to_u32};
use crate::pages::pages_for;
use crate::{Buffer, CudaError, DeviceKv, Gpu, KvFormat};

pub(crate) const MAX_HEAD_DIM: usize = 256;
const SPLIT_BLOCK: u32 = 128;
/// Most query heads that may share one key/value head in the split decode kernel.
const MAX_GROUP: usize = 8;
const MIN_CHUNK: usize = 32;
const MAX_CHUNK: usize = 256;
/// Most new tokens, and most (token, query head) rows, the shared-read kernel for a few tokens takes.
pub(crate) const MULTI_MAX_TOKENS: usize = 8;
const MULTI_MAX_ROWS: usize = 32;
/// Positions each block of the few-token attention takes: small, so many blocks keep every multiprocessor busy and
/// enough loads in flight (64 beat 128 and 256 in measurements). Shared memory for `MULTI_MAX_ROWS` rows stays under 48 KB.
const MULTI_CHUNK: usize = 64;
/// Blocks per multiprocessor the split aims for, so a long context keeps every one of them busy.
const SPLIT_BLOCKS_PER_SM: usize = 4;

/// Positions each block of the split decode attention takes, and how many blocks that makes along the context.
fn split_plan(visible: usize, kv_heads: usize, sm_count: usize) -> (usize, usize) {
    let wanted = sm_count
        .saturating_mul(SPLIT_BLOCKS_PER_SM)
        .div_ceil(kv_heads.max(1))
        .max(1);
    let chunk = visible.div_ceil(wanted).clamp(MIN_CHUNK, MAX_CHUNK);
    (chunk, visible.div_ceil(chunk))
}

/// Floats in the two scratch buffers the split decode kernels need for any context up to `capacity`; zero when the
/// head layout takes the general kernel instead.
pub(crate) fn scratch_lens(
    (heads, kv_heads, head_dim): (usize, usize, usize),
    capacity: usize,
) -> Option<(usize, usize)> {
    let uses_split = heads
        .checked_div(kv_heads)
        .is_some_and(|group| group <= MAX_GROUP);
    // The few-token kernel writes a partial result per new token as well.
    let splits = if uses_split {
        capacity
            .div_ceil(MIN_CHUNK)
            .max(MULTI_MAX_TOKENS.saturating_mul(capacity.div_ceil(MULTI_CHUNK)))
    } else {
        0
    };
    let len = |per_head: usize| heads.checked_mul(splits)?.checked_mul(per_head);
    Some((len(head_dim)?, len(2)?))
}

pub(crate) fn split_scratch(
    gpu: &Gpu,
    dims: (usize, usize, usize),
    capacity: usize,
) -> Result<(Buffer, Buffer), CudaError> {
    let (acc, ml) = scratch_lens(dims, capacity).ok_or(CudaError::Overflow {
        what: "attention scratch",
    })?;
    Ok((gpu.zeros(acc)?, gpu.zeros(ml)?))
}

/// Where a layer's keys and values are, for the kernels.
struct KvView<'a> {
    pages: &'a CudaSlice<u64>,
    key_offset: u32,
    row_bytes: u32,
    format: usize,
}

fn kernel(set: &[CudaFunction], format: usize) -> Result<&CudaFunction, CudaError> {
    set.get(format).ok_or(CudaError::Shape {
        what: "kv format",
        expected: KvFormat::ALL.len(),
        actual: format,
    })
}

impl Gpu {
    /// Causal grouped-query attention for the `tokens` new tokens at `start..`, reading
    /// cached keys and values of `layer` (which must already include these tokens).
    pub fn attention(
        &self,
        q: &Buffer,
        kv: &mut DeviceKv,
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
        if heads != kv.heads {
            return Err(CudaError::Shape {
                what: "attention heads",
                expected: kv.heads,
                actual: heads,
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
        let key_offset = kv.key_offset(layer)?;
        let view = KvView {
            pages: &kv.page_ptrs,
            key_offset,
            row_bytes: to_u32(kv.row_bytes(), "kv row")?,
            format: kv.format.index(),
        };
        if tokens == 0 || visible == 0 {
            return Ok(());
        }
        // The kernels follow the page table to every position below `visible`.
        if pages_for(visible) > kv.mapped_pages() {
            return Err(CudaError::Shape {
                what: "attention pages",
                expected: pages_for(visible),
                actual: kv.mapped_pages(),
            });
        }
        if tokens == 1
            && heads
                .checked_div(kv_heads)
                .is_some_and(|group| group <= MAX_GROUP)
        {
            return self.attention_split(
                q,
                &view,
                (&mut kv.partial_acc, &mut kv.partial_ml),
                out,
                (heads, kv_heads, head_dim),
                visible,
            );
        }
        let group = heads.checked_div(kv_heads).unwrap_or(1);
        if (2..=MULTI_MAX_TOKENS).contains(&tokens)
            && group <= MAX_GROUP
            && tokens.saturating_mul(group) <= MULTI_MAX_ROWS
        {
            return self.attention_multi(
                q,
                &view,
                (&mut kv.partial_acc, &mut kv.partial_ml),
                out,
                (heads, kv_heads, head_dim),
                (start, tokens),
            );
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
        let function = kernel(&self.kernels.attention, view.format)?;
        self.launch(function, config, |a| {
            a.arg(&q.data)
                .arg(view.pages)
                .arg(&view.key_offset)
                .arg(&view.row_bytes);
            a.arg(&mut out.data);
            a.arg(&heads)
                .arg(&kv_heads)
                .arg(&head_dim)
                .arg(&start)
                .arg(&scale);
        })
    }

    /// Attention for the one newest token, with the context split across blocks and merged afterwards. The caller
    /// has checked the buffers against `visible` positions, that the page table maps every page below `visible`, the
    /// head layout and `heads / kv_heads <= MAX_GROUP`.
    fn attention_split(
        &self,
        q: &Buffer,
        view: &KvView,
        (partial_acc, partial_ml): (&mut Buffer, &mut Buffer),
        out: &mut Buffer,
        (heads, kv_heads, head_dim): (usize, usize, usize),
        visible: usize,
    ) -> Result<(), CudaError> {
        let group = heads.checked_div(kv_heads).unwrap_or(1);
        let (chunk, splits) = split_plan(visible, kv_heads, self.sm_count);
        let needed = |per_head: usize| heads.saturating_mul(splits).saturating_mul(per_head);
        for (what, buffer, per_head) in [
            ("attention partial sums", &*partial_acc, head_dim),
            ("attention partial maxima", &*partial_ml, 2),
        ] {
            if buffer.len < needed(per_head) {
                return Err(CudaError::Shape {
                    what,
                    expected: needed(per_head),
                    actual: buffer.len,
                });
            }
        }
        let shared = group
            .saturating_mul(head_dim.saturating_add(chunk))
            .saturating_mul(size_of::<f32>());
        let partial = LaunchConfig {
            grid_dim: (
                to_u32(splits, "attention splits")?,
                to_u32(kv_heads, "kv heads")?,
                1,
            ),
            block_dim: (SPLIT_BLOCK, 1, 1),
            shared_mem_bytes: to_u32(shared, "shared memory")?,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (heads_u, kv_heads_u, head_dim_u, visible_u, chunk_u, splits_u) = (
            to_u32(heads, "heads")?,
            to_u32(kv_heads, "kv heads")?,
            to_u32(head_dim, "head size")?,
            to_u32(visible, "visible positions")?,
            to_u32(chunk, "attention chunk")?,
            to_u32(splits, "attention splits")?,
        );
        let function = kernel(&self.kernels.attention_partial, view.format)?;
        self.launch(function, partial, |a| {
            a.arg(&q.data)
                .arg(view.pages)
                .arg(&view.key_offset)
                .arg(&view.row_bytes);
            a.arg(&mut partial_acc.data).arg(&mut partial_ml.data);
            a.arg(&heads_u).arg(&kv_heads_u).arg(&head_dim_u);
            a.arg(&visible_u).arg(&chunk_u).arg(&splits_u).arg(&scale);
        })?;
        let merge = LaunchConfig {
            grid_dim: (heads_u, 1, 1),
            block_dim: (to_u32(head_dim.next_multiple_of(32), "block size")?, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(&self.kernels.attention_merge, merge, |a| {
            a.arg(&partial_acc.data)
                .arg(&partial_ml.data)
                .arg(&mut out.data);
            a.arg(&head_dim_u).arg(&splits_u);
        })
    }

    /// Attention for `tokens` (2 to `MULTI_MAX_TOKENS`) new tokens after `start` cached positions, with the context
    /// split across blocks and every key and value row read once for all of them, then merged. The caller has checked
    /// the buffers, that the page table maps every page below `start + tokens`, the head layout and the number of
    /// rows.
    fn attention_multi(
        &self,
        q: &Buffer,
        view: &KvView,
        (partial_acc, partial_ml): (&mut Buffer, &mut Buffer),
        out: &mut Buffer,
        (heads, kv_heads, head_dim): (usize, usize, usize),
        (start, tokens): (usize, usize),
    ) -> Result<(), CudaError> {
        let group = heads.checked_div(kv_heads).unwrap_or(1);
        let rows = tokens.saturating_mul(group);
        let visible = start.saturating_add(tokens);
        let (chunk, splits) = (MULTI_CHUNK, visible.div_ceil(MULTI_CHUNK));
        let needed = |per_head: usize| {
            tokens
                .saturating_mul(heads)
                .saturating_mul(splits)
                .saturating_mul(per_head)
        };
        for (what, buffer, per_head) in [
            ("attention partial sums", &*partial_acc, head_dim),
            ("attention partial maxima", &*partial_ml, 2),
        ] {
            if buffer.len < needed(per_head) {
                return Err(CudaError::Shape {
                    what,
                    expected: needed(per_head),
                    actual: buffer.len,
                });
            }
        }
        let shared = rows
            .saturating_mul(head_dim.saturating_add(chunk))
            .saturating_mul(size_of::<f32>());
        let partial = LaunchConfig {
            grid_dim: (
                to_u32(splits, "attention splits")?,
                to_u32(kv_heads, "kv heads")?,
                1,
            ),
            block_dim: (SPLIT_BLOCK, 1, 1),
            shared_mem_bytes: to_u32(shared, "shared memory")?,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (heads_u, kv_heads_u, head_dim_u, start_u, tokens_u, chunk_u, splits_u) = (
            to_u32(heads, "heads")?,
            to_u32(kv_heads, "kv heads")?,
            to_u32(head_dim, "head size")?,
            to_u32(start, "start position")?,
            to_u32(tokens, "tokens")?,
            to_u32(chunk, "attention chunk")?,
            to_u32(splits, "attention splits")?,
        );
        let variant = match rows {
            0..=8 => 0,
            9..=16 => 1,
            _ => 2,
        };
        let set =
            self.kernels
                .attention_multi_partial
                .get(variant)
                .ok_or(CudaError::Unsupported {
                    what: "attention rows",
                    max: MULTI_MAX_ROWS,
                    actual: rows,
                })?;
        self.launch(kernel(set, view.format)?, partial, |a| {
            a.arg(&q.data)
                .arg(view.pages)
                .arg(&view.key_offset)
                .arg(&view.row_bytes);
            a.arg(&mut partial_acc.data).arg(&mut partial_ml.data);
            a.arg(&heads_u).arg(&kv_heads_u).arg(&head_dim_u);
            a.arg(&start_u).arg(&tokens_u).arg(&chunk_u).arg(&splits_u);
            a.arg(&scale);
        })?;
        let merge = LaunchConfig {
            grid_dim: (
                to_u32(tokens.saturating_mul(heads), "attention rows")?,
                1,
                1,
            ),
            block_dim: (to_u32(head_dim.next_multiple_of(32), "block size")?, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(&self.kernels.attention_merge, merge, |a| {
            a.arg(&partial_acc.data)
                .arg(&partial_ml.data)
                .arg(&mut out.data);
            a.arg(&head_dim_u).arg(&splits_u);
        })
    }
}
