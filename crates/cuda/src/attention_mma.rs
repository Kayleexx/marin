use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::attention::KvView;
use crate::ops::to_u32;
use crate::{Buffer, CudaError, Gpu};

/// The only head size, query positions per block and block size of the tensor-core prompt attention.
pub(crate) const HEAD_DIM: usize = 128;
const QUERY_ROWS: usize = 64;
const THREADS: u32 = 128;

impl Gpu {
    /// Tensor-core attention for a prompt pass over an f16 cache with head size `HEAD_DIM`. The caller has checked
    /// the buffers, the page table and the layout.
    pub(super) fn attention_prefill_mma(
        &self,
        q: &Buffer,
        view: &KvView,
        out: &mut Buffer,
        (heads, kv_heads): (usize, usize),
        (start, tokens): (usize, usize),
    ) -> Result<(), CudaError> {
        let config = LaunchConfig {
            grid_dim: (
                to_u32(tokens.div_ceil(QUERY_ROWS), "attention tiles")?,
                to_u32(heads, "heads")?,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let scale_log2 = std::f32::consts::LOG2_E / (HEAD_DIM as f32).sqrt();
        let (heads, kv_heads, tokens, start) = (
            to_u32(heads, "heads")?,
            to_u32(kv_heads, "kv heads")?,
            to_u32(tokens, "tokens")?,
            to_u32(start, "start position")?,
        );
        self.launch(&self.kernels.attention_prefill_mma, config, |a| {
            a.arg(&q.data)
                .arg(view.pages)
                .arg(&view.key_offset)
                .arg(&view.row_bytes);
            a.arg(&mut out.data);
            a.arg(&heads)
                .arg(&kv_heads)
                .arg(&tokens)
                .arg(&start)
                .arg(&scale_log2);
        })
    }
}
