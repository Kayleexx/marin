use cuda::{Buffer, Gpu, Tokens};

use super::config::Qwen3Config;
use crate::ModelError;

/// Every intermediate of one forward pass over `count` tokens (logits kept for `rows` of them), allocated once and reused by all layers. A one-token
/// workspace lives as long as the sequence's cache, so decoding never allocates on the GPU.
pub(super) struct Workspace {
    pub(super) ids: Tokens,
    pub(super) hidden: Buffer,
    pub(super) normed: Buffer,
    pub(super) q: Buffer,
    pub(super) k: Buffer,
    pub(super) v: Buffer,
    pub(super) context: Buffer,
    pub(super) attended: Buffer,
    pub(super) gate: Buffer,
    pub(super) up: Buffer,
    pub(super) ff: Buffer,
    pub(super) last: Buffer,
    pub(super) logits: Buffer,
}

fn scaled(rows: usize, per_row: usize) -> Result<usize, ModelError> {
    rows.checked_mul(per_row)
        .ok_or_else(|| ModelError::InvalidConfig("workspace size overflows".into()))
}

impl Workspace {
    pub(super) fn new(
        gpu: &Gpu,
        cfg: &Qwen3Config,
        count: usize,
        rows: usize,
    ) -> Result<Self, ModelError> {
        let size = |per_token: usize| {
            count
                .checked_mul(per_token)
                .ok_or_else(|| ModelError::InvalidConfig("workspace size overflows".into()))
        };
        let (hidden, q, kv) = (cfg.hidden_size, cfg.q_dim()?, cfg.kv_dim()?);
        Ok(Self {
            ids: gpu.tokens(count)?,
            hidden: gpu.zeros(size(hidden)?)?,
            normed: gpu.zeros(size(hidden)?)?,
            q: gpu.zeros(size(q)?)?,
            k: gpu.zeros(size(kv)?)?,
            v: gpu.zeros(size(kv)?)?,
            context: gpu.zeros(size(q)?)?,
            attended: gpu.zeros(size(hidden)?)?,
            gate: gpu.zeros(size(cfg.intermediate_size)?)?,
            up: gpu.zeros(size(cfg.intermediate_size)?)?,
            ff: gpu.zeros(size(hidden)?)?,
            last: gpu.zeros(rows.saturating_mul(hidden))?,
            logits: gpu.zeros(scaled(rows, cfg.vocab_size)?)?,
        })
    }
}
