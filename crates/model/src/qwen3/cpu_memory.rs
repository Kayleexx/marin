use kv::KvCache;

use super::Qwen3Config;
use crate::ModelError;

pub const CPU_SLACK_BYTES: u64 = 256 * 1024 * 1024;
const GROWTH_HEADROOM: u64 = 64 * 1024 * 1024;

/// Named activation buffers and last-token logits; attention packing is budgeted separately.
pub fn cpu_workspace_bytes(cfg: &Qwen3Config, tokens: usize) -> Result<u64, ModelError> {
    let per_token = [
        (cfg.hidden_size, 4_usize),
        (cfg.q_dim()?, 2),
        (cfg.kv_dim()?, 2),
        (cfg.intermediate_size, 2),
    ]
    .into_iter()
    .try_fold(0_usize, |sum, (width, count)| {
        sum.checked_add(width.checked_mul(count)?)
    });
    per_token
        .and_then(|n| n.checked_mul(tokens))
        .and_then(|n| n.checked_add(cfg.vocab_size))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| ModelError::InvalidConfig("CPU workspace size overflows".into()))
}

pub(super) fn ensure_available(needed: u64) -> Result<(), ModelError> {
    if let Some(available) = metrics::cpu_available_bytes().map_err(ModelError::CpuMemoryInfo)?
        && needed > available
    {
        return Err(ModelError::CpuMemory { needed, available });
    }
    Ok(())
}

pub(super) fn before_growth(
    cfg: &Qwen3Config,
    kv: &KvCache,
    end: usize,
    tokens: usize,
) -> Result<(), ModelError> {
    let workspace = cpu_workspace_bytes(cfg, tokens)?;
    let total = KvCache::bytes_for(cfg.layers, cfg.kv_heads, cfg.head_dim, end)
        .ok_or(kv::KvError::Overflow)?;
    let growth = u64::try_from(total.saturating_sub(kv.bytes_initialized()))
        .map_err(|_| kv::KvError::Overflow)?;
    let copy = u64::try_from(kv.growth_copy_bytes(end)?).map_err(|_| kv::KvError::Overflow)?;
    let packing = if tokens > 1 {
        cfg.kv_dim()?
            .checked_mul(
                end.checked_next_multiple_of(64)
                    .and_then(usize::checked_next_power_of_two)
                    .ok_or(kv::KvError::Overflow)?,
            )
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(kv::KvError::Overflow)?
    } else {
        0
    };
    let needed = growth
        .checked_add(copy)
        .and_then(|n| n.checked_add(packing))
        .and_then(|n| n.checked_add(GROWTH_HEADROOM))
        .and_then(|n| n.checked_add(workspace))
        .ok_or(kv::KvError::Overflow)?;
    ensure_available(needed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_workspace_scales_with_the_full_input_and_arithmetic_is_checked() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        let dir = loader::ModelDir::open(&path).unwrap();
        let mut cfg =
            Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref()).unwrap();
        let expected = (4 * cfg.hidden_size
            + 2 * cfg.q_dim().unwrap()
            + 2 * cfg.kv_dim().unwrap()
            + 2 * cfg.intermediate_size)
            * 512
            * 4
            + cfg.vocab_size * 4;
        assert_eq!(cpu_workspace_bytes(&cfg, 512).unwrap(), expected as u64);
        assert_eq!(
            cpu_workspace_bytes(&cfg, 1024).unwrap(),
            (2 * expected - cfg.vocab_size * 4) as u64
        );
        cfg.hidden_size = usize::MAX;
        assert!(cpu_workspace_bytes(&cfg, 512).is_err());
    }
}
