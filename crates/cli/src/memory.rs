use std::fs;

use kv::KvCache;
use loader::ModelDir;
use model::{Precision, Qwen3Config};

use crate::CliError;

/// Extra room for activations and logits on top of weights and the KV cache.
const SLACK_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    pub weights: u64,
    /// The whole `--ctx` window.
    pub kv: u64,
    /// The whole window in the GPU cache's format.
    pub kv_window: u64,
    /// What a GPU cache needs before its first token; the window beyond it is allocated as the conversation grows.
    pub kv_startup: u64,
    pub needed: u64,
    pub available: Option<u64>,
}

pub fn available_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

pub const DEFAULT_CTX: usize = 2048;

pub fn resolve_ctx(
    cfg: &Qwen3Config,
    requested: Option<usize>,
    default: usize,
) -> Result<usize, CliError> {
    match requested {
        Some(0) => Err(CliError::BadOption("--ctx must be above 0".into())),
        Some(ctx) => Ok(ctx),
        None => Ok(default.min(cfg.max_positions)),
    }
}

pub fn footprint(
    dir: &ModelDir,
    cfg: &Qwen3Config,
    ctx: usize,
    precision: Precision,
) -> Result<Footprint, CliError> {
    if ctx > cfg.max_positions {
        return Err(CliError::ContextTooLarge {
            ctx,
            max: cfg.max_positions,
        });
    }
    let kv = KvCache::bytes_for(cfg.layers, cfg.kv_heads, cfg.head_dim, ctx)
        .and_then(|b| u64::try_from(b).ok())
        .ok_or_else(|| CliError::BadOption(format!("--ctx {ctx} is too large")))?;
    let weights = model::weight_bytes(&dir.weights, cfg, precision.weights)?;
    let kv_startup = model::kv_startup_bytes(cfg, ctx, precision.kv).unwrap_or(kv);
    let kv_window = model::kv_window_bytes(cfg, ctx, precision.kv).unwrap_or(kv);
    Ok(Footprint {
        weights,
        kv,
        kv_window,
        kv_startup,
        needed: weights.saturating_add(kv).saturating_add(SLACK_BYTES),
        available: available_bytes(),
    })
}

impl Footprint {
    pub fn ensure_fits(&self) -> Result<(), CliError> {
        match self.available {
            Some(available) if available < self.needed => Err(CliError::NotEnoughMemory {
                needed: self.needed,
                available,
            }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpu_needs_a_slab_to_start_not_the_whole_window() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        let dir = ModelDir::open(&path).unwrap();
        let cfg =
            Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref()).unwrap();
        let long = footprint(&dir, &cfg, 1024, Precision::default()).unwrap();
        assert!(
            long.kv_startup < long.kv,
            "a long window is not reserved at the start"
        );
        let short = footprint(&dir, &cfg, 64, Precision::default()).unwrap();
        assert!(
            short.kv_startup > short.kv,
            "a window under a slab still takes the slab"
        );
    }
}
