use kv::KvCache;
use loader::ModelDir;
use model::{Precision, Qwen3Config};

use crate::CliError;

#[derive(Debug, Clone, Copy)]
pub struct Footprint {
    pub weights: u64,
    pub parameters: Option<u64>,
    pub recurrent: u64,
    pub quant: Option<tensor::QuantScheme>,
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
    metrics::cpu_available_bytes().ok().flatten()
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

/// Prompts are read in `SLICE` tokens per pass. A pass is admitted again against free memory for a second copy of the
/// state on top of the one held, which `footprint` counts as three windows.
pub fn pass_workspace(c: &model::qwen35::TextConfig, ctx: usize) -> Result<u64, CliError> {
    Ok(c.workspace_bytes(crate::session::prefill::SLICE.min(ctx), ctx)? as u64)
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
    if model::granite::is_granite(&dir.config_json) {
        if dir.weights.packed_group().is_some()
            && dir.weights.packed_group() != precision.weights.map(|q| q.group)
        {
            return Err(CliError::BadOption(
                "Granite artifact and requested INT8 group differ".into(),
            ));
        }
        if precision
            .weights
            .is_some_and(|q| q.bits != model::QuantBits::Int8)
        {
            return Err(CliError::BadOption(
                "Granite CPU supports INT8 weights only".into(),
            ));
        }
        let granite = model::granite::Config::parse(&dir.config_json)?;
        let weights = model::granite::resident_bytes(&dir.weights, &granite, precision.weights)?;
        let kv = KvCache::bytes_for(cfg.layers, cfg.kv_heads, cfg.head_dim, ctx)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| CliError::BadOption("Granite KV size overflows".into()))?;
        let transient = dir
            .weights
            .names()
            .filter_map(|name| dir.weights.info(name))
            .map(|info| info.byte_len())
            .max()
            .unwrap_or(0);
        let needed = weights
            .checked_add(kv)
            .and_then(|n| n.checked_add(model::cpu_workspace_bytes(cfg, ctx).ok()?))
            .and_then(|n| n.checked_add(transient))
            .and_then(|n| n.checked_add(model::CPU_SLACK_BYTES))
            .ok_or_else(|| CliError::BadOption("Granite memory size overflows".into()))?;
        return Ok(Footprint {
            weights,
            parameters: dir
                .weights
                .names()
                .filter(|name| !name.ends_with(".scales"))
                .try_fold(0_u64, |n, name| {
                    let info = dir.weights.info(name)?;
                    let count = info
                        .shape
                        .iter()
                        .try_fold(1_u64, |a, d| a.checked_mul(*d as u64))?;
                    n.checked_add(count)
                }),
            recurrent: 0,
            quant: precision.weights,
            kv,
            kv_window: kv,
            kv_startup: 0,
            needed,
            available: metrics::cpu_available_bytes()?,
        });
    }
    if model::qwen35::is_qwen35(&dir.config_json) {
        let c = model::qwen35::TextConfig::parse(
            &dir.config_json,
            dir.generation_config_json.as_deref(),
        )?;
        let weights = model::qwen35::weight_bytes(&dir.weights, precision.weights)?;
        let kv = c.state_bytes(ctx)? as u64;
        let needed = weights
            .checked_add(kv.saturating_mul(3))
            .and_then(|n| n.checked_add(pass_workspace(&c, ctx).ok()?))
            .and_then(|n| n.checked_add(model::CPU_SLACK_BYTES))
            .ok_or_else(|| CliError::BadOption("hybrid memory overflow".into()))?;
        return Ok(Footprint {
            weights,
            parameters: dir
                .weights
                .names()
                .filter(|n| n.starts_with("model.language_model.") && !n.ends_with(".scales"))
                .try_fold(0_u64, |count, name| {
                    let i = dir.weights.info(name)?;
                    let elements = i
                        .shape
                        .iter()
                        .try_fold(1_u64, |n, x| n.checked_mul(*x as u64))?;
                    count.checked_add(elements)
                }),
            recurrent: c.state_bytes(0)? as u64,
            quant: precision.weights,
            kv,
            kv_window: kv,
            kv_startup: c.state_bytes(0)? as u64,
            needed: needed.max(
                model::qwen35::load_bytes(&dir.weights, &c, precision.weights)?
                    .checked_add(model::CPU_SLACK_BYTES)
                    .ok_or_else(|| CliError::BadOption("load memory overflow".into()))?,
            ),
            available: metrics::cpu_available_bytes()?,
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
        parameters: cfg.parameter_count(),
        recurrent: 0,
        quant: precision.weights,
        kv,
        kv_window,
        kv_startup,
        needed: weights
            .saturating_add(kv)
            .saturating_add(model::CPU_SLACK_BYTES)
            .saturating_add(model::cpu_workspace_bytes(cfg, ctx)?),
        available: metrics::cpu_available_bytes()?,
    })
}

impl Footprint {
    pub fn fit_cpu(
        &mut self,
        cfg: &Qwen3Config,
        ctx: usize,
        adaptive: bool,
    ) -> Result<usize, CliError> {
        self.available = metrics::cpu_available_bytes()?;
        let logits = model::cpu_workspace_bytes(cfg, 0)?;
        let activation = model::cpu_workspace_bytes(cfg, 1)?
            .checked_sub(logits)
            .ok_or_else(|| CliError::BadOption("CPU workspace estimate overflows".into()))?;
        let fixed = self
            .weights
            .checked_add(model::CPU_SLACK_BYTES)
            .and_then(|n| n.checked_add(logits))
            .ok_or_else(|| CliError::BadOption("CPU memory estimate overflows".into()))?;
        let per_position = KvCache::bytes_for(cfg.layers, cfg.kv_heads, cfg.head_dim, 1)
            .and_then(|n| u64::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| CliError::BadOption("CPU KV estimate overflows".into()))?;
        let per_token = per_position
            .checked_add(activation)
            .ok_or_else(|| CliError::BadOption("CPU memory estimate overflows".into()))?;
        let mut chosen = ctx;
        if let (true, Some(available)) = (adaptive, self.available) {
            let fit = available
                .saturating_sub(fixed)
                .checked_div(per_token)
                .unwrap_or(0);
            chosen = chosen.min(usize::try_from(fit).unwrap_or(usize::MAX));
            if chosen == 0 {
                return Err(CliError::NotEnoughMemory {
                    needed: fixed.saturating_add(per_token),
                    available,
                });
            }
        }
        self.kv = per_position
            .checked_mul(
                u64::try_from(chosen)
                    .map_err(|_| CliError::BadOption("CPU context overflows".into()))?,
            )
            .ok_or_else(|| CliError::BadOption("CPU KV estimate overflows".into()))?;
        self.kv_window = self.kv;
        self.kv_startup = 0;
        self.needed = fixed
            .checked_add(self.kv)
            .and_then(|n| {
                activation
                    .checked_mul(u64::try_from(chosen).ok()?)
                    .and_then(|bytes| n.checked_add(bytes))
            })
            .ok_or_else(|| CliError::BadOption("CPU memory estimate overflows".into()))?;
        self.ensure_fits()?;
        Ok(chosen)
    }

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
    fn a_hybrid_pass_is_budgeted_per_slice_not_per_window() {
        let c = model::qwen35::TextConfig::parse(
            include_str!("../../../tests/fixtures/tiny-qwen35/config.json"),
            None,
        )
        .unwrap();
        let slice = crate::session::prefill::SLICE;
        assert_eq!(
            pass_workspace(&c, 4096).unwrap(),
            c.workspace_bytes(slice, 4096).unwrap() as u64
        );
        assert!(pass_workspace(&c, 4096).unwrap() < c.workspace_bytes(4096, 4096).unwrap() as u64);
        assert_eq!(
            pass_workspace(&c, 1).unwrap(),
            c.workspace_bytes(1, 1).unwrap() as u64
        );
    }

    #[test]
    fn cpu_admission_budgets_the_whole_prefill_workspace() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        let dir = ModelDir::open(&path).unwrap();
        let cfg =
            Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref()).unwrap();
        let mut usage = footprint(&dir, &cfg, 1024, Precision::default()).unwrap();
        assert_eq!(
            usage.needed,
            usage.weights
                + usage.kv
                + model::CPU_SLACK_BYTES
                + model::cpu_workspace_bytes(&cfg, 1024).unwrap()
        );
        let mut limited = usage;
        limited.available = Some(512 << 20);
        limited.weights = 1024 << 20;
        limited.needed = limited.weights
            + limited.kv
            + model::CPU_SLACK_BYTES
            + model::cpu_workspace_bytes(&cfg, 1024).unwrap();
        assert!(limited.ensure_fits().is_err());
        assert_eq!(usage.fit_cpu(&cfg, 1024, false).unwrap(), 1024);
        assert_eq!(usage.kv_startup, 0);
        assert_eq!(
            usage.needed,
            usage.weights
                + usage.kv
                + model::CPU_SLACK_BYTES
                + model::cpu_workspace_bytes(&cfg, 1024).unwrap()
        );
        let mut short = footprint(&dir, &cfg, 512, Precision::default()).unwrap();
        short.fit_cpu(&cfg, 512, false).unwrap();
        assert_eq!(
            usage.needed - short.needed,
            usage.kv - short.kv + model::cpu_workspace_bytes(&cfg, 1024).unwrap()
                - model::cpu_workspace_bytes(&cfg, 512).unwrap()
        );
    }

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
