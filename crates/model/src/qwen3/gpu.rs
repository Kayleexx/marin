use cuda::{Buffer, DeviceKv, Gpu, KvFormat};
use loader::ModelDir;
use metrics::{Stage, StageTimes};
use runtime::{CausalLm, RollbackError, SequenceCache};
use tensor::{QuantScheme, quantize};

use super::Observer;
use super::config::Qwen3Config;
use super::gpu_probe::GpuProbe;
use super::weight::{Weight, matmul_pair, matmul_triple};
use super::weights::{Qwen3Weights, Store};
use super::workspace::Workspace;
use crate::ModelError;

struct GpuStore<'a> {
    gpu: &'a Gpu,
    quant: Option<QuantScheme>,
}

impl Store for GpuStore<'_> {
    type Matrix = Weight;
    type Vector = Buffer;

    /// The embedding table and the output head: int8 whenever quantization is on, whatever the layers use, because a
    /// 4-bit head costs far more accuracy than 4-bit layers and the table is a fifth of a decode step.
    fn matrix(&mut self, rows: usize, cols: usize, bits: Vec<u16>) -> Result<Weight, ModelError> {
        let Some(scheme) = self.quant else {
            return Ok(Weight::Bf16(self.gpu.upload_matrix(rows, cols, &bits)?));
        };
        let quantized = quantize(&bits, rows, cols, scheme.table())?;
        Ok(Weight::Quant(self.gpu.upload_quantized(&quantized)?))
    }

    fn projection(
        &mut self,
        rows: usize,
        cols: usize,
        bits: Vec<u16>,
    ) -> Result<Weight, ModelError> {
        let Some(scheme) = self.quant else {
            return self.matrix(rows, cols, bits);
        };
        let quantized = quantize(&bits, rows, cols, scheme)?;
        Ok(Weight::Quant(self.gpu.upload_quantized(&quantized)?))
    }

    fn vector(&mut self, values: Vec<f32>) -> Result<Buffer, ModelError> {
        Ok(self.gpu.upload(&values)?)
    }
}

/// Attention memory on the GPU, plus the one-token workspace decoding reuses.
pub struct GpuCache {
    kv: DeviceKv,
    decode: Workspace,
    /// Workspaces for `forward_rows`, made the first time a size is used: slot `n - 1` serves `n` tokens.
    verify: Vec<Option<Workspace>>,
}

impl GpuCache {
    /// GPU pages the sequence has needed so far, and the memory they take.
    pub fn kv_held(&self) -> (usize, usize) {
        (self.kv.pages_held(), self.kv.bytes_held())
    }
}

impl SequenceCache for GpuCache {
    fn used(&self) -> usize {
        self.kv.len()
    }

    fn capacity(&self) -> usize {
        self.kv.capacity()
    }

    fn clear(&mut self) {
        self.kv.clear();
    }

    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        self.kv.truncate(len).map_err(|_| RollbackError {
            len,
            held: self.kv.len(),
        })
    }
}

/// GPU memory the cache needs before the first token for a window of `capacity` positions; the rest is allocated as the
/// sequence grows.
pub fn kv_startup_bytes(config: &Qwen3Config, capacity: usize, format: KvFormat) -> Option<u64> {
    let dims = (config.heads, config.kv_heads, config.head_dim);
    DeviceKv::startup_bytes(config.layers, dims, capacity, format)
        .and_then(|bytes| u64::try_from(bytes).ok())
}

/// GPU memory the cache takes once a sequence fills the whole window of `capacity` positions.
pub fn kv_window_bytes(config: &Qwen3Config, capacity: usize, format: KvFormat) -> Option<u64> {
    let row = format.row_bytes(config.kv_heads, config.head_dim)?;
    let bytes = row
        .checked_mul(capacity)?
        .checked_mul(config.layers.checked_mul(2)?)?;
    u64::try_from(bytes).ok()
}

/// Most tokens `forward_rows` takes in one pass.
pub const MAX_ROWS: usize = 8;

/// Longest run of prompt tokens handled in one pass, which bounds the workspace (about 100 KB per token) however long
/// the prompt is. Every token's result is independent of how the prompt is split.
const PREFILL_CHUNK: usize = 512;

pub struct Qwen3Gpu {
    config: Qwen3Config,
    weights: Qwen3Weights<Weight, Buffer>,
    gpu: Gpu,
    quant: Option<QuantScheme>,
    kv_format: KvFormat,
}

impl Qwen3Gpu {
    pub fn load(
        dir: &ModelDir,
        gpu: Gpu,
        quant: Option<QuantScheme>,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        let mut store = GpuStore { gpu: &gpu, quant };
        let weights = Qwen3Weights::load(&dir.weights, &config, &mut store, progress)?;
        gpu.synchronize()?;
        Ok(Self {
            config,
            weights,
            gpu,
            quant,
            kv_format: KvFormat::default(),
        })
    }

    /// Caches made from now on store keys and values in `format`.
    pub fn with_kv_format(mut self, format: KvFormat) -> Self {
        self.kv_format = format;
        self
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv_format
    }

    pub fn quantization(&self) -> Option<QuantScheme> {
        self.quant
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn new_cache(&self, capacity: usize) -> Result<GpuCache, ModelError> {
        let cfg = &self.config;
        Ok(GpuCache {
            kv: DeviceKv::new(
                &self.gpu,
                cfg.layers,
                (cfg.heads, cfg.kv_heads, cfg.head_dim),
                capacity,
                self.kv_format,
            )?,
            decode: Workspace::new(&self.gpu, cfg, 1, 1)?,
            verify: (0..MAX_ROWS).map(|_| None).collect(),
        })
    }

    /// Same contract as the CPU model's `forward_observed`; observing copies one row per layer back to the host.
    pub fn forward_observed(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_with(tokens, cache, observe, None)
    }

    /// `forward_observed` that also times the stages of the pass when `times` is given.
    pub fn forward_with(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        observe: Observer<'_>,
        times: Option<&mut StageTimes>,
    ) -> Result<Vec<f32>, ModelError> {
        self.pass(tokens, cache, observe, times, false)
    }

    /// Like `forward`, but returns the logits after every token fed, one vocabulary-sized row each, so the
    /// tokens can be checked against what the model would have chosen. At most `MAX_ROWS` tokens.
    pub fn forward_rows(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
    ) -> Result<Vec<f32>, ModelError> {
        self.pass(tokens, cache, None, None, true)
    }

    fn pass(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        mut observe: Observer<'_>,
        times: Option<&mut StageTimes>,
        all_rows: bool,
    ) -> Result<Vec<f32>, ModelError> {
        if all_rows && tokens.len() > MAX_ROWS {
            return Err(ModelError::Unsupported(format!(
                "forward_rows takes at most {MAX_ROWS} tokens, got {}",
                tokens.len()
            )));
        }
        if tokens.len() > PREFILL_CHUNK {
            let mut logits = Vec::new();
            let mut chunks = tokens.chunks(PREFILL_CHUNK).peekable();
            let mut times = times;
            while let Some(chunk) = chunks.next() {
                let watcher = if chunks.peek().is_none() {
                    observe.take()
                } else {
                    None
                };
                logits = self.pass(chunk, cache, watcher, times.as_deref_mut(), false)?;
            }
            return Ok(logits);
        }
        let mut probe = GpuProbe::new(&self.gpu, times);
        let (cfg, gpu) = (&self.config, &self.gpu);
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        let count = tokens.len();
        let hidden_size = cfg.hidden_size;
        let first_row = if all_rows { 0 } else { count.saturating_sub(1) };
        let last_row_start = count.saturating_sub(1).saturating_mul(hidden_size);
        if let Some(bad) = tokens
            .iter()
            .find(|t| usize::try_from(**t).map_or(true, |t| t >= cfg.vocab_size))
        {
            return Err(ModelError::TokenOutOfRange {
                token: *bad,
                vocab: cfg.vocab_size,
            });
        }
        let GpuCache { kv, decode, verify } = cache;
        let mut longer;
        let ws = if all_rows {
            let slot = verify
                .get_mut(count.saturating_sub(1))
                .ok_or_else(|| ModelError::Unsupported("no workspace for this size".into()))?;
            if slot.is_none() {
                *slot = Some(Workspace::new(gpu, cfg, count, count)?);
            }
            slot.as_mut()
                .ok_or_else(|| ModelError::Unsupported("no workspace for this size".into()))?
        } else if count == 1 {
            decode
        } else {
            longer = Workspace::new(gpu, cfg, count, 1)?;
            &mut longer
        };
        probe.stage(Stage::Embedding, || {
            Ok(self
                .weights
                .embed
                .embed(gpu, tokens, &mut ws.ids, &mut ws.hidden)?)
        })?;
        let mut watch = |index: usize, hidden: &Buffer| -> Result<(), ModelError> {
            if let Some(callback) = observe.as_mut() {
                callback(
                    index,
                    &gpu.download_range(hidden, last_row_start, hidden_size)?,
                );
            }
            Ok(())
        };
        watch(0, &ws.hidden)?;
        let start = kv.len();
        for (index, layer) in self.weights.layers.iter().enumerate() {
            probe.layer(Some(index));
            probe.stage(Stage::Norm, || {
                Ok(gpu.rms_norm_into(&ws.hidden, &layer.input_norm, &mut ws.normed, cfg.rms_eps)?)
            })?;
            probe.stage(Stage::Qkv, || {
                Ok(matmul_triple(
                    gpu,
                    (&layer.q_proj, &layer.k_proj, &layer.v_proj),
                    &ws.normed,
                    count,
                    (&mut ws.q, &mut ws.k, &mut ws.v),
                )?)
            })?;
            probe.stage(Stage::Norm, || {
                gpu.rms_norm(&mut ws.q, &layer.q_norm, cfg.rms_eps)?;
                Ok(gpu.rms_norm(&mut ws.k, &layer.k_norm, cfg.rms_eps)?)
            })?;
            probe.stage(Stage::Rope, || {
                gpu.rope(&mut ws.q, cfg.heads, cfg.head_dim, start, cfg.rope_theta)?;
                Ok(gpu.rope(&mut ws.k, cfg.kv_heads, cfg.head_dim, start, cfg.rope_theta)?)
            })?;
            probe.stage(Stage::KvWrite, || Ok(kv.write(gpu, index, &ws.k, &ws.v)?))?;
            probe.stage(Stage::Attention, || {
                Ok(gpu.attention(&ws.q, kv, index, cfg.heads, count, &mut ws.context)?)
            })?;
            probe.stage(Stage::OutProj, || {
                Ok(layer
                    .o_proj
                    .matmul(gpu, &ws.context, count, &mut ws.attended)?)
            })?;
            probe.stage(Stage::Residual, || {
                Ok(gpu.add(&mut ws.hidden, &ws.attended)?)
            })?;

            probe.stage(Stage::Norm, || {
                Ok(gpu.rms_norm_into(&ws.hidden, &layer.post_norm, &mut ws.normed, cfg.rms_eps)?)
            })?;
            probe.stage(Stage::Mlp, || {
                matmul_pair(
                    gpu,
                    (&layer.gate_proj, &layer.up_proj),
                    &ws.normed,
                    count,
                    (&mut ws.gate, &mut ws.up),
                )?;
                gpu.silu_mul(&mut ws.gate, &ws.up)?;
                Ok(layer.down_proj.matmul(gpu, &ws.gate, count, &mut ws.ff)?)
            })?;
            probe.stage(Stage::Residual, || Ok(gpu.add(&mut ws.hidden, &ws.ff)?))?;
            watch(index.saturating_add(1), &ws.hidden)?;
        }
        probe.layer(None);
        probe.stage(Stage::Norm, || {
            gpu.copy_range_into(
                &ws.hidden,
                first_row.saturating_mul(hidden_size),
                &mut ws.last,
            )?;
            Ok(gpu.rms_norm(&mut ws.last, &self.weights.final_norm, cfg.rms_eps)?)
        })?;
        let head = self.weights.lm_head.as_ref().unwrap_or(&self.weights.embed);
        probe.stage(Stage::LmHead, || {
            Ok(head.matmul(
                gpu,
                &ws.last,
                count.saturating_sub(first_row),
                &mut ws.logits,
            )?)
        })?;
        let logits = gpu.download(&ws.logits)?;
        probe.finish()?;
        tensor::ensure_finite(&logits, "logits")?;
        kv.advance(count)?;
        Ok(logits)
    }
}

impl CausalLm for Qwen3Gpu {
    type Error = ModelError;
    type Cache = GpuCache;

    fn forward(&self, tokens: &[u32], cache: &mut GpuCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }

    fn max_rows(&self) -> usize {
        MAX_ROWS
    }

    fn forward_rows(&self, tokens: &[u32], cache: &mut GpuCache) -> Result<Vec<f32>, ModelError> {
        Qwen3Gpu::forward_rows(self, tokens, cache)
    }
}
