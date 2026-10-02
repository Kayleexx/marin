use cuda::{Buffer, DeviceKv, DeviceMatrix, Gpu};
use loader::ModelDir;
use metrics::{Stage, StageTimes};
use runtime::{CausalLm, SequenceCache};

use super::Observer;
use super::config::Qwen3Config;
use super::gpu_probe::GpuProbe;
use super::weights::{Qwen3Weights, Store};
use crate::ModelError;

struct GpuStore<'a>(&'a Gpu);

impl Store for GpuStore<'_> {
    type Matrix = DeviceMatrix;
    type Vector = Buffer;

    fn matrix(
        &mut self,
        rows: usize,
        cols: usize,
        bits: Vec<u16>,
    ) -> Result<DeviceMatrix, ModelError> {
        Ok(self.0.upload_matrix(rows, cols, &bits)?)
    }

    fn vector(&mut self, values: Vec<f32>) -> Result<Buffer, ModelError> {
        Ok(self.0.upload(&values)?)
    }
}

/// Attention memory on the GPU.
pub struct GpuCache(DeviceKv);

impl SequenceCache for GpuCache {
    fn used(&self) -> usize {
        self.0.len()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

pub struct Qwen3Gpu {
    config: Qwen3Config,
    weights: Qwen3Weights<DeviceMatrix, Buffer>,
    gpu: Gpu,
}

impl Qwen3Gpu {
    pub fn load(
        dir: &ModelDir,
        gpu: Gpu,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        let weights = Qwen3Weights::load(&dir.weights, &config, &mut GpuStore(&gpu), progress)?;
        gpu.synchronize()?;
        Ok(Self {
            config,
            weights,
            gpu,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn new_cache(&self, capacity: usize) -> Result<GpuCache, ModelError> {
        let cfg = &self.config;
        Ok(GpuCache(DeviceKv::new(
            &self.gpu,
            cfg.layers,
            cfg.kv_heads,
            cfg.head_dim,
            capacity,
        )?))
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
        mut observe: Observer<'_>,
        times: Option<&mut StageTimes>,
    ) -> Result<Vec<f32>, ModelError> {
        let mut probe = GpuProbe::new(&self.gpu, times);
        let (cfg, gpu, kv) = (&self.config, &self.gpu, &mut cache.0);
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        let count = tokens.len();
        let (hidden_size, q_dim, kv_dim) = (cfg.hidden_size, cfg.q_dim()?, cfg.kv_dim()?);
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
        let size = |per_token: usize| count.checked_mul(per_token).ok_or_else(overflow);
        let mut hidden = probe.stage(Stage::Embedding, || {
            let mut hidden = gpu.zeros(size(hidden_size)?)?;
            gpu.embed(&self.weights.embed, tokens, &mut hidden)?;
            Ok(hidden)
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
        watch(0, &hidden)?;
        let start = kv.len();
        for (index, layer) in self.weights.layers.iter().enumerate() {
            let normed = probe.stage(Stage::Norm, || {
                let mut normed = gpu.duplicate(&hidden)?;
                gpu.rms_norm(&mut normed, &layer.input_norm, cfg.rms_eps)?;
                Ok(normed)
            })?;
            let (mut q, mut k, v) = probe.stage(Stage::Qkv, || {
                let mut q = gpu.zeros(size(q_dim)?)?;
                let mut k = gpu.zeros(size(kv_dim)?)?;
                let mut v = gpu.zeros(size(kv_dim)?)?;
                gpu.matmul(&layer.q_proj, &normed, count, &mut q)?;
                gpu.matmul(&layer.k_proj, &normed, count, &mut k)?;
                gpu.matmul(&layer.v_proj, &normed, count, &mut v)?;
                Ok((q, k, v))
            })?;
            probe.stage(Stage::Norm, || {
                gpu.rms_norm(&mut q, &layer.q_norm, cfg.rms_eps)?;
                gpu.rms_norm(&mut k, &layer.k_norm, cfg.rms_eps)?;
                Ok(())
            })?;
            probe.stage(Stage::Rope, || {
                gpu.rope(&mut q, cfg.heads, cfg.head_dim, start, cfg.rope_theta)?;
                gpu.rope(&mut k, cfg.kv_heads, cfg.head_dim, start, cfg.rope_theta)?;
                Ok(())
            })?;
            probe.stage(Stage::KvWrite, || Ok(kv.write(gpu, index, &k, &v)?))?;
            let context = probe.stage(Stage::Attention, || {
                let mut context = gpu.zeros(size(q_dim)?)?;
                gpu.attention(&q, kv, index, cfg.heads, count, &mut context)?;
                Ok(context)
            })?;
            let attended = probe.stage(Stage::OutProj, || {
                let mut attended = gpu.zeros(size(hidden_size)?)?;
                gpu.matmul(&layer.o_proj, &context, count, &mut attended)?;
                Ok(attended)
            })?;
            probe.stage(Stage::Residual, || Ok(gpu.add(&mut hidden, &attended)?))?;

            let normed = probe.stage(Stage::Norm, || {
                let mut normed = gpu.duplicate(&hidden)?;
                gpu.rms_norm(&mut normed, &layer.post_norm, cfg.rms_eps)?;
                Ok(normed)
            })?;
            let ff = probe.stage(Stage::Mlp, || {
                let mut gate = gpu.zeros(size(cfg.intermediate_size)?)?;
                let mut up = gpu.zeros(size(cfg.intermediate_size)?)?;
                gpu.matmul(&layer.gate_proj, &normed, count, &mut gate)?;
                gpu.matmul(&layer.up_proj, &normed, count, &mut up)?;
                gpu.silu_mul(&mut gate, &up)?;
                let mut ff = gpu.zeros(size(hidden_size)?)?;
                gpu.matmul(&layer.down_proj, &gate, count, &mut ff)?;
                Ok(ff)
            })?;
            probe.stage(Stage::Residual, || Ok(gpu.add(&mut hidden, &ff)?))?;
            watch(index.saturating_add(1), &hidden)?;
        }
        let last = probe.stage(Stage::Norm, || {
            let mut last = gpu.copy_range(&hidden, last_row_start, hidden_size)?;
            gpu.rms_norm(&mut last, &self.weights.final_norm, cfg.rms_eps)?;
            Ok(last)
        })?;
        let head = self.weights.lm_head.as_ref().unwrap_or(&self.weights.embed);
        let logits = probe.stage(Stage::LmHead, || {
            let mut logits = gpu.zeros(cfg.vocab_size)?;
            gpu.matmul(head, &last, 1, &mut logits)?;
            Ok(logits)
        })?;
        let logits = gpu.download(&logits)?;
        probe.finish()?;
        tensor::ensure_finite(&logits, "logits")?;
        kv.advance(count)?;
        Ok(logits)
    }
}

fn overflow() -> ModelError {
    ModelError::InvalidConfig("input size overflows".into())
}

impl CausalLm for Qwen3Gpu {
    type Error = ModelError;
    type Cache = GpuCache;

    fn forward(&self, tokens: &[u32], cache: &mut GpuCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }
}
