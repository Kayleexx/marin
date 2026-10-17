use super::TextConfig;
use super::gpu_cache::GpuCache;
use super::gpu_weights::{Layer, load_layers};
use super::math::{invalid, size};
use crate::qwen3::weight::Weight;
use crate::{ModelError, Observer, Qwen3Config};
use cuda::{Buffer, Gpu, Tokens};
use loader::ModelDir;
use runtime::CausalLm;
use std::sync::atomic::{AtomicU64, Ordering};
use tensor::QuantScheme;

pub struct Qwen35Gpu {
    pub(super) config: TextConfig,
    summary: Qwen3Config,
    pub(super) embed: Weight,
    pub(super) norm: Buffer,
    pub(super) layers: Vec<Layer>,
    pub(super) gpu: Gpu,
    quant: Option<QuantScheme>,
    owner: u64,
}

pub(super) struct Workspace {
    pub ids: Tokens,
    pub hidden: Buffer,
    pub normed: Buffer,
    pub qkv: Buffer,
    pub z: Buffer,
    pub a: Buffer,
    pub b: Buffer,
    pub mixed: Buffer,
    pub core: Buffer,
    pub projected: Buffer,
    pub q: Buffer,
    pub q_gate: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub context: Buffer,
    pub attended: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub ff: Buffer,
    pub last: Buffer,
    pub logits: Buffer,
}

impl Workspace {
    pub(super) fn new(gpu: &Gpu, config: &TextConfig, count: usize) -> Result<Self, ModelError> {
        let c = &config.block;
        let alloc = |width| -> Result<Buffer, ModelError> { Ok(gpu.zeros(size(count, width)?)?) };
        let q = size(c.num_attention_heads, c.head_dim)?;
        let kv = size(c.num_key_value_heads, c.head_dim)?;
        Ok(Self {
            ids: gpu.tokens(count)?,
            hidden: alloc(c.hidden_size)?,
            normed: alloc(c.hidden_size)?,
            qkv: alloc(c.channels()?)?,
            z: alloc(c.value_dim()?)?,
            a: alloc(c.linear_num_value_heads)?,
            b: alloc(c.linear_num_value_heads)?,
            mixed: alloc(c.channels()?)?,
            core: alloc(c.value_dim()?)?,
            projected: alloc(size(q, 2)?)?,
            q: alloc(q)?,
            q_gate: alloc(q)?,
            k: alloc(kv)?,
            v: alloc(kv)?,
            context: alloc(q)?,
            attended: alloc(c.hidden_size)?,
            gate: alloc(c.intermediate_size)?,
            up: alloc(c.intermediate_size)?,
            ff: alloc(c.hidden_size)?,
            last: gpu.zeros(c.hidden_size)?,
            logits: gpu.zeros(config.summary().vocab_size)?,
        })
    }
}

impl Qwen35Gpu {
    pub fn load(
        dir: &ModelDir,
        gpu: Gpu,
        quant: Option<QuantScheme>,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = TextConfig::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        super::layout::validate(&dir.weights, &config)?;
        super::weight_bytes(&dir.weights, quant)?;
        let (embed, norm, layers) = load_layers(&dir.weights, &gpu, &config, quant, progress)?;
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            summary: config.summary(),
            config,
            embed,
            norm,
            layers,
            gpu,
            quant,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.summary
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
    pub fn quantization(&self) -> Option<QuantScheme> {
        self.quant
    }
    pub fn new_cache(&self, capacity: usize) -> Result<GpuCache, ModelError> {
        GpuCache::new(&self.gpu, &self.config, &self.layers, capacity, self.owner)
    }

    pub fn checkpoint_cache(&self, cache: &GpuCache) -> Result<GpuCache, ModelError> {
        if cache.owner != self.owner {
            return Err(invalid("cache belongs to a different Qwen3.5 GPU model"));
        }
        cache.stage(&self.gpu, &self.config, &self.layers)
    }

    pub fn forward_observed(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        super::gpu_forward::forward(self, tokens, cache, observe, None)
    }

    pub fn forward_profiled(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        times: &mut metrics::StageTimes,
    ) -> Result<Vec<f32>, ModelError> {
        super::gpu_forward::forward(self, tokens, cache, None, Some(times))
    }

    pub(super) fn owner(&self) -> u64 {
        self.owner
    }
}

impl CausalLm for Qwen35Gpu {
    type Error = ModelError;
    type Cache = GpuCache;
    fn forward(&self, tokens: &[u32], cache: &mut GpuCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }
}
