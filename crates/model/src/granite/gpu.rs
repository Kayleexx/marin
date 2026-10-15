use cuda::{Buffer, DeviceKv, Gpu, KvFormat, Tokens};
use loader::ModelDir;
use metrics::{Stage, StageTimes};
use runtime::{CausalLm, RollbackError, SequenceCache};
use tensor::ensure_finite;

use super::Config;
use super::gpu_weights::Weights;
use super::layer::{RouteObserver, selected};
use crate::qwen3::gpu_probe::GpuProbe;
use crate::{ModelError, Observer, Qwen3Config};

const PREFILL_CHUNK: usize = 512;

pub fn workspace_bytes(config: &Config, context: usize) -> Result<u64, ModelError> {
    let h = config.hidden_size;
    let k = config.num_experts_per_tok;
    let per_token = h
        .checked_mul(6)
        .and_then(|n| n.checked_add(config.kv_dim().checked_mul(2)?))
        .and_then(|n| n.checked_add(config.num_local_experts))
        .and_then(|n| {
            n.checked_add(
                k.checked_mul(
                    h.checked_add(config.intermediate_size.checked_mul(2)?)?
                        .checked_add(1)?,
                )?,
            )
        })
        .ok_or_else(|| ModelError::InvalidConfig("Granite GPU workspace overflows".into()))?;
    let floats = context
        .clamp(1, PREFILL_CHUNK)
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(per_token))
        .and_then(|n| n.checked_add(h.checked_add(config.vocab_size)?))
        .ok_or_else(|| ModelError::InvalidConfig("Granite GPU workspace overflows".into()))?;
    u64::try_from(floats)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| ModelError::InvalidConfig("Granite GPU workspace overflows".into()))
}

fn size(rows: usize, cols: usize) -> Result<usize, ModelError> {
    rows.checked_mul(cols)
        .ok_or_else(|| ModelError::InvalidConfig("Granite CUDA workspace overflows".into()))
}

struct Workspace {
    ids: Tokens,
    hidden: Buffer,
    normed: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    context: Buffer,
    attended: Buffer,
    router: Buffer,
    routes: Tokens,
    gates: Buffer,
    projected: Buffer,
    expert_output: Buffer,
    mixture: Buffer,
    last: Buffer,
    logits: Buffer,
}

impl Workspace {
    fn new(gpu: &Gpu, config: &Config, tokens: usize) -> Result<Self, ModelError> {
        let h = config.hidden_size;
        let kv = config.kv_dim();
        let assignments = size(tokens, config.num_experts_per_tok)?;
        Ok(Self {
            ids: gpu.tokens(tokens)?,
            hidden: gpu.zeros(size(tokens, h)?)?,
            normed: gpu.zeros(size(tokens, h)?)?,
            q: gpu.zeros(size(tokens, h)?)?,
            k: gpu.zeros(size(tokens, kv)?)?,
            v: gpu.zeros(size(tokens, kv)?)?,
            context: gpu.zeros(size(tokens, h)?)?,
            attended: gpu.zeros(size(tokens, h)?)?,
            router: gpu.zeros(size(tokens, config.num_local_experts)?)?,
            routes: gpu.tokens(assignments)?,
            gates: gpu.zeros(assignments)?,
            projected: gpu.zeros(size(assignments, size(config.intermediate_size, 2)?)?)?,
            expert_output: gpu.zeros(size(assignments, h)?)?,
            mixture: gpu.zeros(size(tokens, h)?)?,
            last: gpu.zeros(h)?,
            logits: gpu.zeros(config.vocab_size)?,
        })
    }
}

pub struct GpuCache {
    kv: DeviceKv,
    decode: Workspace,
}

impl GpuCache {
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

pub struct GraniteGpu {
    config: Config,
    summary: Qwen3Config,
    weights: Weights,
    gpu: Gpu,
}

impl GraniteGpu {
    pub fn load(
        dir: &ModelDir,
        gpu: Gpu,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = Config::parse(&dir.config_json)?;
        let weights = Weights::load(&dir.weights, &gpu, &config, progress)?;
        Ok(Self {
            summary: config.summary(),
            config,
            weights,
            gpu,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.summary
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn new_cache(&self, capacity: usize) -> Result<GpuCache, ModelError> {
        let c = &self.config;
        Ok(GpuCache {
            kv: DeviceKv::new(
                &self.gpu,
                c.num_hidden_layers,
                (c.num_attention_heads, c.num_key_value_heads, c.head_dim()),
                capacity,
                KvFormat::F32,
            )?,
            decode: Workspace::new(&self.gpu, c, 1)?,
        })
    }

    pub fn forward_observed(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_detailed(tokens, cache, observe, None)
    }

    pub fn forward_detailed(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        observe: Observer<'_>,
        route: RouteObserver<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_with(tokens, cache, observe, route, None)
    }

    pub fn forward_profiled(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        times: &mut StageTimes,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_with(tokens, cache, None, None, Some(times))
    }

    fn forward_with(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        mut observe: Observer<'_>,
        mut route: RouteObserver<'_>,
        mut times: Option<&mut StageTimes>,
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        let start = cache.kv.len();
        let end = start
            .checked_add(tokens.len())
            .ok_or(kv::KvError::Overflow)?;
        if end > cache.kv.capacity() {
            return Err(kv::KvError::ContextFull {
                needed: end,
                capacity: cache.kv.capacity(),
            }
            .into());
        }
        let result = (|| {
            let mut logits = Vec::new();
            let mut chunks = tokens.chunks(PREFILL_CHUNK).peekable();
            while let Some(chunk) = chunks.next() {
                let final_chunk = chunks.peek().is_none();
                logits = self.pass(
                    chunk,
                    cache,
                    if final_chunk { observe.take() } else { None },
                    if final_chunk { route.take() } else { None },
                    times.as_deref_mut(),
                )?;
            }
            Ok(logits)
        })();
        if result.is_err() {
            cache.kv.truncate(start)?;
        }
        result
    }

    fn pass(
        &self,
        tokens: &[u32],
        cache: &mut GpuCache,
        mut observe: Observer<'_>,
        mut route: RouteObserver<'_>,
        times: Option<&mut StageTimes>,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let count = tokens.len();
        let h = c.hidden_size;
        let last = count.saturating_sub(1);
        let gpu = &self.gpu;
        let mut probe = GpuProbe::new(gpu, times);
        let kv = &mut cache.kv;
        let mut prefill;
        let ws = if count == 1 {
            &mut cache.decode
        } else {
            prefill = Workspace::new(gpu, c, count)?;
            &mut prefill
        };
        probe.stage(Stage::Embedding, || {
            gpu.embed(&self.weights.embed, tokens, &mut ws.ids, &mut ws.hidden)?;
            Ok(gpu.granite_scale(&mut ws.hidden, c.embedding_multiplier)?)
        })?;
        if let Some(callback) = observe.as_deref_mut() {
            callback(0, &gpu.download_range(&ws.hidden, size(last, h)?, h)?);
        }
        let start = kv.len();
        let query_scale = c.attention_multiplier * (c.head_dim() as f32).sqrt();
        for (index, layer) in self.weights.layers.iter().enumerate() {
            probe.layer(Some(index));
            probe.stage(Stage::Norm, || {
                Ok(gpu.rms_norm_into(
                    &ws.hidden,
                    &layer.input_norm,
                    &mut ws.normed,
                    c.rms_norm_eps,
                )?)
            })?;
            probe.stage(Stage::Qkv, || {
                gpu.matmul_precise(&layer.q, &ws.normed, count, &mut ws.q)?;
                gpu.matmul_precise(&layer.k, &ws.normed, count, &mut ws.k)?;
                Ok(gpu.matmul_precise(&layer.v, &ws.normed, count, &mut ws.v)?)
            })?;
            probe.stage(Stage::Rope, || {
                gpu.rope(
                    &mut ws.q,
                    c.num_attention_heads,
                    c.head_dim(),
                    start,
                    c.rope_theta,
                )?;
                gpu.rope(
                    &mut ws.k,
                    c.num_key_value_heads,
                    c.head_dim(),
                    start,
                    c.rope_theta,
                )?;
                Ok(gpu.granite_scale(&mut ws.q, query_scale)?)
            })?;
            probe.stage(Stage::KvWrite, || Ok(kv.write(gpu, index, &ws.k, &ws.v)?))?;
            probe.stage(Stage::Attention, || {
                Ok(gpu.attention(
                    &ws.q,
                    kv,
                    index,
                    c.num_attention_heads,
                    count,
                    &mut ws.context,
                )?)
            })?;
            probe.stage(Stage::OutProj, || {
                Ok(gpu.matmul_precise(&layer.o, &ws.context, count, &mut ws.attended)?)
            })?;
            probe.stage(Stage::Residual, || {
                Ok(gpu.granite_scaled_add(&mut ws.hidden, &ws.attended, c.residual_multiplier)?)
            })?;
            probe.stage(Stage::Norm, || {
                Ok(gpu.rms_norm_into(
                    &ws.hidden,
                    &layer.post_norm,
                    &mut ws.normed,
                    c.rms_norm_eps,
                )?)
            })?;
            probe.stage(Stage::Mlp, || {
                gpu.matmul_precise(&layer.router, &ws.normed, count, &mut ws.router)?;
                Ok(gpu.granite_moe(
                    &ws.normed,
                    &ws.router,
                    (&layer.expert_input, &layer.expert_output),
                    (
                        count,
                        c.num_local_experts,
                        c.num_experts_per_tok,
                        h,
                        c.intermediate_size,
                    ),
                    (
                        &mut ws.routes,
                        &mut ws.gates,
                        &mut ws.projected,
                        &mut ws.expert_output,
                        &mut ws.mixture,
                    ),
                )?)
            })?;
            if let Some(callback) = route.as_deref_mut() {
                let router = gpu.download_range(
                    &ws.router,
                    size(last, c.num_local_experts)?,
                    c.num_local_experts,
                )?;
                let mixture = gpu.download_range(&ws.mixture, size(last, h)?, h)?;
                let ids = selected(&router, c.num_experts_per_tok)?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<Vec<_>>();
                let actual = gpu.granite_route_ids(
                    &ws.routes,
                    size(last, c.num_experts_per_tok)?,
                    c.num_experts_per_tok,
                )?;
                let mut expected = ids.clone();
                expected.sort_unstable();
                if actual.iter().map(|id| *id as usize).collect::<Vec<_>>() != expected {
                    return Err(ModelError::InvalidConfig(format!(
                        "Granite CUDA layer {index} route differs"
                    )));
                }
                callback(index, start.saturating_add(last), &router, &ids, &mixture);
            }
            probe.stage(Stage::Residual, || {
                Ok(gpu.granite_scaled_add(&mut ws.hidden, &ws.mixture, c.residual_multiplier)?)
            })?;
            if let Some(callback) = observe.as_deref_mut() {
                callback(
                    index.saturating_add(1),
                    &gpu.download_range(&ws.hidden, size(last, h)?, h)?,
                );
            }
        }
        probe.layer(None);
        probe.stage(Stage::Norm, || {
            gpu.copy_range_into(&ws.hidden, size(last, h)?, &mut ws.last)?;
            Ok(gpu.rms_norm(&mut ws.last, &self.weights.norm, c.rms_norm_eps)?)
        })?;
        probe.stage(Stage::LmHead, || {
            gpu.matmul_precise(&self.weights.embed, &ws.last, 1, &mut ws.logits)?;
            Ok(gpu.granite_scale(&mut ws.logits, 1.0 / c.logits_scaling)?)
        })?;
        let logits = gpu.download(&ws.logits)?;
        probe.finish()?;
        ensure_finite(&logits, "Granite CUDA logits")?;
        kv.advance(count)?;
        Ok(logits)
    }
}

impl CausalLm for GraniteGpu {
    type Error = ModelError;
    type Cache = GpuCache;
    fn forward(&self, tokens: &[u32], cache: &mut GpuCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }
}
