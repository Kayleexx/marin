use super::math::{invalid, size, zeros};
use super::read::Reader;
use super::state::admit;
use super::{Cache, ReferenceBlock, TextConfig};
use crate::{ModelError, Qwen3Config};
use loader::{Dtype, ModelDir, Weights};
use runtime::{CausalLm, SequenceCache};
use std::sync::atomic::{AtomicU64, Ordering};
use tensor::{Matrix, rms_norm_rows};

pub struct Qwen35 {
    pub(super) blocks: Vec<ReferenceBlock>,
    config: TextConfig,
    summary: Qwen3Config,
    embed: Matrix,
    norm: Vec<f32>,
    owner: u64,
}
pub fn weight_bytes(weights: &Weights) -> Result<u64, ModelError> {
    weights
        .names()
        .filter(|n| n.starts_with("model.language_model."))
        .try_fold(0_u64, |n, name| {
            let i = weights
                .info(name)
                .ok_or_else(|| invalid("missing tensor metadata"))?;
            if !matches!(i.dtype, Dtype::Bf16 | Dtype::F32) {
                return Err(invalid("unsupported text dtype"));
            }
            let bytes = if i.shape.len() == 2 {
                i.byte_len()
            } else {
                i.shape
                    .iter()
                    .try_fold(1_usize, |n, x| size(n, *x))
                    .and_then(|n| size(n, 4))? as u64
            };
            n.checked_add(bytes)
                .ok_or_else(|| invalid("weights overflow"))
        })
}
pub fn load_bytes(weights: &Weights, config: &TextConfig) -> Result<u64, ModelError> {
    let transient = weights
        .names()
        .filter(|n| n.starts_with("model.language_model."))
        .filter_map(|n| weights.info(n))
        .map(|i| i.byte_len())
        .max()
        .unwrap_or(0);
    let needed = weight_bytes(weights)?
        .checked_add(transient)
        .and_then(|n| n.checked_add(config.state_bytes(0).ok()? as u64))
        .and_then(|n| n.checked_add(config.workspace_bytes(1, 1).ok()? as u64))
        .ok_or_else(|| invalid("load admission overflow"))?;
    Ok(needed)
}
impl Qwen35 {
    pub fn load(dir: &ModelDir, mut progress: impl FnMut(u64, u64)) -> Result<Self, ModelError> {
        if dir.weights.packed_group().is_some() {
            return Err(invalid("Qwen3.5 packed quantization is unsupported"));
        }
        let config = TextConfig::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        let summary = config.summary();
        let bytes = weight_bytes(&dir.weights)?;
        let needed = load_bytes(&dir.weights, &config)?;
        admit(
            usize::try_from(needed).map_err(|_| invalid("weights exceed address space"))?,
            usize::MAX,
        )?;
        let mut root = Reader::source(&dir.weights, "model.language_model.".into());
        // The full text namespace is validated by per-block readers plus the two root tensors.
        let name = "model.language_model.embed_tokens.weight";
        let info = dir
            .weights
            .info(name)
            .ok_or_else(|| invalid("missing embeddings"))?;
        if info.shape != [summary.vocab_size, summary.hidden_size] || info.dtype != Dtype::Bf16 {
            return Err(invalid("expected BF16 tied text embeddings"));
        }
        let bits = dir.weights.read_bf16(name)?;
        if bits.iter().any(|v| !tensor::bf16_to_f32(*v).is_finite()) {
            return Err(invalid("nonfinite embeddings"));
        }
        let embed = Matrix::new(summary.vocab_size, summary.hidden_size, bits)?;
        let norm = root.norm("norm.weight", summary.hidden_size)?;
        let expected = config.layer_types.iter().fold(2_usize, |n, k| {
            n.saturating_add(match k {
                super::LayerKind::LinearAttention => 14,
                super::LayerKind::FullAttention => 11,
            })
        });
        if dir
            .weights
            .names()
            .filter(|n| n.starts_with("model.language_model."))
            .count()
            != expected
        {
            return Err(invalid("unexpected/missing text tensors"));
        }
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(config.layer_types.len())
            .map_err(|e| invalid(e.to_string()))?;
        for (i, kind) in config.layer_types.iter().enumerate() {
            let reader = Reader::source(&dir.weights, format!("model.language_model.layers.{i}."));
            let (common, mixer) = super::weights::load_reader(&config.block, *kind, reader)?;
            blocks.push(ReferenceBlock {
                config: config.block.clone(),
                kind: *kind,
                common,
                mixer,
            });
            progress(
                (i.saturating_add(1) as u64)
                    .saturating_mul(bytes)
                    .checked_div(config.layer_types.len() as u64)
                    .unwrap_or(0),
                bytes,
            );
        }
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            blocks,
            config,
            summary,
            embed,
            norm,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
        })
    }
    pub fn config(&self) -> &Qwen3Config {
        &self.summary
    }
    pub fn new_cache(&self, capacity: usize) -> Result<Cache, ModelError> {
        if capacity == 0 || capacity > self.summary.max_positions {
            return Err(invalid("invalid Qwen3.5 context"));
        }
        admit(self.config.state_bytes(0)?, usize::MAX)?;
        let mut states = Vec::new();
        states
            .try_reserve_exact(self.blocks.len())
            .map_err(|e| invalid(e.to_string()))?;
        for b in &self.blocks {
            states.push(b.new_state(capacity, usize::MAX)?);
        }
        Ok(Cache {
            states,
            tokens: Vec::new(),
            capacity,
            replay: false,
            owner: self.owner,
        })
    }
    pub fn forward_observed(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        mut observe: crate::Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_full_observed(
            tokens,
            cache,
            Some(&mut |i, hidden: &[f32]| {
                if let Some(f) = &mut observe {
                    f(
                        i,
                        hidden
                            .chunks_exact(self.summary.hidden_size)
                            .last()
                            .unwrap_or_default(),
                    );
                }
            }),
        )
    }
    pub fn forward_full_observed(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        mut observe: crate::Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        if cache.owner != self.owner {
            return Err(invalid("cache belongs to a different Qwen3.5 model"));
        }
        let end = cache
            .used()
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("context overflow"))?;
        if end > cache.capacity {
            return Err(kv::KvError::ContextFull {
                needed: end,
                capacity: cache.capacity,
            }
            .into());
        }
        for id in tokens {
            if *id as usize >= self.summary.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token: *id,
                    vocab: self.summary.vocab_size,
                });
            }
        }
        let count = if cache.replay { end } else { tokens.len() };
        let window = end
            .checked_next_multiple_of(256)
            .ok_or_else(|| invalid("context overflow"))?;
        let needed = self
            .config
            .state_bytes(window)?
            .checked_mul(2)
            .ok_or_else(|| invalid("state growth overflow"))?
            .checked_add(self.config.workspace_bytes(count, end)?)
            .ok_or_else(|| invalid("pass memory overflow"))?;
        admit(needed, usize::MAX)?;
        let mut staged = cache.stage(self, end)?;
        let mut input = Vec::new();
        input
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        if cache.replay {
            input.extend_from_slice(&cache.tokens);
        }
        input.extend_from_slice(tokens);
        let mut hidden = zeros(size(input.len(), self.summary.hidden_size)?)?;
        for (id, row) in input
            .iter()
            .zip(hidden.chunks_exact_mut(self.summary.hidden_size))
        {
            self.embed.row_f32(*id as usize, row)?;
        }
        notify(&mut observe, 0, &hidden);
        for (i, (block, state)) in self.blocks.iter().zip(&mut staged.states).enumerate() {
            hidden = block.forward_in_place(&hidden, state, None)?;
            notify(&mut observe, i.saturating_add(1), &hidden);
        }
        let mut last = hidden
            .chunks_exact(self.summary.hidden_size)
            .last()
            .ok_or(ModelError::NoTokens)?
            .to_vec();
        rms_norm_rows(&mut last, &self.norm, self.summary.rms_eps)?;
        let mut logits = zeros(self.summary.vocab_size)?;
        self.embed.matmul(&last, &mut logits)?;
        tensor::ensure_finite(&logits, "Qwen3.5 logits")?;
        staged.tokens.extend_from_slice(tokens);
        *cache = staged;
        Ok(logits)
    }
}
fn notify(observe: &mut crate::Observer<'_>, index: usize, hidden: &[f32]) {
    if let Some(f) = observe {
        f(index, hidden);
    }
}
impl CausalLm for Qwen35 {
    type Error = ModelError;
    type Cache = Cache;
    fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }
}
