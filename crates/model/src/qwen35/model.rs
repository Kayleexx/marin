use super::math::{invalid, size, zeros};
use super::read::Reader;
use super::state::admit;
use super::{Cache, CpuMark, ReferenceBlock, State, TextConfig};
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
    quant: Option<tensor::QuantScheme>,
}
pub fn weight_bytes(
    weights: &Weights,
    quant: Option<tensor::QuantScheme>,
) -> Result<u64, ModelError> {
    let quant = quant
        .map(|q| tensor::QuantScheme::new(q.bits, q.group))
        .transpose()?;
    if let Some(q) = quant
        && q.bits != tensor::QuantBits::Int8
    {
        return Err(invalid("CPU Qwen3.5 supports INT8 only"));
    }
    if let Some(group) = weights.packed_group()
        && quant != Some(tensor::QuantScheme::new(tensor::QuantBits::Int8, group)?)
    {
        return Err(invalid(
            "packed artifact requires matching opt-in INT8/group",
        ));
    }
    weights
        .names()
        .filter(|n| n.starts_with("model.language_model."))
        .try_fold(0_u64, |n, name| {
            let i = weights
                .info(name)
                .ok_or_else(|| invalid("missing tensor metadata"))?;
            if !(matches!(i.dtype, Dtype::Bf16 | Dtype::F32)
                || (weights.packed_group().is_some() && i.dtype == Dtype::I8))
            {
                return Err(invalid("unsupported text dtype"));
            }
            let bytes = if weights.packed_group().is_some() {
                i.byte_len()
            } else if let (Some(q), [rows, cols]) = (quant, i.shape.as_slice()) {
                if super::layout::quantized(name) {
                    if i.dtype != Dtype::Bf16 || cols.checked_rem(q.group) != Some(0) {
                        return Err(invalid(format!("incompatible quantized matrix {name}")));
                    }
                    q.matrix_bytes(*rows, *cols)
                        .ok_or_else(|| invalid("Q8 size overflow"))?
                } else {
                    size(size(*rows, *cols)?, 4)? as u64
                }
            } else if i.shape.len() == 2 {
                i.byte_len()
            } else {
                size(i.shape.iter().try_fold(1_usize, |n, x| size(n, *x))?, 4)? as u64
            };
            n.checked_add(bytes)
                .ok_or_else(|| invalid("weights overflow"))
        })
}
pub fn load_bytes(
    weights: &Weights,
    config: &TextConfig,
    quant: Option<tensor::QuantScheme>,
) -> Result<u64, ModelError> {
    let transient = weights
        .names()
        .filter(|n| n.starts_with("model.language_model."))
        .filter_map(|n| weights.info(n))
        .map(|i| i.byte_len())
        .max()
        .unwrap_or(0);
    let transient = if weights.packed_group().is_some() {
        1 << 20
    } else {
        transient
    };
    let needed = weight_bytes(weights, quant)?
        .checked_add(transient)
        .and_then(|n| n.checked_add(config.state_bytes(0).ok()? as u64))
        .and_then(|n| n.checked_add(config.workspace_bytes(1, 1).ok()? as u64))
        .ok_or_else(|| invalid("load admission overflow"))?;
    Ok(needed)
}
impl Qwen35 {
    pub fn load(dir: &ModelDir, mut progress: impl FnMut(u64, u64)) -> Result<Self, ModelError> {
        Self::load_quantized(dir, None, &mut progress)
    }
    pub fn load_quantized(
        dir: &ModelDir,
        quant: Option<tensor::QuantScheme>,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = TextConfig::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        super::layout::validate(&dir.weights, &config)?;
        let summary = config.summary();
        let bytes = weight_bytes(&dir.weights, quant)?;
        let needed = load_bytes(&dir.weights, &config, quant)?;
        admit(
            usize::try_from(needed).map_err(|_| invalid("weights exceed address space"))?,
            usize::MAX,
        )?;
        let mut root = Reader::source(&dir.weights, "model.language_model.".into(), quant);
        let embed = super::read::matrix(
            &dir.weights,
            "model.language_model.embed_tokens.weight",
            summary.vocab_size,
            summary.hidden_size,
            quant,
        )?;
        let norm = root.norm("norm.weight", summary.hidden_size)?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(config.layer_types.len())
            .map_err(|e| invalid(e.to_string()))?;
        for (i, kind) in config.layer_types.iter().enumerate() {
            let reader = Reader::source(
                &dir.weights,
                format!("model.language_model.layers.{i}."),
                quant,
            );
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
            quant,
        })
    }
    pub fn quantization(&self) -> Option<tensor::QuantScheme> {
        self.quant
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
        Ok(Cache::new(states, capacity, self.owner))
    }
    /// A mark of `cache` to `rewind_cache` to.
    pub fn mark_cache(&self, cache: &Cache) -> Result<CpuMark, ModelError> {
        if cache.owner != self.owner {
            return Err(invalid("cache belongs to a different Qwen3.5 model"));
        }
        cache.mark()
    }
    pub fn rewind_cache(&self, cache: &mut Cache, mark: CpuMark) -> Result<(), ModelError> {
        if cache.owner != self.owner {
            return Err(invalid("cache belongs to a different Qwen3.5 model"));
        }
        cache.rewind(mark)
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
        observe: crate::Observer<'_>,
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
        let mut input = Vec::new();
        input
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        if cache.replay {
            input.extend_from_slice(&cache.tokens);
        }
        input.extend_from_slice(tokens);
        let undo = cache.begin(self, end)?;
        match self.run(&input, &mut cache.states, observe) {
            Ok(logits) => {
                cache.commit(tokens);
                Ok(logits)
            }
            Err(error) => {
                cache.undo(undo);
                Err(error)
            }
        }
    }
    fn run(
        &self,
        input: &[u32],
        states: &mut [State],
        mut observe: crate::Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let mut hidden = zeros(size(input.len(), self.summary.hidden_size)?)?;
        for (id, row) in input
            .iter()
            .zip(hidden.chunks_exact_mut(self.summary.hidden_size))
        {
            self.embed.row_f32(*id as usize, row)?;
        }
        notify(&mut observe, 0, &hidden);
        for (i, (block, state)) in self.blocks.iter().zip(states).enumerate() {
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
