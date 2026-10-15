use kv::KvCache;
use loader::ModelDir;
use runtime::CausalLm;
use tensor::{ensure_finite, rms_norm_rows};

use super::Config;
use super::batch;
use super::layer::{RouteObserver, Scratch};
use super::weights::{ResidentWeights, resident_bytes};
use crate::{ModelError, Observer, Qwen3Config};

pub struct Granite {
    config: Config,
    summary: Qwen3Config,
    weights: ResidentWeights,
}

impl Granite {
    pub fn load(dir: &ModelDir, progress: impl FnMut(u64, u64)) -> Result<Self, ModelError> {
        Self::load_quantized(dir, None, progress)
    }

    pub fn load_quantized(
        dir: &ModelDir,
        quant: Option<tensor::QuantScheme>,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let config = Config::parse(&dir.config_json)?;
        if dir.weights.packed_group().is_some()
            && dir.weights.packed_group() != quant.map(|q| q.group)
        {
            return Err(ModelError::Unsupported(
                "Granite artifact and requested INT8 group differ".into(),
            ));
        }
        if quant.is_some_and(|q| q.bits != tensor::QuantBits::Int8) {
            return Err(ModelError::Unsupported(
                "Granite CPU supports INT8 weights only".into(),
            ));
        }
        let bytes = resident_bytes(&dir.weights, &config, quant)?;
        let transient = dir
            .weights
            .names()
            .filter_map(|name| dir.weights.info(name))
            .map(|info| info.byte_len())
            .max()
            .unwrap_or(0);
        crate::qwen3::ensure_cpu_available(
            bytes
                .checked_add(transient)
                .and_then(|n| n.checked_add(crate::CPU_SLACK_BYTES))
                .ok_or_else(|| ModelError::InvalidConfig("Granite memory overflows".into()))?,
        )?;
        let weights = ResidentWeights::load(&dir.weights, &config, progress, quant)?;
        Ok(Self {
            summary: config.summary(),
            config,
            weights,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.summary
    }

    pub fn quantization(&self) -> Option<tensor::QuantScheme> {
        self.weights.quant
    }

    pub fn new_cache(&self, capacity: usize) -> Result<KvCache, ModelError> {
        Ok(KvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim(),
            capacity,
        )?)
    }

    pub fn forward_observed(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_detailed(tokens, kv, observe, None)
    }

    pub fn forward_detailed(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        mut observe: Observer<'_>,
        mut route: RouteObserver<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        let start = kv.len();
        let end = start
            .checked_add(tokens.len())
            .ok_or(kv::KvError::Overflow)?;
        if end > kv.capacity() {
            return Err(kv::KvError::ContextFull {
                needed: end,
                capacity: kv.capacity(),
            }
            .into());
        }
        for token in tokens {
            if usize::try_from(*token).unwrap_or(usize::MAX) >= self.config.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token: *token,
                    vocab: self.config.vocab_size,
                });
            }
        }
        if tokens.len() >= 32 {
            let workspace = batch::workspace_bytes(&self.config, tokens.len())?.max(
                crate::qwen3::cpu_workspace_bytes(&self.summary, tokens.len())?,
            );
            crate::qwen3::before_cpu_growth_with_workspace(
                &self.summary,
                kv,
                end,
                tokens.len(),
                workspace,
            )?;
        } else {
            crate::qwen3::before_cpu_growth(&self.summary, kv, end, tokens.len())?;
        }
        kv.reserve(end)?;
        let result = (|| {
            let mut hidden = if tokens.len() >= 32 {
                batch::run(
                    &self.config,
                    &self.weights,
                    tokens,
                    kv,
                    &mut observe,
                    &mut route,
                )?
            } else {
                let mut scratch = Scratch::new(&self.config);
                let mut hidden = vec![0.0_f32; self.config.hidden_size];
                for (offset, token) in tokens.iter().enumerate() {
                    let id = usize::try_from(*token).unwrap_or(usize::MAX);
                    self.weights.embed.row_f32(id, &mut hidden)?;
                    for value in &mut hidden {
                        *value *= self.config.embedding_multiplier;
                    }
                    if offset == tokens.len().saturating_sub(1)
                        && let Some(callback) = observe.as_deref_mut()
                    {
                        callback(0, &hidden);
                    }
                    for (index, layer) in self.weights.layers.iter().enumerate() {
                        layer.forward(
                            &self.config,
                            index,
                            &mut hidden,
                            kv,
                            &mut scratch,
                            &mut route,
                        )?;
                        if offset == tokens.len().saturating_sub(1)
                            && let Some(callback) = observe.as_deref_mut()
                        {
                            callback(index.saturating_add(1), &hidden);
                        }
                    }
                    kv.advance(1)?;
                }
                hidden
            };
            rms_norm_rows(&mut hidden, &self.weights.norm, self.config.rms_norm_eps)?;
            let mut logits = vec![0.0_f32; self.config.vocab_size];
            self.weights.embed.matmul(&hidden, &mut logits)?;
            for value in &mut logits {
                *value /= self.config.logits_scaling;
            }
            ensure_finite(&logits, "Granite logits")?;
            Ok(logits)
        })();
        if result.is_err() {
            kv.truncate(start)?;
        }
        result
    }
}

impl CausalLm for Granite {
    type Error = ModelError;
    type Cache = KvCache;

    fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, kv, None)
    }
}
