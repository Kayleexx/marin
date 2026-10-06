use kv::KvCache;
use loader::ModelDir;
use metrics::{Probe, Stage};
use runtime::CausalLm;
use tensor::{Matrix, ensure_finite, rms_norm_rows};

use super::Observer;
use super::config::Qwen3Config;
use super::weights::{CpuStore, Qwen3Weights};
use crate::ModelError;

pub struct Qwen3 {
    config: Qwen3Config,
    weights: Qwen3Weights<Matrix, Vec<f32>>,
}

impl Qwen3 {
    /// `progress(loaded_bytes, total_bytes)` is called after each tensor is read.
    pub fn load(dir: &ModelDir, progress: impl FnMut(u64, u64)) -> Result<Self, ModelError> {
        let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        let weights = Qwen3Weights::load(&dir.weights, &config, &mut CpuStore, progress)?;
        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn new_cache(&self, capacity: usize) -> Result<KvCache, ModelError> {
        let kv_heads = self.config.kv_heads;
        Ok(KvCache::new(
            self.config.layers,
            kv_heads,
            self.config.head_dim,
            capacity,
        )?)
    }

    /// Like `forward`, but also reports the last token's hidden state before layer 0
    /// (index 0) and after each layer (index `n`), for comparing against a reference.
    pub fn forward_observed(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        self.forward_with(tokens, kv, observe, &mut Probe::off())
    }

    /// `forward_observed` that also times the stages of the pass when `probe` is on.
    pub fn forward_with(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        mut observe: Observer<'_>,
        probe: &mut Probe<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(ModelError::NoTokens);
        }
        let start = kv.len();
        let mut hidden = probe.time(Stage::Embedding, || self.embed(tokens))?;
        notify(&mut observe, 0, last_row(&hidden, cfg.hidden_size));
        for (index, layer) in self.weights.layers.iter().enumerate() {
            probe.layer(Some(index));
            layer.forward(cfg, index, &mut hidden, start, kv, probe)?;
            notify(
                &mut observe,
                index.saturating_add(1),
                last_row(&hidden, cfg.hidden_size),
            );
        }
        probe.layer(None);
        let mut last = last_row(&hidden, cfg.hidden_size).to_vec();
        probe.time(Stage::Norm, || {
            rms_norm_rows(&mut last, &self.weights.final_norm, cfg.rms_eps)
        })?;
        let head = self.weights.lm_head.as_ref().unwrap_or(&self.weights.embed);
        let mut logits = vec![0.0; cfg.vocab_size];
        probe.time(Stage::LmHead, || head.matmul(&last, &mut logits))?;
        ensure_finite(&logits, "logits")?;
        kv.advance(tokens.len())?;
        Ok(logits)
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        let size = tokens
            .len()
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| ModelError::InvalidConfig("input size overflows".into()))?;
        let mut hidden = vec![0.0; size];
        for (token, row) in tokens
            .iter()
            .zip(hidden.chunks_exact_mut(self.config.hidden_size))
        {
            let id = usize::try_from(*token).unwrap_or(usize::MAX);
            if id >= self.config.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token: *token,
                    vocab: self.config.vocab_size,
                });
            }
            self.weights.embed.row_f32(id, row)?;
        }
        Ok(hidden)
    }
}

fn notify(observe: &mut Observer<'_>, index: usize, hidden: &[f32]) {
    if let Some(callback) = observe {
        callback(index, hidden);
    }
}

fn last_row(hidden: &[f32], width: usize) -> &[f32] {
    hidden.chunks_exact(width).last().unwrap_or(&[])
}

impl CausalLm for Qwen3 {
    type Error = ModelError;
    type Cache = KvCache;

    fn forward(&self, tokens: &[u32], kv: &mut KvCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, kv, None)
    }
}
