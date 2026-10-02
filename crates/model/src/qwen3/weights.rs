use loader::{Dtype, Weights};
use tensor::{Matrix, bf16_to_f32};

use super::config::Qwen3Config;
use crate::ModelError;

const MAX_REPORTED_PROBLEMS: usize = 12;

/// Where loaded weights end up: host memory for the CPU path, GPU memory for CUDA.
pub(crate) trait Store {
    type Matrix;
    type Vector;

    fn matrix(
        &mut self,
        rows: usize,
        cols: usize,
        bits: Vec<u16>,
    ) -> Result<Self::Matrix, ModelError>;
    fn vector(&mut self, values: Vec<f32>) -> Result<Self::Vector, ModelError>;
}

pub(crate) struct CpuStore;

impl Store for CpuStore {
    type Matrix = Matrix;
    type Vector = Vec<f32>;

    fn matrix(&mut self, rows: usize, cols: usize, bits: Vec<u16>) -> Result<Matrix, ModelError> {
        Ok(Matrix::new(rows, cols, bits)?)
    }

    fn vector(&mut self, values: Vec<f32>) -> Result<Vec<f32>, ModelError> {
        Ok(values)
    }
}

pub(crate) struct LayerWeights<M, V> {
    pub input_norm: V,
    pub q_proj: M,
    pub k_proj: M,
    pub v_proj: M,
    pub o_proj: M,
    pub q_norm: V,
    pub k_norm: V,
    pub post_norm: V,
    pub gate_proj: M,
    pub up_proj: M,
    pub down_proj: M,
}

pub(crate) struct Qwen3Weights<M, V> {
    pub embed: M,
    pub layers: Vec<LayerWeights<M, V>>,
    pub final_norm: V,
    /// None when the output head shares the embedding matrix.
    pub lm_head: Option<M>,
}

struct Expected {
    name: String,
    shape: Vec<usize>,
}

fn expected(name: String, shape: &[usize]) -> Expected {
    Expected {
        name,
        shape: shape.to_vec(),
    }
}

fn expected_tensors(cfg: &Qwen3Config) -> Result<Vec<Expected>, ModelError> {
    let (h, i, hd) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim);
    let (q, kv) = (cfg.q_dim()?, cfg.kv_dim()?);
    let mut all = vec![
        expected("model.embed_tokens.weight".into(), &[cfg.vocab_size, h]),
        expected("model.norm.weight".into(), &[h]),
    ];
    if !cfg.tie_embeddings {
        all.push(expected("lm_head.weight".into(), &[cfg.vocab_size, h]));
    }
    for n in 0..cfg.layers {
        let layer = |suffix: &str| format!("model.layers.{n}.{suffix}");
        all.extend([
            expected(layer("input_layernorm.weight"), &[h]),
            expected(layer("self_attn.q_proj.weight"), &[q, h]),
            expected(layer("self_attn.k_proj.weight"), &[kv, h]),
            expected(layer("self_attn.v_proj.weight"), &[kv, h]),
            expected(layer("self_attn.o_proj.weight"), &[h, q]),
            expected(layer("self_attn.q_norm.weight"), &[hd]),
            expected(layer("self_attn.k_norm.weight"), &[hd]),
            expected(layer("post_attention_layernorm.weight"), &[h]),
            expected(layer("mlp.gate_proj.weight"), &[i, h]),
            expected(layer("mlp.up_proj.weight"), &[i, h]),
            expected(layer("mlp.down_proj.weight"), &[h, i]),
        ]);
    }
    Ok(all)
}

/// Lists every missing, mis-shaped or wrongly typed tensor at once, plus the bytes to be read.
fn validate(weights: &Weights, expected: &[Expected]) -> Result<u64, ModelError> {
    let mut problems = Vec::new();
    let mut bytes = 0_u64;
    for want in expected {
        let Some(info) = weights.info(&want.name) else {
            problems.push(format!("missing tensor {}", want.name));
            continue;
        };
        if info.shape != want.shape {
            problems.push(format!(
                "tensor {} has shape {:?} but config.json implies {:?}",
                want.name, info.shape, want.shape
            ));
        } else if info.dtype != Dtype::Bf16 {
            problems.push(format!(
                "tensor {} is {} but only BF16 is supported",
                want.name,
                info.dtype_name()
            ));
        }
        bytes = bytes.saturating_add(info.byte_len());
    }
    if problems.is_empty() {
        return Ok(bytes);
    }
    let hidden = problems.len().saturating_sub(MAX_REPORTED_PROBLEMS);
    problems.truncate(MAX_REPORTED_PROBLEMS);
    if hidden > 0 {
        problems.push(format!("... and {hidden} more"));
    }
    Err(ModelError::InvalidWeights(problems))
}

struct Reader<'a, S: Store, F: FnMut(u64, u64)> {
    weights: &'a Weights,
    store: &'a mut S,
    loaded: u64,
    total: u64,
    progress: F,
}

impl<S: Store, F: FnMut(u64, u64)> Reader<'_, S, F> {
    fn read(&mut self, name: &str) -> Result<Vec<u16>, ModelError> {
        let bits = self.weights.read_bf16(name)?;
        let bytes = u64::try_from(bits.len()).unwrap_or(0).saturating_mul(2);
        self.loaded = self.loaded.saturating_add(bytes);
        (self.progress)(self.loaded, self.total);
        Ok(bits)
    }

    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<S::Matrix, ModelError> {
        let bits = self.read(name)?;
        self.store.matrix(rows, cols, bits)
    }

    fn vector(&mut self, name: &str) -> Result<S::Vector, ModelError> {
        let bits = self.read(name)?;
        self.store
            .vector(bits.into_iter().map(bf16_to_f32).collect())
    }
}

/// Bytes of the tensors this model actually loads (shards may hold extras such as an unused `lm_head`).
pub(crate) fn loaded_bytes(weights: &Weights, cfg: &Qwen3Config) -> Result<u64, ModelError> {
    validate(weights, &expected_tensors(cfg)?)
}

impl<M, V> Qwen3Weights<M, V> {
    /// Checks every tensor against the config, then reads them one at a time into `store`.
    pub fn load<S: Store<Matrix = M, Vector = V>>(
        weights: &Weights,
        cfg: &Qwen3Config,
        store: &mut S,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        let total = validate(weights, &expected_tensors(cfg)?)?;
        let mut read = Reader {
            weights,
            store,
            loaded: 0,
            total,
            progress,
        };
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let (q, kv) = (cfg.q_dim()?, cfg.kv_dim()?);
        let embed = read.matrix("model.embed_tokens.weight", cfg.vocab_size, h)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for n in 0..cfg.layers {
            let name = |suffix: &str| format!("model.layers.{n}.{suffix}");
            layers.push(LayerWeights {
                input_norm: read.vector(&name("input_layernorm.weight"))?,
                q_proj: read.matrix(&name("self_attn.q_proj.weight"), q, h)?,
                k_proj: read.matrix(&name("self_attn.k_proj.weight"), kv, h)?,
                v_proj: read.matrix(&name("self_attn.v_proj.weight"), kv, h)?,
                o_proj: read.matrix(&name("self_attn.o_proj.weight"), h, q)?,
                q_norm: read.vector(&name("self_attn.q_norm.weight"))?,
                k_norm: read.vector(&name("self_attn.k_norm.weight"))?,
                post_norm: read.vector(&name("post_attention_layernorm.weight"))?,
                gate_proj: read.matrix(&name("mlp.gate_proj.weight"), i, h)?,
                up_proj: read.matrix(&name("mlp.up_proj.weight"), i, h)?,
                down_proj: read.matrix(&name("mlp.down_proj.weight"), h, i)?,
            });
        }
        let final_norm = read.vector("model.norm.weight")?;
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(read.matrix("lm_head.weight", cfg.vocab_size, h)?)
        };
        Ok(Self {
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }
}
