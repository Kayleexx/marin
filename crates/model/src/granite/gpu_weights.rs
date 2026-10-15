use cuda::{Buffer, DeviceMatrix, Gpu};
use loader::Weights as Source;

use super::Config;
use super::weights::weight_bytes;
use crate::ModelError;

pub(super) struct Layer {
    pub input_norm: Buffer,
    pub post_norm: Buffer,
    pub q: DeviceMatrix,
    pub k: DeviceMatrix,
    pub v: DeviceMatrix,
    pub o: DeviceMatrix,
    pub router: DeviceMatrix,
    pub expert_input: DeviceMatrix,
    pub expert_output: DeviceMatrix,
}

pub(super) struct Weights {
    pub embed: DeviceMatrix,
    pub norm: Buffer,
    pub layers: Vec<Layer>,
}

fn matrix(
    source: &Source,
    gpu: &Gpu,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceMatrix, ModelError> {
    let values = source.read_bf16(name)?;
    if values.len()
        != rows
            .checked_mul(cols)
            .ok_or_else(|| ModelError::InvalidConfig("Granite GPU matrix size overflows".into()))?
        || values
            .iter()
            .any(|bits| !tensor::bf16_to_f32(*bits).is_finite())
    {
        return Err(ModelError::InvalidWeights(vec![format!(
            "invalid Granite matrix {name}"
        )]));
    }
    Ok(gpu.upload_matrix(rows, cols, &values)?)
}

fn vector(source: &Source, gpu: &Gpu, name: &str, len: usize) -> Result<Buffer, ModelError> {
    let values: Vec<f32> = source
        .read_bf16(name)?
        .into_iter()
        .map(tensor::bf16_to_f32)
        .collect();
    if values.len() != len || values.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::InvalidWeights(vec![format!(
            "invalid Granite vector {name}"
        )]));
    }
    Ok(gpu.upload(&values)?)
}

impl Weights {
    pub fn load(
        source: &Source,
        gpu: &Gpu,
        config: &Config,
        mut progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        if source.packed_group().is_some() {
            return Err(ModelError::Unsupported(
                "Granite CUDA BF16 requires source BF16 weights".into(),
            ));
        }
        let total = weight_bytes(source, config)?;
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let kv = config.kv_dim();
        let experts = config.num_local_experts;
        let embed = matrix(
            source,
            gpu,
            "model.embed_tokens.weight",
            config.vocab_size,
            h,
        )?;
        let norm = vector(source, gpu, "model.norm.weight", h)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(|error| {
                ModelError::InvalidConfig(format!("Granite GPU layer allocation: {error}"))
            })?;
        for index in 0..config.num_hidden_layers {
            let name = |suffix: &str| format!("model.layers.{index}.{suffix}.weight");
            layers.push(Layer {
                input_norm: vector(source, gpu, &name("input_layernorm"), h)?,
                post_norm: vector(source, gpu, &name("post_attention_layernorm"), h)?,
                q: matrix(source, gpu, &name("self_attn.q_proj"), h, h)?,
                k: matrix(source, gpu, &name("self_attn.k_proj"), kv, h)?,
                v: matrix(source, gpu, &name("self_attn.v_proj"), kv, h)?,
                o: matrix(source, gpu, &name("self_attn.o_proj"), h, h)?,
                router: matrix(
                    source,
                    gpu,
                    &name("block_sparse_moe.router.layer"),
                    experts,
                    h,
                )?,
                expert_input: matrix(
                    source,
                    gpu,
                    &name("block_sparse_moe.input_linear"),
                    experts
                        .checked_mul(i)
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| {
                            ModelError::InvalidConfig("Granite expert size overflows".into())
                        })?,
                    h,
                )?,
                expert_output: matrix(
                    source,
                    gpu,
                    &name("block_sparse_moe.output_linear"),
                    experts.checked_mul(h).ok_or_else(|| {
                        ModelError::InvalidConfig("Granite expert size overflows".into())
                    })?,
                    i,
                )?,
            });
            progress(
                total
                    .saturating_mul(index.saturating_add(1) as u64)
                    .checked_div(config.num_hidden_layers as u64)
                    .unwrap_or(0),
                total,
            );
        }
        gpu.synchronize()?;
        Ok(Self {
            embed,
            norm,
            layers,
        })
    }
}
