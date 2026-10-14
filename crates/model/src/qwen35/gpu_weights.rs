use super::math::{invalid, size};
use super::{Config, LayerKind, TextConfig};
use crate::ModelError;
use crate::qwen3::weight::Weight;
use cuda::{Buffer, Gpu};
use loader::{Dtype, Weights};
use tensor::{QuantBits, QuantScheme, Quantized, quantize};

pub(super) struct Common {
    pub input_norm: Buffer,
    pub post_norm: Buffer,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub(super) struct Delta {
    pub qkv: Weight,
    pub z: Weight,
    pub a: Weight,
    pub b: Weight,
    pub out: Weight,
    pub conv: Buffer,
    pub norm: Buffer,
    pub a_log: Buffer,
    pub dt_bias: Buffer,
}

pub(super) struct Full {
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub out: Weight,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub index: usize,
}

pub(super) enum Mixer {
    Delta(Box<Delta>),
    Full(Box<Full>),
}

pub(super) struct Layer {
    pub common: Common,
    pub mixer: Mixer,
}

struct Reader<'a> {
    weights: &'a Weights,
    gpu: &'a Gpu,
    prefix: String,
    quant: Option<QuantScheme>,
}

impl Reader<'_> {
    fn name(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.prefix)
    }

    fn matrix(&self, suffix: &str, rows: usize, cols: usize) -> Result<Weight, ModelError> {
        let name = self.name(suffix);
        if let Some(group) = self
            .weights
            .packed_group()
            .filter(|_| super::layout::quantized(&name))
        {
            let q = Quantized {
                rows,
                cols,
                scheme: QuantScheme::new(QuantBits::Int8, group)?,
                codes: self.weights.read_i8(&name)?,
                scales: self.weights.read_f32(&format!("{name}.scales"))?,
            };
            q.validate_int8()?;
            return Ok(Weight::Quant(self.gpu.upload_quantized(&q)?));
        }
        if self.weights.packed_group().is_some() {
            let values = self.weights.read_f32(&name)?;
            if values.len() != size(rows, cols)? || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!("invalid FP32 matrix {name}")));
            }
            let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            if bits
                .iter()
                .zip(&values)
                .any(|(b, v)| tensor::bf16_to_f32(*b) != *v)
            {
                return Err(invalid(format!("{name} cannot be represented as BF16")));
            }
            return Ok(Weight::Bf16(self.gpu.upload_matrix(rows, cols, &bits)?));
        }
        let bits = self.weights.read_bf16(&name)?;
        if bits.len() != size(rows, cols)?
            || bits.iter().any(|v| !tensor::bf16_to_f32(*v).is_finite())
        {
            return Err(invalid(format!("invalid BF16 matrix {name}")));
        }
        if let Some(scheme) = self.quant.filter(|_| super::layout::quantized(&name)) {
            let q = quantize(&bits, rows, cols, scheme)?;
            return Ok(Weight::Quant(self.gpu.upload_quantized(&q)?));
        }
        Ok(Weight::Bf16(self.gpu.upload_matrix(rows, cols, &bits)?))
    }

    fn vector(&self, suffix: &str, n: usize, zero_centered: bool) -> Result<Buffer, ModelError> {
        let name = self.name(suffix);
        let values = match self.weights.info(&name).map(|i| i.dtype) {
            Some(Dtype::F32) => self.weights.read_f32(&name)?,
            Some(Dtype::Bf16) => self
                .weights
                .read_bf16(&name)?
                .into_iter()
                .map(tensor::bf16_to_f32)
                .collect(),
            _ => return Err(invalid(format!("invalid vector {name}"))),
        };
        if values.len() != n || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!("invalid values in {name}")));
        }
        let values: Vec<f32> = if zero_centered {
            values.into_iter().map(|v| v + 1.0).collect()
        } else {
            values
        };
        Ok(self.gpu.upload(&values)?)
    }
}

pub(super) fn load_layers(
    weights: &Weights,
    gpu: &Gpu,
    config: &TextConfig,
    quant: Option<QuantScheme>,
    mut progress: impl FnMut(u64, u64),
) -> Result<(Weight, Buffer, Vec<Layer>), ModelError> {
    let summary = config.summary();
    let c = &config.block;
    let root = Reader {
        weights,
        gpu,
        prefix: "model.language_model.".into(),
        quant,
    };
    let embed = root.matrix("embed_tokens.weight", summary.vocab_size, c.hidden_size)?;
    let norm = root.vector("norm.weight", c.hidden_size, true)?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.layer_types.len())
        .map_err(|e| invalid(format!("GPU layer allocation: {e}")))?;
    let total = super::weight_bytes(weights, quant)?;
    let mut full_index = 0;
    for (i, kind) in config.layer_types.iter().enumerate() {
        let r = Reader {
            weights,
            gpu,
            prefix: format!("model.language_model.layers.{i}."),
            quant,
        };
        let common = Common {
            input_norm: r.vector("input_layernorm.weight", c.hidden_size, true)?,
            post_norm: r.vector("post_attention_layernorm.weight", c.hidden_size, true)?,
            gate: r.matrix("mlp.gate_proj.weight", c.intermediate_size, c.hidden_size)?,
            up: r.matrix("mlp.up_proj.weight", c.intermediate_size, c.hidden_size)?,
            down: r.matrix("mlp.down_proj.weight", c.hidden_size, c.intermediate_size)?,
        };
        let mixer = match kind {
            LayerKind::LinearAttention => Mixer::Delta(Box::new(load_delta(&r, c)?)),
            LayerKind::FullAttention => {
                let full = load_full(&r, c, full_index)?;
                full_index = full_index.saturating_add(1);
                Mixer::Full(Box::new(full))
            }
        };
        layers.push(Layer { common, mixer });
        progress(
            (i as u64)
                .saturating_add(1)
                .saturating_mul(total)
                .checked_div(config.layer_types.len() as u64)
                .unwrap_or(0),
            total,
        );
    }
    gpu.synchronize()?;
    Ok((embed, norm, layers))
}

fn load_delta(r: &Reader<'_>, c: &Config) -> Result<Delta, ModelError> {
    let value = c.value_dim()?;
    Ok(Delta {
        qkv: r.matrix(
            "linear_attn.in_proj_qkv.weight",
            c.channels()?,
            c.hidden_size,
        )?,
        z: r.matrix("linear_attn.in_proj_z.weight", value, c.hidden_size)?,
        a: r.matrix(
            "linear_attn.in_proj_a.weight",
            c.linear_num_value_heads,
            c.hidden_size,
        )?,
        b: r.matrix(
            "linear_attn.in_proj_b.weight",
            c.linear_num_value_heads,
            c.hidden_size,
        )?,
        out: r.matrix("linear_attn.out_proj.weight", c.hidden_size, value)?,
        conv: r.vector(
            "linear_attn.conv1d.weight",
            size(c.channels()?, c.linear_conv_kernel_dim)?,
            false,
        )?,
        norm: r.vector("linear_attn.norm.weight", c.linear_value_head_dim, false)?,
        a_log: r.vector("linear_attn.A_log", c.linear_num_value_heads, false)?,
        dt_bias: r.vector("linear_attn.dt_bias", c.linear_num_value_heads, false)?,
    })
}

fn load_full(r: &Reader<'_>, c: &Config, index: usize) -> Result<Full, ModelError> {
    let q = size(c.num_attention_heads, c.head_dim)?;
    let kv = size(c.num_key_value_heads, c.head_dim)?;
    Ok(Full {
        q: r.matrix("self_attn.q_proj.weight", size(q, 2)?, c.hidden_size)?,
        k: r.matrix("self_attn.k_proj.weight", kv, c.hidden_size)?,
        v: r.matrix("self_attn.v_proj.weight", kv, c.hidden_size)?,
        out: r.matrix("self_attn.o_proj.weight", c.hidden_size, q)?,
        q_norm: r.vector("self_attn.q_norm.weight", c.head_dim, true)?,
        k_norm: r.vector("self_attn.k_norm.weight", c.head_dim, true)?,
        index,
    })
}
