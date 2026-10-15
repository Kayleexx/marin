use loader::{Dtype, Weights as Source};
use tensor::{Matrix, QuantBits, QuantScheme, Quantized, bf16_to_f32};

use super::Config;
use crate::ModelError;

pub(super) struct Expert {
    pub input: Matrix,
    pub output: Matrix,
}

pub(super) struct Layer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub router: Matrix,
    pub experts: Vec<Expert>,
}

pub(super) struct ResidentWeights {
    pub embed: Matrix,
    pub norm: Vec<f32>,
    pub layers: Vec<Layer>,
    pub quant: Option<QuantScheme>,
}

pub(crate) fn expected(config: &Config) -> Vec<(String, Vec<usize>)> {
    let h = config.hidden_size;
    let i = config.intermediate_size;
    let e = config.num_local_experts;
    let kv = config.kv_dim();
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, h],
        ),
        ("model.norm.weight".into(), vec![h]),
    ];
    for layer in 0..config.num_hidden_layers {
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}.weight");
        tensors.extend([
            (name("input_layernorm"), vec![h]),
            (name("post_attention_layernorm"), vec![h]),
            (name("self_attn.q_proj"), vec![h, h]),
            (name("self_attn.k_proj"), vec![kv, h]),
            (name("self_attn.v_proj"), vec![kv, h]),
            (name("self_attn.o_proj"), vec![h, h]),
            (name("block_sparse_moe.router.layer"), vec![e, h]),
            (
                name("block_sparse_moe.input_linear"),
                vec![e, i.saturating_mul(2), h],
            ),
            (name("block_sparse_moe.output_linear"), vec![e, h, i]),
        ]);
    }
    tensors
}

pub(crate) fn quantized(name: &str, shape: &[usize]) -> bool {
    shape.len() > 1 && !name.contains("router.layer")
}

pub fn weight_bytes(source: &Source, config: &Config) -> Result<u64, ModelError> {
    let packed = source.packed_group();
    let expected = expected(config);
    let mut problems = Vec::new();
    let mut total = 0_u64;
    for (name, shape) in &expected {
        let quantized = quantized(name, shape);
        let dtype = if packed.is_none() {
            Dtype::Bf16
        } else if quantized {
            Dtype::I8
        } else {
            Dtype::F32
        };
        match source.info(name) {
            None => problems.push(format!("missing tensor {name}")),
            Some(info) if info.dtype != dtype || info.shape != *shape => problems.push(format!(
                "tensor {name}: expected {dtype:?} {shape:?}, got {} {:?}",
                info.dtype_name(),
                info.shape
            )),
            Some(info) => total = total.saturating_add(info.byte_len()),
        }
        if quantized && let Some(group) = packed {
            let mut scales_shape = shape.clone();
            let Some(last) = scales_shape.last_mut() else {
                problems.push(format!("tensor {name}: missing columns"));
                continue;
            };
            if last.checked_rem(group) != Some(0) {
                problems.push(format!("tensor {name}: columns are not divisible by group"));
            } else {
                *last = last.checked_div(group).unwrap_or(0);
                let scale_name = format!("{name}.scales");
                match source.info(&scale_name) {
                    Some(info) if info.dtype == Dtype::F32 && info.shape == scales_shape => {
                        total = total.saturating_add(info.byte_len());
                    }
                    _ => problems.push(format!("missing/invalid scales {scale_name}")),
                }
            }
        }
    }
    let expected_count = expected.len().saturating_add(packed.map_or(0, |_| {
        expected
            .iter()
            .filter(|(name, shape)| quantized(name, shape))
            .count()
    }));
    if source.names().count() != expected_count {
        problems.push("unexpected Granite tensors".into());
    }
    if problems.is_empty() {
        Ok(total)
    } else {
        problems.truncate(12);
        Err(ModelError::InvalidWeights(problems))
    }
}

pub fn resident_bytes(
    source: &Source,
    config: &Config,
    quant: Option<QuantScheme>,
) -> Result<u64, ModelError> {
    let stored = weight_bytes(source, config)?;
    let Some(scheme) = quant else {
        return Ok(stored);
    };
    if source.packed_group().is_some() {
        return Ok(stored);
    }
    expected(config)
        .iter()
        .try_fold(0_u64, |total, (name, shape)| {
            let bytes = if quantized(name, shape) {
                let cols = shape.last().copied().unwrap_or(0);
                if cols.checked_rem(scheme.group) != Some(0) {
                    return Err(ModelError::Unsupported(format!(
                        "{name}: columns are not divisible by INT8 group"
                    )));
                }
                let rows = shape
                    .iter()
                    .take(shape.len().saturating_sub(1))
                    .try_fold(1_usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| {
                        ModelError::InvalidConfig("Granite resident size overflows".into())
                    })?;
                scheme
                    .matrix_bytes(rows, cols)
                    .ok_or_else(|| ModelError::InvalidConfig("Granite Q8 size overflows".into()))?
            } else if shape.len() == 1 {
                u64::try_from(shape.first().copied().unwrap_or(0))
                    .unwrap_or(u64::MAX)
                    .saturating_mul(4)
            } else {
                source.info(name).map_or(0, |info| info.byte_len())
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| ModelError::InvalidConfig("Granite resident size overflows".into()))
        })
}

struct Reader<'a, F: FnMut(u64, u64)> {
    source: &'a Source,
    progress: F,
    loaded: u64,
    total: u64,
    quant: Option<QuantScheme>,
}

impl<F: FnMut(u64, u64)> Reader<'_, F> {
    fn count(&mut self, name: &str, scales: bool) {
        let bytes = self.source.info(name).map_or(0, |info| info.byte_len());
        let scale_bytes = if scales {
            self.source
                .info(&format!("{name}.scales"))
                .map_or(0, |info| info.byte_len())
        } else {
            0
        };
        self.loaded = self
            .loaded
            .saturating_add(bytes)
            .saturating_add(scale_bytes);
        (self.progress)(self.loaded, self.total);
    }

    fn bits(&mut self, name: &str) -> Result<Vec<u16>, ModelError> {
        let bytes = self.source.info(name).map_or(0, |info| info.byte_len());
        let transient = bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| ModelError::InvalidConfig("Granite load memory overflows".into()))?;
        crate::qwen3::ensure_cpu_available(transient)?;
        let bits = self.source.read_bf16(name)?;
        self.count(name, false);
        Ok(bits)
    }

    fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix, ModelError> {
        if name.contains("router.layer") {
            if self.source.packed_group().is_some() {
                let values = self.source.read_f32(name)?;
                self.count(name, false);
                let bits = values
                    .into_iter()
                    .map(|v| {
                        if v.to_bits() & 0xffff != 0 {
                            Err(ModelError::InvalidWeights(vec![format!(
                                "{name}: router value is not BF16-exact"
                            )]))
                        } else {
                            Ok((v.to_bits() >> 16) as u16)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Matrix::new(rows, cols, bits)?);
            }
            return Ok(Matrix::new(rows, cols, self.bits(name)?)?);
        }
        if let Some(group) = self.source.packed_group() {
            let codes = self.source.read_i8(name)?;
            let scales = self.source.read_f32(&format!("{name}.scales"))?;
            self.count(name, true);
            return Ok(Matrix::from_int8(Quantized {
                rows,
                cols,
                scheme: QuantScheme::new(QuantBits::Int8, group)?,
                codes,
                scales,
            })?);
        }
        let bits = self.bits(name)?;
        Ok(if let Some(q) = self.quant {
            Matrix::quantized(rows, cols, bits, q)?
        } else {
            Matrix::new(rows, cols, bits)?
        })
    }

    fn vector(&mut self, name: &str) -> Result<Vec<f32>, ModelError> {
        if self.source.packed_group().is_some() {
            let values = self.source.read_f32(name)?;
            self.count(name, false);
            Ok(values)
        } else {
            Ok(self.bits(name)?.into_iter().map(bf16_to_f32).collect())
        }
    }

    fn experts(
        &mut self,
        name: &str,
        count: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<Matrix>, ModelError> {
        if let Some(group) = self.source.packed_group() {
            let codes = self.source.read_i8(name)?;
            let scales = self.source.read_f32(&format!("{name}.scales"))?;
            self.count(name, true);
            let per_codes = rows
                .checked_mul(cols)
                .ok_or_else(|| ModelError::InvalidConfig("Granite expert size overflows".into()))?;
            let per_scales = per_codes.checked_div(group).unwrap_or(0);
            return codes
                .chunks_exact(per_codes)
                .zip(scales.chunks_exact(per_scales))
                .map(|(c, s)| {
                    Ok(Matrix::from_int8(Quantized {
                        rows,
                        cols,
                        scheme: QuantScheme::new(QuantBits::Int8, group)?,
                        codes: c.to_vec(),
                        scales: s.to_vec(),
                    })?)
                })
                .collect();
        }
        let bits = self.bits(name)?;
        let per = rows
            .checked_mul(cols)
            .ok_or_else(|| ModelError::InvalidConfig("Granite expert size overflows".into()))?;
        let mut matrices = Vec::with_capacity(count);
        for chunk in bits.chunks_exact(per) {
            matrices.push(if let Some(q) = self.quant {
                Matrix::quantized(rows, cols, chunk.to_vec(), q)?
            } else {
                Matrix::new(rows, cols, chunk.to_vec())?
            });
        }
        Ok(matrices)
    }
}

impl ResidentWeights {
    pub fn load(
        source: &Source,
        config: &Config,
        progress: impl FnMut(u64, u64),
        quant: Option<QuantScheme>,
    ) -> Result<Self, ModelError> {
        let total = weight_bytes(source, config)?;
        let mut read = Reader {
            source,
            progress,
            loaded: 0,
            total,
            quant,
        };
        let (h, i, e, kv) = (
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            config.kv_dim(),
        );
        let embed = read.matrix("model.embed_tokens.weight", config.vocab_size, h)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let name = |suffix: &str| format!("model.layers.{layer}.{suffix}.weight");
            let inputs = read.experts(
                &name("block_sparse_moe.input_linear"),
                e,
                i.saturating_mul(2),
                h,
            )?;
            let outputs = read.experts(&name("block_sparse_moe.output_linear"), e, h, i)?;
            let experts = inputs
                .into_iter()
                .zip(outputs)
                .map(|(input, output)| Expert { input, output })
                .collect();
            layers.push(Layer {
                input_norm: read.vector(&name("input_layernorm"))?,
                post_norm: read.vector(&name("post_attention_layernorm"))?,
                q: read.matrix(&name("self_attn.q_proj"), h, h)?,
                k: read.matrix(&name("self_attn.k_proj"), kv, h)?,
                v: read.matrix(&name("self_attn.v_proj"), kv, h)?,
                o: read.matrix(&name("self_attn.o_proj"), h, h)?,
                router: read.matrix(&name("block_sparse_moe.router.layer"), e, h)?,
                experts,
            });
        }
        Ok(Self {
            embed,
            norm: read.vector("model.norm.weight")?,
            layers,
            quant,
        })
    }
}
