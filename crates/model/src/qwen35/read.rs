use super::math::{Linear, invalid, size};
use crate::ModelError;
use loader::{Dtype, Weights};
use std::collections::{BTreeMap, BTreeSet};

pub(super) enum Reader<'a> {
    Fixture(BTreeMap<String, Vec<f32>>),
    Source {
        weights: &'a Weights,
        prefix: String,
        used: BTreeSet<String>,
        quant: Option<tensor::QuantScheme>,
    },
}
impl<'a> Reader<'a> {
    pub(super) fn source(
        weights: &'a Weights,
        prefix: String,
        quant: Option<tensor::QuantScheme>,
    ) -> Self {
        Self::Source {
            weights,
            prefix,
            used: BTreeSet::new(),
            quant,
        }
    }
    fn check(&mut self, name: &str, shape: &[usize]) -> Result<String, ModelError> {
        let Self::Source {
            weights,
            prefix,
            used,
            ..
        } = self
        else {
            return Ok(name.into());
        };
        let key = format!("{prefix}{name}");
        let info = weights
            .info(&key)
            .ok_or_else(|| invalid(format!("missing {key}")))?;
        let valid_dtype = if weights.packed_group().is_some() {
            info.dtype
                == if super::layout::quantized(&key) {
                    Dtype::I8
                } else {
                    Dtype::F32
                }
        } else {
            matches!(info.dtype, Dtype::Bf16 | Dtype::F32)
        };
        if info.shape != shape || !valid_dtype {
            return Err(invalid(format!(
                "{key}: wrong shape/dtype {:?} {} expected {shape:?}",
                info.shape,
                info.dtype_name()
            )));
        }
        used.insert(key.clone());
        Ok(key)
    }
    pub(super) fn vector_shape(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<Vec<f32>, ModelError> {
        let n = shape.iter().try_fold(1_usize, |n, x| size(n, *x))?;
        let key = self.check(name, shape)?;
        let out = match self {
            Self::Fixture(data) => data
                .remove(name)
                .ok_or_else(|| invalid(format!("missing {name}")))?,
            Self::Source { weights, .. } => match weights.info(&key).map(|i| i.dtype) {
                Some(Dtype::F32) => weights.read_f32(&key)?,
                _ => weights
                    .read_bf16(&key)?
                    .into_iter()
                    .map(tensor::bf16_to_f32)
                    .collect(),
            },
        };
        if out.len() != n {
            return Err(invalid(format!(
                "{name}: expected {n} values, got {}",
                out.len()
            )));
        }
        tensor::ensure_finite(&out, "Qwen3.5 weight")?;
        Ok(out)
    }
    pub(super) fn vector(&mut self, name: &str, n: usize) -> Result<Vec<f32>, ModelError> {
        self.vector_shape(name, &[n])
    }
    pub(super) fn norm(&mut self, name: &str, n: usize) -> Result<Vec<f32>, ModelError> {
        let mut out = self.vector(name, n)?;
        for v in &mut out {
            *v += 1.0;
        }
        tensor::ensure_finite(&out, "zero-centered norm")?;
        Ok(out)
    }
    pub(super) fn matrix(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Linear, ModelError> {
        let key = self.check(name, &[rows, cols])?;
        if let Self::Source {
            weights,
            quant,
            used,
            ..
        } = self
            && (super::layout::quantized(&key)
                || (quant.is_none() && weights.packed_group().is_none()))
            && (weights.packed_group().is_some()
                || weights.info(&key).is_some_and(|i| i.dtype == Dtype::Bf16))
        {
            if weights.packed_group().is_some() {
                used.insert(format!("{key}.scales"));
            }
            return Ok(Linear::stored(matrix(weights, &key, rows, cols, *quant)?));
        }
        Ok(Linear::new(self.vector_shape(name, &[rows, cols])?, cols))
    }
    pub(super) fn finish(self) -> Result<(), ModelError> {
        let extra = match self {
            Self::Fixture(data) => data.into_keys().collect::<Vec<_>>(),
            Self::Source {
                weights,
                prefix,
                used,
                ..
            } => weights
                .names()
                .filter(|n| n.starts_with(&prefix) && !used.contains(*n))
                .map(str::to_owned)
                .collect(),
        };
        if !extra.is_empty() {
            return Err(invalid(format!("unexpected block weights: {extra:?}")));
        }
        Ok(())
    }
}

pub(super) fn matrix(
    weights: &Weights,
    name: &str,
    rows: usize,
    cols: usize,
    quant: Option<tensor::QuantScheme>,
) -> Result<tensor::Matrix, ModelError> {
    if let Some(group) = weights.packed_group() {
        return Ok(tensor::Matrix::from_int8(tensor::Quantized {
            rows,
            cols,
            scheme: tensor::QuantScheme::new(tensor::QuantBits::Int8, group)?,
            codes: weights.read_i8(name)?,
            scales: weights.read_f32(&format!("{name}.scales"))?,
        })?);
    }
    let bits = weights.read_bf16(name)?;
    if let Some(q) = quant {
        return Ok(tensor::Matrix::quantized(rows, cols, bits, q)?);
    }
    if bits.iter().any(|v| !tensor::bf16_to_f32(*v).is_finite()) {
        return Err(invalid(format!("nonfinite {name}")));
    }
    Ok(tensor::Matrix::new(rows, cols, bits)?)
}
