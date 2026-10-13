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
    },
}
impl<'a> Reader<'a> {
    pub(super) fn source(weights: &'a Weights, prefix: String) -> Self {
        Self::Source {
            weights,
            prefix,
            used: BTreeSet::new(),
        }
    }
    fn check(&mut self, name: &str, shape: &[usize]) -> Result<String, ModelError> {
        let Self::Source {
            weights,
            prefix,
            used,
        } = self
        else {
            return Ok(name.into());
        };
        let key = format!("{prefix}{name}");
        let info = weights
            .info(&key)
            .ok_or_else(|| invalid(format!("missing {key}")))?;
        if info.shape != shape || !matches!(info.dtype, Dtype::Bf16 | Dtype::F32) {
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
        if let Self::Source { weights, .. } = self
            && weights.info(&key).is_some_and(|i| i.dtype == Dtype::Bf16)
        {
            let data = weights.read_bf16(&key)?;
            if data.iter().any(|v| !tensor::bf16_to_f32(*v).is_finite()) {
                return Err(invalid(format!("nonfinite {key}")));
            }
            return Linear::bf16(rows, cols, data);
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
