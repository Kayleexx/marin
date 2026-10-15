use kv::KvCache;
use tensor::{AttentionShape, attention, rms_norm_rows, rope_tokens, silu_mul, softmax};

use super::Config;
use super::weights::Layer;
use crate::ModelError;

pub type RouteObserver<'a> = Option<&'a mut dyn FnMut(usize, usize, &[f32], &[usize], &[f32])>;

pub(super) struct Scratch {
    pub normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    context: Vec<f32>,
    attended: Vec<f32>,
    router: Vec<f32>,
    input: Vec<f32>,
    expert: Vec<f32>,
    mixture: Vec<f32>,
}

impl Scratch {
    pub fn new(config: &Config) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Self {
            normed: vec![0.0; h],
            q: vec![0.0; h],
            k: vec![0.0; config.kv_dim()],
            v: vec![0.0; config.kv_dim()],
            context: vec![0.0; h],
            attended: vec![0.0; h],
            router: vec![0.0; config.num_local_experts],
            input: vec![0.0; i.saturating_mul(2)],
            expert: vec![0.0; h],
            mixture: vec![0.0; h],
        }
    }
}

pub(super) fn selected(logits: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>, ModelError> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(ModelError::InvalidConfig(
            "nonfinite Granite router logits".into(),
        ));
    }
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|a, b| {
        let left = logits.get(*a).copied().unwrap_or(f32::NEG_INFINITY);
        let right = logits.get(*b).copied().unwrap_or(f32::NEG_INFINITY);
        right.total_cmp(&left).then_with(|| a.cmp(b))
    });
    ids.truncate(top_k);
    let mut weights: Vec<f32> = ids
        .iter()
        .map(|id| logits.get(*id).copied().unwrap_or(f32::NEG_INFINITY))
        .collect();
    softmax(&mut weights);
    Ok(ids.into_iter().zip(weights).collect())
}

impl Layer {
    pub fn forward(
        &self,
        config: &Config,
        index: usize,
        hidden: &mut [f32],
        kv: &mut KvCache,
        scratch: &mut Scratch,
        route: &mut RouteObserver<'_>,
    ) -> Result<(), ModelError> {
        let position = kv.len();
        scratch.normed.copy_from_slice(hidden);
        rms_norm_rows(&mut scratch.normed, &self.input_norm, config.rms_norm_eps)?;
        self.q.matmul(&scratch.normed, &mut scratch.q)?;
        self.k.matmul(&scratch.normed, &mut scratch.k)?;
        self.v.matmul(&scratch.normed, &mut scratch.v)?;
        rope_tokens(
            &mut scratch.q,
            config.hidden_size,
            config.head_dim(),
            position,
            config.rope_theta,
        )?;
        rope_tokens(
            &mut scratch.k,
            config.kv_dim(),
            config.head_dim(),
            position,
            config.rope_theta,
        )?;
        // Marin attention divides by sqrt(head_dim); Granite's checkpoint uses its own scale.
        let query_scale = config.attention_multiplier * (config.head_dim() as f32).sqrt();
        for value in &mut scratch.q {
            *value *= query_scale;
        }
        kv.write(index, &scratch.k, &scratch.v)?;
        let (keys, values) = kv.read(index, position.saturating_add(1))?;
        attention(
            &AttentionShape {
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim(),
            },
            &scratch.q,
            keys,
            values,
            position,
            &mut scratch.context,
        )?;
        self.o.matmul(&scratch.context, &mut scratch.attended)?;
        for (value, update) in hidden.iter_mut().zip(&scratch.attended) {
            *value += config.residual_multiplier * update;
        }
        scratch.normed.copy_from_slice(hidden);
        rms_norm_rows(&mut scratch.normed, &self.post_norm, config.rms_norm_eps)?;
        self.router.matmul(&scratch.normed, &mut scratch.router)?;
        let choices = selected(&scratch.router, config.num_experts_per_tok)?;
        let ids: Vec<usize> = choices.iter().map(|(id, _)| *id).collect();
        let mut execution = choices;
        execution.sort_by_key(|(id, _)| *id);
        scratch.mixture.fill(0.0);
        for (id, weight) in execution {
            let expert = self.experts.get(id).ok_or_else(|| {
                ModelError::InvalidConfig(format!("Granite expert {id} is missing"))
            })?;
            expert.input.matmul(&scratch.normed, &mut scratch.input)?;
            let (gate, up) = scratch.input.split_at_mut(config.intermediate_size);
            silu_mul(gate, up)?;
            expert.output.matmul(gate, &mut scratch.expert)?;
            for (out, value) in scratch.mixture.iter_mut().zip(&scratch.expert) {
                *out += weight * value;
            }
        }
        if let Some(callback) = route.as_deref_mut() {
            callback(index, position, &scratch.router, &ids, &scratch.mixture);
        }
        for (value, update) in hidden.iter_mut().zip(&scratch.mixture) {
            *value += config.residual_multiplier * update;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_softmax_is_finite_stable_and_exact_on_ties() {
        let ranked = selected(&[0.0, 3.0, 3.0, -1.0], 2).unwrap();
        assert_eq!(ranked, [(1, 0.5), (2, 0.5)]);
        assert!(selected(&[f32::NAN], 1).is_err());
    }
}
