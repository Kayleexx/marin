use kv::KvCache;
use tensor::{AttentionShape, attention, rms_norm_rows, rope_tokens, silu_mul};

use super::Config;
use super::layer::{RouteObserver, selected};
use super::weights::{Layer, ResidentWeights};
use crate::{ModelError, Observer};

fn zeros(len: usize) -> Result<Vec<f32>, ModelError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| ModelError::InvalidConfig(format!("Granite workspace allocation: {e}")))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn size(rows: usize, cols: usize) -> Result<usize, ModelError> {
    rows.checked_mul(cols)
        .ok_or_else(|| ModelError::InvalidConfig("Granite workspace size overflows".into()))
}

fn row(data: &[f32], width: usize, index: usize) -> Result<&[f32], ModelError> {
    data.chunks_exact(width)
        .nth(index)
        .ok_or_else(|| ModelError::InvalidConfig("Granite batch row is missing".into()))
}

fn row_mut(data: &mut [f32], width: usize, index: usize) -> Result<&mut [f32], ModelError> {
    data.chunks_exact_mut(width)
        .nth(index)
        .ok_or_else(|| ModelError::InvalidConfig("Granite batch row is missing".into()))
}

pub(super) fn workspace_bytes(config: &Config, tokens: usize) -> Result<u64, ModelError> {
    let floats = config
        .hidden_size
        .checked_mul(5)
        .and_then(|n| n.checked_add(config.kv_dim().checked_mul(2)?))
        .and_then(|n| n.checked_add(config.intermediate_size.checked_mul(2)?))
        .and_then(|n| n.checked_add(config.num_local_experts))
        .and_then(|n| n.checked_mul(tokens))
        .and_then(|n| n.checked_add(config.vocab_size))
        .ok_or_else(|| ModelError::InvalidConfig("Granite workspace size overflows".into()))?;
    let routing = config
        .num_experts_per_tok
        .checked_mul(tokens)
        .and_then(|n| n.checked_add(config.num_local_experts.checked_mul(tokens)?))
        .and_then(|n| n.checked_mul(size_of::<(usize, f32)>()))
        .and_then(|n| n.checked_add(tokens.checked_mul(size_of::<Vec<(usize, f32)>>())?))
        .and_then(|n| {
            n.checked_add(
                config
                    .num_local_experts
                    .checked_mul(size_of::<Vec<(usize, f32)>>())?,
            )
        })
        .ok_or_else(|| ModelError::InvalidConfig("Granite routing size overflows".into()))?;
    u64::try_from(
        floats
            .checked_mul(size_of::<f32>())
            .and_then(|n| n.checked_add(routing))
            .ok_or_else(|| ModelError::InvalidConfig("Granite workspace size overflows".into()))?,
    )
    .map_err(|_| ModelError::InvalidConfig("Granite workspace size overflows".into()))
}

struct Batch {
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    context: Vec<f32>,
    attended: Vec<f32>,
    router: Vec<f32>,
    input: Vec<f32>,
    choices: Vec<Vec<(usize, f32)>>,
    groups: Vec<Vec<(usize, f32)>>,
}

impl Batch {
    fn new(config: &Config, tokens: usize) -> Result<Self, ModelError> {
        let h = size(tokens, config.hidden_size)?;
        let kv = size(tokens, config.kv_dim())?;
        let router = size(tokens, config.num_local_experts)?;
        let input = size(tokens, size(2, config.intermediate_size)?)?;
        let mut choices = Vec::new();
        choices
            .try_reserve_exact(tokens)
            .map_err(|e| ModelError::InvalidConfig(format!("Granite routing allocation: {e}")))?;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(config.num_local_experts)
            .map_err(|e| ModelError::InvalidConfig(format!("Granite routing allocation: {e}")))?;
        for _ in 0..config.num_local_experts {
            let mut group = Vec::new();
            group.try_reserve_exact(tokens).map_err(|e| {
                ModelError::InvalidConfig(format!("Granite routing allocation: {e}"))
            })?;
            groups.push(group);
        }
        Ok(Self {
            normed: zeros(h)?,
            q: zeros(h)?,
            k: zeros(kv)?,
            v: zeros(kv)?,
            context: zeros(h)?,
            attended: zeros(h)?,
            router: zeros(router)?,
            input: zeros(input)?,
            choices,
            groups,
        })
    }
}

impl Layer {
    fn forward_batch(
        &self,
        config: &Config,
        index: usize,
        hidden: &mut [f32],
        kv: &mut KvCache,
        batch: &mut Batch,
        route: &mut RouteObserver<'_>,
    ) -> Result<(), ModelError> {
        let (h, i, e) = (
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
        );
        let start = kv.len();
        let tokens = hidden.chunks_exact(h).len();
        batch.normed.copy_from_slice(hidden);
        rms_norm_rows(&mut batch.normed, &self.input_norm, config.rms_norm_eps)?;
        self.q.matmul(&batch.normed, &mut batch.q)?;
        self.k.matmul(&batch.normed, &mut batch.k)?;
        self.v.matmul(&batch.normed, &mut batch.v)?;
        rope_tokens(&mut batch.q, h, config.head_dim(), start, config.rope_theta)?;
        rope_tokens(
            &mut batch.k,
            config.kv_dim(),
            config.head_dim(),
            start,
            config.rope_theta,
        )?;
        let query_scale = config.attention_multiplier * (config.head_dim() as f32).sqrt();
        for value in &mut batch.q {
            *value *= query_scale;
        }
        kv.write(index, &batch.k, &batch.v)?;
        let (keys, values) = kv.read(index, start.saturating_add(tokens))?;
        attention(
            &AttentionShape {
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim(),
            },
            &batch.q,
            keys,
            values,
            start,
            &mut batch.context,
        )?;
        self.o.matmul(&batch.context, &mut batch.attended)?;
        for (value, update) in hidden.iter_mut().zip(&batch.attended) {
            *value += config.residual_multiplier * update;
        }
        batch.normed.copy_from_slice(hidden);
        rms_norm_rows(&mut batch.normed, &self.post_norm, config.rms_norm_eps)?;
        self.router.matmul(&batch.normed, &mut batch.router)?;
        batch.choices.clear();
        for group in &mut batch.groups {
            group.clear();
        }
        for (token, logits) in batch.router.chunks_exact(e).enumerate() {
            let choices = selected(logits, config.num_experts_per_tok)?;
            for &(id, weight) in &choices {
                let group = batch.groups.get_mut(id).ok_or_else(|| {
                    ModelError::InvalidConfig(format!("Granite expert {id} is missing"))
                })?;
                group.push((token, weight));
            }
            batch.choices.push(choices);
        }
        batch.q.fill(0.0);
        for (id, group) in batch.groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let expert = self.experts.get(id).ok_or_else(|| {
                ModelError::InvalidConfig(format!("Granite expert {id} is missing"))
            })?;
            let count = group.len();
            for (slot, (token, _)) in group.iter().enumerate() {
                row_mut(&mut batch.context, h, slot)?.copy_from_slice(row(
                    &batch.normed,
                    h,
                    *token,
                )?);
            }
            let gathered = batch.context.get(..size(count, h)?).ok_or_else(|| {
                ModelError::InvalidConfig("Granite expert input is missing".into())
            })?;
            let input = batch
                .input
                .get_mut(..size(count, size(2, i)?)?)
                .ok_or_else(|| {
                    ModelError::InvalidConfig("Granite expert workspace is missing".into())
                })?;
            expert.input.matmul(gathered, input)?;
            for row in input.chunks_exact_mut(size(2, i)?) {
                let (gate, up) = row.split_at_mut(i);
                silu_mul(gate, up)?;
            }
            for slot in 0..count {
                let source = size(slot, size(2, i)?)?..size(slot, size(2, i)?)?.saturating_add(i);
                let destination = size(slot, i)?;
                if batch.input.get(source.clone()).is_none()
                    || batch
                        .input
                        .get(destination..destination.saturating_add(i))
                        .is_none()
                {
                    return Err(ModelError::InvalidConfig(
                        "Granite expert workspace is missing".into(),
                    ));
                }
                batch.input.copy_within(source, destination);
            }
            let activated = batch.input.get(..size(count, i)?).ok_or_else(|| {
                ModelError::InvalidConfig("Granite expert activation is missing".into())
            })?;
            let output = batch.attended.get_mut(..size(count, h)?).ok_or_else(|| {
                ModelError::InvalidConfig("Granite expert output is missing".into())
            })?;
            expert.output.matmul(activated, output)?;
            for (slot, (token, weight)) in group.iter().enumerate() {
                for (out, value) in
                    row_mut(&mut batch.q, h, *token)?
                        .iter_mut()
                        .zip(row(&batch.attended, h, slot)?)
                {
                    *out += weight * value;
                }
            }
        }
        if let Some(callback) = route.as_deref_mut() {
            for (token, choices) in batch.choices.iter().enumerate() {
                let ids: Vec<usize> = choices.iter().map(|(id, _)| *id).collect();
                callback(
                    index,
                    start.saturating_add(token),
                    row(&batch.router, e, token)?,
                    &ids,
                    row(&batch.q, h, token)?,
                );
            }
        }
        for (value, update) in hidden.iter_mut().zip(&batch.q) {
            *value += config.residual_multiplier * update;
        }
        Ok(())
    }
}

pub(super) fn run(
    config: &Config,
    weights: &ResidentWeights,
    tokens: &[u32],
    kv: &mut KvCache,
    observe: &mut Observer<'_>,
    route: &mut RouteObserver<'_>,
) -> Result<Vec<f32>, ModelError> {
    let h = config.hidden_size;
    let mut hidden = zeros(size(tokens.len(), h)?)?;
    for (&token, row) in tokens.iter().zip(hidden.chunks_exact_mut(h)) {
        weights.embed.row_f32(token as usize, row)?;
        for value in row {
            *value *= config.embedding_multiplier;
        }
    }
    if let Some(callback) = observe.as_deref_mut() {
        callback(
            0,
            hidden.chunks_exact(h).last().ok_or(ModelError::NoTokens)?,
        );
    }
    let mut batch = Batch::new(config, tokens.len())?;
    for (index, layer) in weights.layers.iter().enumerate() {
        layer.forward_batch(config, index, &mut hidden, kv, &mut batch, route)?;
        if let Some(callback) = observe.as_deref_mut() {
            callback(
                index.saturating_add(1),
                hidden.chunks_exact(h).last().ok_or(ModelError::NoTokens)?,
            );
        }
    }
    kv.advance(tokens.len())?;
    Ok(hidden
        .chunks_exact(h)
        .last()
        .ok_or(ModelError::NoTokens)?
        .to_vec())
}
