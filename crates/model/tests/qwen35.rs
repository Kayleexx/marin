use model::qwen35::{Config, LayerKind, ReferenceBlock, State};
use serde::de::Error;
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
const BUDGET: usize = 32 << 20;
#[derive(Deserialize)]
struct Fixture {
    transformers: String,
    source_sha256: String,
    blocks: Vec<Block>,
    #[serde(deserialize_with = "nested")]
    hybrid: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Block {
    config: Config,
    kind: LayerKind,
    #[serde(deserialize_with = "mapped")]
    weights: BTreeMap<String, Vec<f32>>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    #[serde(deserialize_with = "floats")]
    prefix: Vec<f32>,
    #[serde(deserialize_with = "floats")]
    input: Vec<f32>,
    #[serde(deserialize_with = "mapped")]
    expected: BTreeMap<String, Vec<f32>>,
    #[serde(deserialize_with = "floats")]
    conv_state: Vec<f32>,
    #[serde(deserialize_with = "floats")]
    recurrent: Vec<f32>,
    #[serde(deserialize_with = "floats")]
    keys: Vec<f32>,
    #[serde(deserialize_with = "floats")]
    values: Vec<f32>,
}
#[derive(Deserialize)]
struct Span {
    offset: usize,
    len: usize,
}
fn read_span(s: Span) -> std::result::Result<Vec<f32>, &'static str> {
    let bytes = include_bytes!("../../../tests/reference/qwen35/values.f32");
    let end = s
        .len
        .checked_mul(4)
        .and_then(|n| n.checked_add(s.offset))
        .ok_or("span overflow")?;
    let data = bytes
        .get(s.offset..end)
        .ok_or("invalid FP32 fixture span")?;
    data.chunks_exact(4)
        .map(|b| Ok(f32::from_le_bytes(b.try_into().map_err(|_| "float shape")?)))
        .collect()
}
fn floats<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<f32>, D::Error> {
    read_span(Span::deserialize(d)?).map_err(D::Error::custom)
}
fn mapped<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<String, Vec<f32>>, D::Error> {
    BTreeMap::<String, Span>::deserialize(d)?
        .into_iter()
        .map(|(name, span)| Ok((name, read_span(span).map_err(D::Error::custom)?)))
        .collect()
}
fn nested<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<Vec<f32>>, D::Error> {
    Vec::<Span>::deserialize(d)?
        .into_iter()
        .map(|s| read_span(s).map_err(D::Error::custom))
        .collect()
}
fn fixture() -> Result<Fixture> {
    Ok(serde_json::from_str(include_str!(
        "../../../tests/reference/qwen35/blocks.json"
    ))?)
}
fn block(b: &Block) -> Result<ReferenceBlock> {
    Ok(ReferenceBlock::new(
        b.config.clone(),
        b.kind,
        b.weights.clone(),
    )?)
}
fn close(actual: &[f32], expected: &[f32], name: &str) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len(), "{name} shape");
    assert!(actual.iter().all(|v| v.is_finite()), "{name} nonfinite");
    if actual.is_empty() {
        return (0.0, 0.0);
    }
    let max_ref = expected
        .iter()
        .map(|x| f64::from(*x).abs())
        .fold(1.0, f64::max);
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    let square_error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum();
    let square_ref: f64 = expected.iter().map(|v| f64::from(*v).powi(2)).sum();
    let abs = max_error / max_ref;
    let rel = (square_error / actual.len() as f64).sqrt()
        / (square_ref / actual.len() as f64).sqrt().max(1e-6);
    assert!(
        abs <= 1e-4 && rel <= 1e-5,
        "{name}: normalized max={abs:e}, relative RMS={rel:e}"
    );
    (abs, rel)
}
fn state_close(s: &State, c: &Case) -> Result {
    if let Some(conv) = s.convolution() {
        close(conv, &c.conv_state, "convolution state");
    }
    if let Some(rec) = s.recurrent() {
        close(rec, &c.recurrent, "recurrent state");
    }
    if !c.keys.is_empty() {
        let (k, v) = s.attention_kv()?;
        close(k, &c.keys, "KV keys");
        close(v, &c.values, "KV values");
    }
    Ok(())
}
fn contents(s: &State) -> Vec<f32> {
    let mut out = vec![s.len() as f32];
    if let Some(x) = s.convolution() {
        out.extend_from_slice(x);
    }
    if let Some(x) = s.recurrent() {
        out.extend_from_slice(x);
    }
    if let Ok((k, v)) = s.attention_kv() {
        out.extend_from_slice(k);
        out.extend_from_slice(v);
    }
    out
}

#[test]
fn hf_intermediates_chunked_prefill_and_final_state() -> Result {
    let fixture = fixture()?;
    assert_eq!(fixture.transformers, "5.17.0");
    assert_eq!(fixture.source_sha256.len(), 64);
    let mut worst = (0.0_f64, 0.0_f64);
    for (i, b) in fixture.blocks.iter().enumerate() {
        let block = block(b)?;
        for c in &b.cases {
            let mut state = block.new_state(512, BUDGET)?;
            if !c.prefix.is_empty() {
                block.forward(&c.prefix, &mut state, BUDGET, None)?;
            }
            let mut seen: BTreeMap<String, Vec<f32>> = BTreeMap::new();
            let output = block.forward(
                &c.input,
                &mut state,
                BUDGET,
                Some(&mut |name, x| {
                    seen.entry(name.to_owned())
                        .or_default()
                        .extend_from_slice(x);
                }),
            )?;
            for (name, expected) in &c.expected {
                let label = format!(
                    "block {i} {} tokens {name}",
                    c.input.len() / b.config.hidden_size
                );
                let score = close(
                    seen.get(name).ok_or("missing intermediate")?,
                    expected,
                    &label,
                );
                worst.0 = worst.0.max(score.0);
                worst.1 = worst.1.max(score.1);
            }
            close(&output, &c.expected["output"], "returned output");
            state_close(&state, c)?;
        }
    }
    println!(
        "HF intermediate worst normalized-max={:e} relative-RMS={:e}",
        worst.0, worst.1
    );
    Ok(())
}

#[test]
fn split_decode_reset_and_independent_sessions() -> Result {
    for b in fixture()?.blocks {
        let block = block(&b)?;
        for c in &b.cases {
            let mut whole = block.new_state(512, BUDGET)?;
            let mut incremental = block.new_state(512, BUDGET)?;
            let mut separate = block.new_state(512, BUDGET)?;
            let untouched = contents(&separate);
            if !c.prefix.is_empty() {
                block.forward(&c.prefix, &mut whole, BUDGET, None)?;
                block.forward(&c.prefix, &mut incremental, BUDGET, None)?;
            }
            let expected = block.forward(&c.input, &mut whole, BUDGET, None)?;
            let mut actual = Vec::new();
            // Multi-token continuation followed by single-token decode crosses the conv boundary.
            let prefill = (c.input.len() / b.config.hidden_size).min(4) * b.config.hidden_size;
            actual.extend(block.forward(&c.input[..prefill], &mut incremental, BUDGET, None)?);
            for token in c.input[prefill..].chunks_exact(b.config.hidden_size) {
                actual.extend(block.forward(token, &mut incremental, BUDGET, None)?);
            }
            assert_eq!(actual, expected);
            assert_eq!(contents(&whole), contents(&incremental));
            state_close(&incremental, c)?;
            assert_eq!(contents(&separate), untouched);
            incremental.reset();
            assert!(incremental.is_empty());
            if !c.prefix.is_empty() {
                block.forward(&c.prefix, &mut incremental, BUDGET, None)?;
                block.forward(&c.prefix, &mut separate, BUDGET, None)?;
            }
            assert_eq!(
                block.forward(&c.input, &mut incremental, BUDGET, None)?,
                expected
            );
            let mut tokenwise = Vec::new();
            for token in c.input.chunks_exact(b.config.hidden_size) {
                tokenwise.extend(block.forward(token, &mut separate, BUDGET, None)?);
            }
            assert_eq!(tokenwise, expected);
            assert_eq!(contents(&separate), contents(&whole));
        }
    }
    Ok(())
}

#[test]
fn hybrid_three_delta_one_full_matches_hf() -> Result {
    let f = fixture()?;
    let input = &f.blocks[0].cases[4].input;
    let mut hidden = input.clone();
    for (i, expected) in f.hybrid.iter().enumerate() {
        let b = &f.blocks[usize::from(i == 3)];
        let block = block(b)?;
        let mut state = block.new_state(512, BUDGET)?;
        hidden = block.forward(&hidden, &mut state, BUDGET, None)?;
        close(&hidden, expected, "hybrid layer output");
    }
    Ok(())
}

#[test]
fn shape_context_memory_and_numeric_failures_do_not_commit() -> Result {
    for b in fixture()?.blocks {
        let block = block(&b)?;
        assert!(block.new_state(8, 0).is_err());
        assert!(block.new_state(0, BUDGET).is_err());
        assert!(block.additional_bytes(usize::MAX, usize::MAX).is_err());
        let mut s = block.new_state(8, BUDGET)?;
        let token = &b.cases[0].input;
        block.forward(token, &mut s, BUDGET, None)?;
        let held = contents(&s);
        for invalid in [
            Vec::new(),
            vec![1.0],
            vec![f32::NAN; b.config.hidden_size],
            vec![0.0; b.config.hidden_size * 8],
        ] {
            assert!(block.forward(&invalid, &mut s, BUDGET, None).is_err());
            assert_eq!(contents(&s), held);
        }
        let needed = block.additional_bytes(1, 2)?;
        assert!(block.forward(token, &mut s, needed - 1, None).is_err());
        assert_eq!(contents(&s), held);
        block.forward(token, &mut s, needed, None)?;
        assert_eq!(s.len(), 2);
        let mut weights = b.weights.clone();
        weights
            .get_mut("mlp.gate_proj.weight")
            .unwrap()
            .fill(1000.0);
        weights.get_mut("mlp.up_proj.weight").unwrap().fill(1000.0);
        weights
            .get_mut("mlp.down_proj.weight")
            .unwrap()
            .fill(f32::MAX);
        let broken = ReferenceBlock::new(b.config.clone(), b.kind, weights)?;
        let held = contents(&s);
        let mut observed_core = false;
        assert!(
            broken
                .forward(
                    &vec![1.0; b.config.hidden_size],
                    &mut s,
                    BUDGET,
                    Some(&mut |name, _| {
                        observed_core |= name == "core";
                    })
                )
                .is_err()
        );
        assert!(observed_core, "failure must follow state update");
        assert_eq!(contents(&s), held);
    }
    Ok(())
}

#[test]
fn malformed_weights_config_and_wrong_state_are_rejected() -> Result {
    let f = fixture()?;
    let b = &f.blocks[0];
    for action in 0..4 {
        let mut weights = b.weights.clone();
        match action {
            0 => {
                weights.remove("linear_attn.A_log");
            }
            1 => {
                weights.insert("unused".into(), vec![0.0]);
            }
            2 => {
                weights.get_mut("linear_attn.A_log").unwrap().push(0.0);
            }
            _ => {
                weights.get_mut("linear_attn.A_log").unwrap()[0] = f32::INFINITY;
            }
        }
        assert!(ReferenceBlock::new(b.config.clone(), b.kind, weights).is_err());
    }
    let mut config = b.config.clone();
    config.hidden_size = usize::MAX;
    assert!(ReferenceBlock::new(config, b.kind, b.weights.clone()).is_err());
    let mut config = b.config.clone();
    config.linear_num_key_heads = 0;
    assert!(ReferenceBlock::new(config, b.kind, b.weights.clone()).is_err());
    let mut config = b.config.clone();
    config.rope_parameters.partial_rotary_factor = f32::NAN;
    assert!(ReferenceBlock::new(config, b.kind, b.weights.clone()).is_err());
    for setting in 0..4 {
        let mut config = b.config.clone();
        match setting {
            0 => config.hidden_act = "gelu".into(),
            1 => config.attention_bias = true,
            2 => config.attn_output_gate = false,
            _ => config.rope_parameters.rope_type = "dynamic".into(),
        }
        assert!(ReferenceBlock::new(config, b.kind, b.weights.clone()).is_err());
    }
    let linear = block(b)?;
    let full = block(&f.blocks[1])?;
    let mut wrong = full.new_state(512, BUDGET)?;
    assert!(
        linear
            .forward(&b.cases[0].input, &mut wrong, BUDGET, None)
            .is_err()
    );
    assert!(wrong.is_empty());
    Ok(())
}

#[test]
fn zero_queries_keys_and_inputs_are_finite_and_reset_is_zero() -> Result {
    for b in fixture()?.blocks {
        let block = block(&b)?;
        let mut state = block.new_state(4, BUDGET)?;
        let initial = contents(&state);
        let output = block.forward(
            &vec![0.0; b.config.hidden_size * 4],
            &mut state,
            BUDGET,
            None,
        )?;
        assert!(output.iter().all(|x| *x == 0.0));
        assert_eq!(state.len(), 4);
        assert!(contents(&state).iter().skip(1).all(|v| *v == 0.0));
        state.reset();
        assert_eq!(contents(&state), initial);
    }
    Ok(())
}
