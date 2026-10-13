use super::Cache;
use super::math::Linear;
use super::model::Qwen35;
use super::state::Storage;
use runtime::{CausalLm, SequenceCache};
use std::path::Path;

fn load() -> Qwen35 {
    let dir = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35"),
    )
    .unwrap();
    Qwen35::load(&dir, |_, _| {}).unwrap()
}
fn contents(c: &Cache) -> Vec<f32> {
    let mut x = vec![c.used() as f32, f32::from(c.replay)];
    for state in &c.states {
        match &state.storage {
            Storage::Delta { conv, recurrent } => {
                x.extend(conv);
                x.extend(recurrent);
            }
            Storage::Full(kv) => {
                let (k, v) = kv.read(0, state.len).unwrap();
                x.extend(k);
                x.extend(v);
            }
        }
    }
    x
}
#[test]
fn complete_model_incremental_replay_reset_and_session_isolation() {
    let m = load();
    let mut c = m.new_cache(32).unwrap();
    let mut fresh = m.new_cache(32).unwrap();
    let mut independent = m.new_cache(32).unwrap();
    let untouched = contents(&independent);
    m.forward(&[17, 4, 81], &mut c).unwrap();
    let incremental = m.forward(&[29], &mut c).unwrap();
    assert_eq!(
        incremental,
        m.forward(&[17, 4, 81, 29], &mut fresh).unwrap()
    );
    assert_eq!(contents(&c), contents(&fresh));
    assert_eq!(contents(&independent), untouched);
    c.truncate(2).unwrap();
    assert_eq!(c.used(), 2);
    let replay = m.forward(&[36, 8], &mut c).unwrap();
    fresh.clear();
    assert_eq!(replay, m.forward(&[17, 4, 36, 8], &mut fresh).unwrap());
    assert_eq!(contents(&c), contents(&fresh));
    c.clear();
    assert_eq!(c.used(), 0);
    assert_eq!(
        m.forward(&[17], &mut c).unwrap(),
        m.forward(&[17], &mut independent).unwrap()
    );
}
#[test]
fn last_layer_failure_rolls_back_every_earlier_layer() {
    let mut m = load();
    let mut c = m.new_cache(32).unwrap();
    m.forward(&[17, 4, 81], &mut c).unwrap();
    let before = contents(&c);
    let ids = c.tokens.clone();
    let h = m.config().hidden_size;
    let ff = m.config().intermediate_size;
    let last = m.blocks.last_mut().unwrap();
    last.common.gate = Linear::new(vec![1000.0; h * ff], h);
    last.common.up = Linear::new(vec![1000.0; h * ff], h);
    last.common.down = Linear::new(vec![f32::MAX; h * ff], ff);
    let mut reached = 0;
    assert!(
        m.forward_observed(&[29], &mut c, Some(&mut |i, _| reached = i))
            .is_err()
    );
    assert_eq!(reached, 3);
    assert_eq!(contents(&c), before);
    assert_eq!(c.tokens, ids);
}
#[test]
fn bad_tokens_context_and_foreign_model_cache_preserve_state() {
    let m = load();
    let mut c = m.new_cache(4).unwrap();
    m.forward(&[17, 4], &mut c).unwrap();
    let before = contents(&c);
    for ids in [&[][..], &[320][..], &[1, 2, 3][..]] {
        assert!(m.forward(ids, &mut c).is_err());
        assert_eq!(contents(&c), before);
    }
    assert!(c.truncate(3).is_err());
    assert_eq!(contents(&c), before);
    assert!(load().forward(&[17], &mut c).is_err());
    assert_eq!(contents(&c), before);
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().map(|x| f64::from(*x).abs()).fold(1.0, f64::max);
    let mut max = 0.0_f64;
    let mut error = 0.0;
    let mut reference = 0.0;
    for (a, b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite());
        let difference = f64::from(*a) - f64::from(*b);
        max = max.max(difference.abs());
        error += difference * difference;
        reference += f64::from(*b).powi(2);
    }
    assert!(max / scale <= 1e-4);
    assert!(
        (error / a.len() as f64).sqrt() / (reference / b.len() as f64).sqrt().max(1e-6) <= 1e-5
    );
}
fn greedy(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
#[test]
fn batched_prefill_matches_tokenwise_and_append_with_state_isolation() {
    let m = load();
    for n in [16, 17, 32, 65] {
        let ids: Vec<u32> = (0..n).map(|i| 1 + i % 100).collect();
        let mut batch = m.new_cache(128).unwrap();
        let mut tokenwise = m.new_cache(128).unwrap();
        let independent = m.new_cache(128).unwrap();
        let untouched = contents(&independent);
        let a = m.forward(&ids, &mut batch).unwrap();
        let mut b = Vec::new();
        for id in &ids {
            b = m.forward(&[*id], &mut tokenwise).unwrap();
        }
        close(&a, &b);
        assert_eq!(greedy(&a), greedy(&b));
        close(&contents(&batch), &contents(&tokenwise));
        let mut split = m.new_cache(128).unwrap();
        let mut split_logits = Vec::new();
        for chunk in ids.chunks(16) {
            split_logits = m.forward(chunk, &mut split).unwrap();
        }
        close(&a, &split_logits);
        assert_eq!(greedy(&a), greedy(&split_logits));
        close(&contents(&batch), &contents(&split));
        let a = m.forward(&[29], &mut batch).unwrap();
        let b = m.forward(&[29], &mut tokenwise).unwrap();
        close(&a, &b);
        assert_eq!(greedy(&a), greedy(&b));
        assert_eq!(contents(&independent), untouched);
        if n >= 17 {
            batch.truncate(17).unwrap();
            let replay = m.forward(&[36; 16], &mut batch).unwrap();
            tokenwise.clear();
            let mut edited = ids[..17].to_vec();
            edited.extend_from_slice(&[36; 16]);
            assert_eq!(replay, m.forward(&edited, &mut tokenwise).unwrap());
            assert_eq!(contents(&batch), contents(&tokenwise));
        }
        batch.clear();
        tokenwise.clear();
        assert_eq!(
            m.forward(&ids, &mut batch).unwrap(),
            m.forward(&ids, &mut tokenwise).unwrap()
        );
    }
}
#[test]
fn batched_last_layer_failure_preserves_the_entire_transaction() {
    let mut m = load();
    let mut c = m.new_cache(128).unwrap();
    m.forward(&[17, 4, 81], &mut c).unwrap();
    let before = contents(&c);
    let ids = c.tokens.clone();
    let h = m.config().hidden_size;
    let ff = m.config().intermediate_size;
    let last = m.blocks.last_mut().unwrap();
    last.common.gate = Linear::new(vec![1000.0; h * ff], h);
    last.common.up = Linear::new(vec![1000.0; h * ff], h);
    last.common.down = Linear::new(vec![f32::MAX; h * ff], ff);
    assert!(m.forward(&[29; 17], &mut c).is_err());
    assert_eq!(contents(&c), before);
    assert_eq!(c.tokens, ids);
}

#[test]
fn int8_batched_state_replay_reset_and_late_failure_are_atomic() {
    let dir = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8"),
    )
    .unwrap();
    let scheme = tensor::QuantScheme::new(tensor::QuantBits::Int8, 32).unwrap();
    let mut m = Qwen35::load_quantized(&dir, Some(scheme), |_, _| {}).unwrap();
    let mut batch = m.new_cache(128).unwrap();
    let mut single = m.new_cache(128).unwrap();
    let untouched = m.new_cache(128).unwrap();
    let original = contents(&untouched);
    let ids: Vec<_> = (1..66).collect();
    let a = m.forward(&ids, &mut batch).unwrap();
    let mut b = Vec::new();
    for token in &ids {
        b = m.forward(&[*token], &mut single).unwrap();
    }
    close(&a, &b);
    close(&contents(&batch), &contents(&single));
    assert_eq!(greedy(&a), greedy(&b));
    close(
        &m.forward(&[5], &mut batch).unwrap(),
        &m.forward(&[5], &mut single).unwrap(),
    );
    batch.truncate(17).unwrap();
    let a = m.forward(&[36; 16], &mut batch).unwrap();
    single.clear();
    let mut edited = ids[..17].to_vec();
    edited.extend_from_slice(&[36; 16]);
    assert_eq!(a, m.forward(&edited, &mut single).unwrap());
    assert_eq!(contents(&batch), contents(&single));
    batch.clear();
    single.clear();
    assert_eq!(
        m.forward(&ids, &mut batch).unwrap(),
        m.forward(&ids, &mut single).unwrap()
    );
    assert_eq!(contents(&untouched), original);
    let before = contents(&batch);
    let ledger = batch.tokens.clone();
    let h = m.config().hidden_size;
    let ff = m.config().intermediate_size;
    let last = m.blocks.last_mut().unwrap();
    last.common.gate = Linear::new(vec![1000.0; h * ff], h);
    last.common.up = Linear::new(vec![1000.0; h * ff], h);
    last.common.down = Linear::new(vec![f32::MAX; h * ff], ff);
    let mut reached = 0;
    assert!(
        m.forward_observed(&[29; 17], &mut batch, Some(&mut |i, _| reached = i))
            .is_err()
    );
    assert_eq!(reached, 3);
    assert_eq!(contents(&batch), before);
    assert_eq!(batch.tokens, ledger);
}
