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
fn break_last_layer(m: &mut Qwen35) {
    let h = m.config().hidden_size;
    let ff = m.config().intermediate_size;
    let n = h.checked_mul(ff).unwrap();
    let last = m.blocks.last_mut().unwrap();
    last.common.gate = Linear::new(vec![1000.0; n], h);
    last.common.up = Linear::new(vec![1000.0; n], h);
    last.common.down = Linear::new(vec![f32::MAX; n], ff);
}
#[test]
fn a_non_finite_recurrent_state_fails_the_pass_leaves_the_cache_and_a_retry_succeeds() {
    let m = load();
    let mut c = m.new_cache(32).unwrap();
    let mut reference = m.new_cache(32).unwrap();
    m.forward(&[17, 4, 81], &mut c).unwrap();
    m.forward(&[17, 4, 81], &mut reference).unwrap();
    let poisoned = c
        .states
        .iter_mut()
        .find_map(|s| match &mut s.storage {
            Storage::Delta { recurrent, .. } => recurrent.first_mut(),
            Storage::Full(_) => None,
        })
        .unwrap();
    let original = std::mem::replace(poisoned, f32::INFINITY);
    let before = contents(&c);
    assert!(m.forward(&[29, 5], &mut c).is_err());
    assert_eq!(contents(&c), before);
    let healed = c
        .states
        .iter_mut()
        .find_map(|s| match &mut s.storage {
            Storage::Delta { recurrent, .. } => recurrent.first_mut(),
            Storage::Full(_) => None,
        })
        .unwrap();
    *healed = original;
    assert_eq!(
        m.forward(&[29, 5], &mut c).unwrap(),
        m.forward(&[29, 5], &mut reference).unwrap()
    );
}
#[test]
fn a_replaying_pass_that_fails_stays_to_be_replayed() {
    let mut m = load();
    let mut c = m.new_cache(32).unwrap();
    m.forward(&[17, 4, 81], &mut c).unwrap();
    c.truncate(2).unwrap();
    let ids = c.tokens.clone();
    break_last_layer(&mut m);
    assert!(m.forward(&[29], &mut c).is_err());
    assert!(c.replays());
    assert_eq!(c.tokens, ids);
}
#[test]
fn last_layer_failure_rolls_back_every_earlier_layer() {
    let mut m = load();
    let mut c = m.new_cache(32).unwrap();
    m.forward(&[17, 4, 81], &mut c).unwrap();
    let before = contents(&c);
    let ids = c.tokens.clone();
    break_last_layer(&mut m);
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
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8");
    int8_state_case(&fixture, 32, false);
    if let Some(real) = std::env::var_os("MARIN_QWEN35_4B_Q8") {
        int8_state_case(Path::new(&real), 128, true);
    }
}

fn int8_state_case(path: &Path, group: usize, real: bool) {
    let dir = loader::ModelDir::open(path).unwrap();
    let scheme = tensor::QuantScheme::new(tensor::QuantBits::Int8, group).unwrap();
    let mut m = Qwen35::load_quantized(&dir, Some(scheme), |_, _| {}).unwrap();
    let mut batch = m.new_cache(128).unwrap();
    let mut single = m.new_cache(128).unwrap();
    let untouched = m.new_cache(128).unwrap();
    let original = contents(&untouched);
    let ids: Vec<_> = (1..if real { 7 } else { 66 }).collect();
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
    let prefix = if real { 3 } else { 17 };
    let replacement = vec![36; if real { 2 } else { 16 }];
    batch.truncate(prefix).unwrap();
    let a = m.forward(&replacement, &mut batch).unwrap();
    single.clear();
    let mut edited = ids[..prefix].to_vec();
    edited.extend_from_slice(&replacement);
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
    let n = h.checked_mul(ff).unwrap();
    let last = m.blocks.last_mut().unwrap();
    last.common.gate = Linear::new(vec![1000.0; n], h);
    last.common.up = Linear::new(vec![1000.0; n], h);
    last.common.down = Linear::new(vec![f32::MAX; n], ff);
    let mut reached = 0;
    assert!(
        m.forward_observed(
            &vec![29; if real { 1 } else { 17 }],
            &mut batch,
            Some(&mut |i, _| reached = i)
        )
        .is_err()
    );
    assert_eq!(reached, m.blocks.len().checked_sub(1).unwrap());
    assert_eq!(contents(&batch), before);
    assert_eq!(batch.tokens, ledger);
}
