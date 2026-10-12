//! Tokens appended to a cache must land at the right positions: processing a sequence in one
//! pass, in chunks, or one token at a time must give the same logits, and so must appending to a
//! kept cache (what a chat turn does) or rebuilding it from scratch.

use std::error::Error;
use std::path::Path;

use kv::KvCache;
use loader::ModelDir;
use model::Qwen3;
use runtime::{CausalLm, SequenceCache};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn load() -> Result<(Qwen3, ModelDir)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    Ok((Qwen3::load(&dir, |_, _| {})?, dir))
}

fn tokens(dir: &ModelDir) -> Result<Vec<u32>> {
    let text =
        "<|im_start|>user\nhello world, the cat sat on the mat<|im_end|>\n<|im_start|>assistant\n";
    Ok(dir.tokenizer.encode(text)?)
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4 * (1.0 + y.abs()), "{x} vs {y}");
    }
}

fn last_logits(model: &Qwen3, chunks: &[&[u32]], capacity: usize) -> Result<(Vec<f32>, usize)> {
    let mut cache: KvCache = model.new_cache(capacity)?;
    let mut logits = Vec::new();
    for chunk in chunks {
        logits = model.forward(chunk, &mut cache)?;
    }
    Ok((logits, cache.used()))
}

#[test]
fn chunked_and_token_by_token_processing_equal_one_pass() {
    let (model, dir) = load().unwrap();
    let all = tokens(&dir).unwrap();
    assert!(all.len() > 12, "{}", all.len());
    let (whole, used) = last_logits(&model, &[&all], 256).unwrap();
    assert_eq!(used, all.len(), "the cache holds exactly the tokens fed");

    let (head, tail) = all.split_at(all.len() / 2);
    let (chunked, used) = last_logits(&model, &[head, tail], 256).unwrap();
    assert_eq!(used, all.len());
    close(&chunked, &whole);

    let singles: Vec<&[u32]> = all.chunks(1).collect();
    let (stepped, used) = last_logits(&model, &singles, 256).unwrap();
    assert_eq!(used, all.len());
    close(&stepped, &whole);
}

#[test]
fn a_cache_reused_for_a_longer_prompt_matches_a_fresh_one() {
    let (model, dir) = load().unwrap();
    let first = tokens(&dir).unwrap();
    let mut longer = first.clone();
    longer.extend(dir.tokenizer.encode("and then it left").unwrap());

    let (fresh, _) = last_logits(&model, &[&longer], 256).unwrap();
    let mut cache: KvCache = model.new_cache(256).unwrap();
    model.forward(&first, &mut cache).unwrap();
    let suffix = longer.get(first.len()..).unwrap();
    let appended = model.forward(suffix, &mut cache).unwrap();
    close(&appended, &fresh);
    assert_eq!(cache.used(), longer.len());

    cache.clear();
    assert_eq!(cache.used(), 0);
    let rebuilt = model.forward(&longer, &mut cache).unwrap();
    close(&rebuilt, &fresh);
}

#[test]
fn feeding_more_tokens_than_the_cache_holds_is_an_error_not_corruption() {
    let (model, dir) = load().unwrap();
    let all = tokens(&dir).unwrap();
    let mut cache: KvCache = model.new_cache(all.len() - 1).unwrap();
    assert!(model.forward(&all, &mut cache).is_err());
    assert_eq!(cache.used(), 0, "a failed forward commits nothing");
}

#[test]
fn cpu_growth_and_rollback_match_a_fresh_pass_with_both_weight_formats() {
    let (_, dir) = load().unwrap();
    for quant in [
        None,
        Some(model::QuantScheme {
            bits: model::QuantBits::Int8,
            group: 32,
        }),
    ] {
        let model = Qwen3::load_quantized(&dir, quant, |_, _| {}).unwrap();
        let mut cache = model.new_cache(512).unwrap();
        let tokens = vec![3; 257];
        model.forward(&tokens[..255], &mut cache).unwrap();
        let before = cache.bytes_held();
        let grown = model.forward(&tokens[255..], &mut cache).unwrap();
        assert!(cache.bytes_held() > before);
        let (fresh, _) = last_logits(&model, &[&tokens], 512).unwrap();
        close(&grown, &fresh);
        cache.truncate(255).unwrap();
        assert!(model.forward(&[u32::MAX], &mut cache).is_err());
        assert_eq!(cache.used(), 255);
        let resumed = model.forward(&[4], &mut cache).unwrap();
        let mut expected = vec![3; 255];
        expected.push(4);
        close(&resumed, &last_logits(&model, &[&expected], 512).unwrap().0);
        cache.clear();
        close(&model.forward(&expected, &mut cache).unwrap(), &resumed);
    }
}

#[test]
fn whole_prefill_observes_once_and_preserves_the_prefix_on_failure() {
    let (_, dir) = load().unwrap();
    for quant in [
        None,
        Some(model::QuantScheme::new(model::QuantBits::Int8, 32).unwrap()),
    ] {
        let model = Qwen3::load_quantized(&dir, quant, |_, _| {}).unwrap();
        let mut cache = model.new_cache(1024).unwrap();
        model.forward(&[3; 5], &mut cache).unwrap();
        let prefix: Vec<_> = (0..model.config().layers)
            .map(|layer| {
                let (keys, values) = cache.read(layer, 5).unwrap();
                (keys.to_vec(), values.to_vec())
            })
            .collect();
        let mut bad = vec![3; 513];
        *bad.last_mut().unwrap() = u32::MAX;
        assert!(model.forward(&bad, &mut cache).is_err());
        assert_eq!(cache.len(), 5);
        for (layer, (keys, values)) in prefix.iter().enumerate() {
            assert_eq!(
                cache.read(layer, 5).unwrap(),
                (keys.as_slice(), values.as_slice())
            );
        }
        let input = vec![3; 513];
        let mut observed = Vec::new();
        let mut stages = model::StageTimes::traced();
        let result = model
            .forward_with(
                &input,
                &mut cache,
                Some(&mut |index, _| observed.push(index)),
                &mut model::Probe::on(&mut stages),
            )
            .unwrap();
        assert_eq!(observed, (0..=model.config().layers).collect::<Vec<_>>());
        assert_eq!(stages.calls(model::Stage::Embedding), 1);
        assert_eq!(cache.len(), 518);
        close(
            &result,
            &last_logits(&model, &[&vec![3; 518]], 1024).unwrap().0,
        );
    }
}
