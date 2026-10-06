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
