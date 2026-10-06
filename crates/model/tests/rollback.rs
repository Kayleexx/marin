//! Rolling a cache back must leave exactly what a cache that never saw the rolled-back tokens holds, and
//! `forward_rows` must give, for every token fed, the logits a pass ending at that token gives.

use std::error::Error;
use std::path::Path;

use loader::ModelDir;
use model::{Backend, Engine, EngineCache};
use runtime::{CausalLm, SequenceCache};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn engine(backend: Backend) -> Result<(Engine, Vec<u32>)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let prompt = dir
        .tokenizer
        .encode("<|im_start|>user\nhello world, the cat sat<|im_end|>\n")?;
    Ok((Engine::load(&dir, backend, None, |_, _| {})?, prompt))
}

fn gpu() -> Option<Backend> {
    match cuda::Gpu::new() {
        Ok(gpu) => Some(Backend::Cuda(Box::new(gpu))),
        Err(_) => {
            eprintln!("skipping: no usable GPU");
            None
        }
    }
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(worst < 1e-3, "logits differ by {worst}");
}

fn fresh(engine: &Engine, tokens: &[u32]) -> Result<Vec<f32>> {
    Ok(engine.forward(tokens, &mut engine.new_cache(64)?)?)
}

fn rollback_matches_a_cache_that_never_saw_the_tokens(
    engine: &Engine,
    prompt: &[u32],
) -> Result<()> {
    let mut cache: EngineCache = engine.new_cache(64)?;
    engine.forward(prompt, &mut cache)?;
    engine.forward(&[3, 4, 5, 6], &mut cache)?;
    let kept = prompt.len().saturating_add(1);
    cache.truncate(kept)?;
    assert_eq!(cache.used(), kept);
    let resumed = engine.forward(&[9], &mut cache)?;
    assert_eq!(cache.used(), kept.saturating_add(1));
    let mut expected = prompt.to_vec();
    expected.extend([3, 9]);
    close(&resumed, &fresh(engine, &expected)?);
    assert!(cache.truncate(cache.used().saturating_add(1)).is_err());
    Ok(())
}

#[test]
fn cpu_rollback_matches_a_cache_that_never_saw_the_tokens() {
    let (engine, prompt) = engine(Backend::Cpu).unwrap();
    rollback_matches_a_cache_that_never_saw_the_tokens(&engine, &prompt).unwrap();
}

#[test]
fn gpu_rollback_matches_a_cache_that_never_saw_the_tokens() {
    let Some(backend) = gpu() else { return };
    let (engine, prompt) = engine(backend).unwrap();
    rollback_matches_a_cache_that_never_saw_the_tokens(&engine, &prompt).unwrap();
}

fn rows_match_passes_ending_at_each_token(engine: &Engine, prompt: &[u32]) -> Result<()> {
    let vocab = engine.config().vocab_size;
    let fed = [3_u32, 4, 5, 6, 7];
    let mut cache = engine.new_cache(64)?;
    engine.forward(prompt, &mut cache)?;
    let rows = engine.forward_rows(&fed, &mut cache)?;
    assert_eq!(rows.len(), fed.len().saturating_mul(vocab));
    assert_eq!(cache.used(), prompt.len().saturating_add(fed.len()));
    for (i, row) in rows.chunks(vocab).enumerate() {
        let mut expected = prompt.to_vec();
        expected.extend(fed.get(..=i).ok_or("row out of range")?);
        close(row, &fresh(engine, &expected)?);
    }
    Ok(())
}

#[test]
fn every_row_of_a_rows_pass_matches_a_pass_ending_at_that_token() {
    let Some(backend) = gpu() else { return };
    let (engine, prompt) = engine(backend).unwrap();
    rows_match_passes_ending_at_each_token(&engine, &prompt).unwrap();
}

#[test]
fn a_rows_pass_can_be_rolled_back_and_repeated() {
    let Some(backend) = gpu() else { return };
    let (engine, prompt) = engine(backend).unwrap();
    let mut cache = engine.new_cache(64).unwrap();
    engine.forward(&prompt, &mut cache).unwrap();
    let first = engine.forward_rows(&[3, 4, 5], &mut cache).unwrap();
    cache.truncate(prompt.len()).unwrap();
    let again = engine.forward_rows(&[3, 4, 5], &mut cache).unwrap();
    assert_eq!(
        first, again,
        "the same tokens at the same positions give the same bits"
    );
}

#[test]
fn a_gpu_rows_pass_of_more_than_eight_tokens_is_an_error_and_commits_nothing() {
    let Some(backend) = gpu() else { return };
    let (engine, _) = self::engine(backend).unwrap();
    assert_eq!(engine.max_rows(), 8);
    let mut cache = engine.new_cache(64).unwrap();
    assert!(engine.forward_rows(&[1; 9], &mut cache).is_err());
    assert_eq!(cache.used(), 0);
}

#[test]
fn the_cpu_reports_no_fast_rows_pass_but_its_fallback_matches_single_passes() {
    let (cpu, prompt) = self::engine(Backend::Cpu).unwrap();
    assert_eq!(cpu.max_rows(), 0);
    let vocab = cpu.config().vocab_size;
    let mut cache = cpu.new_cache(64).unwrap();
    cpu.forward(&prompt, &mut cache).unwrap();
    let rows = cpu.forward_rows(&[3, 4, 5], &mut cache).unwrap();
    assert_eq!(rows.len(), 3 * vocab);
    for (i, row) in rows.chunks(vocab).enumerate() {
        let mut expected = prompt.clone();
        expected.extend(&[3, 4, 5][..=i]);
        close(row, &fresh(&cpu, &expected).unwrap());
    }
}
