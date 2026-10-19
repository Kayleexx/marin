//! A stored prefix cache must give every request the logits of a full prefill and stay unchanged however the copies
//! are used.
use model::{Backend, Engine, QuantBits, QuantScheme, export_int8};
use runtime::{CausalLm, SequenceCache};
use std::path::{Path, PathBuf};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gpu() -> Result<Option<model::Gpu>> {
    match model::Gpu::new() {
        Ok(gpu) => Ok(Some(gpu)),
        Err(error) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("CUDA unavailable: {error}");
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

fn check(engine: &Engine, prefix: &[u32], tail: &[u32], ctx: usize) -> Result {
    let all: Vec<u32> = prefix.iter().chain(tail).copied().collect();
    let expected = engine.forward(&all, &mut engine.new_cache(ctx)?)?;
    let mut stored = engine.new_cache(ctx)?;
    engine.forward(prefix, &mut stored)?;
    for _ in 0..2 {
        let mut copy = engine.session_checkpoint(&stored)?.ok_or("no checkpoint")?;
        assert!(engine.forward(tail, &mut copy)? == expected);
        assert_eq!(copy.used(), all.len());
        engine.forward(&[1, 2, 3], &mut copy)?;
        assert_eq!(stored.used(), prefix.len());
    }
    let mut last = engine.session_checkpoint(&stored)?.ok_or("no checkpoint")?;
    assert!(engine.forward(tail, &mut last)? == expected);
    Ok(())
}

#[test]
fn restored_prefix_matches_a_full_prefill_on_the_fixture() -> Result {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let source = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8"),
    )?;
    let temp = std::env::temp_dir().join(format!("marin-prefix-{}", std::process::id()));
    std::fs::create_dir(&temp)?;
    export_int8(&source, &temp.join("q8"), 32)?;
    let packed = loader::ModelDir::open(&temp.join("q8"))?;
    let scheme = QuantScheme::new(QuantBits::Int8, 32)?;
    let engine = Engine::load(
        &packed,
        Backend::Cuda(Box::new(gpu)),
        Some(scheme),
        |_, _| {},
    )?;
    let prefix: Vec<u32> = (10..41).collect();
    let tail: Vec<u32> = (50..66).collect();
    let outcome = check(&engine, &prefix, &tail, 128);
    std::fs::remove_dir_all(&temp)?;
    outcome
}

/// Needs a packed Qwen3.5 INT8 folder in `MARIN_QWEN35_Q8_DIR`.
#[test]
fn restored_prefix_matches_a_full_prefill_on_the_real_model() -> Result {
    let Some(dir) = std::env::var_os("MARIN_QWEN35_Q8_DIR").map(PathBuf::from) else {
        eprintln!("MARIN_QWEN35_Q8_DIR is not set");
        return Ok(());
    };
    let Some(gpu) = gpu()? else { return Ok(()) };
    let packed = loader::ModelDir::open(&dir)?;
    let scheme = QuantScheme::new(QuantBits::Int8, model::DEFAULT_GROUP)?;
    let engine = Engine::load(
        &packed,
        Backend::Cuda(Box::new(gpu)),
        Some(scheme),
        |_, _| {},
    )?;
    let prefix: Vec<u32> = (1000..1031).collect();
    let tail: Vec<u32> = (2000..2016).collect();
    check(&engine, &prefix, &tail, 2048)
}
