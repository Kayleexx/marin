//! A Qwen3.5 GPU cache exported block by block and imported into a fresh cache must continue bit for bit as the original
//! does, and as a full replay does within the rounding that splitting a prompt into passes already causes; anything
//! that does not fit must be refused.
use model::{
    Backend, Engine, EngineCache, PAGE_TOKENS, QuantBits, QuantScheme, export_int8, pages_for,
};
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

fn round_trip(engine: &Engine, cache: &EngineCache, capacity: usize) -> Result<EngineCache> {
    let tokens = engine.kept_tokens(cache)?.to_vec();
    let layout = engine.state_layout().ok_or("no layout")?;
    let mut copy = engine.begin_state_import(capacity, tokens.len())?;
    for index in 0..layout.delta_layers {
        let bytes = engine.export_state_delta(cache, index)?;
        assert_eq!(bytes.len(), layout.delta_bytes());
        engine.import_state_delta(&mut copy, index, &bytes)?;
    }
    for page in 0..pages_for(tokens.len()) {
        let rows = PAGE_TOKENS.min(
            tokens
                .len()
                .saturating_sub(page.saturating_mul(PAGE_TOKENS)),
        );
        let bytes = engine.export_state_page(cache, page, rows)?;
        assert_eq!(bytes.len(), layout.page_bytes(rows));
        engine.import_state_page(&mut copy, page, rows, &bytes)?;
    }
    engine.finish_state_import(&mut copy, &tokens)?;
    Ok(copy)
}

fn check(engine: &Engine, prefix: &[u32], tail: &[u32], ctx: usize) -> Result {
    let all: Vec<u32> = prefix.iter().chain(tail).copied().collect();
    let expected = engine.forward(&all, &mut engine.new_cache(ctx)?)?;
    let mut original = engine.new_cache(ctx)?;
    engine.forward(prefix, &mut original)?;
    let mut restored = round_trip(engine, &original, ctx)?;
    assert_eq!(restored.used(), prefix.len());
    let after_restore = engine.forward(tail, &mut restored)?;
    let after_original = engine.forward(tail, &mut original)?;
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    };
    let sizes = (prefix.len(), tail.len());
    assert!(
        after_restore == after_original,
        "{sizes:?}: restored vs original differ by {}",
        worst(&after_restore, &after_original)
    );
    let scale = expected.iter().fold(1.0_f32, |m, x| m.max(x.abs()));
    assert!(
        worst(&after_original, &expected) <= 1e-5 * scale,
        "{sizes:?}: reading in two passes instead of one moves a logit by {}",
        worst(&after_original, &expected)
    );
    let again = round_trip(engine, &restored, ctx)?;
    assert_eq!(
        again.used(),
        all.len(),
        "a restored cache can be kept again"
    );
    Ok(())
}

fn refusals(engine: &Engine, prefix: &[u32], ctx: usize) -> Result {
    let mut cache = engine.new_cache(ctx)?;
    assert!(
        engine.kept_tokens(&cache).is_err(),
        "an empty cache holds nothing to keep"
    );
    engine.forward(prefix, &mut cache)?;
    let layout = engine.state_layout().ok_or("no layout")?;
    let good = engine.export_state_delta(&cache, 0)?;
    let mut fresh = engine.begin_state_import(ctx, prefix.len())?;
    assert!(
        engine
            .import_state_delta(&mut fresh, 0, good.get(1..).unwrap_or_default())
            .is_err(),
        "a short block"
    );
    assert!(
        engine
            .import_state_delta(&mut fresh, layout.delta_layers, &good)
            .is_err(),
        "no such layer"
    );
    assert!(
        engine
            .export_state_delta(&cache, layout.delta_layers)
            .is_err()
    );
    assert!(
        engine
            .begin_state_import(ctx, ctx.saturating_add(1))
            .is_err(),
        "more than the context holds"
    );
    assert!(engine.begin_state_import(ctx, 0).is_err());
    cache.truncate(prefix.len().saturating_sub(1))?;
    assert!(
        engine.kept_tokens(&cache).is_err(),
        "a cache that must be replayed is not kept"
    );
    Ok(())
}

#[test]
fn a_restored_cache_continues_like_a_full_replay_on_the_fixture() -> Result {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let source = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8"),
    )?;
    let temp = std::env::temp_dir().join(format!("marin-state-{}", std::process::id()));
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
    let outcome = (|| -> Result {
        for (prefix, tail) in [(31, 16), (64, 5), (65, 1), (130, 40)] {
            let prefix: Vec<u32> = (10..).take(prefix).collect();
            let tail: Vec<u32> = (200..).take(tail).collect();
            check(&engine, &prefix, &tail, 256)?;
        }
        refusals(&engine, &(10..41).collect::<Vec<u32>>(), 128)
    })();
    std::fs::remove_dir_all(&temp)?;
    outcome
}

/// Needs a packed Qwen3.5 INT8 folder in `MARIN_QWEN35_Q8_DIR`.
#[test]
fn a_restored_cache_continues_like_a_full_replay_on_the_real_model() -> Result {
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
    let prefix: Vec<u32> = (1000..1200).collect();
    let tail: Vec<u32> = (2000..2016).collect();
    check(&engine, &prefix, &tail, 2048)
}
