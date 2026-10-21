//! A pass runs in the cache itself and is undone by cutting back the keys and values and putting back a mark of the
//! DeltaNet state. These tests fail passes after some of their layers and chunks have written state, and check that
//! every byte of the committed cache is as it was.

use super::gpu_forward::FAIL_AFTER_LAYERS;
use super::gpu_tests::contents;
use super::{GpuCache, Qwen35Gpu};
use cuda::Gpu;
use loader::ModelDir;
use runtime::{CausalLm, SequenceCache};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn model() -> Result<Option<Qwen35Gpu>> {
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping Qwen3.5 GPU test: {e}");
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35");
    Ok(Some(Qwen35Gpu::load(
        &ModelDir::open(&path)?,
        gpu,
        None,
        |_, _| {},
    )?))
}

fn ids(from: u32, count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| from.wrapping_add(i as u32) % 300)
        .collect()
}

/// A cache and an identical twin that is never put through anything.
fn twins(model: &Qwen35Gpu, history: &[u32]) -> Result<(GpuCache, GpuCache)> {
    let (mut live, mut twin) = (model.new_cache(1024)?, model.new_cache(1024)?);
    model.forward(history, &mut live)?;
    model.forward(history, &mut twin)?;
    Ok((live, twin))
}

#[test]
fn a_pass_that_fails_after_writing_some_chunks_and_layers_changes_nothing() -> Result {
    let Some(model) = model()? else { return Ok(()) };
    let layers = model.layers.len();
    let (mut live, mut twin) = twins(&model, &ids(10, 40))?;
    let before = contents(&model, &live)?;
    // 700 tokens are two chunks; fail inside the second one, after the first has committed.
    for fail_at in [0, 2, layers, layers + 1, layers * 2 - 1] {
        FAIL_AFTER_LAYERS.set(Some(fail_at));
        assert!(
            model.forward(&ids(100, 700), &mut live).is_err(),
            "failure after {fail_at} layers"
        );
        FAIL_AFTER_LAYERS.set(None);
        assert!(!live.replay, "a failed pass needs no replay");
        assert_eq!((live.used(), live.tokens.len()), (40, 40));
        assert_eq!(
            contents(&model, &live)?,
            before,
            "every committed byte after {fail_at} layers"
        );
    }
    let again = model.forward(&ids(100, 700), &mut live)?;
    assert!(
        again == model.forward(&ids(100, 700), &mut twin)?,
        "the retry is bit for bit the twin's"
    );
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    Ok(())
}

#[test]
fn a_replaying_pass_that_fails_stays_to_be_replayed_and_then_succeeds() -> Result {
    let Some(model) = model()? else { return Ok(()) };
    let (mut live, mut twin) = twins(&model, &ids(10, 40))?;
    live.truncate(30)?;
    twin.truncate(30)?;
    FAIL_AFTER_LAYERS.set(Some(2));
    assert!(model.forward(&ids(60, 5), &mut live).is_err());
    FAIL_AFTER_LAYERS.set(None);
    assert!(
        live.replay && live.tokens.len() == 30,
        "the ledger is kept, the state is to be rebuilt"
    );
    let retried = model.forward(&ids(60, 5), &mut live)?;
    assert!(retried == model.forward(&ids(60, 5), &mut twin)?);
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    Ok(())
}

#[test]
fn a_mark_returns_the_cache_exactly_and_is_refused_once_the_history_changed() -> Result {
    let Some(model) = model()? else { return Ok(()) };
    let (mut live, mut twin) = twins(&model, &ids(10, 70))?;
    let before = contents(&model, &live)?;
    let mark = model.mark_cache(&live)?;
    assert!(mark_bytes_are_the_fixed_state(&model, &live));
    model.forward(&ids(200, 600), &mut live)?;
    model.forward(&[7], &mut live)?;
    assert!(live.used() > 600);
    model.rewind_cache(&mut live, mark)?;
    assert_eq!((live.used(), live.tokens.len()), (70, 70));
    assert_eq!(contents(&model, &live)?, before);
    assert!(
        model.forward(&[9], &mut live)? == model.forward(&[9], &mut twin)?,
        "bit for bit the twin's"
    );
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);

    let mark = model.mark_cache(&live)?;
    live.truncate(40)?;
    assert!(
        model.rewind_cache(&mut live, mark).is_err(),
        "the history was cut back"
    );
    let mark = model.mark_cache(&twin)?;
    twin.clear();
    assert!(
        model.rewind_cache(&mut twin, mark).is_err(),
        "the history was restarted"
    );
    let (mut a, b) = twins(&model, &ids(10, 20))?;
    let foreign = model.mark_cache(&b)?;
    assert!(
        model.rewind_cache(&mut a, foreign).is_err(),
        "a mark belongs to the cache it came from"
    );
    let again = model.mark_cache(&a)?;
    model.forward(&[4], &mut a)?;
    model.rewind_cache(&mut a, again)?;
    Ok(())
}

fn mark_bytes_are_the_fixed_state(model: &Qwen35Gpu, cache: &GpuCache) -> bool {
    let layout = model.state_layout().ok();
    layout.is_some_and(|l| cache.mark_bytes() == l.delta_layers.saturating_mul(l.delta_bytes()))
}
