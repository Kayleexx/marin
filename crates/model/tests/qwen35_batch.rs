//! A batched decode step must equal one `forward` per sequence, bit for bit, whatever positions the sequences are at.
use model::qwen35::Qwen35Gpu;
use model::{QuantBits, QuantScheme, export_int8};
use runtime::{CausalLm, SequenceCache};
use std::path::{Path, PathBuf};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn argmax(logits: &[f32]) -> u32 {
    let best = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1));
    best.map_or(0, |(i, _)| u32::try_from(i).unwrap_or(0))
}

/// Prefills `lengths.len()` sequences (distinct tokens each), then decodes `steps` greedy tokens both ways.
fn check(model: &Qwen35Gpu, lengths: &[usize], steps: usize, ctx: usize) -> Result {
    let mut batched = Vec::new();
    let mut single = Vec::new();
    let mut next = Vec::new();
    for (seq, len) in lengths.iter().enumerate() {
        let base = u32::try_from(seq)?.saturating_mul(37).saturating_add(10);
        let prompt: Vec<u32> = (0..u32::try_from(*len)?)
            .map(|i| base.saturating_add(i))
            .collect();
        let mut a = model.new_cache(ctx)?;
        let mut b = model.new_cache(ctx)?;
        let first = model.forward(&prompt, &mut a)?;
        assert_eq!(first, model.forward(&prompt, &mut b)?);
        next.push(argmax(&first));
        batched.push(a);
        single.push(b);
    }
    for step in 0..steps {
        let mut refs: Vec<&mut model::qwen35::GpuCache> = batched.iter_mut().collect();
        let rows = model.forward_batch(&next, &mut refs)?;
        for ((row, token), reference) in rows.iter().zip(next.iter_mut()).zip(single.iter_mut()) {
            let expected = model.forward(&[*token], reference)?;
            assert!(*row == expected, "step {step}: logits differ");
            *token = argmax(row);
        }
        for (a, b) in batched.iter().zip(&single) {
            assert_eq!(a.used(), b.used());
        }
    }
    Ok(())
}

fn full_cache_changes_nothing(model: &Qwen35Gpu) -> Result {
    let mut full = model.new_cache(6)?;
    let mut roomy = model.new_cache(64)?;
    model.forward(&[1, 2, 3, 4, 5, 6], &mut full)?;
    model.forward(&[7, 8, 9], &mut roomy)?;
    let mut shadow = model.new_cache(64)?;
    model.forward(&[7, 8, 9], &mut shadow)?;
    let failed = model.forward_batch(&[1, 2], &mut [&mut full, &mut roomy]);
    assert!(failed.is_err());
    assert_eq!((full.used(), roomy.used()), (6, 3));
    assert!(model.forward(&[4], &mut roomy)? == model.forward(&[4], &mut shadow)?);
    Ok(())
}

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

#[test]
fn batches_match_single_sequences_on_the_fixture() -> Result {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let source = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8"),
    )?;
    let temp = std::env::temp_dir().join(format!("marin-batch-{}", std::process::id()));
    std::fs::create_dir(&temp)?;
    export_int8(&source, &temp.join("q8"), 32)?;
    let packed = loader::ModelDir::open(&temp.join("q8"))?;
    let scheme = QuantScheme::new(QuantBits::Int8, 32)?;
    let model = Qwen35Gpu::load(&packed, gpu, Some(scheme), |_, _| {})?;
    let outcome = check(&model, &[9, 40], 6, 128)
        .and_then(|()| check(&model, &[3, 17, 33], 6, 128))
        .and_then(|()| check(&model, &[5, 6, 7, 8], 4, 128))
        .and_then(|()| full_cache_changes_nothing(&model));
    std::fs::remove_dir_all(&temp)?;
    outcome
}

/// Needs a packed Qwen3.5 INT8 folder in `MARIN_QWEN35_Q8_DIR`.
#[test]
fn batches_match_single_sequences_on_the_real_model() -> Result {
    let Some(dir) = std::env::var_os("MARIN_QWEN35_Q8_DIR").map(PathBuf::from) else {
        eprintln!("MARIN_QWEN35_Q8_DIR is not set");
        return Ok(());
    };
    let Some(gpu) = gpu()? else { return Ok(()) };
    let packed = loader::ModelDir::open(&dir)?;
    let scheme = QuantScheme::new(QuantBits::Int8, model::DEFAULT_GROUP)?;
    let model = Qwen35Gpu::load(&packed, gpu, Some(scheme), |_, _| {})?;
    check(&model, &[47, 130], 12, 512)?;
    check(&model, &[47, 130, 311], 12, 512)?;
    full_cache_changes_nothing(&model)
}
