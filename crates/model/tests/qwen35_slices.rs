//! The server feeds long prompts to the GPU in slices so other requests can run between them. A sliced prefill must
//! give the logits of one pass and leave a state that decodes the same.
use model::{Backend, Engine, QuantBits, QuantScheme, export_int8};
use runtime::CausalLm;
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

fn agrees(sliced: &[f32], whole: &[f32]) -> Result<(f32, f32)> {
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|t| t.0)
    };
    assert_eq!(top(sliced), top(whole), "greedy token differs");
    let peak = whole.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    let max = sliced
        .iter()
        .zip(whole)
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()))
        / peak;
    let rms = |v: &mut dyn Iterator<Item = f32>| {
        (v.map(|x| x * x).sum::<f32>() / whole.len() as f32).sqrt()
    };
    let rel =
        rms(&mut sliced.iter().zip(whole).map(|(a, b)| a - b)) / rms(&mut whole.iter().copied());
    Ok((max, rel))
}

fn check(engine: &Engine, tokens: &[u32], slice: usize, ctx: usize) -> Result<(f32, f32)> {
    let mut whole = engine.new_cache(ctx)?;
    let mut sliced = engine.new_cache(ctx)?;
    let expected = engine.forward(tokens, &mut whole)?;
    let mut got = Vec::new();
    for part in tokens.chunks(slice) {
        got = engine.forward(part, &mut sliced)?;
    }
    let first = agrees(&got, &expected)?;
    let next = tokens.first().copied().unwrap_or(1);
    let after = agrees(
        &engine.forward(&[next], &mut sliced)?,
        &engine.forward(&[next], &mut whole)?,
    )?;
    Ok((first.0.max(after.0), first.1.max(after.1)))
}

#[test]
fn sliced_prefill_matches_one_pass_on_the_fixture() -> Result {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let source = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8"),
    )?;
    let temp = std::env::temp_dir().join(format!("marin-slices-{}", std::process::id()));
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
    let tokens: Vec<u32> = (1..100).collect();
    let mut worst = (0.0_f32, 0.0_f32);
    for slice in [1, 7, 15, 16, 17, 64] {
        let (max, rms) = check(&engine, &tokens, slice, 128)?;
        worst = (worst.0.max(max), worst.1.max(rms));
    }
    std::fs::remove_dir_all(&temp)?;
    assert!(
        worst.0 <= 1e-4 && worst.1 <= 1e-5,
        "normalized max {}, relative RMS {}",
        worst.0,
        worst.1
    );
    Ok(())
}

/// Needs a packed Qwen3.5 INT8 folder in `MARIN_QWEN35_Q8_DIR` and the text file `MARIN_SLICE_PROMPT`.
#[test]
fn sliced_prefill_matches_one_pass_on_the_real_model() -> Result {
    let (Some(dir), Some(prompt)) = (
        std::env::var_os("MARIN_QWEN35_Q8_DIR").map(PathBuf::from),
        std::env::var_os("MARIN_SLICE_PROMPT").map(PathBuf::from),
    ) else {
        eprintln!("MARIN_QWEN35_Q8_DIR and MARIN_SLICE_PROMPT are not set");
        return Ok(());
    };
    let Some(gpu) = gpu()? else { return Ok(()) };
    let packed = loader::ModelDir::open(&dir)?;
    let tokens = packed.tokenizer.encode(&std::fs::read_to_string(prompt)?)?;
    let scheme = QuantScheme::new(QuantBits::Int8, model::DEFAULT_GROUP)?;
    let engine = Engine::load(
        &packed,
        Backend::Cuda(Box::new(gpu)),
        Some(scheme),
        |_, _| {},
    )?;
    for slice in [64, 128] {
        let (max, rms) = check(&engine, &tokens, slice, 2048)?;
        eprintln!(
            "slice {slice}: {} tokens, normalized max {max:.3e}, relative RMS {rms:.3e}",
            tokens.len()
        );
        assert!(
            max <= 1e-4 && rms <= 1e-5,
            "normalized max {max}, relative RMS {rms}"
        );
    }
    Ok(())
}
