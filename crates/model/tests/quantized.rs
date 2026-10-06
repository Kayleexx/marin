//! A quantized engine on the tiny fixture: close to the bf16 one with 8 bits, finite and usable with 4, deterministic
//! across prompt and one-token passes, and refused on the CPU.

use std::error::Error;
use std::path::Path;

use loader::ModelDir;
use model::{Backend, Engine, ModelError, QuantBits, QuantScheme};
use runtime::CausalLm;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn load(quant: Option<QuantScheme>) -> Result<Option<(Engine, Vec<u32>)>> {
    let Ok(gpu) = cuda::Gpu::new() else {
        eprintln!("skipping: no usable GPU");
        return Ok(None);
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let tokens = dir
        .tokenizer
        .encode("<|im_start|>user\nhello world<|im_end|>\n")?;
    let engine = Engine::load(&dir, Backend::Cuda(Box::new(gpu)), quant, |_, _| {})?;
    Ok(Some((engine, tokens)))
}

fn worst_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The prompt, then three tokens decoded one at a time; the logits after each pass.
fn run(engine: &Engine, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut cache = engine.new_cache(64)?;
    let mut passes = vec![engine.forward(tokens, &mut cache)?];
    for token in [3, 5, 7] {
        passes.push(engine.forward(&[token], &mut cache)?);
    }
    Ok(passes)
}

#[test]
fn eight_bit_weights_stay_close_to_bf16_and_four_bit_weights_stay_usable() -> Result<()> {
    let Some((plain, tokens)) = load(None)? else {
        return Ok(());
    };
    let reference = run(&plain, &tokens)?;
    let scale = reference
        .iter()
        .flatten()
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    for (bits, limit) in [(QuantBits::Int8, 0.05), (QuantBits::Int4, 0.5)] {
        let scheme = QuantScheme { bits, group: 32 };
        let Some((engine, _)) = load(Some(scheme))? else {
            return Ok(());
        };
        assert_eq!(engine.quantization(), Some(scheme));
        let passes = run(&engine, &tokens)?;
        for (pass, (got, want)) in passes.iter().zip(&reference).enumerate() {
            assert!(got.iter().all(|v| v.is_finite()), "{bits:?} pass {pass}");
            let worst = worst_difference(got, want);
            assert!(
                worst <= limit * scale,
                "{bits:?} pass {pass}: {worst} against logits up to {scale}"
            );
        }
        assert_eq!(passes, run(&engine, &tokens)?, "{bits:?} repeats exactly");
    }
    Ok(())
}

#[test]
fn a_quantized_decode_step_agrees_with_reading_everything_at_once() -> Result<()> {
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    let Some((engine, tokens)) = load(Some(scheme))? else {
        return Ok(());
    };
    let mut longer = tokens.clone();
    longer.extend([3, 5]);
    let whole = engine.forward(&longer, &mut engine.new_cache(64)?)?;
    let mut cache = engine.new_cache(64)?;
    engine.forward(&tokens, &mut cache)?;
    engine.forward(&[3], &mut cache)?;
    let step = engine.forward(&[5], &mut cache)?;
    let worst = worst_difference(&whole, &step);
    assert!(
        worst < 1e-3,
        "one-token and prompt passes disagree by {worst}"
    );
    Ok(())
}

#[test]
fn a_quantized_decode_step_allocates_nothing() -> Result<()> {
    let scheme = QuantScheme {
        bits: QuantBits::Int4,
        group: 32,
    };
    let Some((engine, tokens)) = load(Some(scheme))? else {
        return Ok(());
    };
    let mut cache = engine.new_cache(64)?;
    engine.forward(&tokens, &mut cache)?;
    engine.forward(&[3], &mut cache)?;
    let before = engine.traffic().ok_or("no traffic counters")?;
    engine.forward(&[5], &mut cache)?;
    let delta = engine
        .traffic()
        .ok_or("no traffic counters")?
        .since(&before);
    assert_eq!(delta.allocations, 0, "{delta:?}");
    Ok(())
}

#[test]
fn the_cpu_backend_refuses_quantized_weights() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    let Err(error) = Engine::load(&dir, Backend::Cpu, Some(scheme), |_, _| {}) else {
        return Err("the CPU backend accepted quantized weights".into());
    };
    assert!(matches!(error, ModelError::Unsupported(_)), "{error}");
    assert!(error.to_string().contains("GPU"), "{error}");
    Ok(())
}
