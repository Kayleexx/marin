//! Timing a pass must not change it: profiled and plain forward passes give identical logits, every stage of
//! every layer is seen the expected number of times, and the GPU path (when there is one) agrees.

use std::error::Error;
use std::path::Path;

use loader::ModelDir;
use model::{Backend, Engine, Stage, StageTimes};
use runtime::CausalLm;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn engine(backend: Backend) -> Result<(Engine, Vec<u32>)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let tokens = dir
        .tokenizer
        .encode("<|im_start|>user\nhello world<|im_end|>\n")?;
    Ok((Engine::load(&dir, backend, None, |_, _| {})?, tokens))
}

fn check(engine: &Engine, tokens: &[u32]) -> Result<()> {
    let layers = u64::try_from(engine.config().layers)?;
    let mut plain = engine.new_cache(64)?;
    let mut profiled = engine.new_cache(64)?;
    let mut times = StageTimes::default();
    let expected = engine.forward(tokens, &mut plain)?;
    let actual = engine.forward_profiled(tokens, &mut profiled, &mut times)?;
    assert_eq!(expected, actual, "profiling changed the logits");
    for stage in [
        Stage::Qkv,
        Stage::Rope,
        Stage::KvWrite,
        Stage::Attention,
        Stage::OutProj,
        Stage::Mlp,
    ] {
        assert_eq!(times.calls(stage), layers, "{stage:?} once per layer");
    }
    assert_eq!(
        times.calls(Stage::Norm),
        layers.saturating_mul(3).saturating_add(1),
        "input, q/k and post norms per layer plus the final one"
    );
    assert_eq!(times.calls(Stage::Residual), layers.saturating_mul(2));
    assert_eq!(
        (times.calls(Stage::Embedding), times.calls(Stage::LmHead)),
        (1, 1)
    );
    assert!(times.sum() > std::time::Duration::ZERO);
    check_trace(engine, tokens, &expected)
}

/// A traced pass matches the plain one and lists, in order, 11 calls per layer plus embedding, final norm and head.
fn check_trace(engine: &Engine, tokens: &[u32], expected: &[f32]) -> Result<()> {
    let layers = engine.config().layers;
    let mut cache = engine.new_cache(64)?;
    let mut times = StageTimes::traced();
    let logits = engine.forward_profiled(tokens, &mut cache, &mut times)?;
    assert_eq!(expected, logits, "tracing changed the logits");
    let trace = times.trace();
    assert_eq!(trace.len(), layers.saturating_mul(11).saturating_add(3));
    let first = trace.first().ok_or("empty trace")?;
    assert_eq!((first.layer, first.stage), (None, Stage::Embedding));
    let last = trace.last().ok_or("empty trace")?;
    assert_eq!((last.layer, last.stage), (None, Stage::LmHead));
    let tagged: Vec<usize> = trace.iter().filter_map(|e| e.layer).collect();
    assert!(tagged.is_sorted(), "layers out of order");
    for layer in 0..layers {
        let calls = trace.iter().filter(|e| e.layer == Some(layer)).count();
        assert_eq!(calls, 11, "layer {layer}");
    }
    let untagged = trace.iter().filter(|e| e.layer.is_none()).count();
    assert_eq!(untagged, 3, "embedding, final norm and lm head");
    Ok(())
}

#[test]
fn cpu_profiling_leaves_the_logits_alone_and_sees_every_stage() {
    let (engine, tokens) = engine(Backend::Cpu).unwrap();
    check(&engine, &tokens).unwrap();
}

#[test]
fn gpu_profiling_leaves_the_logits_alone_and_sees_every_stage() {
    let Ok(gpu) = cuda::Gpu::new() else {
        assert!(
            std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
            "MARIN_REQUIRE_CUDA is set but CUDA is unavailable"
        );
        eprintln!("skipping: no usable GPU");
        return;
    };
    let (engine, tokens) = engine(Backend::Cuda(Box::new(gpu))).unwrap();
    let before = engine.traffic().unwrap();
    check(&engine, &tokens).unwrap();
    let delta = engine.traffic().unwrap().since(&before);
    assert!(delta.launches > 0 && delta.d2h_bytes > 0, "{delta:?}");
}

#[test]
fn a_gpu_decode_step_allocates_nothing_and_matches_a_fresh_pass() {
    let Ok(gpu) = cuda::Gpu::new() else {
        eprintln!("skipping: no usable GPU");
        return;
    };
    let (engine, tokens) = engine(Backend::Cuda(Box::new(gpu))).unwrap();
    let mut cache = engine.new_cache(64).unwrap();
    engine.forward(&tokens, &mut cache).unwrap();
    engine.forward(&[3], &mut cache).unwrap();
    let before = engine.traffic().unwrap();
    let step = engine.forward(&[5], &mut cache).unwrap();
    let delta = engine.traffic().unwrap().since(&before);
    assert_eq!(delta.allocations, 0, "{delta:?}");

    let mut replay = tokens.clone();
    replay.extend([3, 5]);
    let full = engine
        .forward(&replay, &mut engine.new_cache(64).unwrap())
        .unwrap();
    let worst = step
        .iter()
        .zip(&full)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(worst < 1e-3, "decode and prefill disagree by {worst}");
}

#[test]
fn a_gpu_prompt_longer_than_one_chunk_gives_the_same_logits_as_a_different_split() {
    let Ok(gpu) = cuda::Gpu::new() else {
        eprintln!("skipping: no usable GPU");
        return;
    };
    let (engine, tokens) = engine(Backend::Cuda(Box::new(gpu))).unwrap();
    let prompt: Vec<u32> = tokens.iter().copied().cycle().take(1030).collect();
    let whole = engine
        .forward(&prompt, &mut engine.new_cache(1100).unwrap())
        .unwrap();
    let mut cache = engine.new_cache(1100).unwrap();
    engine.forward(&prompt[..100], &mut cache).unwrap();
    let split = engine.forward(&prompt[100..], &mut cache).unwrap();
    let worst = whole
        .iter()
        .zip(&split)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        worst < 1e-4,
        "chunked and split prompts disagree by {worst}"
    );
}
