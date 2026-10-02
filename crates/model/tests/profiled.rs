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
    Ok((Engine::load(&dir, backend, |_, _| {})?, tokens))
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
    let (engine, tokens) = engine(Backend::Cuda(gpu)).unwrap();
    let before = engine.traffic().unwrap();
    check(&engine, &tokens).unwrap();
    let delta = engine.traffic().unwrap().since(&before);
    assert!(delta.launches > 0 && delta.d2h_bytes > 0, "{delta:?}");
}
