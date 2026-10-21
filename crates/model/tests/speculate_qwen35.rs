//! Guessing ahead on the Qwen3.5 GPU engine, whose recurrent state cannot be cut back by a length: the tokens, the stop
//! and the cache after a reply equal plain decoding's, for rules that guess often and so cut back again and again, and
//! a reply cancelled in the middle of a burst leaves the cache plain decoding would have left. Needs a GPU.

use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use loader::ModelDir;
use model::{Backend, Engine, EngineCache};
use runtime::{CausalLm, GenerateParams, SequenceCache, Speculation, StopReason, generate};
use sampler::{Sampler, SamplerConfig};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Fixture {
    engine: Engine,
    dir: ModelDir,
}

fn fixture() -> Result<Option<Fixture>> {
    let gpu = match cuda::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping: no usable GPU ({e})");
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35");
    let dir = ModelDir::open(&path)?;
    let engine = Engine::load(&dir, Backend::Cuda(Box::new(gpu)), None, |_, _| {})?;
    Ok(Some(Fixture { engine, dir }))
}

struct Reply {
    tokens: Vec<u32>,
    stop: StopReason,
    cache: EngineCache,
    guessed_passes: usize,
    rejected: usize,
}

/// Decodes `prompt`, cancelling after `cancel_after` tokens if given.
fn reply(
    f: &Fixture,
    prompt: &[u32],
    sampling: SamplerConfig,
    max_tokens: usize,
    speculation: Option<Speculation>,
    cancel_after: Option<usize>,
) -> Result<Reply> {
    let mut cache = f.engine.new_cache(200)?;
    let params = GenerateParams {
        max_tokens,
        stop_tokens: f.engine.config().eos_tokens.clone(),
        usable_vocab: f.dir.tokenizer.vocab_size(),
        speculation,
    };
    let cancel = AtomicBool::new(false);
    let mut tokens = Vec::new();
    let outcome = generate(
        &f.engine,
        &mut cache,
        &mut Sampler::new(sampling)?,
        prompt,
        &params,
        &cancel,
        |t| {
            tokens.push(t);
            if Some(tokens.len()) == cancel_after {
                cancel.store(true, Ordering::Relaxed);
            }
        },
    )?;
    let stats = &outcome.speculation;
    Ok(Reply {
        tokens,
        stop: outcome.stop,
        cache,
        guessed_passes: stats.guessed_passes,
        rejected: stats.drafted.saturating_sub(stats.accepted),
    })
}

fn prompts(f: &Fixture) -> Result<Vec<Vec<u32>>> {
    [
        "<|im_start|>user\nhello world, the cat sat on the mat. hello world, the cat sat",
        "<|im_start|>user\nhello",
        "a b a b a b a b a b a b",
    ]
    .iter()
    .map(|text| Ok(f.dir.tokenizer.encode(text)?))
    .collect()
}

fn samplings() -> [SamplerConfig; 2] {
    let config = |temperature, top_k, top_p, seed| SamplerConfig {
        temperature,
        top_k,
        top_p,
        seed,
    };
    [config(0.0, 0, 1.0, 0), config(0.7, 20, 0.8, 5)]
}

const RULES: [Speculation; 3] = [
    Speculation {
        match_len: 1,
        max_draft: 3,
    },
    Speculation {
        match_len: 2,
        max_draft: 3,
    },
    Speculation::SHORT_PASS,
];

/// What the next token would be if the reply went on: the cache's whole state, read through one more pass.
fn next_logits(f: &Fixture, mut cache: EngineCache) -> Result<Vec<f32>> {
    Ok(f.engine.forward(&[7], &mut cache)?)
}

#[test]
fn guessing_gives_the_tokens_stop_and_cache_of_plain_decoding() -> Result<()> {
    let Some(f) = fixture()? else { return Ok(()) };
    let (mut guessed, mut rejected) = (0_usize, 0_usize);
    for prompt in prompts(&f)? {
        for sampling in samplings() {
            for max_tokens in [1, 9, 60] {
                let plain = reply(&f, &prompt, sampling, max_tokens, None, None)?;
                for rule in RULES {
                    let label = format!("{sampling:?} max {max_tokens} rule {rule:?}");
                    let spec = reply(&f, &prompt, sampling, max_tokens, Some(rule), None)?;
                    assert_eq!(spec.tokens, plain.tokens, "{label}");
                    assert_eq!(spec.stop, plain.stop, "{label}");
                    assert_eq!(spec.cache.used(), plain.cache.used(), "{label}");
                    guessed += spec.guessed_passes;
                    rejected += spec.rejected;
                    let (a, b) = (
                        next_logits(&f, spec.cache)?,
                        next_logits(&f, plain_copy(&f, &prompt, sampling, max_tokens)?)?,
                    );
                    let worst = a
                        .iter()
                        .zip(&b)
                        .fold(0.0_f32, |w, (x, y)| w.max((x - y).abs()));
                    assert!(worst < 1e-4, "{label}: the cache differs by {worst:e}");
                }
            }
        }
    }
    assert!(
        guessed > 20 && rejected > 20,
        "guessed {guessed} rejected {rejected}"
    );
    Ok(())
}

fn plain_copy(
    f: &Fixture,
    prompt: &[u32],
    sampling: SamplerConfig,
    max_tokens: usize,
) -> Result<EngineCache> {
    Ok(reply(f, prompt, sampling, max_tokens, None, None)?.cache)
}

#[test]
fn cancelling_in_the_middle_of_a_burst_leaves_the_cache_plain_decoding_leaves() -> Result<()> {
    let Some(f) = fixture()? else { return Ok(()) };
    let prompt = prompts(&f)?.remove(0);
    let sampling = samplings()[0];
    for after in [1, 2, 3, 5, 8] {
        let plain = reply(&f, &prompt, sampling, 40, None, Some(after))?;
        let rule = RULES[0];
        let spec = reply(&f, &prompt, sampling, 40, Some(rule), Some(after))?;
        assert_eq!(spec.tokens, plain.tokens, "cancelled after {after}");
        assert_eq!(spec.stop, StopReason::Cancelled);
        assert_eq!(
            spec.cache.used(),
            plain.cache.used(),
            "cancelled after {after}"
        );
        let (a, b) = (next_logits(&f, spec.cache)?, next_logits(&f, plain.cache)?);
        let worst = a
            .iter()
            .zip(&b)
            .fold(0.0_f32, |w, (x, y)| w.max((x - y).abs()));
        assert!(
            worst < 1e-4,
            "cancelled after {after}: the cache differs by {worst:e}"
        );
    }
    Ok(())
}
