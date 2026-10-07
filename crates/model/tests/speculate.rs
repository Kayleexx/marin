//! Guessing ahead on the real engine must give exactly the tokens plain decoding gives: bf16 and int8 weights,
//! greedy and sampled, for rules that guess often and so roll the cache back again and again. The CPU never guesses.

use std::error::Error;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use loader::ModelDir;
use model::{Backend, Engine, KvFormat, QuantBits, QuantScheme};
use runtime::{GenerateParams, SequenceCache, Speculation, generate};
use sampler::{Sampler, SamplerConfig};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Fixture {
    engine: Engine,
    dir: ModelDir,
}

fn fixture(backend: Backend, quant: Option<QuantScheme>, kv: KvFormat) -> Result<Fixture> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let engine = Engine::load(&dir, backend, quant, |_, _| {})?.with_kv_format(kv)?;
    Ok(Fixture { engine, dir })
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

struct Reply {
    tokens: Vec<u32>,
    stop: runtime::StopReason,
    used: usize,
    guessed_passes: usize,
    rejected: usize,
    passes: usize,
}

fn reply(
    f: &Fixture,
    prompt: &[u32],
    sampling: SamplerConfig,
    max_tokens: usize,
    speculation: Option<Speculation>,
) -> Result<Reply> {
    let mut cache = f.engine.new_cache(160)?;
    let params = GenerateParams {
        max_tokens,
        stop_tokens: f.engine.config().eos_tokens.clone(),
        usable_vocab: f.dir.tokenizer.vocab_size(),
        speculation,
    };
    let mut tokens = Vec::new();
    let outcome = generate(
        &f.engine,
        &mut cache,
        &mut Sampler::new(sampling)?,
        prompt,
        &params,
        &AtomicBool::new(false),
        |t| tokens.push(t),
    )?;
    let stats = &outcome.speculation;
    Ok(Reply {
        tokens,
        stop: outcome.stop,
        used: cache.used(),
        guessed_passes: stats.guessed_passes,
        rejected: stats.drafted.saturating_sub(stats.accepted),
        passes: stats.passes,
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

fn samplings() -> [SamplerConfig; 3] {
    let config = |temperature, top_k, top_p, seed| SamplerConfig {
        temperature,
        top_k,
        top_p,
        seed,
    };
    [
        config(0.0, 0, 1.0, 0),
        config(0.7, 20, 0.8, 5),
        config(1.2, 0, 1.0, 9),
    ]
}

/// Plain and guessing decodes agree on every prompt, sampler and rule; returns how often guessing was used and wrong.
fn check(f: &Fixture) -> Result<(usize, usize)> {
    let (mut guessed, mut rejected) = (0_usize, 0_usize);
    for prompt in prompts(f)? {
        for sampling in samplings() {
            for max_tokens in [1, 9, 70] {
                let plain = reply(f, &prompt, sampling, max_tokens, None)?;
                for (match_len, max_draft) in [(1, 7), (2, 3), (3, 7), (8, 7)] {
                    let rule = Speculation {
                        match_len,
                        max_draft,
                    };
                    let guessing = reply(f, &prompt, sampling, max_tokens, Some(rule))?;
                    let label = format!("{sampling:?} max {max_tokens} rule {rule:?}");
                    assert_eq!(guessing.tokens, plain.tokens, "{label}");
                    assert_eq!(guessing.stop, plain.stop, "{label}");
                    assert_eq!(guessing.used, plain.used, "{label}: cache length");
                    assert!(guessing.passes <= plain.passes, "{label}");
                    guessed = guessed.saturating_add(guessing.guessed_passes);
                    rejected = rejected.saturating_add(guessing.rejected);
                }
            }
        }
    }
    Ok((guessed, rejected))
}

#[test]
fn bf16_guessing_gives_plain_decodings_tokens() {
    let Some(backend) = gpu() else { return };
    let f = fixture(backend, None, KvFormat::F32).unwrap();
    let (guessed, rejected) = check(&f).unwrap();
    assert!(
        guessed > 50 && rejected > 50,
        "guessed {guessed} rejected {rejected}"
    );
}

#[test]
fn int8_guessing_gives_plain_decodings_tokens() {
    let Some(backend) = gpu() else { return };
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    let f = fixture(backend, Some(scheme), KvFormat::F32).unwrap();
    let (guessed, rejected) = check(&f).unwrap();
    assert!(
        guessed > 50 && rejected > 50,
        "guessed {guessed} rejected {rejected}"
    );
}

#[test]
fn the_cpu_decodes_plainly_when_asked_to_guess() {
    let f = fixture(Backend::Cpu, None, KvFormat::F32).unwrap();
    let prompt = prompts(&f).unwrap().remove(0);
    let sampling = samplings()[0];
    let plain = reply(&f, &prompt, sampling, 30, None).unwrap();
    let asked = reply(
        &f,
        &prompt,
        sampling,
        30,
        Some(Speculation {
            match_len: 1,
            max_draft: 7,
        }),
    )
    .unwrap();
    assert_eq!(asked.tokens, plain.tokens);
    assert_eq!((asked.guessed_passes, asked.used), (0, plain.used));
}

#[test]
fn lower_precision_kv_guessing_gives_plain_decodings_tokens() {
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    for kv in [KvFormat::F16] {
        for quant in [None, Some(scheme)] {
            let Some(backend) = gpu() else { return };
            let (guessed, rejected) = check(&fixture(backend, quant, kv).unwrap()).unwrap();
            assert!(
                guessed > 50 && rejected > 50,
                "{kv:?}: guessed {guessed} rejected {rejected}"
            );
        }
    }
}
