//! Keeping the cache between turns on the real engine must give the tokens a fresh read of the same prompt gives:
//! bf16 and int8 weights on the GPU, greedy and sampled, with and without guessing ahead, across a stop token, a
//! cancelled reply and an edited history. A kept position was computed in a pass of another length than a fresh read
//! uses, so this is where a rounding difference would flip a token.

use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use loader::ModelDir;
use model::{Backend, Engine, EngineCache, QuantBits, QuantScheme};
use runtime::{
    GenerateParams, Kept, Retained, SequenceCache, Speculation, StopReason, generate,
    generate_reusing,
};
use sampler::{Sampler, SamplerConfig};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CAPACITY: usize = 300;

struct Fixture {
    engine: Engine,
    dir: ModelDir,
}

fn fixture(backend: Backend, quant: Option<QuantScheme>) -> Result<Fixture> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let dir = ModelDir::open(&path)?;
    let engine = Engine::load(&dir, backend, quant, |_, _| {})?;
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

fn encode(f: &Fixture, text: &str) -> Result<Vec<u32>> {
    Ok(f.dir.tokenizer.encode(text)?)
}

fn params(
    f: &Fixture,
    max_tokens: usize,
    stop: Vec<u32>,
    speculation: Option<Speculation>,
) -> GenerateParams {
    GenerateParams {
        max_tokens,
        stop_tokens: stop,
        usable_vocab: f.dir.tokenizer.vocab_size(),
        speculation,
    }
}

struct Reply {
    tokens: Vec<u32>,
    stop: StopReason,
    reused: usize,
}

fn kept(
    f: &Fixture,
    cache: &mut EngineCache,
    retained: &mut Retained,
    prompt: &[u32],
    sampling: SamplerConfig,
    params: &GenerateParams,
    cancel_after: Option<usize>,
) -> Result<Reply> {
    let cancel = AtomicBool::new(false);
    let mut tokens = Vec::new();
    let outcome = generate_reusing(
        &f.engine,
        Kept { cache, retained },
        &mut Sampler::new(sampling)?,
        prompt,
        params,
        &cancel,
        |t| {
            tokens.push(t);
            if cancel_after == Some(tokens.len()) {
                cancel.store(true, Ordering::Relaxed);
            }
        },
    )?;
    assert_eq!(retained.len(), cache.used(), "the record covers the cache");
    Ok(Reply {
        tokens,
        stop: outcome.stop,
        reused: outcome.reused,
    })
}

fn fresh(
    f: &Fixture,
    prompt: &[u32],
    sampling: SamplerConfig,
    params: &GenerateParams,
) -> Result<Reply> {
    let mut cache = f.engine.new_cache(CAPACITY)?;
    let mut tokens = Vec::new();
    let outcome = generate(
        &f.engine,
        &mut cache,
        &mut Sampler::new(sampling)?,
        prompt,
        params,
        &AtomicBool::new(false),
        |t| tokens.push(t),
    )?;
    Ok(Reply {
        tokens,
        stop: outcome.stop,
        reused: outcome.reused,
    })
}

fn same(a: &Reply, b: &Reply, label: &str) {
    assert_eq!(a.tokens, b.tokens, "{label}");
    assert_eq!(a.stop, b.stop, "{label}");
}

fn start(f: &Fixture) -> Result<Vec<u32>> {
    encode(
        f,
        "<|im_start|>user\nhello world, the cat sat on the mat. hello world, the cat sat",
    )
}

/// Turns that each carry the last prompt and reply, then an edit in the middle of the history.
fn conversation(
    f: &Fixture,
    sampling: SamplerConfig,
    rule: Option<Speculation>,
    label: &str,
) -> Result<()> {
    let mut cache = f.engine.new_cache(CAPACITY)?;
    let mut retained = Retained::default();
    let p = params(f, 24, Vec::new(), rule);
    let mut prompt = start(f)?;
    let mut previous = 0;
    for (turn, more) in [" and then", " a b a b a b a b a b", " x"]
        .iter()
        .enumerate()
    {
        let mine = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, None)?;
        same(&mine, &fresh(f, &prompt, sampling, &p)?, label);
        assert!(mine.reused >= previous, "{label}: turn {turn}");
        previous = prompt.len();
        prompt.extend(&mine.tokens);
        prompt.extend(encode(f, more)?);
    }
    let middle = prompt.len() / 2;
    let edited = prompt.get_mut(middle).ok_or("prompt too short")?;
    *edited = edited.wrapping_add(1) % 320;
    let mine = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, None)?;
    same(&mine, &fresh(f, &prompt, sampling, &p)?, label);
    assert!(mine.reused <= middle, "{label}: kept past the edit");
    Ok(())
}

fn cancelled(
    f: &Fixture,
    sampling: SamplerConfig,
    rule: Option<Speculation>,
    label: &str,
) -> Result<()> {
    let mut cache = f.engine.new_cache(CAPACITY)?;
    let mut retained = Retained::default();
    let p = params(f, 40, Vec::new(), rule);
    let mut prompt = start(f)?;
    let cut = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, Some(4))?;
    assert_eq!(cut.stop, StopReason::Cancelled, "{label}");
    let before = prompt.len();
    prompt.extend(&cut.tokens);
    prompt.extend(encode(f, " more")?);
    let mine = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, None)?;
    same(&mine, &fresh(f, &prompt, sampling, &p)?, label);
    assert!(
        mine.reused >= before,
        "{label}: the prompt before the cut is kept"
    );
    Ok(())
}

fn stopped(
    f: &Fixture,
    sampling: SamplerConfig,
    rule: Option<Speculation>,
    label: &str,
) -> Result<()> {
    let probe = fresh(f, &start(f)?, sampling, &params(f, 12, Vec::new(), None))?;
    let stop = *probe.tokens.get(3).ok_or("reply too short")?;
    let p = params(f, 40, vec![stop], rule);
    let mut cache = f.engine.new_cache(CAPACITY)?;
    let mut retained = Retained::default();
    let mut prompt = start(f)?;
    let first = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, None)?;
    assert_eq!(first.stop, StopReason::EndOfText, "{label}");
    same(&first, &fresh(f, &prompt, sampling, &p)?, label);
    prompt.extend(&first.tokens);
    prompt.extend(encode(f, " again")?);
    let mine = kept(f, &mut cache, &mut retained, &prompt, sampling, &p, None)?;
    same(&mine, &fresh(f, &prompt, sampling, &p)?, label);
    Ok(())
}

fn every_case(f: &Fixture) -> Result<()> {
    let config = |temperature, top_k, top_p, seed| SamplerConfig {
        temperature,
        top_k,
        top_p,
        seed,
    };
    let rules = [
        None,
        Some(Speculation {
            match_len: 1,
            max_draft: 7,
        }),
        Some(Speculation {
            match_len: 3,
            max_draft: 7,
        }),
    ];
    for sampling in [
        config(0.0, 0, 1.0, 0),
        config(0.7, 20, 0.8, 5),
        config(1.2, 0, 1.0, 9),
    ] {
        for rule in rules {
            let label = format!("{sampling:?} {rule:?}");
            conversation(f, sampling, rule, &label)?;
            cancelled(f, sampling, rule, &label)?;
            stopped(f, sampling, rule, &label)?;
        }
    }
    Ok(())
}

#[test]
fn bf16_turns_on_a_kept_cache_equal_fresh_reads() {
    let Some(backend) = gpu() else { return };
    every_case(&fixture(backend, None).unwrap()).unwrap();
}

#[test]
fn int8_turns_on_a_kept_cache_equal_fresh_reads() {
    let Some(backend) = gpu() else { return };
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    every_case(&fixture(backend, Some(scheme)).unwrap()).unwrap();
}

#[test]
fn cpu_turns_on_a_kept_cache_equal_fresh_reads() {
    every_case(&fixture(Backend::Cpu, None).unwrap()).unwrap();
}
