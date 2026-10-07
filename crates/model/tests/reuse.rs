//! A kept position was computed in a pass of another length than a fresh read uses, so rounding could flip a token.

use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use loader::ModelDir;
use model::{Backend, Engine, EngineCache, KvFormat, QuantBits, QuantScheme};
use runtime::{
    CausalLm, GenerateParams, Kept, Retained, SequenceCache, Speculation, StopReason, generate,
    generate_reusing,
};
use sampler::{Sampler, SamplerConfig};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CAPACITY: usize = 300;

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

fn every_case(f: &Fixture, greedy_only: bool) -> Result<()> {
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
    let all = [
        config(0.0, 0, 1.0, 0),
        config(0.7, 20, 0.8, 5),
        config(1.2, 0, 1.0, 9),
    ];
    for sampling in all.into_iter().take(if greedy_only { 1 } else { 3 }) {
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
    every_case(&fixture(backend, None, KvFormat::F32).unwrap(), false).unwrap();
}

#[test]
fn int8_turns_on_a_kept_cache_equal_fresh_reads() {
    let Some(backend) = gpu() else { return };
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    every_case(
        &fixture(backend, Some(scheme), KvFormat::F32).unwrap(),
        false,
    )
    .unwrap();
}

#[test]
fn cpu_turns_on_a_kept_cache_equal_fresh_reads() {
    every_case(&fixture(Backend::Cpu, None, KvFormat::F32).unwrap(), false).unwrap();
}

/// Rounding a key or value to fewer bits turns the tiny differences between a prefill pass and a decode pass into
/// occasional jumps of one rounding step, so a sampled reply can differ from a fresh read at the noise floor. Greedy
/// replies must still match exactly, and the logits of a kept cache must stay within a few times the f32 noise.
fn logits_stay_close(f: &Fixture, label: &str) -> Result<()> {
    let ids = encode(
        f,
        "<|im_start|>user\nhello world, the cat sat on the mat. hello world, the cat sat and then a b a b a b a b",
    )?;
    let fresh = f.engine.forward(&ids, &mut f.engine.new_cache(CAPACITY)?)?;
    for split in [1, 5, 17, ids.len().saturating_sub(1)] {
        let mut cache = f.engine.new_cache(CAPACITY)?;
        let (head, tail) = ids.split_at(split);
        f.engine.forward(head, &mut cache)?;
        let kept = f.engine.forward(tail, &mut cache)?;
        let worst = fresh
            .iter()
            .zip(&kept)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst < 5e-3,
            "{label} split {split}: largest logit change {worst}"
        );
    }
    Ok(())
}

#[test]
fn lower_precision_kv_turns_on_a_kept_cache_match_fresh_reads() {
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 32,
    };
    for kv in [KvFormat::F16] {
        for quant in [None, Some(scheme)] {
            let Some(backend) = gpu() else { return };
            let f = fixture(backend, quant, kv).unwrap();
            every_case(&f, true).unwrap();
            logits_stay_close(&f, &format!("{kv:?} {}", quant.is_some())).unwrap();
        }
    }
}

#[test]
fn a_lower_precision_kv_cache_needs_the_gpu() {
    assert!(fixture(Backend::Cpu, None, KvFormat::F16).is_err());
}
