//! marin's chat path against official Hugging Face `Qwen3-1.7B` (float32 math, non-thinking mode).
//! Needs the real model, so it runs when MARIN_QWEN3_CHAT_DIR points at a Qwen3-1.7B folder;
//! set MARIN_REQUIRE_QWEN3_CHAT=1 to make a missing folder a failure.
//! Reference file: scripts/gen_chat_reference.py.

use std::path::{Path, PathBuf};

use chat::DEFAULT_SYSTEM;
use clap::Parser;
use cli::chat::Chat;
use cli::load::load;
use cli::marin_args::MarinCli;
use cli::session::{Session, Settings};
use cli::signal::Interrupt;
use cli::style::Style;
use model::BackendChoice;
use runtime::StopReason;
use sampler::{Sampler, SamplerConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    system: String,
    sampling: Sampling,
    max_new_tokens: usize,
    stop_tokens: Vec<u32>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Sampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    user: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    stopped_on_eos: bool,
    step0_distribution: Vec<(u32, f32)>,
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("MARIN_QWEN3_CHAT_DIR").map(PathBuf::from);
    if dir.is_none() {
        assert!(
            std::env::var_os("MARIN_REQUIRE_QWEN3_CHAT").is_none(),
            "MARIN_QWEN3_CHAT_DIR is not set"
        );
        eprintln!("skipping: set MARIN_QWEN3_CHAT_DIR to a Qwen3-1.7B folder to run this check");
    }
    dir
}

fn reference() -> Result<Reference, Box<dyn std::error::Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference/qwen3-chat-differential.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn greedy(max_tokens: usize) -> Settings {
    Settings {
        max_tokens,
        temperature: 0.0,
        top_k: 20,
        top_p: 0.8,
        seed: Some(1),
    }
}

#[test]
fn marin_defaults_are_the_official_non_thinking_recommendation_not_the_reference_settings() {
    let reference = reference().unwrap();
    let defaults = MarinCli::try_parse_from(["marin"]).unwrap().options;
    assert_eq!(
        (defaults.temp, defaults.top_k, defaults.top_p),
        (
            reference.sampling.temperature,
            reference.sampling.top_k,
            reference.sampling.top_p
        ),
        "Qwen3 non-thinking: temperature 0.7, top_k 20, top_p 0.8"
    );
    assert!(
        defaults.temp > 0.0,
        "greedy decoding is only for the reference and debug path"
    );
    assert_eq!(
        reference.system, DEFAULT_SYSTEM,
        "the reference used the same system message as marin"
    );
}

/// The debug CLI's defaults (top_k off, top_p 0.9, 128 tokens) must never become the assistant's.
#[test]
fn the_debug_cli_defaults_do_not_leak_into_marin() {
    let marin = MarinCli::try_parse_from(["marin"]).unwrap().options;
    let debug = cli::args::Cli::try_parse_from(["marin-run", "./model"]).unwrap();
    assert_eq!((marin.temp, marin.top_k, marin.top_p), (0.7, 20, 0.8));
    assert_eq!(
        (debug.run.temp, debug.run.top_k, debug.run.top_p),
        (0.7, 0, 0.9)
    );
    assert!(
        cli::args::Cli::try_parse_from([
            "marin-run",
            "verify",
            "./m",
            "--reference",
            "r",
            "--temp",
            "1"
        ])
        .is_err(),
        "verify takes no sampling flags: it is greedy by construction"
    );
    assert_ne!(marin.top_p, debug.run.top_p);
    assert_ne!(marin.top_k, debug.run.top_k);
}

#[test]
fn marin_matches_hugging_face_on_prompts_replies_stops_and_sampling_distributions() {
    let Some(dir) = model_dir() else { return };
    let reference = reference().unwrap();
    let loaded = load(
        &dir,
        Some(1024),
        1024,
        BackendChoice::Auto,
        None,
        &Style::plain(),
        false,
    )
    .unwrap();
    let mut session =
        Session::new(loaded, greedy(reference.max_new_tokens), Style::plain()).unwrap();
    assert_eq!(
        session.loaded.model.config().eos_tokens,
        reference.stop_tokens,
        "stop tokens come from generation_config.json"
    );
    let interrupt = Interrupt::install().unwrap();
    for case in &reference.cases {
        let mut chat = Chat::new(&session);
        for (n, turn) in case.turns.iter().enumerate() {
            let label = format!("{} / turn {}: {:?}", case.name, n + 1, turn.user);
            let reply = chat
                .ask(&mut session, &turn.user, &interrupt, &mut Vec::new())
                .unwrap();
            assert_eq!(
                session.last_prompt, turn.prompt_ids,
                "prompt token ids differ from apply_chat_template: {label}"
            );
            assert_eq!(
                reply.tokens, turn.greedy_ids,
                "greedy reply differs from Hugging Face: {label}"
            );
            let stop = if turn.stopped_on_eos {
                StopReason::EndOfText
            } else {
                StopReason::MaxTokens
            };
            assert_eq!(reply.outcome.stop, stop, "{label}");

            let logits = session.next_logits(&turn.prompt_ids).unwrap();
            let config = SamplerConfig {
                temperature: reference.sampling.temperature,
                top_k: reference.sampling.top_k,
                top_p: reference.sampling.top_p,
                seed: 7,
            };
            let ours = Sampler::new(config)
                .unwrap()
                .distribution(&logits, session.loaded.dir.tokenizer.vocab_size())
                .unwrap();
            let ids = |d: &[(u32, f32)]| {
                d.iter()
                    .map(|(id, _)| *id)
                    .collect::<std::collections::BTreeSet<u32>>()
            };
            assert_eq!(
                ids(&ours),
                ids(&turn.step0_distribution),
                "sampling candidate set differs: {label}"
            );
            for (id, theirs) in &turn.step0_distribution {
                let mine = ours.iter().find(|(i, _)| i == id).map(|(_, p)| *p).unwrap();
                assert!(
                    (mine - theirs).abs() < 2e-3,
                    "probability of token {id} is {mine} vs {theirs}: {label}"
                );
            }
        }
    }
}
