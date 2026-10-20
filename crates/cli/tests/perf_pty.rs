//! `/bench` and `/verify` from the palette of the real terminal: asked for by typing after `/`, run while the terminal
//! is handed back, and followed by a prompt that works; and a run stopped by Ctrl-C without harm.

#[path = "pty/keys.rs"]
mod keys;
#[path = "common/long.rs"]
mod long;
mod pty;

use std::fs;
use std::path::Path;

use keys::{PROMPT, REPLY, run_in};
use pty::{Res, Sandbox};

fn model(sandbox: &Sandbox) -> Res<String> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let to = sandbox.0.join("model");
    fs::create_dir_all(&to)?;
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(fixtures.join("tiny-qwen3").join(name), to.join(name))?;
    }
    fs::copy(
        fixtures.join("tiny-chat/tokenizer_config.json"),
        to.join("tokenizer_config.json"),
    )?;
    fs::copy(
        fixtures.join("tiny-qwen3/reference.json"),
        to.join("reference.json"),
    )?;
    Ok(format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 3",
        env!("CARGO_BIN_EXE_marin"),
        to.display()
    ))
}

#[test]
fn the_palette_runs_a_benchmark_and_a_check_and_the_prompt_survives() -> Res<()> {
    let keys = [
        PROMPT,
        ("/", "Mode: chat"),
        ("bench", ""),
        ("\r", "Workload"),
        ("8 4 runs=1 quick\r", "peak process memory"),
        ("/", "Mode: chat"),
        ("verify", ""),
        ("\r", "Reference file"),
        ("\r", "result: PASS"),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
    ];
    let Some((screen, status, _)) = run_in("perf-pty", &keys, model)? else {
        return Ok(());
    };
    assert!(
        status.success(),
        "the terminal is back and Ctrl-D leaves: {screen:?}"
    );
    assert!(screen.contains("not measured by this run"), "{screen:?}");
    assert!(screen.contains("verifying model on cpu"), "{screen:?}");
    Ok(())
}

/// The tiny chat model with a window of 8,192 tokens, so that a benchmark takes seconds rather than milliseconds.
fn long_model(sandbox: &Sandbox) -> Res<String> {
    let tiny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let to = sandbox.0.join("long");
    long::long_tiny(&tiny, &to)?;
    Ok(format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 3",
        env!("CARGO_BIN_EXE_marin"),
        to.display()
    ))
}

#[test]
fn ctrl_c_stops_a_benchmark_and_the_prompt_and_the_conversation_survive() -> Res<()> {
    let keys = [
        PROMPT,
        ("hello there\r", REPLY),
        ("/", "Mode: chat"),
        ("bench", ""),
        ("\r", "Workload"),
        ("3000 500 runs=20\r", "run "),
        ("\x03", "cancelled; nothing was changed"),
        ("hello again\r", REPLY),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
    ];
    let Some((screen, status, _)) = run_in("perf-cancel", &keys, long_model)? else {
        return Ok(());
    };
    assert!(
        status.success(),
        "the terminal is back and Ctrl-D leaves: {screen:?}"
    );
    assert!(
        !screen.contains("results (median"),
        "a cancelled run prints no results: {screen:?}"
    );
    Ok(())
}
