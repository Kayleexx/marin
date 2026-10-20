//! `/bench` and `/verify` through the real `marin` on a pipe: they measure and check the model that is loaded, say what
//! they cannot do, appear in `/help`, and leave the conversation as it was in either mode.

mod common;
#[path = "common/long.rs"]
mod long;

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use common::{Sandbox, fixtures, text};
use long::long_tiny;

/// A chat model with the weights the fixtures' reference was made from: the tiny Qwen3 with a chat template.
fn reference_model(into: &Path, with_reference: bool) -> io::Result<PathBuf> {
    let to = into.join("reference-model");
    fs::create_dir_all(&to)?;
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(fixtures().join("tiny-qwen3").join(name), to.join(name))?;
    }
    fs::copy(
        fixtures().join("tiny-chat/tokenizer_config.json"),
        to.join("tokenizer_config.json"),
    )?;
    if with_reference {
        fs::copy(
            fixtures().join("tiny-qwen3/reference.json"),
            to.join("reference.json"),
        )?;
    }
    Ok(to)
}

fn marin(sandbox: &Sandbox, model: &Path, script: &str, cwd: &Path) -> io::Result<Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
        .current_dir(cwd)
        .env("MARIN_HOME", sandbox.0.join("home"))
        .env("NO_COLOR", "1")
        .args([
            "--backend",
            "cpu",
            "--temp",
            "0",
            "--max-tokens",
            "6",
            "--model",
        ])
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(script.as_bytes())?;
    }
    child.wait_with_output()
}

fn both(out: &Output) -> String {
    format!("{}{}", text(&out.stdout), text(&out.stderr))
}

#[test]
fn bench_and_verify_are_listed_and_do_their_job_on_the_loaded_model() -> io::Result<()> {
    let sandbox = Sandbox::new("perf-basic")?;
    let model = reference_model(&sandbox.0, true)?;
    let out = marin(
        &sandbox,
        &model,
        "/help\n/bench 16 4 runs=2\n/verify\n",
        &sandbox.0,
    )?;
    let shown = both(&out);
    assert!(out.status.success(), "{shown}");
    for needle in [
        "/bench",
        "/verify",
        "not measured by this run: loading the model took",
        "it was not loaded again",
        "first token (TTFT)",
        "decode",
        "verifying reference-model on cpu against",
        "reference.json",
        "result: PASS (3 cases)",
    ] {
        assert!(shown.contains(needle), "{needle}: {shown}");
    }
    Ok(())
}

#[test]
fn problems_are_explained_and_the_run_goes_on() -> io::Result<()> {
    let sandbox = Sandbox::new("perf-problems")?;
    let model = reference_model(&sandbox.0, false)?;
    let wrong = fixtures().join("tiny-qwen35/reference.json");
    let script = format!(
        "/verify\n/verify {}\n/verify nowhere.json\n/bench 5000\n/bench banana\n/status\n",
        wrong.display()
    );
    let out = marin(&sandbox, &model, &script, &sandbox.0)?;
    let shown = both(&out);
    assert!(out.status.success(), "{shown}");
    for needle in [
        "no reference made for reference-model with bfloat16 weights",
        "/verify PATH",
        "it has 5 layers per prompt and this model has 2",
        "nowhere.json",
        "this session's window is 128",
        "usage: /bench",
    ] {
        assert!(shown.contains(needle), "{needle}: {shown}");
    }
    assert!(!shown.contains("result: PASS"), "nothing passed: {shown}");
    assert!(
        !text(&out.stdout).is_empty(),
        "the chat still answered after every refusal"
    );
    Ok(())
}

#[test]
fn a_reference_that_disagrees_fails_and_never_passes() -> io::Result<()> {
    let sandbox = Sandbox::new("perf-fail")?;
    let model = reference_model(&sandbox.0, false)?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(fixtures().join("tiny-qwen3/reference.json"))?)?;
    if let Some(logit) = value.pointer_mut("/cases/0/steps/0/top/0/1") {
        *logit = serde_json::json!(logit.as_f64().unwrap_or(0.0) + 3.0);
    }
    let path = sandbox.0.join("reference.json");
    fs::write(&path, value.to_string())?;
    let out = marin(&sandbox, &model, "/verify reference.json\n", &sandbox.0)?;
    let shown = both(&out);
    assert!(
        shown.contains("FAIL") && shown.contains("result: FAIL"),
        "{shown}"
    );
    assert!(!shown.contains("result: PASS"), "{shown}");
    Ok(())
}

#[test]
fn benchmarking_between_questions_changes_neither_mode_s_conversation() -> io::Result<()> {
    let sandbox = Sandbox::new("perf-modes")?;
    let models = sandbox.0.join("models");
    long_tiny(&fixtures().join("tiny-chat"), &models.join("alpha"))?;
    let model = models.join("alpha");
    let plain = marin(
        &sandbox,
        &model,
        "hello there\nand then\n/status\n",
        &sandbox.0,
    )?;
    let benched = marin(
        &sandbox,
        &model,
        "hello there\n/bench 16 4 quick\nand then\n/bench 8 4 quick\n/status\n",
        &sandbox.0,
    )?;
    let context = |out: &Output| {
        text(&out.stdout)
            .lines()
            .find(|l| l.contains(" used"))
            .unwrap_or("")
            .to_owned()
    };
    assert!(
        both(&benched).contains("results (median of runs)"),
        "{}",
        both(&benched)
    );
    assert_eq!(
        context(&plain),
        context(&benched),
        "the window holds what it held"
    );
    assert!(!context(&plain).is_empty());

    let sizes = |script: &str| -> io::Result<Vec<String>> {
        let out = marin(&sandbox, &model, script, &sandbox.0)?;
        Ok(text(&out.stdout)
            .split("context ")
            .skip(1)
            .filter_map(|r| r.split_once('/').map(|(used, _)| used.to_owned()))
            .filter(|used| used.chars().all(|c| c.is_ascii_digit() || c == ','))
            .collect())
    };
    let alone = sizes("/mode code\nwhere is main?\n")?;
    let mixed = sizes(
        "hi\n/mode code\n/bench 8 4 quick\nwhere is main?\n/mode chat\n/bench 8 4 quick\n/clear\n/mode code\n",
    )?;
    assert!(!alone.is_empty());
    assert_eq!(
        mixed.first(),
        alone.first(),
        "a benchmark did not reach the code prompt: {alone:?} {mixed:?}"
    );
    Ok(())
}
