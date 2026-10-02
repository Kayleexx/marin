mod common;

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};

use common::{Sandbox, fixtures, text};

impl Sandbox {
    fn command(&self, models: &Path) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
        command
            .env("MARIN_HOME", &self.0)
            .env("MARIN_MODELS", models)
            .env("NO_COLOR", "1")
            .args(["--backend", "cpu"]);
        command
    }

    fn config(&self) -> String {
        fs::read_to_string(self.0.join("config.json")).unwrap_or_default()
    }
}

fn run(mut command: Command, stdin: &str) -> io::Result<Output> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(stdin.as_bytes())?;
    }
    child.wait_with_output()
}

#[test]
fn run_finds_the_only_model_remembers_it_and_prints_just_the_reply() {
    let sandbox = Sandbox::new("run").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.args([
        "run",
        "the",
        "cat",
        "sat",
        "--temp",
        "0",
        "--max-tokens",
        "6",
    ]);
    let out = run(command, "").unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(!out.stdout.is_empty());
    assert!(out.stderr.is_empty(), "{}", text(&out.stderr));
    assert!(
        sandbox.config().contains("tiny-chat"),
        "{}",
        sandbox.config()
    );

    let mut again = sandbox.command(Path::new("/nonexistent"));
    again.args(["run", "the cat sat", "--temp", "0", "--max-tokens", "6"]);
    let second = run(again, "").unwrap();
    assert_eq!(
        second.stdout, out.stdout,
        "the remembered default model is used and output repeats"
    );
}

#[test]
fn run_reads_the_prompt_from_stdin() {
    let sandbox = Sandbox::new("stdin").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.args(["run", "--temp", "0", "--max-tokens", "4"]);
    let out = run(command, "hello world\n").unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(!out.stdout.is_empty());
}

#[test]
fn run_without_any_prompt_fails_with_a_hint() {
    let sandbox = Sandbox::new("noprompt").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.arg("run");
    let out = run(command, "   \n").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("prompt is empty"),
        "{}",
        text(&out.stderr)
    );
}

#[test]
fn missing_and_ambiguous_models_get_short_actionable_errors() {
    let sandbox = Sandbox::new("errors").unwrap();
    let empty = sandbox.0.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let mut command = sandbox.command(&empty);
    command.args(["run", "hi"]);
    let out = run(command, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("error: no chat model found") && err.contains("hint:"),
        "{err}"
    );
    assert_eq!(err.lines().count(), 2, "{err}");

    let two = sandbox.0.join("two");
    for name in ["a", "b"] {
        let dir = two.join(name);
        fs::create_dir_all(&dir).unwrap();
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ] {
            fs::copy(fixtures().join("tiny-chat").join(file), dir.join(file)).unwrap();
        }
    }
    let mut command = sandbox.command(&two);
    command.args(["run", "hi"]);
    let err = text(&run(command, "").unwrap().stderr);
    assert!(err.contains("several models found: a, b"), "{err}");

    let mut command = sandbox.command(&two);
    command.args(["run", "hi", "--model", "/definitely/not/here"]);
    assert!(text(&run(command, "").unwrap().stderr).contains("not a directory"));
}

#[test]
fn model_command_lists_sets_and_validates_the_default() {
    let sandbox = Sandbox::new("model").unwrap();
    let out = run(
        {
            let mut c = sandbox.command(&fixtures());
            c.arg("model");
            c
        },
        "",
    )
    .unwrap();
    let listing = text(&out.stdout);
    assert!(
        listing.contains("default: none yet")
            && listing.contains("chat:    tiny-chat")
            && listing.contains("base:    tiny-qwen3 (use marin-run)"),
        "{listing}"
    );

    let tiny = fixtures().join("tiny-chat");
    let mut set = sandbox.command(&fixtures());
    set.arg("model").arg(&tiny);
    let out = run(set, "").unwrap();
    assert!(text(&out.stdout).contains("default model: tiny-chat"));
    assert!(sandbox.config().contains("tiny-chat"));
    let listing = text(
        &run(
            {
                let mut c = sandbox.command(&fixtures());
                c.arg("model");
                c
            },
            "",
        )
        .unwrap()
        .stdout,
    );
    assert!(listing.starts_with("default: tiny-chat"), "{listing}");

    let mut bad = sandbox.command(&fixtures());
    bad.arg("model").arg(&sandbox.0);
    let out = run(bad, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("not a model folder"));
}

#[test]
fn piped_input_runs_each_line_as_a_prompt_without_ui() {
    let sandbox = Sandbox::new("lines").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.args(["--temp", "0", "--max-tokens", "3"]);
    let out = run(command, "the cat\n\nhello world\n").unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(
        text(&out.stdout).lines().count(),
        2,
        "{}",
        text(&out.stdout)
    );
}

#[test]
fn invalid_sampling_options_fail_before_loading() {
    let sandbox = Sandbox::new("sampling").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.args(["run", "hi", "--temp=-1"]);
    let out = run(command, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("temperature"));
}

#[test]
fn help_shows_examples_and_points_to_the_debug_tool() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
    command.arg("--help");
    let out = run(command, "").unwrap();
    let help = text(&out.stdout);
    assert!(
        help.contains("Examples:") && help.contains("marin run") && help.contains("marin-run"),
        "{help}"
    );
}

/// Drives the real interactive UI through a pseudo-terminal supplied by util-linux `script`.
#[test]
fn base_models_are_refused_with_an_explanation_instead_of_being_used_silently() {
    let sandbox = Sandbox::new("base").unwrap();
    let base = fixtures().join("tiny-qwen3");

    let mut run_base = sandbox.command(&fixtures());
    run_base.args(["run", "hi", "--model"]).arg(&base);
    let err = text(&run(run_base, "").unwrap().stderr);
    assert!(
        err.contains("tiny-qwen3 is a base model") && err.contains("marin-run"),
        "{err}"
    );

    let mut set_base = sandbox.command(&fixtures());
    set_base.arg("model").arg(&base);
    let out = run(set_base, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        sandbox.config().is_empty(),
        "a base model never becomes the default"
    );

    let only_base = sandbox.0.join("only-base");
    fs::create_dir_all(&only_base).unwrap();
    fs::create_dir_all(only_base.join("tiny-qwen3-copy")).unwrap();
    for file in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "model.safetensors",
    ] {
        fs::copy(
            base.join(file),
            only_base.join("tiny-qwen3-copy").join(file),
        )
        .unwrap();
    }
    let mut none = sandbox.command(&only_base);
    none.args(["run", "hi"]);
    let err = text(&run(none, "").unwrap().stderr);
    assert!(
        err.contains("no chat model found; tiny-qwen3-copy is a base model"),
        "{err}"
    );
    assert!(
        err.contains("hint:") && err.contains("Qwen/Qwen3-1.7B"),
        "{err}"
    );
    assert_eq!(err.lines().count(), 2, "{err}");
}

#[test]
fn the_end_of_turn_token_from_generation_config_stops_the_reply() {
    let sandbox = Sandbox::new("eos").unwrap();
    let model = sandbox.0.join("tiny-chat-eos");
    fs::create_dir_all(&model).unwrap();
    for file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ] {
        fs::copy(fixtures().join("tiny-chat").join(file), model.join(file)).unwrap();
    }
    let everything: Vec<u32> = (0..297).collect();
    let config = serde_json::json!({ "eos_token_id": everything }).to_string();
    fs::write(model.join("generation_config.json"), config).unwrap();

    let mut command = sandbox.command(Path::new("/nonexistent"));
    command
        .args(["run", "hi", "--temp", "0", "--max-tokens", "50", "--model"])
        .arg(&model);
    let out = run(command, "").unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        out.stdout.is_empty(),
        "the stop token ends the reply and is never printed: {:?}",
        text(&out.stdout)
    );

    let mut without = sandbox.command(Path::new("/nonexistent"));
    without
        .args(["run", "hi", "--temp", "0", "--max-tokens", "5", "--model"])
        .arg(fixtures().join("tiny-chat"));
    assert!(
        !run(without, "").unwrap().stdout.is_empty(),
        "without stop tokens the same prompt does generate text"
    );
}

#[test]
fn a_piped_conversation_answers_every_line() {
    let sandbox = Sandbox::new("convo").unwrap();
    let mut command = sandbox.command(&fixtures());
    command.args(["--temp", "0", "--max-tokens", "3"]);
    let out = run(command, "hi\nwho are you?\n2+2?\n").unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(
        text(&out.stdout).lines().count(),
        3,
        "{}",
        text(&out.stdout)
    );
}
