mod common;

use std::io::{self, Write};
use std::process::{Command, Output, Stdio};

use common::{Sandbox, fixtures, text};
use serde_json::Value;

fn marin_run(args: &[&str], stdin: &str) -> io::Result<Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(args)
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(stdin.as_bytes())?;
    }
    child.wait_with_output()
}

fn trace(args: &[&str]) -> io::Result<Output> {
    let model = fixtures().join("tiny-qwen3").display().to_string();
    let mut all = vec!["trace", &model, "--backend", "cpu"];
    all.extend_from_slice(args);
    marin_run(&all, "")
}

#[test]
fn a_decode_trace_lists_every_layer_and_the_other_stages() -> io::Result<()> {
    let out = trace(&["--prompt-tokens", "8"])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let report = text(&out.stdout);
    for part in [
        "trace: one-token pass 1 (the token at position 8)",
        "embedding",
        "layer ",
        "sum ",
        "final norm",
        "lm head",
        "was outside the timed stages",
        "reading this:",
    ] {
        assert!(report.contains(part), "missing {part:?} in\n{report}");
    }
    assert!(!report.contains("NaN"));
    Ok(())
}

#[test]
fn a_prefill_trace_and_a_later_decode_pass_both_work() -> io::Result<()> {
    let prefill = trace(&["--prompt-tokens", "5", "--pass", "prefill"])?;
    assert!(prefill.status.success(), "{}", text(&prefill.stderr));
    assert!(text(&prefill.stdout).contains("the pass that reads the 5-token prompt"));
    let later = trace(&["--prompt-tokens", "5", "--pass", "4"])?;
    assert!(later.status.success(), "{}", text(&later.stderr));
    assert!(text(&later.stdout).contains("one-token pass 4 (the token at position 8)"));
    Ok(())
}

#[test]
fn json_holds_every_timed_call_in_order() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("trace-json")?;
    let file = sandbox.0.join("trace.json");
    let path = file.display().to_string();
    let out = trace(&["--prompt-tokens", "6", "--json", &path])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("wrote "));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert_eq!(json["format"], 1);
    assert_eq!(json["pass"], "1");
    assert_eq!(json["prompt_tokens"], 6);
    let entries = json["entries"].as_array().ok_or("entries is not a list")?;
    assert_eq!(entries[0]["stage"], "embedding");
    assert!(entries[0]["layer"].is_null());
    assert_eq!(entries[entries.len() - 1]["stage"], "lm head");
    let layers = entries.iter().filter_map(|e| e["layer"].as_u64()).max();
    let layers = layers.ok_or("no layer entries")? + 1;
    assert_eq!(entries.len() as u64, layers * 11 + 3);
    let wall = json["wall_ms"].as_f64().ok_or("no wall time")?;
    let timed: f64 = entries.iter().filter_map(|e| e["ms"].as_f64()).sum();
    assert!(
        timed <= wall,
        "stages nest inside the pass: {timed} ms of {wall} ms"
    );
    assert!(
        timed >= wall / 2.0,
        "most of the pass is inside timed stages: {timed} ms of {wall} ms"
    );
    Ok(())
}

#[test]
fn unusable_options_fail_before_loading_anything() -> io::Result<()> {
    let cases: [(&[&str], i32, &str); 5] = [
        (
            &["--prompt-tokens", "100", "--pass", "100"],
            1,
            "is above what this model supports",
        ),
        (
            &["--prompt-tokens", "0"],
            1,
            "error: trace needs --prompt-tokens above 0",
        ),
        (
            &["--prompt-tokens", "8", "--pass", "5", "--ctx", "10"],
            1,
            "error: --ctx 10 is too small",
        ),
        (&["--pass", "0"], 2, "decode passes count from 1"),
        (&["--pass", "soon"], 2, "not `prefill` or a whole number"),
    ];
    for (args, code, message) in cases {
        let out = trace(args)?;
        assert_eq!(
            out.status.code(),
            Some(code),
            "{args:?}: {}",
            text(&out.stderr)
        );
        assert!(
            text(&out.stderr).contains(message),
            "{args:?}: {}",
            text(&out.stderr)
        );
        assert!(out.stdout.is_empty(), "{args:?}");
        assert!(!text(&out.stderr).contains("loading"), "{args:?}");
    }
    Ok(())
}

#[test]
fn an_unwritable_json_path_is_reported_before_the_trace_runs() -> io::Result<()> {
    let sandbox = Sandbox::new("trace-badjson")?;
    let path = sandbox
        .0
        .join("missing-folder")
        .join("t.json")
        .display()
        .to_string();
    let out = trace(&["--json", &path])?;
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(err.contains("error: cannot write "), "{err}");
    assert!(err.contains("hint: check that the folder exists"), "{err}");
    assert!(out.stdout.is_empty());
    assert!(!err.contains("loading"), "{err}");
    Ok(())
}

#[test]
fn profile_in_the_repl_needs_a_prompt_then_shows_stage_tables() -> io::Result<()> {
    let model = fixtures().join("tiny-qwen3").display().to_string();
    let args = ["--backend", "cpu", "--temp", "0", "--max-tokens", "3"];
    let mut all = vec![model.as_str()];
    all.extend_from_slice(&args);
    let out = marin_run(&all, "/profile\nhello world\n/profile\n/quit\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let shown = text(&out.stdout);
    assert!(shown.contains("nothing generated yet"), "{shown}");
    for part in ["where the time goes", "prefill:", "decode:", "mlp"] {
        assert!(shown.contains(part), "missing {part:?} in\n{shown}");
    }
    Ok(())
}
