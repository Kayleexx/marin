//! The quantization options and `marin-run quality`, on the tiny test model (INT4 weights need a GPU, so what
//! is checked here is what happens before any weights load, and the scoring on the CPU).

mod common;

use std::io;
use std::process::{Command, Output};

use common::{Sandbox, fixtures, text};

fn run(args: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()
}

fn tiny() -> String {
    fixtures().join("tiny-chat").display().to_string()
}

fn sample_text(sandbox: &Sandbox, name: &str, sentence: &str) -> io::Result<String> {
    let path = sandbox.0.join(name);
    std::fs::write(&path, sentence.repeat(60))?;
    Ok(path.display().to_string())
}

#[test]
fn bad_quantization_options_are_refused_before_anything_loads() -> io::Result<()> {
    let model = tiny();
    let cpu = run(&["bench", &model, "--backend", "cpu", "--quantize", "int4"])?;
    assert_eq!(cpu.status.code(), Some(1));
    assert!(
        text(&cpu.stderr).contains("int4 needs the GPU"),
        "{}",
        text(&cpu.stderr)
    );

    let group = run(&["bench", &model, "--quantize", "int4", "--quant-group", "48"])?;
    assert_eq!(group.status.code(), Some(1));
    assert!(
        text(&group.stderr).contains("multiple of 32"),
        "{}",
        text(&group.stderr)
    );

    let alone = run(&["bench", &model, "--quant-group", "64"])?;
    assert_eq!(alone.status.code(), Some(2), "a usage error");
    assert!(
        text(&alone.stderr).contains("--quantize"),
        "{}",
        text(&alone.stderr)
    );

    let width = run(&["bench", &model, "--quantize", "int3"])?;
    assert_eq!(width.status.code(), Some(2));
    Ok(())
}

#[test]
fn inspect_shows_what_quantizing_does_to_the_weights() -> io::Result<()> {
    let model = tiny();
    let plain = text(&run(&["inspect", &model])?.stdout);
    let out = run(&[
        "inspect",
        &model,
        "--quantize",
        "int8",
        "--quant-group",
        "32",
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let quantized = text(&out.stdout);
    assert!(
        quantized.contains("int8 weights, groups of 32"),
        "{quantized}"
    );
    assert!(!plain.contains("quantized"), "{plain}");
    let too_wide = run(&[
        "inspect",
        &model,
        "--quantize",
        "int8",
        "--quant-group",
        "128",
    ])?;
    assert!(!too_wide.status.success());
    assert!(
        text(&too_wide.stderr).contains("not a multiple"),
        "{}",
        text(&too_wide.stderr)
    );
    Ok(())
}

#[test]
fn a_saved_scoring_run_agrees_with_itself_and_refuses_other_text()
-> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("quality")?;
    let model = tiny();
    let words = sample_text(
        &sandbox,
        "a.txt",
        "the quick brown fox jumps over the lazy dog. ",
    )?;
    let other = sample_text(
        &sandbox,
        "b.txt",
        "pack my box with five dozen liquor jugs. ",
    )?;
    let saved = sandbox.0.join("scores.json").display().to_string();
    let common = [
        "quality",
        model.as_str(),
        "--backend",
        "cpu",
        "--tokens",
        "200",
    ];

    let first = run(&[&common[..], &["--text", &words, "--save", &saved]].concat())?;
    assert!(first.status.success(), "{}", text(&first.stderr));
    let report = text(&first.stdout);
    for line in ["loss ", "right guess", "the first 64 are the prompt"] {
        assert!(report.contains(line), "missing {line:?} in\n{report}");
    }
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&saved)?)?;
    assert_eq!(json["format"], 1);
    assert_eq!(json["quantization"], "none");
    assert_eq!(
        json["top1"].as_array().map(Vec::len),
        json["target_logprob"].as_array().map(Vec::len)
    );

    let again = run(&[&common[..], &["--text", &words, "--against", &saved]].concat())?;
    assert!(again.status.success(), "{}", text(&again.stderr));
    let report = text(&again.stdout);
    assert!(
        report.contains("100.0% of positions pick the same first token"),
        "{report}"
    );
    assert!(report.contains("loss change   +0.0000"), "{report}");

    let different = run(&[&common[..], &["--text", &other, "--against", &saved]].concat())?;
    assert!(!different.status.success());
    assert!(
        text(&different.stderr).contains("did not score the same text"),
        "{}",
        text(&different.stderr)
    );
    Ok(())
}

#[test]
fn scoring_refuses_missing_short_or_unreadable_input() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("quality-bad")?;
    let model = tiny();
    let short = sandbox.0.join("short.txt");
    std::fs::write(&short, "too short")?;
    let short = short.display().to_string();
    let args = |extra: &[&str]| {
        run(&[&["quality", model.as_str(), "--backend", "cpu"][..], extra].concat())
    };
    let missing = args(&["--text", "/definitely/not/here.txt"])?;
    assert!(
        text(&missing.stderr).contains("cannot read"),
        "{}",
        text(&missing.stderr)
    );
    let small = args(&["--text", &short, "--tokens", "10"])?;
    assert!(
        text(&small.stderr).contains("--tokens must be at least"),
        "{}",
        text(&small.stderr)
    );
    let few = args(&["--text", &short, "--tokens", "100"])?;
    assert!(text(&few.stderr).contains("only"), "{}", text(&few.stderr));
    let unreadable = sandbox.0.join("scores.json");
    std::fs::write(&unreadable, "not json")?;
    let bad = args(&[
        "--text",
        &short,
        "--against",
        &unreadable.display().to_string(),
    ])?;
    assert!(
        text(&bad.stderr).contains("cannot use result file"),
        "{}",
        text(&bad.stderr)
    );
    let folder = args(&["--text", &short, "--save", &sandbox.0.display().to_string()])?;
    assert!(
        text(&folder.stderr).contains("it is a folder"),
        "{}",
        text(&folder.stderr)
    );
    Ok(())
}

#[test]
fn int8_cpu_selection_and_auto_fallback_are_recorded() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("int8-cpu")?;
    for backend in ["cpu", "auto"] {
        let path = sandbox.0.join(format!("{backend}.json"));
        let out = Command::new(env!("CARGO_BIN_EXE_marin-run"))
            .env("CUDA_VISIBLE_DEVICES", "-1")
            .args([
                "bench",
                &tiny(),
                "--backend",
                backend,
                "--quantize",
                "int8",
                "--quant-group",
                "32",
                "--threads",
                "8",
                "--prompt-tokens",
                "7",
                "--tokens",
                "4",
                "--runs",
                "1",
                "--warmup",
                "0",
                "--json",
            ])
            .arg(&path)
            .output()?;
        assert!(out.status.success(), "{}", text(&out.stderr));
        let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_eq!(record["quantization"], "int8 weights, groups of 32");
        assert_eq!(record["hardware"]["backend"], "cpu");
        assert!(text(&out.stdout).contains("int8 weights, groups of 32, f32 math"));
    }
    Ok(())
}

#[test]
fn int8_repl_names_its_weight_format() -> io::Result<()> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args([
            &tiny(),
            "--backend",
            "cpu",
            "--quantize",
            "int8",
            "--quant-group",
            "32",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        use std::io::Write;
        input.write_all(b"/backend\n/quit\n")?;
    }
    let out = child.wait_with_output()?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("int8 weights, groups of 32"));
    assert!(text(&out.stdout).contains("int8 weights, groups of 32"));
    Ok(())
}
