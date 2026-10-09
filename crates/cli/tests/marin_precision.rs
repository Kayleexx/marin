//! `--kv-type` and `--quantize` on the `marin` command line: what they accept, what the CPU refuses, and that a GPU
//! run with them works and keeps snapshots apart. The inspect-panel tests in `marin_pty_kv.rs` show the choice
//! reaching the loaded model. Without a GPU the GPU tests skip unless `MARIN_REQUIRE_CUDA` is set.

mod common;

use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Output, Stdio};

use common::{Sandbox, fixtures, text};

fn cpu_command(sandbox: &Sandbox, models: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
    command
        .env("MARIN_HOME", &sandbox.0)
        .env("MARIN_MODELS", models)
        .env("NO_COLOR", "1")
        .args(["--backend", "cpu"]);
    command
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
fn kv_type_defaults_to_f32_and_rejects_unknown_values() {
    let sandbox = Sandbox::new("kvtype").unwrap();
    let mut command = cpu_command(&sandbox, &fixtures());
    command.args(["run", "hi", "--kv-type", "bf16"]);
    let out = run(command, "").unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = text(&out.stderr);
    assert!(err.contains("f32") && err.contains("f16"), "{err}");

    let mut f32 = cpu_command(&sandbox, &fixtures());
    f32.args(["run", "the cat", "--temp", "0", "--max-tokens", "4"]);
    let plain = run(f32, "").unwrap();
    let mut explicit = cpu_command(&sandbox, &fixtures());
    explicit.args(["run", "the cat", "--temp", "0", "--max-tokens", "4"]);
    explicit.args(["--kv-type", "f32"]);
    let named = run(explicit, "").unwrap();
    assert!(plain.status.success() && named.status.success());
    assert_eq!(plain.stdout, named.stdout);
}

#[test]
fn an_f16_cache_needs_the_gpu() {
    let sandbox = Sandbox::new("kvtype-cpu").unwrap();
    let mut command = cpu_command(&sandbox, &fixtures());
    command.args(["run", "hi", "--kv-type", "f16"]);
    let out = run(command, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("--kv-type"),
        "{}",
        text(&out.stderr)
    );
}

/// Runs `marin run` on the GPU; `None` when there is no usable GPU.
fn gpu_run(sandbox: &Sandbox, extra: &[&str]) -> Option<Output> {
    let out = Command::new(env!("CARGO_BIN_EXE_marin"))
        .env("MARIN_HOME", &sandbox.0)
        .env("MARIN_MODELS", fixtures())
        .env("NO_COLOR", "1")
        .args(["--backend", "cuda", "run", "the cat sat"])
        .args(["--temp", "0", "--max-tokens", "8"])
        .args(extra)
        .output()
        .ok()?;
    if !out.status.success() {
        assert!(
            std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
            "MARIN_REQUIRE_CUDA is set but the GPU run failed: {}",
            text(&out.stderr)
        );
        return None;
    }
    Some(out)
}

#[test]
fn an_f16_cache_runs_on_the_gpu_and_a_snapshot_is_only_used_by_its_own_type() {
    let sandbox = Sandbox::new("kvtype-gpu").unwrap();
    let file = sandbox.0.join("start.snap");
    let path = file.display().to_string();
    let Some(plain) = gpu_run(&sandbox, &["--kv-type", "f16"]) else {
        eprintln!("skipping: no usable GPU");
        return;
    };
    assert!(!plain.stdout.is_empty());
    let saved = gpu_run(&sandbox, &["--kv-type", "f16", "--snapshot", &path]).unwrap();
    assert_eq!(saved.stdout, plain.stdout);
    let reused = gpu_run(&sandbox, &["--kv-type", "f16", "--snapshot", &path]).unwrap();
    assert_eq!(reused.stdout, plain.stdout);
    let by_default = gpu_run(&sandbox, &[]).unwrap();
    assert_eq!(by_default.stdout, plain.stdout);
    let other = gpu_run(&sandbox, &["--kv-type", "f32", "--snapshot", &path]).unwrap();
    let f32_plain = gpu_run(&sandbox, &["--kv-type", "f32"]).unwrap();
    assert_eq!(other.stdout, f32_plain.stdout);
}

#[test]
fn quantize_takes_none_or_int8_and_needs_the_gpu() {
    let sandbox = Sandbox::new("quantize").unwrap();
    let mut bad = cpu_command(&sandbox, &fixtures());
    bad.args(["run", "hi", "--quantize", "int4"]);
    let out = run(bad, "").unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = text(&out.stderr);
    assert!(err.contains("none") && err.contains("int8"), "{err}");

    let mut group = cpu_command(&sandbox, &fixtures());
    group.args(["run", "hi", "--quant-group", "32"]);
    let out = run(group, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("--quant-group"),
        "{}",
        text(&out.stderr)
    );

    let mut cpu = cpu_command(&sandbox, &fixtures());
    cpu.args(["run", "hi", "--quantize", "int8", "--quant-group", "32"]);
    let out = run(cpu, "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("--quantize"),
        "{}",
        text(&out.stderr)
    );

    let mut plain = cpu_command(&sandbox, &fixtures());
    plain.args(["run", "the cat", "--temp", "0", "--max-tokens", "4"]);
    let mut none = cpu_command(&sandbox, &fixtures());
    none.args(["run", "the cat", "--temp", "0", "--max-tokens", "4"]);
    none.args(["--quantize", "none"]);
    let (plain, none) = (run(plain, "").unwrap(), run(none, "").unwrap());
    assert!(plain.status.success() && none.status.success());
    assert_eq!(plain.stdout, none.stdout);
}

#[test]
fn an_int8_chat_runs_on_the_gpu_and_a_snapshot_is_only_used_by_its_own_weights() {
    let sandbox = Sandbox::new("int8-gpu").unwrap();
    let file = sandbox.0.join("start.snap");
    let path = file.display().to_string();
    let int8 = ["--quantize", "int8", "--quant-group", "32"];
    let snapshot = [
        "--quantize",
        "int8",
        "--quant-group",
        "32",
        "--snapshot",
        &path,
    ];
    let Some(plain) = gpu_run(&sandbox, &int8) else {
        eprintln!("skipping: no usable GPU");
        return;
    };
    assert!(!plain.stdout.is_empty());
    let saved = gpu_run(&sandbox, &snapshot).unwrap();
    let reused = gpu_run(&sandbox, &snapshot).unwrap();
    assert_eq!(saved.stdout, plain.stdout);
    assert_eq!(reused.stdout, plain.stdout);
    let bf16_plain = gpu_run(&sandbox, &[]).unwrap();
    let bf16_after = gpu_run(&sandbox, &["--snapshot", &path]).unwrap();
    assert_eq!(bf16_after.stdout, bf16_plain.stdout);
}
