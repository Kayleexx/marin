mod common;

use std::process::Command;

use common::{Sandbox, fixtures, text};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn call(args: &[&str]) -> Result<std::process::Output> {
    Ok(Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()?)
}

#[test]
fn plan_and_execution_agree_on_explicit_cpu_settings() -> Result {
    let model = fixtures().join("tiny-chat").display().to_string();
    let flags = ["--backend", "cpu", "--ctx", "64", "--threads", "2"];
    let mut plan_args = vec!["plan", model.as_str()];
    plan_args.extend(flags);
    let planned = call(&plan_args)?;
    assert!(planned.status.success(), "{}", text(&planned.stderr));
    let report = text(&planned.stdout);
    for expected in [
        "artifact    BF16 source",
        "backend     cpu",
        "threads     2 CPU prefill",
        "context     64 tokens",
        "CPU-only need",
    ] {
        assert!(report.contains(expected), "{report}");
    }
    let chat = Command::new(env!("CARGO_BIN_EXE_marin"))
        .args([
            "--model",
            &model,
            "--backend",
            "cpu",
            "--ctx",
            "64",
            "--threads",
            "2",
            "plan",
        ])
        .env("NO_COLOR", "1")
        .output()?;
    assert!(chat.status.success(), "{}", text(&chat.stderr));
    assert!(text(&chat.stdout).contains("threads     2 CPU prefill"));
    assert!(text(&chat.stdout).contains("context     64 tokens"));
    let mut run_args = vec![model.as_str()];
    run_args.extend(flags);
    run_args.extend([
        "--max-tokens",
        "2",
        "--temp",
        "0",
        "-p",
        "Hello",
        "--metrics",
    ]);
    let actual = call(&run_args)?;
    assert!(actual.status.success(), "{}", text(&actual.stderr));
    assert!(text(&actual.stderr).contains("generated       2 tokens"));
    let q8 = call(&[
        "plan",
        &model,
        "--backend",
        "cpu",
        "--quantize",
        "int8",
        "--quant-group",
        "32",
    ])?;
    assert!(q8.status.success(), "{}", text(&q8.stderr));
    assert!(
        text(&q8.stdout).contains("BF16 source, int8"),
        "{}",
        text(&q8.stdout)
    );
    Ok(())
}

#[test]
fn packed_artifact_requires_matching_explicit_opt_in() -> Result {
    let sandbox = Sandbox::new("plan-packed")?;
    let source = fixtures().join("tiny-qwen35-int8");
    let packed = sandbox.0.join("packed");
    model::export_int8(&loader::ModelDir::open(&source)?, &packed, 32)?;
    let path = packed.display().to_string();
    let plain = call(&["plan", &path, "--backend", "cpu"])?;
    assert!(!plain.status.success());
    assert!(text(&plain.stderr).contains("requires matching opt-in INT8"));
    let q8 = call(&[
        "plan",
        &path,
        "--backend",
        "cpu",
        "--quantize",
        "int8",
        "--quant-group",
        "32",
    ])?;
    assert!(q8.status.success(), "{}", text(&q8.stderr));
    assert!(
        text(&q8.stdout).contains("artifact    packed INT8, group 32"),
        "{}",
        text(&q8.stdout)
    );
    Ok(())
}
