mod bench_support;
mod common;

use std::io;
use std::process::{Command, Output};

use bench_support::bench;
use common::{Sandbox, text};

fn compare(old: &str, new: &str) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(["compare", old, new])
        .env("NO_COLOR", "1")
        .output()
}

fn saved(sandbox: &Sandbox, name: &str, prompt_tokens: &str) -> io::Result<String> {
    let path = sandbox.0.join(name).display().to_string();
    let out = bench(&[
        "--prompt-tokens",
        prompt_tokens,
        "--tokens",
        "6",
        "--runs",
        "3",
        "--json",
        &path,
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    Ok(path)
}

#[test]
fn compare_reads_two_results_of_the_same_setup() -> io::Result<()> {
    let sandbox = Sandbox::new("bench-compare")?;
    let (old, new) = (
        saved(&sandbox, "a.json", "8")?,
        saved(&sandbox, "b.json", "8")?,
    );
    let out = compare(&old, &new)?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let shown = text(&out.stdout);
    for part in [
        "compare:",
        "median of runs",
        "first token (TTFT)",
        "weight bandwidth (est.)",
        "reading this:",
    ] {
        assert!(shown.contains(part), "missing {part:?} in\n{shown}");
    }
    assert!(!shown.contains("NaN"));
    assert!(
        shown.contains("same build") || shown.contains("uncommitted code"),
        "{shown}"
    );
    assert!(out.stderr.is_empty());
    Ok(())
}

#[test]
fn compare_refuses_results_from_different_setups() -> io::Result<()> {
    let sandbox = Sandbox::new("bench-mismatch")?;
    let (old, new) = (
        saved(&sandbox, "a.json", "8")?,
        saved(&sandbox, "b.json", "9")?,
    );
    let out = compare(&old, &new)?;
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("error: these results were measured under different setups"),
        "{err}"
    );
    assert!(err.contains("prompt tokens: 8 vs 9"), "{err}");
    assert!(err.contains("hint: run both benchmarks again"), "{err}");
    assert!(out.stdout.is_empty());
    Ok(())
}

#[test]
fn compare_explains_files_it_cannot_use() -> io::Result<()> {
    let sandbox = Sandbox::new("bench-badfiles")?;
    let good = saved(&sandbox, "a.json", "8")?;
    let old_format = sandbox.0.join("old.json");
    std::fs::write(&old_format, r#"{"format": 1}"#)?;
    let junk = sandbox.0.join("junk.json");
    std::fs::write(&junk, "not json")?;
    let missing = sandbox.0.join("missing.json");
    let cases = [
        (
            old_format.display().to_string(),
            "it is format 1, compare needs format 2",
        ),
        (junk.display().to_string(), "it is not a result written by"),
        (missing.display().to_string(), "No such file"),
    ];
    for (bad, reason) in cases {
        for (first, second) in [(&bad, &good), (&good, &bad)] {
            let out = compare(first, second)?;
            assert_eq!(out.status.code(), Some(1), "{bad}");
            let err = text(&out.stderr);
            assert!(err.contains("error: cannot use result file"), "{err}");
            assert!(err.contains(reason), "{err}");
            assert!(err.contains("hint: pass files written by"), "{err}");
        }
    }
    Ok(())
}
