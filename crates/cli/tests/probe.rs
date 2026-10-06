use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn probe(args: &[&str]) -> io::Result<Output> {
    Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .arg("probe")
        .args(args)
        .env("NO_COLOR", "1")
        .output()
}

fn scratch(name: &str) -> io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("marin-probe-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[test]
fn unusable_durations_are_refused_before_touching_the_gpu() -> io::Result<()> {
    for seconds in ["0", "0.5", "61", "-3", "nan"] {
        let out = probe(&[&format!("--seconds={seconds}")])?;
        assert_eq!(
            out.status.code(),
            Some(1),
            "{seconds}: {}",
            text(&out.stderr)
        );
        assert!(
            text(&out.stderr).contains("error: --seconds must be between 1 and 60"),
            "{seconds}"
        );
        assert!(out.stdout.is_empty());
    }
    Ok(())
}

#[test]
fn an_unwritable_json_path_is_reported_first() -> io::Result<()> {
    let dir = scratch("badjson")?;
    let path = dir.join("missing").join("p.json");
    let out = probe(&["--json", &path.display().to_string()])?;
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("error: cannot write ") && err.contains("hint: check that the folder exists"),
        "{err}"
    );
    assert!(out.stdout.is_empty());
    let _ = std::fs::remove_dir_all(dir);
    Ok(())
}

#[test]
fn a_probe_run_reports_every_test_and_saves_the_numbers() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = scratch("run")?;
    let file = dir.join("probe.json");
    let out = probe(&["--seconds", "1", "--json", &file.display().to_string()])?;
    if !out.status.success() {
        assert!(
            std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
            "MARIN_REQUIRE_CUDA is set but the probe failed: {}",
            text(&out.stderr)
        );
        eprintln!("skipping: no usable GPU");
        return Ok(());
    }
    let shown = text(&out.stdout);
    for part in [
        "gpu probe:",
        "memory read",
        "fp32 math",
        "bf16 tensor cores",
        "gpu to the host",
        "reading this:",
    ] {
        assert!(shown.contains(part), "missing {part:?} in\n{shown}");
    }
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert_eq!(json["format"], 1);
    let rate = |key: &str| json[key]["sustained_per_s"].as_f64().unwrap_or(0.0);
    assert!(
        (20e9..5e12).contains(&rate("memory_read_bytes_per_s")),
        "{}",
        rate("memory_read_bytes_per_s")
    );
    assert!(
        (1e11..1e14).contains(&rate("fp32_flop_per_s")),
        "{}",
        rate("fp32_flop_per_s")
    );
    let copies = json["copies_to_host"].as_array().ok_or("no copies")?;
    assert_eq!(copies.len(), 3);
    assert_eq!(
        copies[1]["bytes"], 607_744,
        "the logits of a 151,936-token vocabulary, as 32-bit floats"
    );
    assert!(
        copies[2]["median_micros"].as_f64().unwrap_or(0.0)
            > copies[0]["median_micros"].as_f64().unwrap_or(f64::MAX)
    );
    assert!(json["gpu_sampling"]["status"].is_string());
    let _ = std::fs::remove_dir_all(dir);
    Ok(())
}
