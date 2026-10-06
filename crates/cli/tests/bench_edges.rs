mod common;

use std::io;
use std::process::Command;

use common::{Sandbox, fixtures, text};
use serde_json::Value;

fn wait_for(
    child: &mut std::process::Child,
    seconds: u64,
) -> io::Result<Option<std::process::ExitStatus>> {
    let started = std::time::Instant::now();
    let limit = std::time::Duration::from_secs(seconds);
    while started.elapsed() < limit {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Ok(None)
}

#[test]
fn ctrl_c_stops_a_long_bench_cleanly_and_leaves_no_file_behind() -> io::Result<()> {
    use std::os::unix::process::ExitStatusExt;

    let sandbox = Sandbox::new("bench-ctrl-c")?;
    let json = sandbox.0.join("run.json");
    let model = fixtures().join("tiny-qwen3");
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(["bench"])
        .arg(&model)
        .args([
            "--backend",
            "cpu",
            "--prompt-tokens",
            "8",
            "--tokens",
            "8",
            "--runs",
            "10000000",
        ])
        .arg("--json")
        .arg(&json)
        .env("NO_COLOR", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    std::thread::sleep(std::time::Duration::from_millis(1500));
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()?;
    let Some(status) = wait_for(&mut child, 20)? else {
        child.kill()?;
        panic!("bench did not stop after Ctrl-C");
    };
    let out = child.wait_with_output()?;
    assert_eq!(
        status.code(),
        Some(130),
        "signal {:?}: {}",
        status.signal(),
        text(&out.stderr)
    );
    assert!(
        text(&out.stderr).contains("cancelled"),
        "{}",
        text(&out.stderr)
    );
    assert!(out.stdout.is_empty());
    assert_eq!(
        std::fs::read_dir(&sandbox.0)?.count(),
        0,
        "no result or temporary file is left"
    );
    Ok(())
}

#[test]
fn a_closed_output_pipe_ends_quietly() -> io::Result<()> {
    let model = fixtures().join("tiny-qwen3");
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(["trace"])
        .arg(&model)
        .args(["--backend", "cpu", "--prompt-tokens", "8"])
        .env("NO_COLOR", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    drop(child.stdout.take());
    let out = child.wait_with_output()?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        !text(&out.stderr).contains("panicked"),
        "{}",
        text(&out.stderr)
    );
    Ok(())
}

/// Ids of every running process named exactly `nvidia-smi`; empty if `pgrep` is not installed.
fn nvidia_smi_pids() -> Vec<String> {
    Command::new("pgrep")
        .args(["-x", "nvidia-smi"])
        .output()
        .map(|out| {
            text(&out.stdout)
                .split_whitespace()
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn gpu_runs_record_the_device_upload_and_sampled_readings() -> Result<(), Box<dyn std::error::Error>>
{
    let before = nvidia_smi_pids();
    let sandbox = Sandbox::new("bench-gpu")?;
    let file = sandbox.0.join("gpu.json");
    let path = file.display().to_string();
    let model = fixtures().join("tiny-qwen3");
    let out = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .arg("bench")
        .arg(model)
        .args([
            "--backend",
            "cuda",
            "--prompt-tokens",
            "32",
            "--tokens",
            "32",
            "--runs",
            "200",
        ])
        .args(["--sample-gpu", "--json", &path])
        .env("NO_COLOR", "1")
        .output()?;
    if !out.status.success() {
        assert!(
            std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
            "MARIN_REQUIRE_CUDA is set but the GPU run failed: {}",
            text(&out.stderr)
        );
        eprintln!("skipping: no usable GPU");
        return Ok(());
    }
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert!(
        json["hardware"]["gpu"]["name"]
            .as_str()
            .is_some_and(|n| !n.is_empty())
    );
    assert!(
        json["hardware"]["gpu"]["vram_total_bytes"]
            .as_u64()
            .is_some_and(|v| v > 0)
    );
    assert!(
        json["load"]["gpu_upload"]["bytes"]
            .as_u64()
            .is_some_and(|v| v > 0)
    );
    let weights = json["load"]["weights_bytes"].as_u64().ok_or("no weights")?;
    assert!(
        json["results"]["gpu_device_used_peak_bytes"]
            .as_u64()
            .is_some_and(|v| v > weights)
    );
    assert!(
        json["results"]["gpu"]["decode"]["passes"]
            .as_u64()
            .is_some_and(|p| p == 31)
    );
    let sampling = &json["results"]["gpu_sampling"];
    match sampling["status"].as_str() {
        Some("sampled") => {
            assert!(sampling["samples"].as_u64().is_some_and(|n| n >= 1));
            assert!(
                sampling["utilization_percent"]["max"]
                    .as_f64()
                    .is_some_and(|v| (0.0..=100.0).contains(&v))
            );
        }
        Some("unavailable") => assert!(sampling["reason"].as_str().is_some_and(|r| !r.is_empty())),
        other => panic!("unexpected sampling status {other:?}"),
    }
    std::thread::sleep(std::time::Duration::from_millis(300));
    let after = nvidia_smi_pids();
    assert!(
        after.iter().all(|pid| before.contains(pid)),
        "nvidia-smi was left running: {after:?}"
    );
    Ok(())
}

#[test]
fn a_terminal_sees_a_progress_line_that_is_erased_before_the_report() -> io::Result<()> {
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("skipping: `script` is not installed");
        return Ok(());
    }
    let model = fixtures().join("tiny-qwen3");
    let line = format!(
        "{} bench {} --backend cpu --prompt-tokens 8 --tokens 4 --runs 30",
        env!("CARGO_BIN_EXE_marin-run"),
        model.display()
    );
    let out = Command::new("script")
        .args(["-qec", &line, "/dev/null"])
        .env("NO_COLOR", "1")
        .env("TERM", "xterm")
        .env("COLUMNS", "100")
        .env("LANG", "C.UTF-8")
        .output()?;
    let screen = text(&out.stdout);
    assert!(out.status.success(), "{screen}");
    let first = screen.find("warmup 1/1").unwrap_or(usize::MAX);
    let last = screen.find("run 30/30").unwrap_or(usize::MAX);
    let report = screen
        .find("results (median of runs)")
        .unwrap_or(usize::MAX);
    assert!(
        first < last && last < report,
        "progress comes before the report:\n{screen:?}"
    );
    let erased = screen.get(last..report).unwrap_or_default();
    assert!(
        erased.contains('\r'),
        "the line is erased with a carriage return: {erased:?}"
    );
    assert!(!screen.contains("NaN"));
    Ok(())
}
