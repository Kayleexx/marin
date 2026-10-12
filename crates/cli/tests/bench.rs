mod bench_support;
mod common;

use std::io;

use bench_support::bench;
use common::{Sandbox, text};
use serde_json::Value;

#[test]
fn bench_prints_report_and_writes_json() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("bench-json")?;
    let file = sandbox.0.join("run.json");
    let path = file.display().to_string();
    let out = bench(&[
        "--prompt-tokens",
        "8",
        "--tokens",
        "4",
        "--runs",
        "2",
        "--profile",
        "--json",
        &path,
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let report = text(&out.stdout);
    for section in ["load:", "results (median of runs):", "where the time goes"] {
        assert!(report.contains(section), "missing {section:?} in\n{report}");
    }
    assert!(text(&out.stderr).contains("wrote "));

    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert_eq!(json["format"], 2);
    assert_eq!(json["prompt_tokens"], 8);
    assert_eq!(json["output_tokens"], 4);
    assert_eq!(json["measured_runs"], 2);
    assert_eq!(json["batch_size"], 1);
    assert!(
        json["results"]["first_token_ms"]["median"]
            .as_f64()
            .is_some_and(|v| v > 0.0)
    );
    assert!(json["load"]["gpu_device_growth_bytes"].is_null());
    assert!(json["results"]["gpu"].is_null());
    assert_eq!(json["profile"]["decode"]["tokens"], 3);
    assert!(
        json["results"]["weight_bandwidth_gb_per_s"]["median"]
            .as_f64()
            .is_some_and(|v| v > 0.0)
    );
    assert!(
        json["results"]["cpu_busy_cores"]
            .as_f64()
            .is_some_and(|v| v > 0.0)
    );
    assert!(json["results"]["gpu_device_used_peak_bytes"].is_null());
    assert!(json["load"]["gpu_upload"].is_null());
    assert!(json["load"]["disk_read_bytes"].is_u64());
    assert_eq!(json["results"]["gpu_sampling"]["status"], "off");
    assert!(
        !report.contains("gpu sampling"),
        "sampling was not asked for"
    );
    Ok(())
}

#[test]
fn sampling_the_gpu_on_a_cpu_run_says_why_it_cannot() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("bench-sample")?;
    let file = sandbox.0.join("run.json");
    let path = file.display().to_string();
    let out = bench(&[
        "--prompt-tokens",
        "4",
        "--tokens",
        "3",
        "--runs",
        "1",
        "--sample-gpu",
        "--json",
        &path,
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("gpu sampling: n/a (this run uses the CPU)"));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert_eq!(json["results"]["gpu_sampling"]["status"], "unavailable");
    assert!(
        json["results"]["gpu_sampling"]["reason"]
            .as_str()
            .is_some_and(|r| r.contains("CPU"))
    );
    Ok(())
}

#[test]
fn bench_rejects_unusable_options_with_a_clear_error() -> io::Result<()> {
    for args in [["--runs", "0"], ["--tokens", "1"], ["--prompt-tokens", "0"]] {
        let out = bench(&args)?;
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert!(text(&out.stderr).contains("error: bench needs"), "{args:?}");
        assert!(out.stdout.is_empty());
    }
    Ok(())
}

#[test]
fn bench_checks_the_json_path_before_measuring() -> io::Result<()> {
    let sandbox = Sandbox::new("bench-badjson")?;
    let path = sandbox
        .0
        .join("missing")
        .join("run.json")
        .display()
        .to_string();
    let out = bench(&["--json", &path])?;
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(err.contains("error: cannot write "), "{err}");
    assert!(err.contains("hint: check that the folder exists"), "{err}");
    assert!(out.stdout.is_empty());
    assert!(!err.contains("loading"), "{err}");
    Ok(())
}

#[test]
fn the_record_keeps_every_run_and_says_what_it_was_measured_on()
-> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("bench-record")?;
    let path = sandbox.0.join("r.json").display().to_string();
    let out = bench(&[
        "--prompt-tokens",
        "8",
        "--tokens",
        "6",
        "--runs",
        "3",
        "--json",
        &path,
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let runtime = json["runtime"].as_str().ok_or("no runtime")?;
    assert!(
        runtime.starts_with("marin 0.") && runtime.ends_with(')'),
        "{runtime}"
    );
    assert_eq!(json["sampler"]["temperature"], 0.0);
    assert_eq!(json["sampler"]["seed"], 0);
    assert_eq!(json["precision"], "bf16 weights, f32 math");
    assert_eq!(json["quantization"], "none");
    assert!(json["hardware"]["gpu"].is_null(), "a CPU run has no gpu");
    assert!(json["load"]["kv_held_pages"].is_null());
    assert!(
        json["load"]["kv_held_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
    let runs = json["runs"].as_array().ok_or("no runs")?;
    assert_eq!(runs.len(), 3);
    for run in runs {
        assert_eq!(run["generated"], 6);
        assert_eq!(run["token_gaps_ms"].as_array().map(Vec::len), Some(5));
        let total = run["total_ms"].as_f64().ok_or("no total")?;
        let parts = run["first_token_ms"].as_f64().ok_or("no ttft")?
            + run["decode_ms"].as_f64().ok_or("no decode")?;
        assert!((total - parts).abs() < 1e-6, "{total} vs {parts}");
    }
    Ok(())
}

fn keys(value: &Value) -> Vec<&str> {
    value
        .as_object()
        .map_or_else(Vec::new, |o| o.keys().map(String::as_str).collect())
}

/// Changing this list means the result format changed: bump `format` in bench/mod.rs and `compare`'s FORMAT.
#[test]
fn format_2_has_exactly_these_fields() -> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("bench-keys")?;
    let path = sandbox.0.join("k.json").display().to_string();
    let out = bench(&[
        "--prompt-tokens",
        "4",
        "--tokens",
        "3",
        "--runs",
        "1",
        "--profile",
        "--json",
        &path,
    ])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    assert_eq!(
        keys(&json),
        [
            "batch_size",
            "context_length",
            "format",
            "hardware",
            "load",
            "measured_runs",
            "model",
            "output_tokens",
            "precision",
            "profile",
            "prompt_tokens",
            "quantization",
            "results",
            "runs",
            "runtime",
            "sampler",
            "token_pause_ms",
            "unix_time",
            "warmup_runs",
        ]
    );
    assert_eq!(
        keys(&json["hardware"]),
        [
            "backend",
            "cpu",
            "gpu",
            "gpu_power_limit_watts",
            "logical_cores",
            "on_ac_power",
            "os",
            "ram_bytes"
        ]
    );
    assert_eq!(
        keys(&json["load"]),
        [
            "disk_read_bytes",
            "gpu_device_growth_bytes",
            "gpu_upload",
            "kv_held_bytes",
            "kv_held_pages",
            "kv_reserved_bytes",
            "rss_growth_bytes",
            "seconds",
            "weights_bytes",
        ]
    );
    assert_eq!(
        keys(&json["results"]),
        [
            "cpu_busy_cores",
            "decode_tokens_per_s",
            "first_token_ms",
            "gpu",
            "gpu_device_used_peak_bytes",
            "gpu_energy_per_token_joules",
            "gpu_sampling",
            "peak_rss_bytes",
            "prefill_tokens_per_s",
            "sampling_ms_per_token",
            "token_latency_p50_ms",
            "token_latency_p95_ms",
            "weight_bandwidth_gb_per_s",
        ]
    );
    assert_eq!(
        keys(&json["runs"][0]),
        [
            "decode_ms",
            "first_token_ms",
            "generated",
            "prompt_forward_ms",
            "sampling_ms",
            "token_gaps_ms",
            "total_ms",
        ]
    );
    assert_eq!(keys(&json["profile"]), ["clock", "decode", "prefill"]);
    Ok(())
}

#[test]
fn piped_bench_prints_no_progress_line() -> io::Result<()> {
    let out = bench(&["--prompt-tokens", "4", "--tokens", "3", "--runs", "2"])?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let err = text(&out.stderr);
    assert!(!err.contains('\r'), "{err:?}");
    assert!(
        !err.contains("warmup") && !err.contains("run 1/"),
        "{err:?}"
    );
    Ok(())
}

/// The slowest first token or gap of the runs in a saved result, in milliseconds.
fn slowest_timing(json: &Value) -> Result<f64, Box<dyn std::error::Error>> {
    let mut slowest = 0.0_f64;
    for run in json["runs"].as_array().ok_or("no runs")? {
        let first = run["first_token_ms"].as_f64().ok_or("no first token")?;
        slowest = slowest.max(first);
        for gap in run["token_gaps_ms"].as_array().ok_or("no gaps")? {
            slowest = slowest.max(gap.as_f64().ok_or("gap is not a number")?);
        }
    }
    Ok(slowest)
}

#[test]
fn a_pause_after_each_token_is_recorded_and_kept_out_of_the_engine_timings()
-> Result<(), Box<dyn std::error::Error>> {
    let sandbox = Sandbox::new("bench-pause")?;
    let args = ["--prompt-tokens", "8", "--tokens", "6", "--runs", "2"];
    let baseline_file = sandbox.0.join("baseline.json");
    let baseline_path = baseline_file.display().to_string();
    let out = bench(&[&args[..], &["--json", &baseline_path]].concat())?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let baseline: Value = serde_json::from_str(&std::fs::read_to_string(&baseline_file)?)?;

    // The pause belongs to the consumer, so it must stay out of every timing. How long an ordinary first token or
    // gap takes depends on the machine and its load, so this run measures that first (the slowest one without a
    // pause) and pauses for ten times as long, capped at the largest pause allowed. A leaked pause then adds ten
    // times the machine's own timing to a gap (with the exclusion broken on purpose and a 100 ms pause the gaps
    // read about 103 ms), and half the pause is the limit: five times the machine's own timing. Normal timings
    // are far below it and a leak is far above it, whatever the machine is doing.
    let pause = (slowest_timing(&baseline)? * 10.0)
        .ceil()
        .clamp(100.0, 1000.0);
    let limit = pause / 2.0;
    let file = sandbox.0.join("pause.json");
    let path = file.display().to_string();
    let out = bench(
        &[
            &args[..],
            &["--token-pause-ms", &pause.to_string(), "--json", &path],
        ]
        .concat(),
    )?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        text(&out.stdout).contains(&format!("{pause:.1} ms pause after every token")),
        "{}",
        text(&out.stdout)
    );
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
    assert_eq!(json["token_pause_ms"], pause);
    for run in json["runs"].as_array().ok_or("no runs")? {
        let gaps = run["token_gaps_ms"].as_array().ok_or("no gaps")?;
        assert!(
            gaps.iter().all(|g| g.as_f64().is_some_and(|ms| ms < limit)),
            "the pause leaked into the gaps (limit {limit} ms): {gaps:?}"
        );
        assert!(
            run["first_token_ms"].as_f64().is_some_and(|ms| ms < limit),
            "the pause leaked into the first token (limit {limit} ms): {}",
            run["first_token_ms"]
        );
    }
    assert!(
        json["results"]["gpu_energy_per_token_joules"].is_null(),
        "a CPU run has no GPU energy"
    );
    assert!(json["hardware"]["gpu_power_limit_watts"].is_null());
    Ok(())
}

#[test]
fn pauses_outside_the_allowed_range_are_refused() -> std::io::Result<()> {
    for pause in ["-1", "1001", "nan", "inf"] {
        let out = bench(&[&format!("--token-pause-ms={pause}")])?;
        assert_eq!(out.status.code(), Some(1), "{pause}: {}", text(&out.stderr));
        assert!(
            text(&out.stderr).contains("error: --token-pause-ms must be between 0 and 1000"),
            "{pause}"
        );
        assert!(out.stdout.is_empty());
    }
    Ok(())
}
