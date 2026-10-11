use super::super::record::StageRow;
use super::*;

fn phase(tokens: usize) -> Phase {
    Phase {
        tokens,
        wall_ms: 10.0,
        stages: Vec::new(),
        outside_stages_ms: 10.0,
    }
}

#[test]
fn sampling_lines_show_what_was_read_and_what_was_not() {
    assert_eq!(sampling_lines(&GpuSampling::Off), "");
    let unavailable = GpuSampling::Unavailable {
        reason: "nvidia-smi could not start".into(),
    };
    assert_eq!(
        sampling_lines(&unavailable),
        "\ngpu sampling: n/a (nvidia-smi could not start)\n"
    );
    let sampled = GpuSampling::Sampled {
        samples: 4,
        busy_samples: 3,
        utilization_percent: Some(Reading {
            mean: 91.0,
            max: 100.0,
        }),
        power_watts: None,
        clock_mhz: Some(Reading {
            mean: 1980.0,
            max: 2100.0,
        }),
        max_clock_mhz: Some(3000.0),
        temperature_c: Some(Reading {
            mean: 60.0,
            max: 63.0,
        }),
        power_cap_percent: Some(100.0),
        thermal_slowdown_percent: Some(0.0),
        hw_slowdown_percent: None,
    };
    let text = sampling_lines(&sampled);
    assert!(text.contains("4 samples"));
    assert!(text.contains("load        mean 91% · max 100%"), "{text}");
    assert!(text.contains("power       n/a"), "{text}");
    assert!(
        text.contains("clock       mean 1980 MHz (66% of the 3000 MHz maximum) · max 2100 MHz"),
        "{text}"
    );
    assert!(
        text.contains("limits      power cap active in 100% of 3 busy samples · thermal slowdown 0% · hardware slowdown n/a"),
        "{text}"
    );
}

#[test]
fn stage_rows_show_gb_per_s_in_decode_and_tflop_per_s_in_prefill() {
    let row = |gb, tflop| StageRow {
        stage: "mlp",
        total_ms: 10.0,
        calls: 28,
        share_of_wall_percent: 50.0,
        gb_per_s: gb,
        tflop_per_s: tflop,
    };
    let phase = |row| Phase {
        tokens: 1,
        wall_ms: 20.0,
        stages: vec![row],
        outside_stages_ms: 0.0,
    };
    assert!(phase_table("decode", &phase(row(Some(313.4), None)), 1, "token").contains("313 GB/s"));
    assert!(
        phase_table("prefill", &phase(row(None, Some(1.087))), 1, "pass").contains("1.09 TFLOP/s")
    );
    assert!(!phase_table("decode", &phase(row(None, None)), 1, "token").contains("/s\n"));
}

#[test]
fn the_logits_download_rate_is_shown_when_there_was_time_to_measure() {
    let work = |ms| GpuPhaseWork {
        passes: 1,
        launches_per_pass: 1.0,
        allocations_per_pass: 1.0,
        upload_bytes_per_pass: 0.0,
        upload_ms_per_pass: 0.0,
        download_bytes_per_pass: 607_744.0,
        download_ms_per_pass: ms,
        wait_for_gpu_ms_per_pass: 0.0,
    };
    assert!(gpu_line("x", &work(0.11)).contains("608 KB downloaded in 0.110 ms (5.5 GB/s)"));
    assert!(!gpu_line("x", &work(0.0)).contains("GB/s"));
}

#[test]
fn a_profile_without_one_token_passes_has_no_decode_table() {
    let profile = Profile {
        clock: "CPU wall clock",
        prefill: phase(8),
        decode: phase(0),
    };
    let text = profile_section(&profile);
    assert!(text.contains("prefill: 8 tokens"));
    assert!(!text.contains("decode:"));
    let both = Profile {
        decode: phase(3),
        ..profile
    };
    assert!(profile_section(&both).contains("decode: 3 tokens"));
}

fn stat(value: f64) -> Stat {
    Stat {
        min: value,
        median: value,
        max: value,
    }
}

fn sample_record() -> Record {
    use super::super::record::{Hardware, Load, Results, SamplerSettings};
    Record {
        format: 2,
        unix_time: 0,
        model: "m".into(),
        precision: "bf16",
        quantization: "none".to_string(),
        context_length: 10,
        batch_size: 1,
        prompt_tokens: 4,
        output_tokens: 3,
        warmup_runs: 1,
        measured_runs: 1,
        sampler: SamplerSettings {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        },
        token_pause_ms: 0.0,
        runtime: "marin".into(),
        hardware: Hardware {
            cpu: "cpu".into(),
            logical_cores: 4,
            ram_bytes: 1_000_000_000,
            backend: "cpu".into(),
            os: "linux".into(),
            gpu: None,
            on_ac_power: None,
            gpu_power_limit_watts: None,
        },
        load: Load {
            seconds: 0.1,
            weights_bytes: 1000,
            kv_reserved_bytes: 1000,
            kv_held_pages: None,
            kv_held_bytes: None,
            rss_growth_bytes: None,
            disk_read_bytes: None,
            gpu_upload: None,
            gpu_device_growth_bytes: None,
        },
        results: Results {
            first_token_ms: stat(1.0),
            prefill_tokens_per_s: stat(1.0),
            decode_tokens_per_s: stat(1.0),
            token_latency_p50_ms: stat(1.0),
            token_latency_p95_ms: stat(1.0),
            sampling_ms_per_token: stat(0.1),
            weight_bandwidth_gb_per_s: stat(1.0),
            cpu_busy_cores: None,
            gpu: None,
            gpu_device_used_peak_bytes: None,
            gpu_sampling: GpuSampling::Off,
            gpu_energy_per_token_joules: None,
            peak_rss_bytes: None,
        },
        runs: Vec::new(),
        profile: None,
    }
}

fn record(on_ac: Option<bool>, limit: Option<f64>, pause_ms: f64, energy: Option<f64>) -> String {
    let mut record = sample_record();
    record.hardware.on_ac_power = on_ac;
    record.hardware.gpu_power_limit_watts = limit;
    record.token_pause_ms = pause_ms;
    record.results.gpu_energy_per_token_joules = energy;
    text(&record)
}

#[test]
fn the_report_says_how_the_machine_was_powered_and_what_energy_a_token_cost() {
    let shown = record(Some(true), Some(40.0), 0.0, Some(0.7149));
    assert!(
        shown.contains("on AC power · gpu power limit 40 W\n"),
        "{shown}"
    );
    assert!(
        shown.contains("gpu energy per token") && shown.contains("0.71 J"),
        "{shown}"
    );
    let slow = record(Some(false), None, 2.5, None);
    assert!(
        slow.contains("on battery · 2.5 ms pause after every token"),
        "{slow}"
    );
    assert!(!slow.contains("gpu energy per token"));
    let unknown = record(None, None, 0.0, None);
    assert!(
        !unknown.contains("power limit")
            && !unknown.contains("AC power")
            && !unknown.contains("battery")
    );
}

#[test]
fn quantized_header_names_the_stored_weights() {
    let mut record = sample_record();
    record.quantization = "int8 weights, groups of 128".into();
    let header = text(&record);
    assert!(header.contains("int8 weights, groups of 128, f32 math"));
    assert!(!header.contains("bf16"));
}
