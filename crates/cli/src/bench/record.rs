use std::fs;

use metrics::{on_ac_power, power_limit_watts};
use model::GpuInfo;
use serde::{Deserialize, Serialize};

/// Minimum, median and maximum of one measurement over the runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Stat {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

impl Stat {
    pub fn of(values: &[f64]) -> Option<Self> {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len().checked_div(2)?;
        let median = if sorted.len().checked_rem(2) == Some(0) {
            let lower = sorted.get(middle.checked_sub(1)?)?;
            (lower + sorted.get(middle)?) / 2.0
        } else {
            *sorted.get(middle)?
        };
        Some(Self {
            min: *sorted.first()?,
            median,
            max: *sorted.last()?,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct GpuHardware {
    pub name: String,
    pub compute_capability: String,
    pub vram_total_bytes: u64,
    pub driver_version: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Hardware {
    pub cpu: String,
    pub logical_cores: usize,
    pub ram_bytes: u64,
    pub backend: String,
    pub os: String,
    pub gpu: Option<GpuHardware>,
    /// Whether the machine was on mains power; `None` where that cannot be read.
    pub on_ac_power: Option<bool>,
    /// The power limit the driver enforces on the GPU. A GPU held to it runs at a speed set by energy, not time.
    pub gpu_power_limit_watts: Option<f64>,
}

/// The first dotted number in the NVIDIA kernel module's version line, e.g. `610.57.04`.
fn parse_driver_version(text: &str) -> Option<String> {
    text.lines()
        .next()?
        .split_whitespace()
        .find(|word| {
            word.contains('.')
                && word.starts_with(|c: char| c.is_ascii_digit())
                && word.chars().all(|c| c.is_ascii_digit() || c == '.')
        })
        .map(str::to_string)
}

fn gpu_hardware(info: GpuInfo) -> GpuHardware {
    let driver = fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| parse_driver_version(&text));
    GpuHardware {
        name: info.name,
        compute_capability: format!(
            "{}.{}",
            info.compute_capability.0, info.compute_capability.1
        ),
        vram_total_bytes: u64::try_from(info.total_bytes).unwrap_or(u64::MAX),
        driver_version: driver,
    }
}

fn proc_value(file: &str, key: &str) -> Option<String> {
    let text = fs::read_to_string(file).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    Some(line.split_once(':')?.1.trim().to_string())
}

pub fn hardware(backend: String, gpu: Option<GpuInfo>) -> Hardware {
    let cores = fs::read_to_string("/proc/cpuinfo")
        .map(|t| t.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(0);
    let ram_kib: u64 = proc_value("/proc/meminfo", "MemTotal")
        .and_then(|v| v.split_whitespace().next().and_then(|n| n.parse().ok()))
        .unwrap_or(0);
    let kernel = fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default();
    Hardware {
        cpu: proc_value("/proc/cpuinfo", "model name").unwrap_or_else(|| "unknown".into()),
        logical_cores: cores,
        ram_bytes: ram_kib.saturating_mul(1024),
        backend,
        os: format!("{} {}", std::env::consts::OS, kernel.trim()),
        on_ac_power: on_ac_power(),
        gpu_power_limit_watts: gpu.is_some().then(power_limit_watts).flatten(),
        gpu: gpu.map(gpu_hardware),
    }
}

/// A copy of `bytes` that took `seconds`, as the host saw it (so it includes waiting on the driver).
#[derive(Debug, Serialize)]
pub struct Transfer {
    pub bytes: u64,
    pub seconds: f64,
    pub gb_per_s: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Load {
    pub seconds: f64,
    pub weights_bytes: u64,
    pub kv_reserved_bytes: u64,
    /// Physical GPU pages; CPU storage has no page table.
    pub kv_held_pages: Option<u64>,
    /// Actual KV allocation after the runs on either backend.
    pub kv_held_bytes: Option<u64>,
    pub rss_growth_bytes: Option<u64>,
    /// Bytes fetched from storage during the load; files already in the page cache count as 0.
    pub disk_read_bytes: Option<u64>,
    /// Weights copied to the GPU during the load; `None` on the CPU.
    pub gpu_upload: Option<Transfer>,
    /// Device-wide drop in free GPU memory across the load: this model plus any other program using the GPU.
    pub gpu_device_growth_bytes: Option<u64>,
}

/// What the GPU did per forward pass within one phase of a run.
#[derive(Debug, Serialize)]
pub struct GpuPhaseWork {
    pub passes: usize,
    pub launches_per_pass: f64,
    pub allocations_per_pass: f64,
    pub upload_bytes_per_pass: f64,
    pub upload_ms_per_pass: f64,
    pub download_bytes_per_pass: f64,
    pub download_ms_per_pass: f64,
    pub wait_for_gpu_ms_per_pass: f64,
}

#[derive(Debug, Serialize)]
pub struct GpuWork {
    pub prefill: GpuPhaseWork,
    pub decode: GpuPhaseWork,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Reading {
    pub mean: f64,
    pub max: f64,
}

/// What `--sample-gpu` found: GPU load, power, clock and temperature over the runs.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GpuSampling {
    Off,
    Unavailable {
        reason: String,
    },
    Sampled {
        samples: usize,
        /// Samples with the GPU at least half busy, which the three shares below are taken over.
        busy_samples: usize,
        utilization_percent: Option<Reading>,
        power_watts: Option<Reading>,
        clock_mhz: Option<Reading>,
        max_clock_mhz: Option<f64>,
        temperature_c: Option<Reading>,
        /// Percent of busy samples in which the driver was lowering clocks to stay under the power limit.
        power_cap_percent: Option<f64>,
        thermal_slowdown_percent: Option<f64>,
        hw_slowdown_percent: Option<f64>,
    },
}

#[derive(Debug, Serialize)]
pub struct Results {
    pub first_token_ms: Stat,
    pub prefill_tokens_per_s: Stat,
    pub decode_tokens_per_s: Stat,
    pub token_latency_p50_ms: Stat,
    pub token_latency_p95_ms: Stat,
    pub sampling_ms_per_token: Stat,
    /// Estimate: weights bytes times decode speed, since each one-token pass reads every weight once.
    pub weight_bandwidth_gb_per_s: Stat,
    /// Average number of CPU cores kept busy over all runs, warmup included.
    pub cpu_busy_cores: Option<f64>,
    pub gpu: Option<GpuWork>,
    /// Most GPU memory in use on the device (all programs) seen between tokens.
    pub gpu_device_used_peak_bytes: Option<u64>,
    pub gpu_sampling: GpuSampling,
    /// Joules the GPU used per generated token, over the whole measured loop: mean power times its duration,
    /// divided by every token it generated. It includes prompt processing, warmup and any pauses between tokens.
    pub gpu_energy_per_token_joules: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct StageRow {
    pub stage: &'static str,
    pub total_ms: f64,
    pub calls: u64,
    pub share_of_wall_percent: f64,
    /// Weight bytes read per second, for weight-reading stages of a decode pass.
    pub gb_per_s: Option<f64>,
    /// Math done per second, for weight-reading stages of the prompt pass.
    pub tflop_per_s: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Phase {
    pub tokens: usize,
    pub wall_ms: f64,
    pub stages: Vec<StageRow>,
    /// Wall time not inside any timed stage: allocation, copies and launch overhead on the host.
    pub outside_stages_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Profile {
    pub clock: &'static str,
    pub prefill: Phase,
    pub decode: Phase,
}

/// The sampler settings of every run: always greedy, so repeats give the same tokens.
#[derive(Debug, Serialize)]
pub struct SamplerSettings {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

/// One measured run, unsummarized, so any statistic can be recomputed later.
#[derive(Debug, Serialize)]
pub struct RunRecord {
    pub first_token_ms: f64,
    pub prompt_forward_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub sampling_ms: f64,
    pub generated: usize,
    /// Time from each generated token to the next, in order.
    pub token_gaps_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct Record {
    pub format: u32,
    pub unix_time: u64,
    pub model: String,
    pub precision: &'static str,
    pub quantization: String,
    pub context_length: usize,
    pub batch_size: usize,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub sampler: SamplerSettings,
    /// Time the benchmark waited after each token, standing in for a slower consumer; 0 normally.
    pub token_pause_ms: f64,
    pub runtime: String,
    pub hardware: Hardware,
    pub load: Load,
    pub results: Results,
    pub runs: Vec<RunRecord>,
    pub profile: Option<Profile>,
}

/// `marin <crate version> (<git commit>)`; the commit says `-dirty` when the code was edited since.
pub fn runtime_version() -> String {
    format!(
        "marin {} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("MARIN_REVISION")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_versions_come_from_the_first_dotted_number() {
        let line = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.57.04  Release Build  (root@)\nGCC version: gcc 16.2.1\n";
        assert_eq!(parse_driver_version(line).as_deref(), Some("610.57.04"));
        assert_eq!(parse_driver_version("NVRM version: unknown"), None);
        assert_eq!(parse_driver_version(""), None);
    }

    #[test]
    fn the_runtime_version_names_the_commit_or_says_unknown() {
        let version = runtime_version();
        let revision = version
            .split_once('(')
            .and_then(|(_, rest)| rest.strip_suffix(')'))
            .unwrap_or_default();
        let hash = revision.strip_suffix("-dirty").unwrap_or(revision);
        let hex = hash.len() == 12 && hash.chars().all(|c| c.is_ascii_hexdigit());
        assert!(hex || hash == "unknown", "{version}");
        assert!(version.starts_with("marin 0."), "{version}");
    }

    #[test]
    fn stats_of_odd_even_single_and_empty_sets() {
        let odd = Stat::of(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!((odd.min, odd.median, odd.max), (1.0, 2.0, 3.0));
        let even = Stat::of(&[4.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(even.median, 2.5);
        let one = Stat::of(&[7.0]).unwrap();
        assert_eq!((one.min, one.median, one.max), (7.0, 7.0, 7.0));
        assert!(Stat::of(&[]).is_none());
    }
}
