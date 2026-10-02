use std::fs;

use serde::Serialize;

/// Minimum, median and maximum of one measurement over the runs.
#[derive(Debug, Clone, Copy, Serialize)]
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
pub struct Hardware {
    pub cpu: String,
    pub logical_cores: usize,
    pub ram_bytes: u64,
    pub backend: String,
    pub os: String,
}

fn proc_value(file: &str, key: &str) -> Option<String> {
    let text = fs::read_to_string(file).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    Some(line.split_once(':')?.1.trim().to_string())
}

pub fn hardware(backend: String) -> Hardware {
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
    }
}

#[derive(Debug, Serialize)]
pub struct Load {
    pub seconds: f64,
    pub weights_bytes: u64,
    pub kv_reserved_bytes: u64,
    pub rss_growth_bytes: Option<u64>,
    /// GPU memory the load took beyond the CUDA context: weights, KV cache and working buffers.
    pub gpu_growth_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct GpuWork {
    pub launches_per_token: f64,
    pub allocations_per_token: f64,
    pub download_bytes_per_token: f64,
    pub download_ms_per_token: f64,
    pub wait_for_gpu_ms_per_token: f64,
}

#[derive(Debug, Serialize)]
pub struct Results {
    pub first_token_ms: Stat,
    pub prefill_tokens_per_s: Stat,
    pub decode_tokens_per_s: Stat,
    pub token_latency_p50_ms: Stat,
    pub token_latency_p95_ms: Stat,
    pub sampling_ms_per_token: Stat,
    pub gpu: Option<GpuWork>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct StageRow {
    pub stage: &'static str,
    pub total_ms: f64,
    pub calls: u64,
    pub share_of_wall_percent: f64,
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

#[derive(Debug, Serialize)]
pub struct Record {
    pub format: u32,
    pub unix_time: u64,
    pub model: String,
    pub precision: &'static str,
    pub quantization: &'static str,
    pub context_length: usize,
    pub batch_size: usize,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub runtime: String,
    pub hardware: Hardware,
    pub load: Load,
    pub results: Results,
    pub profile: Option<Profile>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
