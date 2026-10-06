use std::io::Write;
use std::path::Path;

use serde::Deserialize;

use super::record::{Reading, Stat};
use crate::CliError;

/// The result format `compare` reads; older files lack fields it needs.
const FORMAT: u32 = 2;

#[derive(Deserialize)]
struct FormatOnly {
    format: u32,
}

#[derive(Deserialize)]
struct Gpu {
    name: String,
}

#[derive(Deserialize)]
struct Hardware {
    cpu: String,
    backend: String,
    gpu: Option<Gpu>,
    #[serde(default)]
    on_ac_power: Option<bool>,
    #[serde(default)]
    gpu_power_limit_watts: Option<f64>,
}

#[derive(Deserialize)]
struct Sampler {
    temperature: f64,
    top_k: u64,
    top_p: f64,
    seed: u64,
}

/// What `--sample-gpu` saw, when it was used and the GPU could be read.
#[derive(Deserialize)]
struct Sampling {
    status: String,
    clock_mhz: Option<Reading>,
    power_cap_percent: Option<f64>,
}

#[derive(Deserialize)]
struct Results {
    #[serde(default)]
    gpu_sampling: Option<Sampling>,
    first_token_ms: Stat,
    prefill_tokens_per_s: Stat,
    decode_tokens_per_s: Stat,
    token_latency_p50_ms: Stat,
    token_latency_p95_ms: Stat,
    sampling_ms_per_token: Stat,
    weight_bandwidth_gb_per_s: Stat,
}

#[derive(Deserialize)]
struct Saved {
    model: String,
    precision: String,
    quantization: String,
    context_length: usize,
    batch_size: usize,
    prompt_tokens: usize,
    output_tokens: usize,
    sampler: Sampler,
    #[serde(default)]
    token_pause_ms: f64,
    runtime: String,
    hardware: Hardware,
    results: Results,
}

fn unusable(path: &Path, reason: impl ToString) -> CliError {
    CliError::Saved {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}

fn read(path: &Path) -> Result<Saved, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| unusable(path, e))?;
    let format: FormatOnly = serde_json::from_str(&text).map_err(|_| {
        unusable(
            path,
            "it is not a result written by `marin-run bench --json`",
        )
    })?;
    if format.format != FORMAT {
        return Err(unusable(
            path,
            format!(
                "it is format {}, compare needs format {FORMAT}",
                format.format
            ),
        ));
    }
    serde_json::from_str(&text).map_err(|e| unusable(path, e))
}

/// What differs between the two setups, so a comparison is only made between like and like. Quantization is left out:
/// it is what a before-and-after of a quantized run changes, and the report states it.
fn mismatches(old: &Saved, new: &Saved) -> Vec<String> {
    let gpu = |s: &Saved| s.hardware.gpu.as_ref().map(|g| g.name.clone());
    let mut found = Vec::new();
    let mut check = |label: &str, old: String, new: String| {
        if old != new {
            found.push(format!("{label}: {old} vs {new}"));
        }
    };
    check("model", old.model.clone(), new.model.clone());
    check(
        "backend",
        old.hardware.backend.clone(),
        new.hardware.backend.clone(),
    );
    check(
        "gpu",
        gpu(old).unwrap_or_default(),
        gpu(new).unwrap_or_default(),
    );
    check("cpu", old.hardware.cpu.clone(), new.hardware.cpu.clone());
    check("precision", old.precision.clone(), new.precision.clone());
    check(
        "context",
        old.context_length.to_string(),
        new.context_length.to_string(),
    );
    check(
        "batch size",
        old.batch_size.to_string(),
        new.batch_size.to_string(),
    );
    check(
        "prompt tokens",
        old.prompt_tokens.to_string(),
        new.prompt_tokens.to_string(),
    );
    check(
        "output tokens",
        old.output_tokens.to_string(),
        new.output_tokens.to_string(),
    );
    let sampler = |s: &Saved| {
        let c = &s.sampler;
        format!(
            "temperature {} top-k {} top-p {} seed {}",
            c.temperature, c.top_k, c.top_p, c.seed
        )
    };
    check("sampler", sampler(old), sampler(new));
    check(
        "pause after each token",
        format!("{} ms", old.token_pause_ms),
        format!("{} ms", new.token_pause_ms),
    );
    found
}

struct Metric {
    label: &'static str,
    unit: &'static str,
    digits: usize,
    higher_is_better: bool,
}

const METRICS: [Metric; 7] = [
    Metric {
        label: "first token (TTFT)",
        unit: "ms",
        digits: 1,
        higher_is_better: false,
    },
    Metric {
        label: "prefill",
        unit: "tok/s",
        digits: 1,
        higher_is_better: true,
    },
    Metric {
        label: "decode",
        unit: "tok/s",
        digits: 2,
        higher_is_better: true,
    },
    Metric {
        label: "token latency p50",
        unit: "ms",
        digits: 2,
        higher_is_better: false,
    },
    Metric {
        label: "token latency p95",
        unit: "ms",
        digits: 2,
        higher_is_better: false,
    },
    Metric {
        label: "sampling per token",
        unit: "ms",
        digits: 3,
        higher_is_better: false,
    },
    Metric {
        label: "weight bandwidth (est.)",
        unit: "GB/s",
        digits: 1,
        higher_is_better: true,
    },
];

/// A change only counts when the runs' min..max ranges do not overlap; otherwise it may be noise.
fn verdict(old: &Stat, new: &Stat, higher_is_better: bool) -> &'static str {
    if old.min <= new.max && new.min <= old.max {
        return "same (ranges overlap)";
    }
    if (new.median > old.median) == higher_is_better {
        "faster"
    } else {
        "slower"
    }
}

fn change(old: &Stat, new: &Stat) -> String {
    if old.median == 0.0 {
        return "n/a".into();
    }
    format!("{:+.1}%", (new.median - old.median) / old.median * 100.0)
}

fn row(metric: &Metric, old: &Stat, new: &Stat) -> String {
    let value = |s: &Stat| format!("{:.*} {}", metric.digits, s.median, metric.unit);
    format!(
        "  {:<24}{:>14}{:>14}{:>9}  {}\n",
        metric.label,
        value(old),
        value(new),
        change(old, new),
        verdict(old, new, metric.higher_is_better)
    )
}

fn sampled(s: &Saved) -> Option<&Sampling> {
    s.results
        .gpu_sampling
        .as_ref()
        .filter(|g| g.status == "sampled")
}

/// Warnings for differences between the two runs that move speed without any change in the code: how the machine
/// was powered, the GPU's power limit, and the clocks and power cap the GPU actually ran under.
fn conditions(old: &Saved, new: &Saved) -> Option<String> {
    let mut notes = Vec::new();
    let (a, b) = (&old.hardware, &new.hardware);
    if let (Some(x), Some(y)) = (a.on_ac_power, b.on_ac_power)
        && x != y
    {
        let power = |on| if on { "on AC power" } else { "on battery" };
        notes.push(format!(
            "one run was {} and the other {}",
            power(x),
            power(y)
        ));
    }
    if let (Some(x), Some(y)) = (a.gpu_power_limit_watts, b.gpu_power_limit_watts)
        && (x - y).abs() >= 1.0
    {
        notes.push(format!("the GPU power limit was {x:.0} W and {y:.0} W"));
    }
    if let (Some(x), Some(y)) = (sampled(old), sampled(new))
        && let (Some(clock_a), Some(clock_b)) = (x.clock_mhz.as_ref(), y.clock_mhz.as_ref())
        && let (Some(cap_a), Some(cap_b)) = (x.power_cap_percent, y.power_cap_percent)
    {
        let change = if clock_a.mean > 0.0 {
            (clock_b.mean - clock_a.mean) / clock_a.mean * 100.0
        } else {
            0.0
        };
        if change.abs() >= 10.0 || (cap_b - cap_a).abs() >= 25.0 {
            notes.push(format!(
                "the GPU clock averaged {:.0} vs {:.0} MHz ({change:+.0}%) with the power cap active in {cap_a:.0}% vs {cap_b:.0}% of busy samples",
                clock_a.mean, clock_b.mean
            ));
        }
    }
    (!notes.is_empty()).then(|| {
        format!(
            "note: the runs were not made under the same conditions: {}. A speed change may come from the GPU's power limit, not the code.\n",
            notes.join("; ")
        )
    })
}

fn render(old: &Saved, new: &Saved, old_path: &Path, new_path: &Path) -> String {
    let (o, n) = (&old.results, &new.results);
    let stats = [
        (&o.first_token_ms, &n.first_token_ms),
        (&o.prefill_tokens_per_s, &n.prefill_tokens_per_s),
        (&o.decode_tokens_per_s, &n.decode_tokens_per_s),
        (&o.token_latency_p50_ms, &n.token_latency_p50_ms),
        (&o.token_latency_p95_ms, &n.token_latency_p95_ms),
        (&o.sampling_ms_per_token, &n.sampling_ms_per_token),
        (&o.weight_bandwidth_gb_per_s, &n.weight_bandwidth_gb_per_s),
    ];
    let quantization = if old.quantization == new.quantization {
        String::new()
    } else {
        format!(
            "  quantization: {} -> {}\n",
            old.quantization, new.quantization
        )
    };
    let mut out = format!(
        "compare: {} -> {}\n  {}  ·  {}  ·  prompt {} tokens, output {} tokens\n  old {}\n  new {}\n{quantization}\n",
        old_path.display(),
        new_path.display(),
        new.model,
        new.hardware.backend,
        new.prompt_tokens,
        new.output_tokens,
        old.runtime,
        new.runtime
    );
    out.push_str(&format!(
        "  {:<24}{:>14}{:>14}{:>9}\n",
        "median of runs", "old", "new", "change"
    ));
    for (metric, (old, new)) in METRICS.iter().zip(stats) {
        out.push_str(&row(metric, old, new));
    }
    out.push_str(
        "\nreading this: \"same\" means the old and new min..max ranges overlap, so the difference\n\
         may only be run-to-run noise. Take more --runs to tighten the ranges.\n",
    );
    if let Some(note) = conditions(old, new) {
        out.push_str(&note);
    }
    if old.runtime == new.runtime {
        out.push_str(if new.runtime.contains("-dirty") {
            "both results carry the same label but come from uncommitted code, so the code may still differ.\n"
        } else {
            "both results come from the same build, so any difference here is noise or machine state.\n"
        });
    }
    out
}

pub fn run(old_path: &Path, new_path: &Path) -> Result<(), CliError> {
    let (old, new) = (read(old_path)?, read(new_path)?);
    let differences = mismatches(&old, &new);
    if !differences.is_empty() {
        return Err(CliError::NotComparable(differences.join("; ")));
    }
    std::io::stdout()
        .lock()
        .write_all(render(&old, &new, old_path, new_path).as_bytes())?;
    Ok(())
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
