use std::fmt::Write;

use super::record::{GpuPhaseWork, GpuSampling, Phase, Profile, Reading, Record, Stat};
use crate::units;

fn stat(label: &str, s: &Stat, unit: &str, digits: usize) -> String {
    format!(
        "  {label:<26}{:>10.digits$} {unit}   (min {:.digits$}, max {:.digits$})\n",
        s.median, s.min, s.max
    )
}

fn phase_table(name: &str, phase: &Phase, per: usize, per_label: &str) -> String {
    let mut out = String::new();
    let divisor = per.max(1) as f64;
    let _ = writeln!(
        out,
        "  {name}: {} tokens, {:.1} ms wall ({:.2} ms {per_label})",
        phase.tokens,
        phase.wall_ms,
        phase.wall_ms / divisor
    );
    let mut rows: Vec<_> = phase.stages.iter().filter(|r| r.calls > 0).collect();
    rows.sort_by(|a, b| b.total_ms.total_cmp(&a.total_ms));
    for row in rows {
        let rate = match (row.gb_per_s, row.tflop_per_s) {
            (Some(gb), _) => format!("   {gb:>6.0} GB/s"),
            (_, Some(tflop)) => format!("   {tflop:>6.2} TFLOP/s"),
            _ => String::new(),
        };
        let _ = writeln!(
            out,
            "    {:<24}{:>9.2} ms/{per_label:<5}{:>6.1}%   {:>5} calls{rate}",
            row.stage,
            row.total_ms / divisor,
            row.share_of_wall_percent,
            row.calls
        );
    }
    let outside = if phase.wall_ms > 0.0 {
        phase.outside_stages_ms / phase.wall_ms * 100.0
    } else {
        0.0
    };
    let _ = writeln!(
        out,
        "    {:<24}{:>9.2} ms/{per_label:<5}{outside:>6.1}%",
        "outside timed stages",
        phase.outside_stages_ms / divisor
    );
    out
}

pub(crate) fn sampling_lines(sampling: &GpuSampling) -> String {
    let both = |reading: &Option<Reading>, unit: &str, digits: usize| {
        let gap = if unit == "%" { "" } else { " " };
        reading.as_ref().map(|r| {
            format!(
                "mean {:.digits$}{gap}{unit} · max {:.digits$}{gap}{unit}",
                r.mean, r.max
            )
        })
    };
    match sampling {
        GpuSampling::Off => String::new(),
        GpuSampling::Unavailable { reason } => format!("\ngpu sampling: n/a ({reason})\n"),
        GpuSampling::Sampled {
            samples,
            busy_samples,
            utilization_percent,
            power_watts,
            clock_mhz,
            max_clock_mhz,
            temperature_c,
            power_cap_percent,
            thermal_slowdown_percent,
            hw_slowdown_percent,
        } => {
            let mut out = format!("\ngpu while running ({samples} samples from nvidia-smi):\n");
            let clock = clock_mhz.as_ref().map(|c| {
                let of_max = max_clock_mhz
                    .filter(|m| *m > 0.0)
                    .map_or(String::new(), |m| {
                        format!(" ({:.0}% of the {m:.0} MHz maximum)", c.mean / m * 100.0)
                    });
                format!("mean {:.0} MHz{of_max} · max {:.0} MHz", c.mean, c.max)
            });
            let percent =
                |value: &Option<f64>| value.map_or("n/a".to_string(), |v| format!("{v:.0}%"));
            let limits = format!(
                "power cap active in {} of {busy_samples} busy samples · thermal slowdown {} · hardware slowdown {}",
                percent(power_cap_percent),
                percent(thermal_slowdown_percent),
                percent(hw_slowdown_percent)
            );
            let rows = [
                ("load", both(utilization_percent, "%", 0)),
                ("power", both(power_watts, "W", 1)),
                ("clock", clock),
                ("limits", Some(limits)),
                ("temperature", both(temperature_c, "C", 0)),
            ];
            for (label, text) in rows {
                let _ = writeln!(out, "  {label:<12}{}", text.unwrap_or_else(|| "n/a".into()));
            }
            out
        }
    }
}

/// How to read the numbers, for the sections the report actually shows.
fn notes(r: &Record) -> String {
    let mut out = String::from("\nreading this:\n");
    out.push_str(
        "  each one-token pass reads every weight once, so a weight bandwidth near the memory's\n  \
         rated speed means decoding is limited by memory, not by arithmetic.\n",
    );
    if r.results.gpu_device_used_peak_bytes.is_some() {
        out.push_str("  gpu memory is read for the whole device, so other programs on the GPU are included.\n");
    }
    if matches!(r.results.gpu_sampling, GpuSampling::Sampled { .. }) {
        out.push_str(
            "  when the power cap is active, the driver is holding the GPU to its power limit: speed then\n  \
             depends on energy per token as much as on time, so a faster step may not mean more tokens a second.\n",
        );
    }
    out
}

fn gpu_line(label: &str, g: &GpuPhaseWork) -> String {
    let rate = if g.download_ms_per_pass > 0.0 {
        format!(
            " ({:.1} GB/s)",
            g.download_bytes_per_pass / (g.download_ms_per_pass / 1000.0) / 1e9
        )
    } else {
        String::new()
    };
    format!(
        "  {label:<16}{:>6.0} kernel launches · {:>4.0} device allocations · {:.0} KB downloaded in {:.3} ms{rate} · {:.2} ms waiting for queued GPU work\n",
        g.launches_per_pass,
        g.allocations_per_pass,
        g.download_bytes_per_pass / 1000.0,
        g.download_ms_per_pass,
        g.wait_for_gpu_ms_per_pass
    )
}

/// How the machine was powered: on AC or battery, the GPU's power limit and any pause between tokens.
fn power_line(r: &Record) -> String {
    let mut parts = Vec::new();
    match r.hardware.on_ac_power {
        Some(true) => parts.push("on AC power".to_string()),
        Some(false) => parts.push("on battery".to_string()),
        None => {}
    }
    if let Some(watts) = r.hardware.gpu_power_limit_watts {
        parts.push(format!("gpu power limit {watts:.0} W"));
    }
    if r.token_pause_ms > 0.0 {
        parts.push(format!(
            "{:.1} ms pause after every token",
            r.token_pause_ms
        ));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  {}\n", parts.join(" · "))
    }
}

pub fn profile_section(p: &Profile) -> String {
    let mut out = format!(
        "where the time goes (profiled pass, stage times on the {}):\n",
        p.clock
    );
    out.push_str(&phase_table("prefill", &p.prefill, 1, "pass"));
    if p.decode.tokens > 0 {
        out.push_str(&phase_table("decode", &p.decode, p.decode.tokens, "token"));
    }
    out
}

pub fn text(r: &Record) -> String {
    let mut out = format!(
        "{}  ·  {}  ·  {}\n  prompt {} tokens, output {} tokens, context {}, batch {}, {} runs after {} warmup\n  {} · {} cores · {} RAM · {}\n",
        r.model,
        r.hardware.backend,
        r.precision,
        r.prompt_tokens,
        r.output_tokens,
        r.context_length,
        r.batch_size,
        r.measured_runs,
        r.warmup_runs,
        r.hardware.cpu,
        r.hardware.logical_cores,
        units::bytes(r.hardware.ram_bytes),
        r.runtime
    );
    out.push_str(&power_line(r));
    out.push('\n');
    let _ = writeln!(out, "load: {:.2} s", r.load.seconds);
    let _ = writeln!(
        out,
        "  weights {} · kv cache reserved {}",
        units::bytes(r.load.weights_bytes),
        units::bytes(r.load.kv_reserved_bytes)
    );
    if let Some(rss) = r.load.rss_growth_bytes {
        let _ = writeln!(out, "  process memory grew by {}", units::bytes(rss));
    }
    if let Some(disk) = r.load.disk_read_bytes {
        let _ = writeln!(
            out,
            "  read from disk {} (files already in memory count as 0)",
            units::bytes(disk)
        );
    }
    if let Some(up) = &r.load.gpu_upload {
        let rate = up
            .gb_per_s
            .map_or(String::new(), |g| format!(" ({g:.1} GB/s)"));
        let _ = writeln!(
            out,
            "  copied to the gpu {} in {:.2} s{rate}",
            units::bytes(up.bytes),
            up.seconds
        );
    }
    if let Some(gpu) = r.load.gpu_device_growth_bytes {
        let _ = writeln!(
            out,
            "  gpu memory in use grew by {} (device-wide, includes other programs)",
            units::bytes(gpu)
        );
    }
    out.push_str("\nresults (median of runs):\n");
    out.push_str(&stat(
        "first token (TTFT)",
        &r.results.first_token_ms,
        "ms",
        1,
    ));
    out.push_str(&stat(
        "prefill",
        &r.results.prefill_tokens_per_s,
        "tok/s",
        1,
    ));
    out.push_str(&stat("decode", &r.results.decode_tokens_per_s, "tok/s", 2));
    out.push_str(&stat(
        "token latency p50",
        &r.results.token_latency_p50_ms,
        "ms",
        2,
    ));
    out.push_str(&stat(
        "token latency p95",
        &r.results.token_latency_p95_ms,
        "ms",
        2,
    ));
    out.push_str(&stat(
        "sampling per token",
        &r.results.sampling_ms_per_token,
        "ms",
        3,
    ));
    out.push_str(&stat(
        "weight bandwidth (est.)",
        &r.results.weight_bandwidth_gb_per_s,
        "GB/s",
        1,
    ));
    if let Some(joules) = r.results.gpu_energy_per_token_joules {
        let _ = writeln!(
            out,
            "  gpu energy per token      {joules:>10.2} J   (power times time over the runs, prompt and any pauses included)"
        );
    }
    if let Some(cores) = r.results.cpu_busy_cores {
        let _ = writeln!(
            out,
            "  cpu busy during runs      {cores:>10.1} of {} cores",
            r.hardware.logical_cores
        );
    }
    if let Some(peak) = r.results.gpu_device_used_peak_bytes {
        let _ = writeln!(
            out,
            "  gpu memory in use (peak)  {:>10} (device-wide, seen between tokens)",
            units::bytes(peak)
        );
    }
    if let Some(peak) = r.results.peak_rss_bytes {
        let _ = writeln!(
            out,
            "  peak process memory       {:>10}",
            units::bytes(peak)
        );
    }
    out.push_str(&sampling_lines(&r.results.gpu_sampling));
    if let Some(g) = &r.results.gpu {
        out.push_str("\ngpu work per forward pass:\n");
        out.push_str(&gpu_line("prompt pass", &g.prefill));
        out.push_str(&gpu_line("one-token pass", &g.decode));
    }
    if let Some(p) = &r.profile {
        out.push('\n');
        out.push_str(&profile_section(p));
    }
    out.push_str(&notes(r));
    out
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
