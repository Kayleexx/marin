use std::fmt::Write;

use super::record::{Phase, Record, Stat};
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
        let _ = writeln!(
            out,
            "    {:<24}{:>9.2} ms/{per_label:<5}{:>6.1}%   {:>5} calls",
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

pub fn text(r: &Record) -> String {
    let mut out = format!(
        "{}  ·  {}  ·  {}\n  prompt {} tokens, output {} tokens, context {}, batch {}, {} runs after {} warmup\n  {} · {} cores · {} RAM · {}\n\n",
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
    if let Some(gpu) = r.load.gpu_growth_bytes {
        let _ = writeln!(
            out,
            "  gpu memory grew by {} (beyond the CUDA context)",
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
    if let Some(peak) = r.results.peak_rss_bytes {
        let _ = writeln!(
            out,
            "  peak process memory       {:>10}",
            units::bytes(peak)
        );
    }
    if let Some(g) = &r.results.gpu {
        let _ = writeln!(
            out,
            "\ngpu work per forward pass: {:.0} kernel launches · {:.0} device allocations · {:.0} KB downloaded in {:.3} ms · {:.2} ms waiting for queued GPU work",
            g.launches_per_token,
            g.allocations_per_token,
            g.download_bytes_per_token / 1000.0,
            g.download_ms_per_token,
            g.wait_for_gpu_ms_per_token
        );
    }
    if let Some(p) = &r.profile {
        let _ = write!(
            out,
            "\nwhere the time goes (profiled pass, stage times on the {}):\n",
            p.clock
        );
        out.push_str(&phase_table("prefill", &p.prefill, 1, "pass"));
        out.push_str(&phase_table("decode", &p.decode, p.decode.tokens, "token"));
    }
    out
}
