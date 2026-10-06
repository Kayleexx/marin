use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use model::{CopyTimes, Gpu, ModelError, Window};
use serde::Serialize;

use crate::CliError;
use crate::bench::record::GpuSampling;
use crate::bench::render::sampling_lines;
use crate::bench::resources::GpuWatch;
use crate::json_out::JsonOut;
use crate::signal::Interrupt;
use crate::units;

/// How long each timed window of a probe lasts.
const WINDOW: Duration = Duration::from_millis(500);
/// Biggest buffer the memory probe reads: far larger than any cache, small enough to leave room for other work.
const READ_BUFFER_BYTES: u64 = 1 << 30;
/// Values copied back in the copy probe: one, the logits of a 151,936-token vocabulary, and a large block.
const COPY_SIZES: [usize; 3] = [1, 151_936, 16_777_216];
const COPY_REPEATS: usize = 30;

pub struct Options {
    pub seconds: f64,
    pub json: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Rate {
    first_per_s: f64,
    sustained_per_s: f64,
    best_per_s: f64,
    windows: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Probed {
    Measured(Rate),
    Unavailable { reason: String },
}

#[derive(Debug, Serialize)]
struct Copy {
    bytes: usize,
    min_micros: f64,
    median_micros: f64,
    gb_per_s: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    format: u32,
    gpu: String,
    compute_capability: String,
    multiprocessors: usize,
    vram_bytes: u64,
    window_seconds: f64,
    memory_read_bytes: u64,
    memory_read_bytes_per_s: Probed,
    fp32_flop_per_s: Probed,
    bf16_tensor_flop_per_s: Probed,
    copies_to_host: Vec<Copy>,
    gpu_sampling: GpuSampling,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match (values.get(middle.wrapping_sub(1)), values.get(middle)) {
        (Some(low), Some(high)) if values.len().is_multiple_of(2) => (low + high) / 2.0,
        (_, Some(mid)) => *mid,
        _ => 0.0,
    }
}

/// The opening window, what the GPU holds afterwards (median of the later ones), and the best window.
fn rate(windows: &[Window]) -> Probed {
    let each: Vec<f64> = windows.iter().map(Window::per_second).collect();
    let (Some(first), Some(best)) = (
        each.first().copied(),
        each.iter().copied().max_by(f64::total_cmp),
    ) else {
        return Probed::Unavailable {
            reason: "no window was measured".into(),
        };
    };
    let mut later: Vec<f64> = each.iter().skip(1).copied().collect();
    let sustained = if later.is_empty() {
        first
    } else {
        median(&mut later)
    };
    Probed::Measured(Rate {
        first_per_s: first,
        sustained_per_s: sustained,
        best_per_s: best,
        windows: each,
    })
}

fn copy(times: &CopyTimes) -> Copy {
    let mut micros: Vec<f64> = times.times.iter().map(|t| t.as_secs_f64() * 1e6).collect();
    let min = micros.iter().copied().fold(f64::INFINITY, f64::min);
    let mid = median(&mut micros);
    Copy {
        bytes: times.bytes,
        min_micros: if min.is_finite() { min } else { 0.0 },
        median_micros: mid,
        gb_per_s: if mid > 0.0 {
            times.bytes as f64 / (mid / 1e6) / 1e9
        } else {
            0.0
        },
    }
}

fn unit_line(label: &str, probed: &Probed, scale: f64, unit: &str) -> String {
    match probed {
        Probed::Measured(r) => format!(
            "  {label:<34}{:>9.1} {unit}{:>9.1} {unit}{:>9.1} {unit}\n",
            r.first_per_s / scale,
            r.sustained_per_s / scale,
            r.best_per_s / scale
        ),
        Probed::Unavailable { reason } => format!("  {label:<34}n/a ({reason})\n"),
    }
}

fn text(r: &Report) -> String {
    let mut out = format!(
        "gpu probe: {} · compute {} · {} multiprocessors · {} memory\n  each test runs in windows of {:.1} s: \"first\" is the opening burst, \"sustained\" is the median of the windows after it\n\n",
        r.gpu,
        r.compute_capability,
        r.multiprocessors,
        units::bytes(r.vram_bytes),
        r.window_seconds
    );
    out.push_str(&format!(
        "  {:<34}{:>14}{:>16}{:>16}\n",
        "what this gpu can do", "first", "sustained", "best"
    ));
    out.push_str(&unit_line(
        &format!("memory read ({} buffer)", units::bytes(r.memory_read_bytes)),
        &r.memory_read_bytes_per_s,
        1e9,
        "GB/s",
    ));
    out.push_str(&unit_line(
        "fp32 math (fused multiply-add)",
        &r.fp32_flop_per_s,
        1e12,
        "TFLOP/s",
    ));
    out.push_str(&unit_line(
        "bf16 tensor cores",
        &r.bf16_tensor_flop_per_s,
        1e12,
        "TFLOP/s",
    ));
    out.push_str(
        "\n  copying results from the gpu to the host (the path the logits take every token):\n",
    );
    for c in &r.copies_to_host {
        out.push_str(&format!(
            "    {:<12}{:>9.1} µs typical · {:>7.1} µs best · {:>6.2} GB/s\n",
            units::bytes(c.bytes as u64),
            c.median_micros,
            c.min_micros,
            c.gb_per_s
        ));
    }
    out.push_str(&sampling_lines(&r.gpu_sampling));
    out.push_str(
        "\nreading this: compare these with what the engine reaches (`marin-run bench --profile` shows GB/s and TFLOP/s\n  \
         per stage). \"sustained\" is the honest ceiling on a laptop GPU, which is held to a power limit.\n",
    );
    out
}

pub fn run(options: &Options, interrupt: &Interrupt) -> Result<(), CliError> {
    if !(options.seconds.is_finite() && options.seconds >= 1.0 && options.seconds <= 60.0) {
        return Err(CliError::BadOption(
            "--seconds must be between 1 and 60".into(),
        ));
    }
    interrupt.generating();
    let json = options.json.as_deref().map(JsonOut::create).transpose()?;
    let gpu = Gpu::new().map_err(ModelError::from)?;
    let info = gpu.info().map_err(ModelError::from)?;
    let windows = (options.seconds / WINDOW.as_secs_f64()).ceil().max(3.0) as usize;
    let cancelled = || -> Result<(), CliError> {
        if interrupt.cancel_flag().load(Ordering::SeqCst) {
            Err(CliError::Cancelled)
        } else {
            Ok(())
        }
    };
    let watch = GpuWatch::watching();
    let free = u64::try_from(info.free_bytes).unwrap_or(0);
    let bytes = READ_BUFFER_BYTES.min(free / 2);
    let reading = gpu
        .probe_read_bandwidth(usize::try_from(bytes).unwrap_or(0), WINDOW, windows)
        .map(|w| rate(&w))
        .map_err(ModelError::from)?;
    cancelled()?;
    let fp32 = rate(&gpu.probe_fp32(WINDOW, windows).map_err(ModelError::from)?);
    cancelled()?;
    let tensor = gpu.probe_bf16_tensor(WINDOW, windows).map_or_else(
        |e| Probed::Unavailable {
            reason: e.to_string(),
        },
        |w| rate(&w),
    );
    cancelled()?;
    let sampling = watch.finish();
    let mut copies = Vec::new();
    for floats in COPY_SIZES {
        copies.push(copy(
            &gpu.probe_download(floats, COPY_REPEATS)
                .map_err(ModelError::from)?,
        ));
    }
    let report = Report {
        format: 1,
        gpu: info.name.clone(),
        compute_capability: format!(
            "{}.{}",
            info.compute_capability.0, info.compute_capability.1
        ),
        multiprocessors: gpu.sm_count().map_err(ModelError::from)?,
        vram_bytes: u64::try_from(info.total_bytes).unwrap_or(u64::MAX),
        window_seconds: WINDOW.as_secs_f64(),
        memory_read_bytes: bytes,
        memory_read_bytes_per_s: reading,
        fp32_flop_per_s: fp32,
        bf16_tensor_flop_per_s: tensor,
        copies_to_host: copies,
        gpu_sampling: sampling,
    };
    std::io::Write::write_all(&mut std::io::stdout().lock(), text(&report).as_bytes())?;
    if let (Some(json), Some(path)) = (json, &options.json) {
        json.finish(&report)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(gb: f64) -> Window {
        Window {
            work: gb * 1e9,
            time: Duration::from_secs(1),
        }
    }

    fn rates(probed: &Probed) -> (f64, f64, f64) {
        match probed {
            Probed::Measured(r) => (
                r.first_per_s / 1e9,
                r.sustained_per_s / 1e9,
                r.best_per_s / 1e9,
            ),
            Probed::Unavailable { reason } => panic!("{reason}"),
        }
    }

    #[test]
    fn the_opening_window_is_kept_apart_from_what_the_gpu_sustains() {
        let probed = rate(&[window(340.0), window(300.0), window(290.0), window(295.0)]);
        assert_eq!(
            rates(&probed),
            (340.0, 295.0, 340.0),
            "sustained is the median of the later windows"
        );
        assert_eq!(
            rates(&rate(&[window(100.0)])),
            (100.0, 100.0, 100.0),
            "one window is both"
        );
        assert!(matches!(rate(&[]), Probed::Unavailable { .. }));
    }

    #[test]
    fn medians_handle_every_length() {
        assert_eq!(median(&mut []), 0.0);
        assert_eq!(median(&mut [5.0]), 5.0);
        assert_eq!(median(&mut [9.0, 1.0]), 5.0);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn a_copy_reports_typical_and_best_time_and_the_rate() {
        let times = CopyTimes {
            bytes: 607_744,
            times: [100, 90, 110, 95]
                .iter()
                .map(|us| Duration::from_micros(*us))
                .collect(),
        };
        let c = copy(&times);
        assert_eq!((c.min_micros, c.median_micros), (90.0, 97.5));
        assert!((c.gb_per_s - 6.23).abs() < 0.01, "{}", c.gb_per_s);
        let empty = copy(&CopyTimes {
            bytes: 4,
            times: Vec::new(),
        });
        assert_eq!(
            (empty.min_micros, empty.median_micros, empty.gb_per_s),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn the_report_names_each_test_and_says_when_one_is_not_available() {
        let report = Report {
            format: 1,
            gpu: "Test GPU".into(),
            compute_capability: "12.0".into(),
            multiprocessors: 20,
            vram_bytes: 8_000_000_000,
            window_seconds: 0.5,
            memory_read_bytes: 1_070_000_000,
            memory_read_bytes_per_s: rate(&[window(320.0), window(290.0)]),
            fp32_flop_per_s: Probed::Measured(Rate {
                first_per_s: 1.1e13,
                sustained_per_s: 1.0e13,
                best_per_s: 1.1e13,
                windows: vec![],
            }),
            bf16_tensor_flop_per_s: Probed::Unavailable {
                reason: "needs compute 8.0".into(),
            },
            copies_to_host: vec![copy(&CopyTimes {
                bytes: 607_744,
                times: vec![Duration::from_micros(90)],
            })],
            gpu_sampling: GpuSampling::Off,
        };
        let shown = text(&report);
        for part in [
            "gpu probe: Test GPU · compute 12.0 · 20 multiprocessors",
            "memory read (1.07 GB buffer)",
            "320.0 GB/s",
            "290.0 GB/s",
            "fp32 math (fused multiply-add)",
            "10.0 TFLOP/s",
            "bf16 tensor cores",
            "n/a (needs compute 8.0)",
            "608 KB",
            "90.0 µs typical",
        ] {
            assert!(shown.contains(part), "missing {part:?} in\n{shown}");
        }
        assert!(!shown.contains("NaN"));
    }
}
