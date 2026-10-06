//! The GPU probes check their own results (a wrong answer is an error, not a fast number), so these tests run
//! them briefly and look at what comes back. Without a GPU they skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

use std::time::Duration;

use cuda::{CudaError, Gpu};

const SHORT: Duration = Duration::from_millis(60);

fn gpu() -> Result<Option<Gpu>, String> {
    match Gpu::new() {
        Ok(gpu) => Ok(Some(gpu)),
        Err(CudaError::Unavailable(why)) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping GPU test: {why}");
            Ok(None)
        }
        Err(e) => Err(e.to_string()),
    }
}

#[test]
fn memory_reads_report_a_plausible_rate_and_pass_their_checksum() -> Result<(), String> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let windows = gpu
        .probe_read_bandwidth(64_000_000, SHORT, 3)
        .map_err(|e| e.to_string())?;
    assert_eq!(windows.len(), 3);
    for window in &windows {
        let gb_per_s = window.per_second() / 1e9;
        assert!((20.0..5_000.0).contains(&gb_per_s), "{gb_per_s} GB/s");
        assert!(window.work > 0.0 && window.time > Duration::ZERO);
    }
    Ok(())
}

#[test]
fn buffers_too_small_to_say_anything_about_memory_are_refused() -> Result<(), String> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for bytes in [0, 16, 1_000_000, 3_999_999] {
        let refused = gpu.probe_read_bandwidth(bytes, SHORT, 1);
        assert!(
            matches!(refused, Err(CudaError::Unsupported { .. })),
            "{bytes}: {refused:?}"
        );
    }
    Ok(())
}

#[test]
fn the_math_probes_report_positive_rates_or_say_the_gpu_cannot_run_them() -> Result<(), String> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let fp32 = gpu.probe_fp32(SHORT, 2).map_err(|e| e.to_string())?;
    let tflops = fp32
        .iter()
        .map(|w| w.per_second() / 1e12)
        .fold(0.0, f64::max);
    assert!((0.1..100.0).contains(&tflops), "{tflops} TFLOP/s");
    match gpu.probe_bf16_tensor(SHORT, 2) {
        Ok(windows) => {
            let best = windows
                .iter()
                .map(|w| w.per_second() / 1e12)
                .fold(0.0, f64::max);
            assert!(
                best > tflops,
                "tensor cores out-run plain fp32 math: {best} vs {tflops}"
            );
        }
        Err(CudaError::Compile(_) | CudaError::Driver(_)) => {
            eprintln!("tensor-core probe unavailable here")
        }
        Err(other) => return Err(other.to_string()),
    }
    Ok(())
}

#[test]
fn downloads_time_each_repeat_and_bigger_copies_take_longer() -> Result<(), String> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let small = gpu.probe_download(1, 10).map_err(|e| e.to_string())?;
    let large = gpu
        .probe_download(8_000_000, 10)
        .map_err(|e| e.to_string())?;
    assert_eq!((small.times.len(), large.times.len()), (10, 10));
    assert_eq!((small.bytes, large.bytes), (4, 32_000_000));
    let typical = |times: &[Duration]| {
        let mut sorted = times.to_vec();
        sorted.sort();
        sorted.get(sorted.len() / 2).copied().unwrap_or_default()
    };
    assert!(typical(&large.times) > typical(&small.times) * 10);
    Ok(())
}
