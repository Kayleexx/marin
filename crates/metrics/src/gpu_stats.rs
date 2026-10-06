use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;

const QUERY: &str = "utilization.gpu,power.draw,clocks.sm,temperature.gpu,\
clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,\
clocks_throttle_reasons.hw_slowdown,clocks.max.sm";

/// A sample counts as busy when the GPU was working at least this much: idle samples say nothing about limits.
const BUSY_PERCENT: f64 = 50.0;

/// One reading from the GPU's own sensors; a field is `None` when this GPU does not report it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpuSample {
    pub utilization_percent: Option<f64>,
    pub power_watts: Option<f64>,
    pub clock_mhz: Option<f64>,
    pub temperature_c: Option<f64>,
    /// The driver is lowering clocks to stay under the power limit.
    pub power_cap: Option<bool>,
    pub thermal_slowdown: Option<bool>,
    pub hw_slowdown: Option<bool>,
    pub max_clock_mhz: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuStats {
    pub samples: usize,
    pub utilization_percent: Option<Summary>,
    pub power_watts: Option<Summary>,
    pub clock_mhz: Option<Summary>,
    pub temperature_c: Option<Summary>,
    /// Samples with the GPU at least half busy; the shares below are taken over these when there are any.
    pub busy_samples: usize,
    pub power_cap_percent: Option<f64>,
    pub thermal_slowdown_percent: Option<f64>,
    pub hw_slowdown_percent: Option<f64>,
    pub max_clock_mhz: Option<f64>,
}

/// `[N/A]`, `N/A` and anything else that is not a number mean "not reported".
fn number(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

fn active(text: &str) -> Option<bool> {
    match text.trim() {
        "Active" => Some(true),
        "Not Active" => Some(false),
        _ => None,
    }
}

/// One `--format=csv,noheader,nounits` line with exactly the queried fields, at least one of them a reading.
pub fn parse_sample(line: &str) -> Option<GpuSample> {
    let mut fields = line.split(',');
    let sample = GpuSample {
        utilization_percent: number(fields.next()?),
        power_watts: number(fields.next()?),
        clock_mhz: number(fields.next()?),
        temperature_c: number(fields.next()?),
        power_cap: active(fields.next()?),
        thermal_slowdown: active(fields.next()?),
        hw_slowdown: active(fields.next()?),
        max_clock_mhz: number(fields.next()?),
    };
    (fields.next().is_none() && sample != GpuSample::default()).then_some(sample)
}

fn summary(values: impl Iterator<Item = f64>) -> Option<Summary> {
    let (count, sum, max) = values.fold((0_u32, 0.0, f64::MIN), |(n, sum, max), v| {
        (n.saturating_add(1), sum + v, max.max(v))
    });
    (count > 0).then(|| Summary {
        mean: sum / f64::from(count),
        max,
    })
}

/// Share of `samples` (of those that reported) in which `flag` was active, in percent.
fn share(samples: &[&GpuSample], flag: fn(&GpuSample) -> Option<bool>) -> Option<f64> {
    let reported: Vec<bool> = samples.iter().filter_map(|s| flag(s)).collect();
    (!reported.is_empty())
        .then(|| reported.iter().filter(|on| **on).count() as f64 / reported.len() as f64 * 100.0)
}

pub fn summarize(samples: &[GpuSample]) -> GpuStats {
    let pick = |field: fn(&GpuSample) -> Option<f64>| summary(samples.iter().filter_map(field));
    let busy: Vec<&GpuSample> = samples
        .iter()
        .filter(|s| s.utilization_percent.is_some_and(|u| u >= BUSY_PERCENT))
        .collect();
    let judged: Vec<&GpuSample> = if busy.is_empty() {
        samples.iter().collect()
    } else {
        busy.clone()
    };
    GpuStats {
        busy_samples: busy.len(),
        power_cap_percent: share(&judged, |s| s.power_cap),
        thermal_slowdown_percent: share(&judged, |s| s.thermal_slowdown),
        hw_slowdown_percent: share(&judged, |s| s.hw_slowdown),
        max_clock_mhz: pick(|s| s.max_clock_mhz).map(|m| m.max),
        samples: samples.len(),
        utilization_percent: pick(|s| s.utilization_percent),
        power_watts: pick(|s| s.power_watts),
        clock_mhz: pick(|s| s.clock_mhz),
        temperature_c: pick(|s| s.temperature_c),
    }
}

/// The power limit the driver enforces on GPU 0, in watts, or `None` when it cannot be read.
pub fn power_limit_watts() -> Option<f64> {
    if std::env::var_os("CUDA_VISIBLE_DEVICES").is_some() {
        return None;
    }
    let out = Command::new("nvidia-smi")
        .args([
            "-i",
            "0",
            "--query-gpu=enforced.power.limit",
            "--format=csv,noheader,nounits",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|out| out.status.success())?;
    parse_limit(&String::from_utf8_lossy(&out.stdout))
}

fn parse_limit(output: &str) -> Option<f64> {
    number(output.lines().next()?).filter(|watts| *watts > 0.0)
}

/// Reads GPU 0's load, power, clock and temperature twice a second from one `nvidia-smi` child process,
/// for as long as it lives. The child is stopped on `stop` and when this is dropped.
pub struct Sampler {
    child: Child,
    reader: Option<JoinHandle<Vec<GpuSample>>>,
}

impl Sampler {
    /// The error says in plain words why sampling is not possible here.
    pub fn start() -> Result<Self, String> {
        if std::env::var_os("CUDA_VISIBLE_DEVICES").is_some() {
            return Err(
                "CUDA_VISIBLE_DEVICES is set, so it is unclear which GPU nvidia-smi should watch"
                    .into(),
            );
        }
        let mut child = Command::new("nvidia-smi")
            .args(["-i", "0", "--query-gpu", QUERY])
            .args(["--format=csv,noheader,nounits", "-lms", "500"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nvidia-smi could not start ({e})"))?;
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err("nvidia-smi gave no output stream".into());
        };
        let reader = std::thread::spawn(move || {
            BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
                .filter_map(|line| parse_sample(&line))
                .collect()
        });
        Ok(Self {
            child,
            reader: Some(reader),
        })
    }

    fn shutdown(&mut self) -> Vec<GpuSample> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.reader
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default()
    }

    pub fn stop(mut self) -> Result<GpuStats, String> {
        let samples = self.shutdown();
        if samples.is_empty() {
            return Err("nvidia-smi gave no readings (the run may have been too short)".into());
        }
        Ok(summarize(&samples))
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// load, power, clock, temperature, power cap, thermal slowdown, hardware slowdown, max clock.
    fn line(load: &str, cap: &str) -> String {
        format!("{load}, 30.0, 1500, 55, {cap}, Not Active, Not Active, 3090")
    }

    #[test]
    fn parses_a_full_line_and_not_reported_fields() {
        let full =
            parse_sample("37, 24.50, 1500, 55, Active, Not Active, Not Active, 3090").unwrap();
        assert_eq!(full.utilization_percent, Some(37.0));
        assert_eq!(full.power_watts, Some(24.5));
        assert_eq!(full.clock_mhz, Some(1500.0));
        assert_eq!(full.temperature_c, Some(55.0));
        assert_eq!(full.power_cap, Some(true));
        assert_eq!(full.thermal_slowdown, Some(false));
        assert_eq!(full.max_clock_mhz, Some(3090.0));
        let partial =
            parse_sample("0, [N/A], 210, N/A, [N/A], Not Active, [Not Supported], [N/A]").unwrap();
        assert_eq!(partial.power_watts, None);
        assert_eq!(partial.temperature_c, None);
        assert_eq!(partial.power_cap, None);
        assert_eq!(partial.hw_slowdown, None);
        assert_eq!(partial.utilization_percent, Some(0.0));
    }

    #[test]
    fn rejects_lines_without_the_right_fields_or_without_any_reading() {
        let blank = "[N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]";
        for text in [
            "",
            "Failed to initialize NVML",
            "1, 2, 3, 4",
            "37, 24.5, 1500, 55, Active, Not Active, Not Active, 3090, 7",
            " , , , , , , , ",
            blank,
        ] {
            assert_eq!(parse_sample(text), None, "{text:?}");
        }
        let odd = parse_sample("NaN, inf, 3, 4, Active, Active, Active, 5").unwrap();
        assert_eq!((odd.utilization_percent, odd.power_watts), (None, None));
    }

    #[test]
    fn summaries_ignore_missing_readings() {
        let samples = [
            parse_sample("10, 20, 1000, 50, Not Active, Not Active, Not Active, 3000").unwrap(),
            parse_sample("30, [N/A], 2000, 60, Not Active, Not Active, Not Active, 3000").unwrap(),
        ];
        let stats = summarize(&samples);
        assert_eq!(stats.samples, 2);
        assert_eq!(
            stats.utilization_percent,
            Some(Summary {
                mean: 20.0,
                max: 30.0
            })
        );
        assert_eq!(
            stats.power_watts,
            Some(Summary {
                mean: 20.0,
                max: 20.0
            })
        );
        assert_eq!(stats.temperature_c.map(|s| s.max), Some(60.0));
        assert_eq!(stats.max_clock_mhz, Some(3000.0));
        assert_eq!(summarize(&[]).utilization_percent, None);
        assert_eq!(summarize(&[]).power_cap_percent, None);
    }

    #[test]
    fn the_enforced_power_limit_is_read_from_one_line_or_not_at_all() {
        assert_eq!(parse_limit("40.00\n"), Some(40.0));
        assert_eq!(parse_limit("  115.00 \n"), Some(115.0));
        for bad in ["", "[N/A]\n", "N/A", "0.00", "-5", "garbage"] {
            assert_eq!(parse_limit(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_power_cap_share_counts_busy_samples_only() {
        let samples: Vec<GpuSample> = [
            line("0", "Not Active"),
            line("2", "Not Active"),
            line("90", "Active"),
            line("95", "Active"),
            line("100", "Active"),
            line("80", "Not Active"),
        ]
        .iter()
        .map(|l| parse_sample(l).unwrap())
        .collect();
        let stats = summarize(&samples);
        assert_eq!((stats.samples, stats.busy_samples), (6, 4));
        assert_eq!(
            stats.power_cap_percent,
            Some(75.0),
            "the two idle samples do not dilute it"
        );
        assert_eq!(stats.thermal_slowdown_percent, Some(0.0));
    }

    #[test]
    fn with_no_busy_samples_every_sample_is_judged_and_unreported_flags_give_nothing() {
        let idle: Vec<GpuSample> = [line("1", "Active"), line("3", "Not Active")]
            .iter()
            .map(|l| parse_sample(l).unwrap())
            .collect();
        let stats = summarize(&idle);
        assert_eq!(
            (stats.busy_samples, stats.power_cap_percent),
            (0, Some(50.0))
        );
        let unknown = parse_sample("90, 30, 1500, 55, [N/A], [N/A], [N/A], 3090").unwrap();
        assert_eq!(summarize(&[unknown]).power_cap_percent, None);
    }
}
