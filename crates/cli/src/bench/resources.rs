use std::time::{Duration, Instant};

use metrics::{Sampler, cpu_time};
use model::Engine;

use super::record::{GpuSampling, Reading};

/// Average CPU cores kept busy between `start` and `busy_cores`.
pub struct CpuWatch {
    cpu: Option<Duration>,
    started: Instant,
}

impl CpuWatch {
    pub fn start() -> Self {
        Self {
            cpu: cpu_time(),
            started: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// `None` when the platform gives no CPU time, or no time passed.
    pub fn busy_cores(&self) -> Option<f64> {
        let used = cpu_time()?.saturating_sub(self.cpu?);
        let wall = self.started.elapsed();
        (!wall.is_zero()).then(|| used.as_secs_f64() / wall.as_secs_f64())
    }
}

/// `--sample-gpu`: an `nvidia-smi` child that watches the GPU while the runs go.
pub enum GpuWatch {
    Off,
    Unavailable(String),
    Running(Sampler),
}

impl GpuWatch {
    pub fn start(requested: bool, engine: &Engine) -> Self {
        if !requested {
            return Self::Off;
        }
        if matches!(engine, Engine::Cpu(_)) {
            return Self::Unavailable("this run uses the CPU".into());
        }
        Self::watching()
    }

    /// Watches the GPU now, whatever is running on it.
    pub fn watching() -> Self {
        match Sampler::start() {
            Ok(sampler) => Self::Running(sampler),
            Err(reason) => Self::Unavailable(reason),
        }
    }

    pub fn finish(self) -> GpuSampling {
        let reading = |s: Option<metrics::Summary>| {
            s.map(|s| Reading {
                mean: s.mean,
                max: s.max,
            })
        };
        match self {
            Self::Off => GpuSampling::Off,
            Self::Unavailable(reason) => GpuSampling::Unavailable { reason },
            Self::Running(sampler) => match sampler.stop() {
                Err(reason) => GpuSampling::Unavailable { reason },
                Ok(stats) => GpuSampling::Sampled {
                    samples: stats.samples,
                    busy_samples: stats.busy_samples,
                    utilization_percent: reading(stats.utilization_percent),
                    power_watts: reading(stats.power_watts),
                    clock_mhz: reading(stats.clock_mhz),
                    max_clock_mhz: stats.max_clock_mhz,
                    temperature_c: reading(stats.temperature_c),
                    power_cap_percent: stats.power_cap_percent,
                    thermal_slowdown_percent: stats.thermal_slowdown_percent,
                    hw_slowdown_percent: stats.hw_slowdown_percent,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_cores_is_a_small_non_negative_number_for_a_busy_loop() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let watch = CpuWatch::start();
        let until = Instant::now() + Duration::from_millis(60);
        while Instant::now() < until {
            std::hint::spin_loop();
        }
        let cores = watch.busy_cores().unwrap();
        assert!(cores > 0.2 && cores < 64.0, "{cores}");
    }

    #[test]
    fn an_unrequested_watch_says_off_and_a_cpu_run_says_why_not() {
        assert!(matches!(GpuWatch::Off.finish(), GpuSampling::Off));
        let finished = GpuWatch::Unavailable("this run uses the CPU".into()).finish();
        assert!(matches!(finished, GpuSampling::Unavailable { reason } if reason.contains("CPU")));
    }
}
