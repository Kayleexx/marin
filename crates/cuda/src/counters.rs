use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cudarc::driver::CudaEvent;
use cudarc::driver::sys::CUevent_flags;

use crate::CudaError;
use crate::gpu::Gpu;

/// Always-on tallies of what the GPU path does: cheap atomic adds, read as snapshots.
#[derive(Default)]
pub(crate) struct Counters {
    launches: AtomicU64,
    allocations: AtomicU64,
    h2d_bytes: AtomicU64,
    h2d_nanos: AtomicU64,
    d2h_bytes: AtomicU64,
    d2h_nanos: AtomicU64,
    wait_nanos: AtomicU64,
}

fn nanos(time: Duration) -> u64 {
    u64::try_from(time.as_nanos()).unwrap_or(u64::MAX)
}

impl Counters {
    pub(crate) fn launch(&self) {
        self.launches.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn allocation(&self) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn upload(&self, bytes: usize, started: Instant) {
        self.h2d_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.h2d_nanos
            .fetch_add(nanos(started.elapsed()), Ordering::Relaxed);
    }

    /// `waited` is the time spent for earlier GPU work to finish; `copied` is the transfer itself.
    pub(crate) fn download(&self, bytes: usize, waited: Duration, copied: Duration) {
        self.d2h_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.d2h_nanos.fetch_add(nanos(copied), Ordering::Relaxed);
        self.wait_nanos.fetch_add(nanos(waited), Ordering::Relaxed);
    }

    fn snapshot(&self) -> Traffic {
        let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Traffic {
            launches: load(&self.launches),
            allocations: load(&self.allocations),
            h2d_bytes: load(&self.h2d_bytes),
            h2d_time: Duration::from_nanos(load(&self.h2d_nanos)),
            d2h_bytes: load(&self.d2h_bytes),
            d2h_time: Duration::from_nanos(load(&self.d2h_nanos)),
            sync_wait: Duration::from_nanos(load(&self.wait_nanos)),
        }
    }
}

/// What the GPU path did so far: kernel launches, device allocations and host/device copies.
/// `sync_wait` is time the host spent waiting for queued GPU work before a download could start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Traffic {
    pub launches: u64,
    pub allocations: u64,
    pub h2d_bytes: u64,
    pub h2d_time: Duration,
    pub d2h_bytes: u64,
    pub d2h_time: Duration,
    pub sync_wait: Duration,
}

impl Traffic {
    /// The difference between two snapshots, `self` being the later one.
    pub fn since(&self, earlier: &Traffic) -> Traffic {
        Traffic {
            launches: self.launches.saturating_sub(earlier.launches),
            allocations: self.allocations.saturating_sub(earlier.allocations),
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            h2d_time: self.h2d_time.saturating_sub(earlier.h2d_time),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            d2h_time: self.d2h_time.saturating_sub(earlier.d2h_time),
            sync_wait: self.sync_wait.saturating_sub(earlier.sync_wait),
        }
    }
}

/// A point on the GPU's own clock, recorded in stream order.
pub struct Mark(CudaEvent);

impl Mark {
    /// GPU time between `earlier` and this mark.
    pub fn since(&self, earlier: &Mark) -> Result<Duration, CudaError> {
        let millis = earlier.0.elapsed_ms(&self.0)?;
        Ok(Duration::from_secs_f64(f64::from(millis.max(0.0)) / 1000.0))
    }
}

impl Gpu {
    pub fn traffic(&self) -> Traffic {
        self.counters.snapshot()
    }

    pub fn mark(&self) -> Result<Mark, CudaError> {
        Ok(Mark(
            self.stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_subtract_without_underflow() {
        let counters = Counters::default();
        let before = counters.snapshot();
        counters.launch();
        counters.launch();
        counters.allocation();
        counters.download(8, Duration::from_micros(5), Duration::from_micros(2));
        let delta = counters.snapshot().since(&before);
        assert_eq!(
            (delta.launches, delta.allocations, delta.d2h_bytes),
            (2, 1, 8)
        );
        assert_eq!(delta.sync_wait, Duration::from_micros(5));
        assert_eq!(before.since(&counters.snapshot()), Traffic::default());
    }
}
