/// Memory the GPU needs beyond weights and the KV cache: activations and kernel code.
pub const GPU_SLACK_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum GpuFit {
    /// The weights and the whole `--ctx` window fit.
    Window,
    /// The weights and the first KV slab fit, but a conversation that fills the window may not.
    Startup,
    No,
}

/// `startup` and `window` are the bytes of weights plus the cache the GPU needs to start, and to hold the whole window.
pub(super) fn gpu_fit(free: u64, startup: u64, window: u64) -> GpuFit {
    if free < startup.saturating_add(GPU_SLACK_BYTES) {
        GpuFit::No
    } else if free < window.saturating_add(GPU_SLACK_BYTES) {
        GpuFit::Startup
    } else {
        GpuFit::Window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn the_whole_window_does_not_have_to_fit_to_start() {
        assert_eq!(gpu_fit(8 * GB, 3 * GB, 5 * GB), GpuFit::Window);
        assert_eq!(gpu_fit(6 * GB, 3 * GB, 8 * GB), GpuFit::Startup);
        assert_eq!(gpu_fit(2 * GB, 3 * GB, 5 * GB), GpuFit::No);
    }

    #[test]
    fn the_slack_counts_at_both_edges() {
        let (startup, window) = (3 * GB, 5 * GB);
        assert_eq!(
            gpu_fit(startup + GPU_SLACK_BYTES, startup, window),
            GpuFit::Startup
        );
        assert_eq!(
            gpu_fit(startup + GPU_SLACK_BYTES - 1, startup, window),
            GpuFit::No
        );
        assert_eq!(
            gpu_fit(window + GPU_SLACK_BYTES, startup, window),
            GpuFit::Window
        );
        assert_eq!(
            gpu_fit(window + GPU_SLACK_BYTES - 1, startup, window),
            GpuFit::Startup
        );
    }

    #[test]
    fn a_window_that_is_only_the_startup_needs_no_more() {
        let startup = 3 * GB;
        assert_eq!(
            gpu_fit(startup + GPU_SLACK_BYTES, startup, startup),
            GpuFit::Window
        );
        assert_eq!(
            gpu_fit(startup + GPU_SLACK_BYTES - 1, startup, startup),
            GpuFit::No
        );
    }

    #[test]
    fn nothing_overflows() {
        assert_eq!(gpu_fit(u64::MAX, u64::MAX, u64::MAX), GpuFit::Window);
        assert_eq!(gpu_fit(0, u64::MAX, u64::MAX), GpuFit::No);
    }
}
