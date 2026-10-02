use std::io::Write;
use std::time::Instant;

use crate::style::{Style, stderr_is_terminal, terminal_width};
use crate::units;

const BAR_WIDTH: usize = 20;
const MIN_WIDTH_FOR_BAR: usize = 60;

/// Weight-loading progress on stderr: a bar on terminals, one line otherwise.
pub struct Progress {
    style: Style,
    interactive: bool,
    started: Instant,
    last_percent: Option<u64>,
    drew: bool,
}

impl Progress {
    pub fn new(style: Style, enabled: bool) -> Self {
        Self {
            style,
            interactive: enabled && stderr_is_terminal(),
            started: Instant::now(),
            last_percent: None,
            drew: false,
        }
    }

    pub fn update(&mut self, loaded: u64, total: u64) {
        if !self.interactive || total == 0 {
            return;
        }
        let percent = loaded
            .saturating_mul(100)
            .checked_div(total)
            .unwrap_or(0)
            .min(100);
        if self.last_percent == Some(percent) {
            return;
        }
        self.last_percent = Some(percent);
        let filled = (percent as f64 / 100.0 * BAR_WIDTH as f64) as usize;
        let bar = if terminal_width() >= MIN_WIDTH_FOR_BAR {
            let done = "█".repeat(filled);
            let todo = "░".repeat(BAR_WIDTH.saturating_sub(filled));
            format!("{} {done}{todo} ", self.style.dim("loading"))
        } else {
            format!("{} ", self.style.dim("loading"))
        };
        let sizes = format!("{} / {}", units::bytes(loaded), units::bytes(total));
        let mut err = std::io::stderr();
        let _ = write!(err, "\r {bar}{sizes}   ");
        let _ = err.flush();
        self.drew = true;
    }

    pub fn finish(&self) {
        if self.drew {
            let _ = write!(
                std::io::stderr(),
                "\r{}\r",
                " ".repeat(terminal_width().min(79))
            );
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.started.elapsed()
    }
}
