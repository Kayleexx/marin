use std::io::Write;

use crate::style::stderr_is_terminal;

/// One self-erasing progress line on stderr, only when a person is watching it.
pub struct Status {
    live: bool,
}

impl Status {
    pub fn new() -> Self {
        Self {
            live: stderr_is_terminal(),
        }
    }

    pub fn show(&self, text: &str) {
        if self.live {
            eprint!("\r  {text:<40}");
            let _ = std::io::stderr().flush();
        }
    }

    pub fn clear(&self) {
        if self.live {
            eprint!("\r{:<44}\r", "");
            let _ = std::io::stderr().flush();
        }
    }
}
