use std::io::{self, Write};
use std::panic;
use std::sync::Once;

use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

static PANIC_HOOK: Once = Once::new();

/// A panic message printed while the terminal is raw would be garbled and the shell left unusable if the
/// process then aborts, so the terminal is restored before the previous hook reports the panic.
fn restore_on_panic() {
    PANIC_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let _ = execute!(io::stdout(), DisableBracketedPaste);
            let _ = disable_raw_mode();
            previous(info);
        }));
    });
}

/// Raw keyboard mode for the prompt. Dropping it restores the terminal.
pub struct RawMode {
    active: bool,
}

impl RawMode {
    pub fn enable() -> io::Result<Self> {
        restore_on_panic();
        let mut mode = Self { active: false };
        mode.resume()?;
        Ok(mode)
    }

    /// Back to normal line mode, e.g. while a reply streams (so Ctrl+C is a real signal).
    pub fn suspend(&mut self) -> io::Result<()> {
        if self.active {
            execute!(io::stdout(), DisableBracketedPaste)?;
            disable_raw_mode()?;
            self.active = false;
        }
        Ok(())
    }

    pub fn resume(&mut self) -> io::Result<()> {
        if !self.active {
            enable_raw_mode()?;
            execute!(io::stdout(), EnableBracketedPaste)?;
            self.active = true;
        }
        Ok(())
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = self.suspend();
        let _ = io::stdout().flush();
    }
}
