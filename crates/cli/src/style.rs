use std::env;
use std::io::IsTerminal;

/// Terminal colors, switched off for pipes, `NO_COLOR`, `TERM=dumb` and `--no-color`.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    color: bool,
}

impl Style {
    pub fn detect(no_color_flag: bool) -> Self {
        let disabled = no_color_flag
            || env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty())
            || env::var("TERM").is_ok_and(|t| t == "dumb");
        Self {
            color: !disabled && std::io::stderr().is_terminal(),
        }
    }

    pub fn plain() -> Self {
        Self { color: false }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }

    pub fn bold(&self, text: &str) -> String {
        self.paint("1", text)
    }

    pub fn dim(&self, text: &str) -> String {
        self.paint("2", text)
    }

    pub fn red(&self, text: &str) -> String {
        self.paint("1;31", text)
    }

    pub fn green(&self, text: &str) -> String {
        self.paint("32", text)
    }

    pub fn yellow(&self, text: &str) -> String {
        self.paint("33", text)
    }

    /// The user's role color: a soft periwinkle that sits next to Marin's cyan in the cool palette.
    pub fn periwinkle(&self, text: &str) -> String {
        self.paint("38;5;147", text)
    }

    pub fn cyan(&self, text: &str) -> String {
        self.paint("36", text)
    }
}

pub fn stderr_is_terminal() -> bool {
    std::io::stderr().is_terminal()
}

pub fn stdin_is_terminal() -> bool {
    std::io::stdin().is_terminal()
}

pub fn terminal_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse().ok())
        .filter(|w| *w > 0)
        .unwrap_or(80)
}
