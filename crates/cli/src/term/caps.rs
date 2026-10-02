/// How much color the terminal can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Truecolor,
    Ansi256,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    pub color: Color,
    pub unicode: bool,
    /// No cursor control at all: plain lines only.
    pub dumb: bool,
}

const RESET: &str = "\x1b[0m";

impl Caps {
    pub fn detect(no_color_flag: bool) -> Self {
        Self::from_env(|name| std::env::var(name).ok(), no_color_flag)
    }

    pub fn from_env(var: impl Fn(&str) -> Option<String>, no_color_flag: bool) -> Self {
        let get = |name: &str| var(name).filter(|v| !v.is_empty());
        let term = get("TERM").unwrap_or_default();
        let dumb = term.is_empty() || term == "dumb";
        let colorterm = get("COLORTERM").unwrap_or_default().to_lowercase();
        let color = if no_color_flag || get("NO_COLOR").is_some() || dumb || term == "linux" {
            Color::None
        } else if colorterm.contains("truecolor") || colorterm.contains("24bit") {
            Color::Truecolor
        } else {
            Color::Ansi256
        };
        let locale = get("LC_ALL")
            .or_else(|| get("LC_CTYPE"))
            .or_else(|| get("LANG"))
            .unwrap_or_default();
        let locale = locale.to_lowercase();
        Self {
            color,
            unicode: locale.contains("utf-8") || locale.contains("utf8"),
            dumb,
        }
    }

    fn sequence(self, base: u16, rgb: [u8; 3]) -> String {
        match self.color {
            Color::None => String::new(),
            Color::Truecolor => format!("\x1b[{base};2;{};{};{}m", rgb[0], rgb[1], rgb[2]),
            Color::Ansi256 => format!("\x1b[{base};5;{}m", ansi256(rgb)),
        }
    }

    pub fn fg(self, rgb: [u8; 3]) -> String {
        self.sequence(38, rgb)
    }

    pub fn bg(self, rgb: [u8; 3]) -> String {
        self.sequence(48, rgb)
    }

    /// Back to the terminal's own background color.
    pub fn default_bg(self) -> &'static str {
        if self.color == Color::None {
            ""
        } else {
            "\x1b[49m"
        }
    }

    pub fn reset(self) -> &'static str {
        if self.color == Color::None { "" } else { RESET }
    }

    pub fn paint(self, rgb: [u8; 3], text: &str) -> String {
        if self.color == Color::None {
            text.to_string()
        } else {
            format!("{}{text}{RESET}", self.fg(rgb))
        }
    }
}

/// Nearest xterm-256 palette entry: the gray ramp for near-grays, else the 6x6x6 cube.
fn ansi256(rgb: [u8; 3]) -> u8 {
    let [r, g, b] = rgb.map(u16::from);
    let spread = r.max(g).max(b).saturating_sub(r.min(g).min(b));
    if spread < 10 {
        let gray = (r.saturating_add(g).saturating_add(b)) / 3;
        return match gray {
            0..=7 => 16,
            248..=255 => 231,
            _ => u8::try_from(232_u16.saturating_add(gray.saturating_sub(8) / 10)).unwrap_or(255),
        };
    }
    let level = |v: u16| v.saturating_mul(5).saturating_add(127) / 255;
    let index = 16_u16
        .saturating_add(level(r).saturating_mul(36))
        .saturating_add(level(g).saturating_mul(6))
        .saturating_add(level(b));
    u8::try_from(index).unwrap_or(231)
}

#[cfg(test)]
pub(super) fn strip_ansi(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(vars: &[(&str, &str)], flag: bool) -> Caps {
        Caps::from_env(
            |name| {
                vars.iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_string())
            },
            flag,
        )
    }

    #[test]
    fn color_level_follows_colorterm_then_term() {
        let base = [("TERM", "xterm-256color"), ("LANG", "en_US.UTF-8")];
        assert_eq!(caps(&base, false).color, Color::Ansi256);
        let truecolor = [("TERM", "xterm"), ("COLORTERM", "truecolor")];
        assert_eq!(caps(&truecolor, false).color, Color::Truecolor);
        let bits = [("TERM", "xterm"), ("COLORTERM", "24bit")];
        assert_eq!(caps(&bits, false).color, Color::Truecolor);
    }

    #[test]
    fn no_color_dumb_and_console_terminals_get_no_color() {
        let truecolor = [("TERM", "xterm"), ("COLORTERM", "truecolor")];
        assert_eq!(
            caps(&[("NO_COLOR", "1"), ("TERM", "xterm")], false).color,
            Color::None
        );
        assert_eq!(caps(&truecolor, true).color, Color::None, "--no-color flag");
        assert_eq!(
            caps(&[("NO_COLOR", ""), ("TERM", "xterm")], false).color,
            Color::Ansi256,
            "empty NO_COLOR is ignored"
        );
        assert!(caps(&[("TERM", "dumb")], false).dumb);
        assert!(caps(&[], false).dumb);
        assert_eq!(caps(&[("TERM", "linux")], false).color, Color::None);
    }

    #[test]
    fn unicode_comes_from_the_locale() {
        assert!(caps(&[("TERM", "xterm"), ("LANG", "en_US.UTF-8")], false).unicode);
        assert!(
            caps(
                &[("TERM", "xterm"), ("LC_ALL", "C.utf8"), ("LANG", "C")],
                false
            )
            .unicode
        );
        assert!(!caps(&[("TERM", "xterm"), ("LANG", "C")], false).unicode);
        assert!(!caps(&[("TERM", "xterm")], false).unicode);
    }

    #[test]
    fn escape_sequences_match_the_color_level() {
        let t = caps(&[("TERM", "xterm"), ("COLORTERM", "truecolor")], false);
        assert_eq!(t.fg([1, 2, 3]), "\x1b[38;2;1;2;3m");
        assert_eq!(t.bg([4, 5, 6]), "\x1b[48;2;4;5;6m");
        let p = caps(&[("TERM", "xterm-256color")], false);
        assert_eq!(p.fg([255, 0, 0]), "\x1b[38;5;196m");
        assert_eq!(p.fg([128, 128, 128]), "\x1b[38;5;244m");
        let none = caps(&[("TERM", "xterm"), ("NO_COLOR", "1")], false);
        assert_eq!(
            (none.fg([1, 2, 3]), none.reset(), none.paint([1, 2, 3], "x")),
            (String::new(), "", "x".to_string())
        );
    }
}
