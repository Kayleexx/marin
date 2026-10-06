use crate::term::caps::{Caps, Color};

/// Text attributes for one span. `fg` is a plain SGR color code (31 to 36), 0 for none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Look {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub dim: bool,
    pub fg: u8,
}

impl Look {
    pub const NONE: Look = Look {
        bold: false,
        italic: false,
        underline: false,
        dim: false,
        fg: 0,
    };

    pub const fn bold(self) -> Self {
        Self { bold: true, ..self }
    }

    pub const fn italic(self) -> Self {
        Self {
            italic: true,
            ..self
        }
    }

    pub const fn underline(self) -> Self {
        Self {
            underline: true,
            ..self
        }
    }

    pub const fn dim(self) -> Self {
        Self { dim: true, ..self }
    }

    pub const fn fg(self, code: u8) -> Self {
        Self { fg: code, ..self }
    }

    /// `self` with everything `over` sets added on top.
    pub fn merge(self, over: Look) -> Look {
        Look {
            bold: self.bold || over.bold,
            italic: self.italic || over.italic,
            underline: self.underline || over.underline,
            dim: self.dim || over.dim,
            fg: if over.fg == 0 { self.fg } else { over.fg },
        }
    }
}

const RED: u8 = 31;
const GREEN: u8 = 32;
const YELLOW: u8 = 33;
const MAGENTA: u8 = 35;
const CYAN: u8 = 36;

/// Named looks. Only the 8 basic colors are used, so they follow the terminal's own palette.
pub mod looks {
    use super::*;

    pub const H1: Look = Look::NONE.bold().fg(CYAN);
    pub const H2: Look = Look::NONE.bold();
    pub const H3: Look = Look::NONE.fg(CYAN);
    pub const CODE: Look = Look::NONE.fg(YELLOW);
    pub const LINK: Look = Look::NONE.underline().fg(CYAN);
    pub const DIM: Look = Look::NONE.dim();
    pub const KEYWORD: Look = Look::NONE.fg(MAGENTA);
    pub const STRING: Look = Look::NONE.fg(GREEN);
    pub const NUMBER: Look = Look::NONE.fg(YELLOW);
    pub const COMMENT: Look = Look::NONE.dim();
    pub const TYPE: Look = Look::NONE.fg(CYAN);
    pub const ADDED: Look = Look::NONE.fg(GREEN);
    pub const REMOVED: Look = Look::NONE.fg(RED);
    pub const HUNK: Look = Look::NONE.fg(CYAN);
    pub const META: Look = Look::NONE.bold();
    pub const WARN: Look = Look::NONE.fg(YELLOW);
    pub const ERROR: Look = Look::NONE.fg(RED);
}

/// How much styling and which glyphs the terminal gets: colorless terminals get plain text with ASCII cues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub color: bool,
    pub unicode: bool,
}

/// The characters of a table or code frame.
pub struct Frame {
    pub horizontal: &'static str,
    pub vertical: &'static str,
    pub top: [&'static str; 3],
    pub middle: [&'static str; 3],
    pub bottom: [&'static str; 3],
}

const UNICODE_FRAME: Frame = Frame {
    horizontal: "─",
    vertical: "│",
    top: ["┌", "┬", "┐"],
    middle: ["├", "┼", "┤"],
    bottom: ["└", "┴", "┘"],
};

const ASCII_FRAME: Frame = Frame {
    horizontal: "-",
    vertical: "|",
    top: ["+", "+", "+"],
    middle: ["+", "+", "+"],
    bottom: ["+", "+", "+"],
};

impl Theme {
    pub fn new(caps: &Caps) -> Self {
        Self {
            color: caps.color != Color::None,
            unicode: caps.unicode,
        }
    }

    /// No escape codes and ASCII glyphs only.
    pub fn plain() -> Self {
        Self {
            color: false,
            unicode: false,
        }
    }

    pub fn frame(&self) -> &'static Frame {
        if self.unicode {
            &UNICODE_FRAME
        } else {
            &ASCII_FRAME
        }
    }

    pub fn bullet(&self, depth: usize) -> &'static str {
        match (self.unicode, depth) {
            (false, 0) => "-",
            (false, 1) => "*",
            (false, _) => "+",
            (true, 0) => "•",
            (true, 1) => "◦",
            (true, _) => "▪",
        }
    }

    pub fn open(&self, look: &Look) -> String {
        if !self.color || *look == Look::NONE {
            return String::new();
        }
        let mut codes = Vec::new();
        for (on, code) in [
            (look.bold, 1),
            (look.dim, 2),
            (look.italic, 3),
            (look.underline, 4),
        ] {
            if on {
                codes.push(code.to_string());
            }
        }
        if look.fg != 0 {
            codes.push(look.fg.to_string());
        }
        format!("\x1b[{}m", codes.join(";"))
    }

    pub fn paint(&self, look: &Look, text: &str) -> String {
        let open = self.open(look);
        if open.is_empty() {
            text.to_string()
        } else {
            format!("{open}{text}\x1b[0m")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_combine_and_the_stronger_color_wins() {
        let both = looks::H2.merge(looks::LINK);
        assert!(both.bold && both.underline && both.fg == CYAN);
        assert_eq!(Look::NONE.merge(Look::NONE), Look::NONE);
        assert_eq!(looks::CODE.merge(looks::ERROR).fg, RED);
    }

    #[test]
    fn plain_themes_emit_no_escape_codes_and_colored_ones_close_every_span() {
        let plain = Theme::plain();
        assert_eq!(plain.paint(&looks::H1, "x"), "x");
        let color = Theme {
            color: true,
            unicode: true,
        };
        assert_eq!(color.paint(&looks::H1, "x"), "\x1b[1;36mx\x1b[0m");
        assert_eq!(color.paint(&Look::NONE, "x"), "x");
        assert_eq!(
            color.paint(&Look::NONE.italic().dim(), "x"),
            "\x1b[2;3mx\x1b[0m"
        );
    }

    #[test]
    fn glyphs_fall_back_to_ascii_without_unicode() {
        assert_eq!(Theme::plain().bullet(0), "-");
        assert_eq!(Theme::plain().frame().vertical, "|");
        let color = Theme {
            color: true,
            unicode: true,
        };
        assert_eq!(color.bullet(1), "◦");
        assert_eq!(color.frame().vertical, "│");
    }
}
