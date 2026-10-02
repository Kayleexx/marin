use std::io::{self, Write};

use crossterm::cursor::{MoveToColumn, MoveUp};
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal::{self, Clear, ClearType};

use unicode_width::UnicodeWidthStr;

use super::menu::Menu;
use crate::style::Style;

const YOU: &str = "you >";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Dim,
    Strong,
    Accent,
    Error,
    /// An input line: the leading `you >` is colored, the typed text is not.
    Prompt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    pub text: String,
    pub kind: Kind,
    /// Subtle trailing text, always dim.
    pub detail: Option<String>,
}

impl Line {
    pub fn new(kind: Kind, text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind,
            detail: None,
        }
    }

    pub fn with_detail(kind: Kind, text: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            detail: Some(detail.into()),
            ..Self::new(kind, text)
        }
    }
}

/// Terminal columns used by `text` (wide characters count twice, combining marks zero).
pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Pads with spaces to `columns`, measuring display width rather than characters.
pub fn pad(text: &str, columns: usize) -> String {
    format!(
        "{text}{}",
        " ".repeat(columns.saturating_sub(display_width(text)))
    )
}

/// Terminal width; some terminals report 0 columns, so fall back to `COLUMNS`, then 80.
pub fn width() -> usize {
    let reported = terminal::size().map_or(0, |(w, _)| usize::from(w));
    let columns = || std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok());
    (if reported > 0 {
        reported
    } else {
        columns().unwrap_or(80)
    })
    .max(20)
}

/// Terminal height; falls back to `LINES`, then 24.
pub fn height() -> usize {
    let reported = terminal::size().map_or(0, |(_, h)| usize::from(h));
    let lines = || std::env::var("LINES").ok().and_then(|c| c.parse().ok());
    if reported > 0 {
        reported
    } else {
        lines().unwrap_or(24)
    }
}

/// Cuts `text` to `max` terminal columns, marking the cut with an ellipsis. Widths are measured on the
/// growing string, because the width of an emoji sequence is not the sum of its characters.
pub fn fit(text: &str, max: usize) -> String {
    if display_width(text) <= max {
        return text.to_string();
    }
    let keep = max.saturating_sub(1);
    let mut out = String::new();
    for c in text.chars() {
        out.push(c);
        if display_width(&out) > keep {
            out.pop();
            break;
        }
    }
    out.push('…');
    out
}

/// Renders blocks of lines under the cursor without taking over the screen.
/// Between calls the cursor always rests on the first line of the block.
pub struct Screen {
    style: Style,
    /// Blocks start at this column, leaving the startup portrait to their left untouched.
    indent: usize,
}

impl Screen {
    pub fn new(style: Style) -> Self {
        Self { style, indent: 0 }
    }

    pub fn set_indent(&mut self, indent: usize) {
        self.indent = indent;
    }

    pub fn indent(&self) -> usize {
        self.indent
    }

    fn styled(&self, kind: Kind, text: &str) -> String {
        match kind {
            Kind::Plain => text.to_string(),
            Kind::Dim => self.style.dim(text),
            Kind::Strong => self.style.bold(text),
            Kind::Accent => self.style.cyan(text),
            Kind::Error => self.style.red(text),
            Kind::Prompt => match text.strip_prefix(YOU) {
                Some(typed) => format!("{}{typed}", self.style.periwinkle(YOU)),
                None => text.to_string(),
            },
        }
    }

    /// `width` is the number of columns the line may use; callers leave the last column free.
    fn paint(&self, line: &Line, width: usize) -> String {
        let text = fit(&line.text, width);
        let room = width.saturating_sub(display_width(&text));
        let detail = line.detail.as_deref().filter(|_| room > 1);
        let styled = self.styled(line.kind, &text);
        match detail {
            Some(detail) => format!("{styled}{}", self.style.dim(&fit(detail, room))),
            None => styled,
        }
    }

    pub fn draw(&self, lines: &[Line], cursor_col: usize) -> io::Result<()> {
        let width = width();
        let indent = u16::try_from(self.indent).unwrap_or(0);
        let usable = width.saturating_sub(self.indent).saturating_sub(1);
        let mut out = io::stdout().lock();
        queue!(out, MoveToColumn(indent), Clear(ClearType::FromCursorDown))?;
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                queue!(out, Print("\r\n"), MoveToColumn(indent))?;
            }
            queue!(out, Print(self.paint(line, usable)))?;
        }
        let up = u16::try_from(lines.len().saturating_sub(1)).unwrap_or(0);
        if up > 0 {
            queue!(out, MoveUp(up))?;
        }
        let column = u16::try_from(
            self.indent
                .saturating_add(cursor_col)
                .min(width.saturating_sub(1)),
        )
        .unwrap_or(0);
        queue!(out, MoveToColumn(column))?;
        out.flush()
    }

    /// Prints permanent lines where the block was; the next `draw` starts below them.
    pub fn print(&self, lines: &[Line]) -> io::Result<()> {
        let indent = u16::try_from(self.indent).unwrap_or(0);
        let usable = width().saturating_sub(self.indent).saturating_sub(1);
        let mut out = io::stdout().lock();
        queue!(out, MoveToColumn(indent), Clear(ClearType::FromCursorDown))?;
        for line in lines {
            queue!(
                out,
                Print(self.paint(line, usable)),
                Print("\r\n"),
                MoveToColumn(indent)
            )?;
        }
        out.flush()
    }
}

/// Palette or pick-list rows: the selected one is highlighted, descriptions stay dim.
pub fn menu_lines<T>(menu: &Menu<T>, max_rows: usize) -> Vec<Line> {
    let rows = menu.rows(max_rows);
    if rows.is_empty() {
        return vec![Line::new(Kind::Dim, "  no match")];
    }
    rows.iter()
        .map(|row| {
            let marker = if row.selected { ">" } else { " " };
            let kind = if row.selected {
                Kind::Accent
            } else {
                Kind::Plain
            };
            Line::with_detail(
                kind,
                format!("{marker} {}", pad(row.label, LABEL_COLUMNS)),
                row.detail,
            )
        })
        .collect()
}

const LABEL_COLUMNS: usize = 28;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::menu::Item;

    #[test]
    fn fit_cuts_long_text_by_display_width_not_characters() {
        assert_eq!(fit("hello", 10), "hello");
        assert_eq!(fit("hello world", 6), "hello…");
        assert_eq!(fit("héllo wörld", 6), "héllo…");
        assert_eq!(fit("anything", 0), "…");
        assert_eq!(
            fit("你好世界你好", 7),
            "你好世…",
            "wide characters count two columns"
        );
        assert!(display_width(&fit("😀😀😀😀😀", 6)) <= 6);
    }

    #[test]
    fn widths_follow_the_terminal_not_the_character_count() {
        assert_eq!(
            (
                display_width("abc"),
                display_width("你好"),
                display_width("e\u{301}")
            ),
            (3, 4, 1)
        );
        assert_eq!(pad("你好", 6), "你好  ");
        assert_eq!(pad("long text", 3), "long text");
    }

    #[test]
    fn menu_lines_mark_the_selection_and_keep_descriptions_dim() {
        let mut menu = Menu::new(vec![
            Item {
                label: "One".into(),
                detail: "first".into(),
                value: 1,
            },
            Item {
                label: "Two".into(),
                detail: "second".into(),
                value: 2,
            },
        ]);
        let lines = menu_lines(&menu, 5);
        let first = lines.first().unwrap();
        assert!(first.text.starts_with('>') && first.kind == Kind::Accent);
        assert_eq!(first.detail.as_deref(), Some("first"));
        let second = lines.get(1).unwrap();
        assert!(second.text.starts_with(' ') && second.kind == Kind::Plain);
        menu.push('z');
        assert_eq!(menu_lines(&menu, 5), [Line::new(Kind::Dim, "  no match")]);
    }

    #[test]
    fn fit_never_exceeds_the_limit_for_wide_and_joined_characters() {
        for text in [
            "❤️ love you",
            "👨‍👩‍👧👨‍👩‍👧👨‍👩‍👧",
            "你好世界你好世界",
            "🇯🇵🇯🇵🇯🇵",
            "e\u{301}e\u{301}e\u{301}e\u{301}",
        ] {
            for max in 1..14 {
                let cut = fit(text, max);
                assert!(display_width(&cut) <= max, "{text:?} cut to {max}: {cut:?}");
            }
        }
    }
}
