use super::content::ContentKind;
use super::lexer::{Class, Lexer, Piece};
use super::syntax::Lang;
use super::table::{self, Align, Table};
use super::theme::{Look, looks};
use super::wrap::{Prefix, Wrap};
use crate::term::screen::{display_width, fit};

enum Mode {
    Highlight(Lexer),
    Plain,
    Diff,
    Log,
    Table(char),
}

/// Preformatted text: source, data, diffs, logs, delimited rows or plain text. Characters are written as they
/// arrive, long lines break at the width, and an optional gutter and frame set the block apart from prose.
pub struct Code {
    mode: Mode,
    label: String,
    frame: bool,
    gutter: bool,
    look: Option<Look>,
    line: String,
    rows: Vec<String>,
}

fn class_look(class: Class) -> Look {
    match class {
        Class::Plain => Look::NONE,
        Class::Keyword => looks::KEYWORD,
        Class::Str => looks::STRING,
        Class::Number => looks::NUMBER,
        Class::Comment => looks::COMMENT,
        Class::Type => looks::TYPE,
    }
}

fn diff_look(first: char) -> Look {
    match first {
        '+' => looks::ADDED,
        '-' => looks::REMOVED,
        '@' => looks::HUNK,
        ' ' => Look::NONE,
        _ => looks::META,
    }
}

fn log_look(line: &str) -> Look {
    let lower = line.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.contains(w));
    if has(&["error", "fatal", "panic", "critical"]) {
        looks::ERROR
    } else if has(&["warn"]) {
        looks::WARN
    } else if has(&["debug", "trace"]) {
        looks::DIM
    } else {
        Look::NONE
    }
}

/// Fields of one delimited row; double quotes may hold the delimiter and `""` is a quote.
fn fields(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = vec![String::new()];
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                if let Some(f) = fields.last_mut() {
                    f.push('"');
                }
            }
            '"' => quoted = !quoted,
            c if c == delimiter && !quoted => fields.push(String::new()),
            c => {
                if let Some(f) = fields.last_mut() {
                    f.push(c);
                }
            }
        }
    }
    fields.into_iter().map(|f| f.trim().to_string()).collect()
}

impl Code {
    pub fn new(kind: ContentKind, label: &str, frame: bool, gutter: bool) -> Self {
        let mode = match kind {
            ContentKind::Code(lang) | ContentKind::Data(lang) if lang != Lang::Other => {
                Mode::Highlight(Lexer::new(lang))
            }
            ContentKind::Diff => Mode::Diff,
            ContentKind::Log => Mode::Log,
            ContentKind::Delimited(d) => Mode::Table(d),
            _ => Mode::Plain,
        };
        let is_table = matches!(mode, Mode::Table(_));
        Self {
            mode,
            label: label.to_string(),
            frame: frame && !is_table,
            gutter: gutter && !is_table,
            look: None,
            line: String::new(),
            rows: Vec::new(),
        }
    }

    fn prefixes(&self, wrap: &Wrap) -> (Prefix, Prefix) {
        if !self.gutter {
            return (Prefix::default(), Prefix::default());
        }
        let theme = wrap.theme();
        let bar = theme.paint(&looks::DIM, theme.frame().vertical);
        (
            Prefix {
                text: format!("{bar} "),
                width: 2,
            },
            Prefix {
                text: format!("{bar}   "),
                width: 4,
            },
        )
    }

    fn begin_line(&self, wrap: &mut Wrap) {
        let (first, cont) = self.prefixes(wrap);
        wrap.begin(first, cont);
    }

    fn edge(&self, wrap: &mut Wrap, open: bool) {
        let theme = wrap.theme();
        let (corner, rule) = match (theme.unicode, open) {
            (true, true) => ("╭", "─"),
            (true, false) => ("╰", "─"),
            (false, _) => ("+", "-"),
        };
        let label = if open && !self.label.is_empty() {
            format!(" {}", self.label)
        } else {
            String::new()
        };
        let text = fit(&format!("{corner}{rule}{label}"), wrap.room(0));
        let width = display_width(&text);
        wrap.line(&theme.paint(&looks::DIM, &text), width);
    }

    /// Writes the header and gets ready for the first line.
    pub fn open(&mut self, wrap: &mut Wrap) {
        if self.frame {
            self.edge(wrap, true);
        }
        if !matches!(self.mode, Mode::Table(_)) {
            self.begin_line(wrap);
        }
    }

    pub fn push(&mut self, c: char, wrap: &mut Wrap) {
        match c {
            '\r' => {}
            '\n' => self.newline(wrap),
            '\t' => {
                for _ in 0..4 {
                    self.character(' ', wrap);
                }
            }
            c => self.character(c, wrap),
        }
    }

    fn write(&mut self, pieces: Vec<Piece>, wrap: &mut Wrap) {
        for (class, text) in pieces {
            let look = class_look(class);
            for c in text.chars() {
                wrap.put(look, c);
            }
        }
    }

    fn character(&mut self, c: char, wrap: &mut Wrap) {
        let mut pieces = Vec::new();
        match &mut self.mode {
            Mode::Highlight(lexer) => lexer.push(c, &mut pieces),
            Mode::Plain => wrap.put(Look::NONE, c),
            Mode::Diff => {
                let look = *self.look.get_or_insert_with(|| diff_look(c));
                wrap.put(look, c);
            }
            Mode::Log | Mode::Table(_) => self.line.push(c),
        }
        self.write(pieces, wrap);
    }

    fn newline(&mut self, wrap: &mut Wrap) {
        let mut pieces = Vec::new();
        match &mut self.mode {
            Mode::Highlight(lexer) => lexer.newline(&mut pieces),
            Mode::Diff => self.look = None,
            Mode::Log => self.flush_log(wrap),
            Mode::Table(_) => {
                let row = std::mem::take(&mut self.line);
                if !row.trim().is_empty() {
                    self.rows.push(row);
                }
                return;
            }
            Mode::Plain => {}
        }
        self.write(pieces, wrap);
        wrap.end_line_prefixed();
        self.begin_line(wrap);
    }

    fn flush_log(&mut self, wrap: &mut Wrap) {
        let line = std::mem::take(&mut self.line);
        let look = log_look(&line);
        for c in line.chars() {
            wrap.put(look, c);
        }
    }

    /// Writes whatever is held, closes the frame, and for delimited rows lays out the table.
    pub fn finish(&mut self, wrap: &mut Wrap) {
        let mut pieces = Vec::new();
        match &mut self.mode {
            Mode::Highlight(lexer) => lexer.finish(&mut pieces),
            Mode::Log => self.flush_log(wrap),
            Mode::Table(_) => {
                let row = std::mem::take(&mut self.line);
                if !row.trim().is_empty() {
                    self.rows.push(row);
                }
            }
            Mode::Plain | Mode::Diff => {}
        }
        self.write(pieces, wrap);
        wrap.end_line();
        if let Mode::Table(delimiter) = self.mode {
            self.table(delimiter, wrap);
        }
        if self.frame {
            self.edge(wrap, false);
        }
    }

    fn table(&mut self, delimiter: char, wrap: &mut Wrap) {
        let mut rows = self.rows.iter().map(|r| fields(r, delimiter));
        let Some(header) = rows.next() else {
            return;
        };
        let body: Vec<Vec<String>> = rows.collect();
        let columns = body
            .iter()
            .map(Vec::len)
            .chain([header.len()])
            .max()
            .unwrap_or(1);
        let mut header = header;
        header.resize(columns, String::new());
        let data = Table {
            aligns: vec![Align::Left; columns],
            header,
            rows: body,
        };
        let room = wrap.room(0);
        let lines = table::layout(&data, room, &wrap.theme())
            .unwrap_or_else(|| table::stacked(&data, room));
        for (text, width) in lines {
            if text.is_empty() {
                wrap.blank_line();
            } else {
                wrap.line(&text, width);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::theme::Theme;

    fn render(kind: ContentKind, text: &str, width: usize, frame: bool) -> String {
        let mut wrap = Wrap::new(Theme::plain(), width, 0, false);
        let mut code = Code::new(kind, "x", frame, true);
        code.open(&mut wrap);
        for c in text.chars() {
            code.push(c, &mut wrap);
        }
        code.finish(&mut wrap);
        wrap.take()
    }

    #[test]
    fn lines_get_a_gutter_and_an_optional_frame() {
        let out = render(ContentKind::Code(Lang::Other), "a\n\nb\n", 20, true);
        assert_eq!(out, "+- x\n| a\n|\n| b\n+-\n");
    }

    #[test]
    fn a_long_language_label_is_cut_to_fit_the_header_line() {
        let mut wrap = Wrap::new(Theme::plain(), 20, 8, false);
        let mut code = Code::new(
            ContentKind::PlainText,
            "averyveryverylonglanguagenamethatkeepsgoing",
            true,
            true,
        );
        code.open(&mut wrap);
        code.finish(&mut wrap);
        let out = wrap.take();
        assert!(out.lines().all(|l| l.chars().count() <= 20), "{out:?}");
        assert!(out.contains('…'), "{out:?}");
    }

    #[test]
    fn long_lines_break_at_the_width_with_an_indented_continuation() {
        let out = render(
            ContentKind::Code(Lang::Other),
            "abcdefghijklmnop\n",
            10,
            false,
        );
        assert_eq!(out, "| abcdefgh\n|   ijklmn\n|   op\n");
        assert!(out.lines().all(|l| l.chars().count() <= 10));
    }

    #[test]
    fn text_without_a_final_newline_is_still_written_once() {
        assert_eq!(
            render(ContentKind::Code(Lang::Other), "last", 20, false),
            "| last\n"
        );
        assert_eq!(render(ContentKind::Code(Lang::Other), "", 20, false), "");
    }

    #[test]
    fn tabs_become_spaces_and_carriage_returns_vanish() {
        assert_eq!(
            render(ContentKind::PlainText, "a\tb\r\n", 20, false),
            "| a    b\n"
        );
    }

    #[test]
    fn delimited_rows_become_a_table_and_quotes_hold_delimiters() {
        let out = render(
            ContentKind::Delimited(','),
            "name,note\nann,\"a, b\"\n",
            40,
            true,
        );
        assert!(out.contains("| name | note |"), "{out}");
        assert!(out.contains("| ann  | a, b |"), "{out}");
        assert!(!out.contains("+- x"), "tables are not framed");
        assert_eq!(fields("a,\"b\"\"c\",d", ','), ["a", "b\"c", "d"]);
    }

    #[test]
    fn diff_and_log_lines_are_styled_by_their_first_characters() {
        let theme = Theme {
            color: true,
            unicode: true,
        };
        let mut wrap = Wrap::new(theme, 40, 0, false);
        let mut code = Code::new(ContentKind::Diff, "diff", false, false);
        code.open(&mut wrap);
        for c in "+new\n-old\n same\n".chars() {
            code.push(c, &mut wrap);
        }
        code.finish(&mut wrap);
        let out = wrap.take();
        assert!(out.contains("\x1b[32m+new\x1b[0m"), "{out:?}");
        assert!(out.contains("\x1b[31m-old\x1b[0m"), "{out:?}");
        assert!(out.contains(" same\n"), "{out:?}");
        assert_eq!(log_look("2024 ERROR boom"), looks::ERROR);
        assert_eq!(log_look("WARN slow"), looks::WARN);
        assert_eq!(log_look("hello"), Look::NONE);
    }
}
