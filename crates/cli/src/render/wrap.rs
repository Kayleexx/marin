use unicode_width::UnicodeWidthChar;

use super::theme::{Look, Theme, looks};

/// Styled text shown before the first word of a line, and its width in columns.
#[derive(Debug, Clone, Default)]
pub struct Prefix {
    pub text: String,
    pub width: usize,
}

fn columns(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

/// Places words into lines of at most `width` columns, after `margin` blank columns and a per-block prefix.
/// It only ever appends to its output, so a line that has been written is never touched again.
pub struct Wrap {
    theme: Theme,
    width: usize,
    margin: usize,
    out: String,
    col: usize,
    /// Column where text starts on the current line (after margin and prefix).
    base: usize,
    first: Prefix,
    cont: Prefix,
    /// Margin and prefix owed before the next text: `Some(true)` the first-line prefix, `Some(false)` the continuation.
    owed: Option<bool>,
    word: Vec<(Look, String)>,
    word_width: usize,
    space: bool,
    /// The caller already printed `margin` columns of lead-in on the open first line.
    fresh: bool,
    wrote: bool,
    printed: bool,
    blank_last: bool,
    /// Preformatted characters of one look, written out when the look changes or the line ends.
    run: String,
    run_look: Look,
}

impl Wrap {
    pub fn new(theme: Theme, width: usize, margin: usize, midline: bool) -> Self {
        Self {
            theme,
            width: width.max(margin.saturating_add(8)),
            margin,
            out: String::new(),
            col: if midline { margin } else { 0 },
            base: 0,
            first: Prefix::default(),
            cont: Prefix::default(),
            owed: None,
            word: Vec::new(),
            word_width: 0,
            space: false,
            fresh: midline,
            wrote: false,
            printed: false,
            blank_last: false,
            run: String::new(),
            run_look: Look::NONE,
        }
    }

    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// Columns available to text on a line that has `prefix_width` columns of prefix.
    pub fn room(&self, prefix_width: usize) -> usize {
        self.width
            .saturating_sub(self.margin)
            .saturating_sub(prefix_width)
    }

    pub fn take(&mut self) -> String {
        std::mem::take(&mut self.out)
    }

    pub fn printed(&self) -> bool {
        self.printed
    }

    /// Columns of text already on the current line, after its prefix.
    pub fn line_columns(&self) -> usize {
        self.col.saturating_sub(self.base)
    }

    /// Starts a source line: `first` leads it, `cont` leads every wrapped continuation of it.
    pub fn begin(&mut self, first: Prefix, cont: Prefix) {
        self.flush_word();
        self.flush_run();
        if self.wrote {
            self.end_line();
        }
        self.space = false;
        self.first = first;
        self.cont = cont;
        self.owed = Some(true);
    }

    fn start_line(&mut self) {
        let Some(first) = self.owed.take() else {
            return;
        };
        if self.fresh {
            self.fresh = false;
        } else {
            self.out.push_str(&" ".repeat(self.margin));
        }
        let prefix = if first { &self.first } else { &self.cont };
        self.out.push_str(&prefix.text);
        self.col = self.margin.saturating_add(prefix.width);
        self.base = self.col;
    }

    fn flush_run(&mut self) {
        if !self.run.is_empty() {
            self.out
                .push_str(&self.theme.paint(&self.run_look, &self.run));
            self.run.clear();
        }
    }

    fn newline(&mut self) {
        self.flush_run();
        self.out.push('\n');
        self.col = 0;
        self.fresh = false;
        self.owed = Some(false);
        self.wrote = false;
        self.space = false;
    }

    fn write_chars(&mut self, look: Look, text: &str) {
        let mut run = String::new();
        for c in text.chars() {
            let w = columns(c);
            if self.col.saturating_add(w) > self.width && self.col > self.base {
                self.emit(look, &mut run);
                self.newline();
                self.start_line();
            }
            run.push(c);
            self.col = self.col.saturating_add(w);
        }
        self.emit(look, &mut run);
    }

    fn emit(&mut self, look: Look, run: &mut String) {
        if !run.is_empty() {
            self.out.push_str(&self.theme.paint(&look, run));
            run.clear();
        }
    }

    /// Adds a character of running text: whitespace separates words, anything else extends the current word.
    pub fn char(&mut self, look: Look, c: char) {
        if c.is_whitespace() {
            self.flush_word();
            self.space = true;
            return;
        }
        match self.word.last_mut() {
            Some((last, text)) if *last == look => text.push(c),
            _ => self.word.push((look, c.to_string())),
        }
        self.word_width = self.word_width.saturating_add(columns(c));
    }

    /// Writes the word being built; one that does not fit the line moves to the next.
    pub fn flush_word(&mut self) {
        if self.word.is_empty() {
            return;
        }
        self.flush_run();
        let mut spans = std::mem::take(&mut self.word);
        let width = std::mem::take(&mut self.word_width);
        if self.theme.color && is_url(&spans) {
            for (look, _) in &mut spans {
                *look = look.merge(looks::LINK);
            }
        }
        self.start_line();
        let gap = usize::from(self.space && self.col > self.base);
        self.space = false;
        if self.col.saturating_add(gap).saturating_add(width) > self.width && self.col > self.base {
            self.newline();
            self.start_line();
        } else if gap == 1 {
            self.out.push(' ');
            self.col = self.col.saturating_add(1);
        }
        for (look, text) in spans {
            self.write_chars(look, &text);
        }
        self.wrote = true;
        self.printed = true;
        self.blank_last = false;
    }

    /// Writes one character of preformatted text; lines longer than the width break at the width.
    pub fn put(&mut self, look: Look, c: char) {
        self.start_line();
        if look != self.run_look {
            self.flush_run();
            self.run_look = look;
        }
        let w = columns(c);
        if self.col.saturating_add(w) > self.width && self.col > self.base {
            self.newline();
            self.start_line();
            self.run_look = look;
        }
        self.run.push(c);
        self.col = self.col.saturating_add(w);
        self.wrote = true;
        self.printed = true;
        self.blank_last = false;
    }

    /// Writes a finished line of styled text whose display width the caller measured.
    pub fn line(&mut self, styled: &str, width: usize) {
        self.flush_word();
        self.flush_run();
        if self.wrote {
            self.end_line();
        }
        self.owed = Some(true);
        self.first = Prefix::default();
        self.start_line();
        self.out.push_str(styled);
        self.col = self.col.saturating_add(width);
        self.wrote = true;
        self.end_line();
        self.printed = true;
        self.blank_last = false;
    }

    /// Ends the line, keeping a prefix that was owed even if nothing followed it (an empty code line).
    pub fn end_line_prefixed(&mut self) {
        self.flush_word();
        self.flush_run();
        if self.owed.is_some() {
            self.start_line();
            let trimmed = self.out.trim_end_matches(' ').len();
            self.out.truncate(trimmed);
            self.wrote = true;
        }
        self.end_line();
        self.printed = true;
        self.blank_last = false;
    }

    pub fn end_line(&mut self) {
        self.flush_word();
        self.flush_run();
        if self.wrote {
            self.out.push('\n');
            self.fresh = false;
        }
        self.col = if self.fresh { self.margin } else { 0 };
        self.wrote = false;
        self.owed = None;
        self.space = false;
    }

    /// One empty line, unless the output already ends with one or nothing has been printed yet.
    pub fn blank_line(&mut self) {
        self.end_line();
        if self.printed && !self.blank_last {
            self.out.push('\n');
            self.blank_last = true;
        }
    }

    /// Moves a block that must not share the "marin >" line onto its own line.
    pub fn fresh_line(&mut self) {
        self.flush_word();
        self.flush_run();
        if self.fresh {
            self.fresh = false;
            self.out.push('\n');
            self.col = 0;
            self.printed = true;
        }
        self.end_line();
    }

    /// Ends the output at a line start and says whether it left anything on screen.
    pub fn finish(&mut self) {
        self.end_line();
    }
}

fn is_url(spans: &[(Look, String)]) -> bool {
    let text: String = spans.iter().map(|(_, t)| t.as_str()).collect();
    text.starts_with("http://") || text.starts_with("https://")
}

#[cfg(test)]
#[path = "wrap_tests.rs"]
mod tests;
