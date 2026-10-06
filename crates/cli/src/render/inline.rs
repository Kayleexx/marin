use super::theme::{Look, looks};
use super::wrap::Wrap;

/// Longest link label or URL that is held back while waiting for the closing bracket.
const LABEL_LIMIT: usize = 150;
const URL_LIMIT: usize = 300;

enum Link {
    Label(String),
    AfterLabel(String),
    Url { label: String, url: String },
}

/// Bold, italic, code spans, links and escapes inside one line of text.
///
/// Characters arrive one at a time. A marker run (`*`, `_`, backticks) is held until the next character shows
/// whether it opens or closes something; a `[` is held until the link is complete or turns out not to be one.
/// Everything resets at the end of the line, so a marker that never closes cannot style the rest of the reply.
pub struct Inline {
    bold: Option<char>,
    italic: Option<char>,
    code: Option<usize>,
    run: Option<(char, usize)>,
    prev: Option<char>,
    link: Option<Link>,
    escape: bool,
    base: Look,
}

fn blank(c: Option<char>) -> bool {
    c.is_none_or(char::is_whitespace)
}

fn punctuation(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_punctuation())
}

impl Inline {
    pub fn new() -> Self {
        Self {
            bold: None,
            italic: None,
            code: None,
            run: None,
            prev: None,
            link: None,
            escape: false,
            base: Look::NONE,
        }
    }

    /// The look every span of the current block starts from, e.g. a heading's.
    pub fn set_base(&mut self, base: Look) {
        self.base = base;
    }

    fn look(&self) -> Look {
        let mut look = self.base;
        if self.bold.is_some() {
            look.bold = true;
        }
        if self.italic.is_some() {
            look.italic = true;
        }
        if self.code.is_some() {
            look = look.merge(looks::CODE);
        }
        look
    }

    fn emit(&self, c: char, wrap: &mut Wrap) {
        wrap.char(self.look(), c);
    }

    fn emit_str(&self, text: &str, wrap: &mut Wrap) {
        for c in text.chars() {
            self.emit(c, wrap);
        }
    }

    pub fn push(&mut self, c: char, wrap: &mut Wrap) {
        if self.escape {
            self.escape = false;
            self.emit(c, wrap);
            self.prev = Some(c);
            return;
        }
        if let Some(link) = self.link.take() {
            self.link_char(link, c, wrap);
            return;
        }
        if let Some((marker, count)) = self.run {
            if c == marker {
                self.run = Some((marker, count.saturating_add(1)));
                return;
            }
            self.run = None;
            self.resolve(marker, count, Some(c), wrap);
        }
        let in_code = self.code.is_some();
        match c {
            '\\' if !in_code => self.escape = true,
            '`' => self.run = Some((c, 1)),
            '*' | '_' if !in_code => self.run = Some((c, 1)),
            '[' if !in_code => self.link = Some(Link::Label(String::new())),
            _ => {
                self.emit(c, wrap);
                self.prev = Some(c);
            }
        }
    }

    fn resolve(&mut self, marker: char, count: usize, next: Option<char>, wrap: &mut Wrap) {
        if marker == '`' {
            self.backticks(count, wrap);
            return;
        }
        let (mut can_open, mut can_close) = (!blank(next), !blank(self.prev));
        if marker == '_' {
            can_open = can_open && (blank(self.prev) || punctuation(self.prev));
            can_close = can_close && (blank(next) || punctuation(next));
        }
        let handled = match count {
            1 => self.toggle(marker, false, can_open, can_close),
            2 => self.toggle(marker, true, can_open, can_close),
            3 => self.toggle_both(marker, can_open, can_close),
            _ => false,
        };
        if !handled {
            self.literal(marker, count, wrap);
        }
        self.prev = Some(marker);
    }

    fn literal(&self, marker: char, count: usize, wrap: &mut Wrap) {
        for _ in 0..count {
            self.emit(marker, wrap);
        }
    }

    fn backticks(&mut self, count: usize, wrap: &mut Wrap) {
        let plain = !wrap.theme().color;
        match self.code {
            Some(open) if open == count => {
                if plain {
                    wrap.char(self.base, '`');
                }
                self.code = None;
            }
            Some(_) => self.literal('`', count, wrap),
            None => {
                self.code = Some(count);
                if plain {
                    wrap.char(self.base, '`');
                }
            }
        }
        self.prev = Some('`');
    }

    fn toggle(&mut self, marker: char, strong: bool, can_open: bool, can_close: bool) -> bool {
        let slot = if strong {
            &mut self.bold
        } else {
            &mut self.italic
        };
        match *slot {
            Some(opener) if opener == marker && can_close => *slot = None,
            None if can_open => *slot = Some(marker),
            _ => return false,
        }
        true
    }

    fn toggle_both(&mut self, marker: char, can_open: bool, can_close: bool) -> bool {
        if self.bold == Some(marker) && self.italic == Some(marker) && can_close {
            self.bold = None;
            self.italic = None;
        } else if self.bold.is_none() && self.italic.is_none() && can_open {
            self.bold = Some(marker);
            self.italic = Some(marker);
        } else {
            return false;
        }
        true
    }

    fn link_char(&mut self, link: Link, c: char, wrap: &mut Wrap) {
        match link {
            Link::Label(mut label) => match c {
                ']' => self.link = Some(Link::AfterLabel(label)),
                _ if label.chars().count() >= LABEL_LIMIT => {
                    self.flush_literal(&format!("[{label}"), wrap);
                    self.push(c, wrap);
                }
                _ => {
                    label.push(c);
                    self.link = Some(Link::Label(label));
                }
            },
            Link::AfterLabel(label) if c == '(' => {
                self.link = Some(Link::Url {
                    label,
                    url: String::new(),
                });
            }
            Link::AfterLabel(label) => {
                self.flush_literal(&format!("[{label}]"), wrap);
                self.push(c, wrap);
            }
            Link::Url { label, mut url } => match c {
                ')' => self.finish_link(&label, &url, wrap),
                _ if c.is_whitespace() || url.chars().count() >= URL_LIMIT => {
                    self.flush_literal(&format!("[{label}]({url}"), wrap);
                    self.push(c, wrap);
                }
                _ => {
                    url.push(c);
                    self.link = Some(Link::Url { label, url });
                }
            },
        }
    }

    fn flush_literal(&mut self, text: &str, wrap: &mut Wrap) {
        self.emit_str(text, wrap);
        self.prev = text.chars().last().or(self.prev);
    }

    fn finish_link(&mut self, label: &str, url: &str, wrap: &mut Wrap) {
        let link = self.look().merge(looks::LINK);
        let text = if label.is_empty() { url } else { label };
        for c in text.chars() {
            wrap.char(link, c);
        }
        if !label.is_empty() && label != url {
            wrap.char(self.look(), ' ');
            for c in format!("({url})").chars() {
                wrap.char(self.look().merge(looks::DIM), c);
            }
        }
        self.prev = Some(')');
    }

    /// The line ended: held markers become literal text and every style is switched off.
    pub fn end_line(&mut self, wrap: &mut Wrap) {
        if self.escape {
            self.emit('\\', wrap);
        }
        match self.link.take() {
            Some(Link::Label(label)) => self.emit_str(&format!("[{label}"), wrap),
            Some(Link::AfterLabel(label)) => self.emit_str(&format!("[{label}]"), wrap),
            Some(Link::Url { label, url }) => self.emit_str(&format!("[{label}]({url}"), wrap),
            None => {}
        }
        if let Some((marker, count)) = self.run.take() {
            self.resolve(marker, count, None, wrap);
        }
        *self = Self {
            base: self.base,
            ..Self::new()
        };
    }
}
