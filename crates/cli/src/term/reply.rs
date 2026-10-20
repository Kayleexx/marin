use std::io::{self, Write};

use crossterm::cursor::MoveToColumn;
use crossterm::queue;
use crossterm::terminal::{Clear, ClearType};

use super::caps::Caps;
use super::screen::{display_width, width};
use super::view::REPLY_PREFIX;
use crate::render::{MarkdownStream, MarkdownWriter, Theme};
use crate::style::Style;

/// Introduces the reply with the "marin >" prefix, once: a dim placeholder after it until the first token arrives.
/// The reply itself is rendered as markdown, wrapped to the terminal and aligned under the prefix.
pub(crate) struct ReplyWriter {
    placeholder: bool,
    prefix: String,
    out: io::Stdout,
    markdown: MarkdownWriter<io::Stdout>,
}

impl ReplyWriter {
    /// On a terminal: the prefix and a placeholder. Piped: plain text with neither.
    pub(crate) fn new(style: &Style, caps: &Caps, terminal: bool) -> io::Result<Self> {
        let prefix = if terminal {
            format!("{} ", style.cyan(REPLY_PREFIX.trim_end()))
        } else {
            String::new()
        };
        if terminal {
            let mut out = io::stdout().lock();
            write!(out, "{prefix}{}", style.dim("…"))?;
            out.flush()?;
        }
        let stream = MarkdownStream::new(
            if terminal {
                Theme::new(caps)
            } else {
                Theme::plain()
            },
            width().saturating_sub(1),
            if terminal {
                display_width(REPLY_PREFIX)
            } else {
                0
            },
            terminal,
        );
        Ok(Self {
            placeholder: terminal,
            prefix,
            out: io::stdout(),
            markdown: MarkdownWriter::new(io::stdout(), stream),
        })
    }

    /// The reply is complete or cut short: what is held is written and the cursor rests at a line start.
    pub(crate) fn finish_markdown(&mut self) -> io::Result<()> {
        self.markdown.finish()
    }

    /// Replaces the placeholder line by the prefix alone, or by nothing when an error follows.
    pub(crate) fn finish_placeholder(&mut self, keep_prefix: bool) -> io::Result<()> {
        if self.placeholder {
            self.placeholder = false;
            let mut out = self.out.lock();
            queue!(out, MoveToColumn(0), Clear(ClearType::CurrentLine))?;
            if keep_prefix {
                write!(out, "{}", self.prefix)?;
            }
            out.flush()?;
        }
        Ok(())
    }
}

impl Write for ReplyWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.finish_placeholder(true)?;
        self.markdown.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.markdown.flush()
    }
}
