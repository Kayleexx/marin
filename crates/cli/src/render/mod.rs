//! Terminal-native rendering of text as it streams in: markdown for replies, and a small content-type boundary
//! (`ContentKind`, `detect`, `Renderer`) that routes code, data, diffs, tables, logs and plain text.

mod classify;
mod code;
mod content;
mod detect;
mod inline;
mod lexer;
mod stream;
mod syntax;
mod table;
mod theme;
mod wrap;

#[cfg(test)]
mod tests;

use std::io::{self, Write};

pub use content::{ContentKind, Renderer};
pub use detect::{SAMPLE_BYTES, detect};
pub use stream::MarkdownStream;
pub use syntax::Lang;
pub use theme::Theme;

/// Turns byte chunks into text, holding an incomplete character at the end of a chunk until the rest arrives.
/// Bytes that can never be text become U+FFFD.
#[derive(Default)]
pub struct Utf8Tail {
    bytes: Vec<u8>,
}

impl Utf8Tail {
    pub fn push(&mut self, chunk: &[u8]) -> String {
        self.bytes.extend_from_slice(chunk);
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.bytes) {
                Ok(valid) => {
                    text.push_str(valid);
                    self.bytes.clear();
                    return text;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    text.push_str(
                        self.bytes
                            .get(..good)
                            .and_then(|b| std::str::from_utf8(b).ok())
                            .unwrap_or(""),
                    );
                    match e.error_len() {
                        Some(bad) => {
                            text.push('\u{fffd}');
                            self.bytes
                                .drain(..good.saturating_add(bad).min(self.bytes.len()));
                        }
                        None => {
                            self.bytes.drain(..good);
                            return text;
                        }
                    }
                }
            }
        }
    }

    /// The input ended: a character that never completed is replaced.
    pub fn finish(&mut self) -> String {
        if std::mem::take(&mut self.bytes).is_empty() {
            String::new()
        } else {
            "\u{fffd}".to_string()
        }
    }
}

/// A `Write` that renders the markdown written to it onto `inner`, with nothing held back but what the
/// stream needs to decide how to show it. Call `finish` when the text is complete.
pub struct MarkdownWriter<W: Write> {
    stream: MarkdownStream,
    tail: Utf8Tail,
    inner: W,
}

impl<W: Write> MarkdownWriter<W> {
    pub fn new(inner: W, stream: MarkdownStream) -> Self {
        Self {
            stream,
            tail: Utf8Tail::default(),
            inner,
        }
    }

    fn drain(&mut self) -> io::Result<()> {
        let out = self.stream.take();
        if out.is_empty() {
            return Ok(());
        }
        self.inner.write_all(out.as_bytes())
    }

    /// Writes what is still held, ends on a line start and flushes.
    pub fn finish(&mut self) -> io::Result<()> {
        let rest = self.tail.finish();
        self.stream.feed(&rest);
        self.stream.finish();
        self.drain()?;
        self.inner.flush()
    }
}

impl<W: Write> Write for MarkdownWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = self.tail.push(buf);
        self.stream.feed(&text);
        self.drain()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
