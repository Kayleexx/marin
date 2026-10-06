use super::code::Code;
use super::stream::MarkdownStream;
use super::syntax::Lang;
use super::theme::{Theme, looks};
use super::wrap::Wrap;

/// What a piece of text is, and so how it is shown. Unknown things fall back to plain text, never to an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    Markdown,
    Code(Lang),
    Data(Lang),
    Diff,
    /// Rows split at this character: comma or tab.
    Delimited(char),
    Log,
    PlainText,
    /// Text nothing recognized.
    Unknown,
    /// Not text, or not valid text: never printed.
    Binary,
}

impl ContentKind {
    /// From the language word of a code fence. Markdown inside a fence is shown as the source it is.
    pub fn from_hint(hint: &str) -> Self {
        match hint.trim().to_ascii_lowercase().as_str() {
            "" | "text" | "txt" | "plain" | "plaintext" | "markdown" | "md" => Self::PlainText,
            "rust" | "rs" => Self::Code(Lang::Rust),
            "python" | "py" | "python3" => Self::Code(Lang::Python),
            "js" | "javascript" | "ts" | "typescript" | "jsx" | "tsx" | "node" => {
                Self::Code(Lang::Script)
            }
            "c" | "cpp" | "c++" | "cc" | "h" | "hpp" | "java" | "go" | "golang" | "cs"
            | "csharp" | "kotlin" | "swift" => Self::Code(Lang::CLike),
            "sh" | "bash" | "zsh" | "shell" | "console" | "fish" => Self::Code(Lang::Shell),
            "json" | "jsonc" => Self::Data(Lang::Json),
            "toml" => Self::Data(Lang::Toml),
            "yaml" | "yml" => Self::Data(Lang::Yaml),
            "diff" | "patch" => Self::Diff,
            "csv" => Self::Delimited(','),
            "tsv" => Self::Delimited('\t'),
            "log" | "logs" => Self::Log,
            _ => Self::Code(Lang::Other),
        }
    }

    /// From a name typed by a person (`--as`): `markdown` means markdown here, unlike inside a fence.
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "markdown" | "md" => Self::Markdown,
            "binary" => Self::Unknown,
            other => Self::from_hint(other),
        }
    }

    pub(super) fn from_extension(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        let ext = lower
            .rsplit_once('.')
            .map_or(lower.as_str(), |(_, ext)| ext);
        Some(match ext {
            "md" | "markdown" | "mdx" => Self::Markdown,
            "txt" | "text" => Self::PlainText,
            "log" => Self::Log,
            "rs" | "py" | "pyw" | "js" | "mjs" | "cjs" | "jsx" | "ts" | "tsx" | "c" | "h"
            | "cc" | "cpp" | "hpp" | "cxx" | "java" | "go" | "cs" | "kt" | "swift" | "sh"
            | "bash" | "zsh" | "fish" | "json" | "jsonc" | "toml" | "yaml" | "yml" | "diff"
            | "patch" | "csv" | "tsv" => Self::from_hint(ext),
            "rb" | "lua" | "php" | "sql" | "html" | "css" | "xml" | "pl" | "hs" | "ex" | "exs"
            | "zig" | "dart" | "scala" | "vue" | "svelte" | "makefile" | "dockerfile" => {
                Self::Code(Lang::Other)
            }
            _ => return None,
        })
    }

    /// Short name for headers.
    pub fn name(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Code(Lang::Rust) => "rust",
            Self::Code(Lang::Python) => "python",
            Self::Code(Lang::Script) => "javascript",
            Self::Code(Lang::CLike) => "c-like code",
            Self::Code(Lang::Shell) => "shell",
            Self::Code(_) => "code",
            Self::Data(Lang::Json) => "json",
            Self::Data(Lang::Toml) => "toml",
            Self::Data(_) => "yaml",
            Self::Diff => "diff",
            Self::Delimited('\t') => "tsv",
            Self::Delimited(_) => "csv",
            Self::Log => "log",
            Self::PlainText => "text",
            Self::Unknown => "unknown text",
            Self::Binary => "binary",
        }
    }
}

struct CodeDoc {
    code: Code,
    wrap: Wrap,
}

enum Inner {
    Markdown(Box<MarkdownStream>),
    Code(Box<CodeDoc>),
    Silent,
}

/// Shows text of any detected kind as it arrives. Output is append-only, so it can follow a stream.
pub struct Renderer {
    inner: Inner,
}

impl Renderer {
    pub fn new(kind: ContentKind, theme: Theme, width: usize) -> Self {
        let inner = match kind {
            ContentKind::Markdown => {
                Inner::Markdown(Box::new(MarkdownStream::new(theme, width, 0, false)))
            }
            ContentKind::Binary => Inner::Silent,
            kind => {
                let mut wrap = Wrap::new(theme, width, 0, false);
                let gutter = matches!(kind, ContentKind::Code(_) | ContentKind::Data(_));
                let mut code = Code::new(kind, kind.name(), false, gutter);
                code.open(&mut wrap);
                Inner::Code(Box::new(CodeDoc { code, wrap }))
            }
        };
        Self { inner }
    }

    /// A line saying `text` was not shown, for content that cannot be.
    pub fn notice(theme: Theme, text: &str) -> String {
        format!("{}\n", theme.paint(&looks::DIM, text))
    }

    pub fn feed(&mut self, text: &str) {
        match &mut self.inner {
            Inner::Markdown(stream) => stream.feed(text),
            Inner::Code(doc) => text.chars().for_each(|c| doc.code.push(c, &mut doc.wrap)),
            Inner::Silent => {}
        }
    }

    pub fn finish(&mut self) {
        match &mut self.inner {
            Inner::Markdown(stream) => stream.finish(),
            Inner::Code(doc) => {
                doc.code.finish(&mut doc.wrap);
                doc.wrap.finish();
            }
            Inner::Silent => {}
        }
    }

    pub fn take(&mut self) -> String {
        match &mut self.inner {
            Inner::Markdown(stream) => stream.take(),
            Inner::Code(doc) => doc.wrap.take(),
            Inner::Silent => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_renders_something_sane_in_a_narrow_terminal() {
        let cases = [
            (ContentKind::Markdown, "# T\n\ntext **b** and `c`\n"),
            (ContentKind::Code(Lang::Rust), "fn main() {}\n"),
            (ContentKind::Data(Lang::Json), "{\"a\": 1}\n"),
            (ContentKind::Diff, "@@ -1 +1 @@\n-a\n+b\n"),
            (ContentKind::Delimited(','), "a,b\n1,2\n"),
            (ContentKind::Log, "ERROR x\n"),
            (ContentKind::PlainText, "hello world\n"),
            (ContentKind::Unknown, "???\n"),
        ];
        for (kind, text) in cases {
            let mut r = Renderer::new(kind, Theme::plain(), 20);
            r.feed(text);
            r.finish();
            let out = r.take();
            assert!(!out.is_empty(), "{kind:?}");
            assert!(
                out.lines().all(|l| l.chars().count() <= 20),
                "{kind:?}: {out:?}"
            );
        }
        let mut notice = Renderer::new(ContentKind::Binary, Theme::plain(), 20);
        notice.feed("anything");
        notice.finish();
        assert_eq!(notice.take(), "");
        assert_eq!(
            Renderer::notice(Theme::plain(), "binary, 3 bytes"),
            "binary, 3 bytes\n"
        );
    }
}
