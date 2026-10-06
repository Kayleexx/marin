use super::syntax::{Lang, Syntax};

/// What a stretch of code is, for coloring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Plain,
    Keyword,
    Str,
    Number,
    Comment,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Normal,
    Ident,
    Number,
    /// `run` counts the quotes that opened the string while it may still turn out to be `""` or `"""`.
    Str {
        quote: char,
        run: u8,
        triple: bool,
        escaped: bool,
        closing: u8,
    },
    LineComment,
    BlockComment {
        star: bool,
    },
    /// A `/` that may start `//` or `/*`.
    Slash,
}

/// Colors code one character at a time, so a line shows up as it is typed. Words are held until they end;
/// strings and comments are classified from their first character.
pub struct Lexer {
    syntax: Option<&'static Syntax>,
    state: State,
    word: String,
}

pub type Piece = (Class, String);

impl Lexer {
    pub fn new(lang: Lang) -> Self {
        Self {
            syntax: lang.syntax(),
            state: State::Normal,
            word: String::new(),
        }
    }

    fn put(out: &mut Vec<Piece>, class: Class, c: char) {
        match out.last_mut() {
            Some((last, text)) if *last == class => text.push(c),
            _ => out.push((class, c.to_string())),
        }
    }

    pub fn push(&mut self, c: char, out: &mut Vec<Piece>) {
        let Some(syntax) = self.syntax else {
            Self::put(out, Class::Plain, c);
            return;
        };
        match self.state {
            State::Normal => self.normal(syntax, c, out),
            State::Ident => {
                if c.is_alphanumeric() || c == '_' {
                    self.word.push(c);
                } else {
                    self.end_word(syntax, out);
                    self.normal(syntax, c, out);
                }
            }
            State::Number => {
                if c.is_alphanumeric() || c == '.' || c == '_' {
                    Self::put(out, Class::Number, c);
                } else {
                    self.state = State::Normal;
                    self.normal(syntax, c, out);
                }
            }
            State::Slash => self.after_slash(syntax, c, out),
            State::LineComment => Self::put(out, Class::Comment, c),
            State::BlockComment { star } => {
                Self::put(out, Class::Comment, c);
                self.state = if star && c == '/' {
                    State::Normal
                } else {
                    State::BlockComment { star: c == '*' }
                };
            }
            State::Str { .. } => self.string(syntax, c, out),
        }
    }

    fn normal(&mut self, syntax: &'static Syntax, c: char, out: &mut Vec<Piece>) {
        if c.is_alphabetic() || c == '_' {
            self.state = State::Ident;
            self.word.push(c);
        } else if c.is_ascii_digit() {
            self.state = State::Number;
            Self::put(out, Class::Number, c);
        } else if syntax.quotes.contains(&c) {
            self.state = State::Str {
                quote: c,
                run: 1,
                triple: false,
                escaped: false,
                closing: 0,
            };
            Self::put(out, Class::Str, c);
        } else if c == '/' && (syntax.line_comment == "//" || syntax.block_comment) {
            self.state = State::Slash;
        } else if !syntax.line_comment.is_empty()
            && syntax.line_comment.starts_with(c)
            && syntax.line_comment.len() == 1
        {
            self.state = State::LineComment;
            Self::put(out, Class::Comment, c);
        } else {
            Self::put(out, Class::Plain, c);
        }
    }

    fn after_slash(&mut self, syntax: &'static Syntax, c: char, out: &mut Vec<Piece>) {
        match c {
            '/' if syntax.line_comment == "//" => {
                self.state = State::LineComment;
                Self::put(out, Class::Comment, '/');
                Self::put(out, Class::Comment, c);
            }
            '*' if syntax.block_comment => {
                self.state = State::BlockComment { star: false };
                Self::put(out, Class::Comment, '/');
                Self::put(out, Class::Comment, c);
            }
            _ => {
                Self::put(out, Class::Plain, '/');
                self.state = State::Normal;
                self.normal(syntax, c, out);
            }
        }
    }

    fn string(&mut self, syntax: &'static Syntax, c: char, out: &mut Vec<Piece>) {
        let State::Str {
            quote,
            mut run,
            mut triple,
            mut escaped,
            mut closing,
        } = self.state
        else {
            return;
        };
        if syntax.triple && !triple && run < 3 {
            if c == quote {
                run = run.saturating_add(1);
                triple = run == 3;
                Self::put(out, Class::Str, c);
                self.state = State::Str {
                    quote,
                    run,
                    triple,
                    escaped,
                    closing,
                };
                return;
            }
            if run == 2 {
                self.state = State::Normal;
                self.normal(syntax, c, out);
                return;
            }
            run = 3;
        }
        Self::put(out, Class::Str, c);
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if triple {
            closing = if c == quote {
                closing.saturating_add(1)
            } else {
                0
            };
            if closing == 3 {
                self.state = State::Normal;
                return;
            }
        } else if c == quote {
            self.state = State::Normal;
            return;
        }
        self.state = State::Str {
            quote,
            run,
            triple,
            escaped,
            closing,
        };
    }

    fn end_word(&mut self, syntax: &'static Syntax, out: &mut Vec<Piece>) {
        let word = std::mem::take(&mut self.word);
        let class = if syntax.keywords.contains(&word.as_str()) {
            Class::Keyword
        } else if word.chars().next().is_some_and(char::is_uppercase) {
            Class::Type
        } else {
            Class::Plain
        };
        for c in word.chars() {
            Self::put(out, class, c);
        }
        self.state = State::Normal;
    }

    /// The line ended: held text is written, and constructs that cannot span lines are closed.
    pub fn newline(&mut self, out: &mut Vec<Piece>) {
        self.flush(out);
        self.state = match self.state {
            State::LineComment | State::Number => State::Normal,
            State::Str { triple: false, .. } => State::Normal,
            other => other,
        };
    }

    pub fn finish(&mut self, out: &mut Vec<Piece>) {
        self.flush(out);
    }

    fn flush(&mut self, out: &mut Vec<Piece>) {
        match (self.state, self.syntax) {
            (State::Ident, Some(syntax)) => self.end_word(syntax, out),
            (State::Slash, _) => {
                Self::put(out, Class::Plain, '/');
                self.state = State::Normal;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(lang: Lang, text: &str) -> Vec<Piece> {
        let mut lexer = Lexer::new(lang);
        let mut out = Vec::new();
        for c in text.chars() {
            if c == '\n' {
                lexer.newline(&mut out);
                Lexer::put(&mut out, Class::Plain, c);
            } else {
                lexer.push(c, &mut out);
            }
        }
        lexer.finish(&mut out);
        out
    }

    fn classes(lang: Lang, text: &str) -> Vec<(Class, String)> {
        lex(lang, text)
    }

    fn piece(class: Class, text: &str) -> (Class, String) {
        (class, text.to_string())
    }

    #[test]
    fn rust_keywords_types_strings_numbers_and_comments_are_told_apart() {
        let got = classes(Lang::Rust, "let x: Vec<u8> = 42; // done\n");
        assert_eq!(got.first(), Some(&piece(Class::Keyword, "let")));
        assert!(got.contains(&piece(Class::Type, "Vec")));
        assert!(got.contains(&piece(Class::Number, "42")));
        assert!(got.contains(&piece(Class::Comment, "// done")));
        let text: String = got.iter().map(|p| p.1.as_str()).collect();
        assert_eq!(
            text, "let x: Vec<u8> = 42; // done\n",
            "no character is lost or duplicated"
        );
    }

    #[test]
    fn a_keyword_written_in_two_parts_in_the_table_is_still_highlighted() {
        let word = concat!("un", "safe");
        let got = classes(Lang::Rust, &format!("{word} fn f() {{}}"));
        assert_eq!(got.first(), Some(&piece(Class::Keyword, word)), "{got:?}");
    }

    #[test]
    fn strings_may_hold_keywords_and_escaped_quotes() {
        let got = classes(Lang::Rust, r#"let s = "if \" fn";"#);
        assert!(got.contains(&piece(Class::Str, r#""if \" fn""#)), "{got:?}");
    }

    #[test]
    fn python_triple_quotes_span_lines_and_empty_strings_close() {
        let got = classes(Lang::Python, "x = \"\"\"a\nb\"\"\" + ''\n");
        assert!(got.contains(&piece(Class::Str, "\"\"\"a")), "{got:?}");
        assert!(
            got.contains(&piece(Class::Str, "b\"\"\"")),
            "the string is still open on the second line: {got:?}"
        );
        assert!(got.contains(&piece(Class::Str, "''")), "{got:?}");
        let text: String = got.iter().map(|p| p.1.as_str()).collect();
        assert_eq!(text, "x = \"\"\"a\nb\"\"\" + ''\n");
    }

    #[test]
    fn block_comments_span_lines_and_a_lone_slash_is_plain() {
        let got = classes(Lang::Script, "a / b /* x\ny */ c");
        assert!(got.contains(&piece(Class::Comment, "/* x")), "{got:?}");
        assert!(got.contains(&piece(Class::Comment, "y */")), "{got:?}");
        assert_eq!(
            got.iter().filter(|(c, _)| *c == Class::Comment).count(),
            2,
            "one piece per line"
        );
        let text: String = got.iter().map(|p| p.1.as_str()).collect();
        assert_eq!(text, "a / b /* x\ny */ c");
    }

    #[test]
    fn unterminated_strings_and_comments_end_at_the_line() {
        let got = classes(Lang::Shell, "echo \"oops\n# note\nls\n");
        assert!(
            got.iter().any(|(c, t)| *c == Class::Str && t == "\"oops"),
            "{got:?}"
        );
        assert!(
            got.iter()
                .any(|(c, t)| *c == Class::Comment && t == "# note"),
            "{got:?}"
        );
        assert!(
            got.iter()
                .any(|(c, t)| *c == Class::Plain && t.contains("ls")),
            "{got:?}"
        );
    }

    #[test]
    fn unknown_languages_and_data_formats_stay_safe() {
        assert_eq!(
            lex(Lang::Other, "def x(): 1\n"),
            vec![piece(Class::Plain, "def x(): 1\n")]
        );
        let json = lex(Lang::Json, "{\"a\": [1, true, null]}");
        assert!(json.contains(&piece(Class::Str, "\"a\"")));
        assert!(json.contains(&piece(Class::Keyword, "true")));
        assert!(json.contains(&piece(Class::Number, "1")));
    }

    #[test]
    fn every_input_comes_out_unchanged_whatever_the_language() {
        let samples = [
            "",
            "x",
            "\"",
            "/",
            "/*",
            "'''",
            "a/*b*/c//d",
            "0x1f.2e_3",
            "é_ü Ünï",
            "\"\"\"\"\"\"x",
        ];
        for lang in [
            Lang::Rust,
            Lang::Python,
            Lang::Script,
            Lang::CLike,
            Lang::Shell,
            Lang::Json,
            Lang::Toml,
            Lang::Yaml,
            Lang::Other,
        ] {
            for sample in samples {
                let text: String = lex(lang, sample).iter().map(|p| p.1.as_str()).collect();
                assert_eq!(text, sample, "{lang:?} {sample:?}");
            }
        }
    }
}
