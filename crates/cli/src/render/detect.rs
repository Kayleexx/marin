//! Deciding what a file or stream is from its name and its first bytes.

use super::content::ContentKind;
use super::syntax::Lang;

/// How many leading bytes of a file are looked at to decide what it is.
pub const SAMPLE_BYTES: usize = 8192;

fn looks_binary(sample: &[u8]) -> bool {
    if sample.contains(&0) {
        return true;
    }
    if let Err(e) = std::str::from_utf8(sample)
        && e.error_len().is_some()
    {
        return true;
    }
    let control = sample
        .iter()
        .filter(|b| **b < 0x20 && !matches!(**b, b'\n' | b'\r' | b'\t' | 0x0c | 0x1b))
        .count();
    control.saturating_mul(10) > sample.len()
}

/// Decides what `sample` (the start of a file or stream) is. Content settles binary; the file name settles
/// the rest when it is a known one; without a usable name the text itself is looked at.
pub fn detect(name: Option<&str>, sample: &[u8]) -> ContentKind {
    let sample = sample.get(..SAMPLE_BYTES).unwrap_or(sample);
    if looks_binary(sample) {
        return ContentKind::Binary;
    }
    if let Some(kind) = name.and_then(ContentKind::from_extension) {
        return kind;
    }
    sniff(&String::from_utf8_lossy(sample))
}

fn sniff(text: &str) -> ContentKind {
    let lines: Vec<&str> = text.lines().take(40).collect();
    if let Some(first) = lines.first().and_then(|l| l.strip_prefix("#!")) {
        return match () {
            () if first.contains("python") => ContentKind::Code(Lang::Python),
            () if first.contains("node") || first.contains("deno") => {
                ContentKind::Code(Lang::Script)
            }
            () if first.contains("sh") => ContentKind::Code(Lang::Shell),
            () => ContentKind::Code(Lang::Other),
        };
    }
    if lines.iter().any(|l| {
        l.starts_with("diff --git ")
            || l.starts_with("Index: ")
            || (l.starts_with("@@ -") && l.contains(" @@"))
    }) {
        return ContentKind::Diff;
    }
    let trimmed = text.trim();
    if trimmed.starts_with(['{', '[']) {
        match serde_json::from_str::<serde::de::IgnoredAny>(trimmed) {
            Ok(_) => return ContentKind::Data(Lang::Json),
            Err(e) if e.is_eof() => return ContentKind::Data(Lang::Json),
            Err(_) => {}
        }
    }
    if let Some(kind) = delimited(&lines) {
        return kind;
    }
    if markdownish(&lines) {
        return ContentKind::Markdown;
    }
    if keyed(&lines, is_toml_line) {
        return ContentKind::Data(Lang::Toml);
    }
    if lines.first() == Some(&"---") || keyed(&lines, is_yaml_line) {
        return ContentKind::Data(Lang::Yaml);
    }
    if lines.len() >= 3
        && lines
            .iter()
            .filter(|l| is_log_line(l))
            .count()
            .saturating_mul(10)
            >= lines.len().saturating_mul(4)
    {
        return ContentKind::Log;
    }
    ContentKind::Unknown
}

fn delimited(lines: &[&str]) -> Option<ContentKind> {
    let rows: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| !l.trim().is_empty())
        .take(10)
        .collect();
    for delimiter in ['\t', ','] {
        let counts: Vec<usize> = rows.iter().map(|l| l.matches(delimiter).count()).collect();
        let first = *counts.first()?;
        let needed = if delimiter == '\t' { 2 } else { 3 };
        if first > 0 && counts.len() >= needed && counts.iter().all(|c| *c == first) {
            return Some(ContentKind::Delimited(delimiter));
        }
    }
    None
}

fn markdownish(lines: &[&str]) -> bool {
    let headings = lines
        .iter()
        .filter(|l| l.starts_with('#') && l.trim_start_matches('#').starts_with(' '))
        .count();
    let fences = lines
        .iter()
        .filter(|l| l.trim_start().starts_with("```"))
        .count();
    let lists = lines
        .iter()
        .filter(|l| l.starts_with("- ") || l.starts_with("* ") || l.trim_start().starts_with("1. "))
        .count();
    headings >= 2
        || ((headings > 0 || fences > 0)
            && headings.saturating_add(fences).saturating_add(lists) >= 3)
}

/// Whether most non-blank, non-comment lines look like `key<separator>...` according to `is_key_line`.
fn keyed(lines: &[&str], is_key_line: fn(&str) -> bool) -> bool {
    let body: Vec<&&str> = lines
        .iter()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();
    let hits = body.iter().filter(|l| is_key_line(l)).count();
    hits >= 2 && hits.saturating_mul(10) >= body.len().saturating_mul(7)
}

fn is_toml_line(line: &str) -> bool {
    let t = line.trim();
    (t.starts_with('[') && t.ends_with(']') && !t.contains(' '))
        || t.split_once(" = ").is_some_and(|(k, _)| {
            !k.is_empty()
                && k.chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        })
}

fn is_yaml_line(line: &str) -> bool {
    let t = line.trim_start().trim_start_matches("- ");
    t.split_once(':').is_some_and(|(k, rest)| {
        !k.is_empty()
            && k.chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
            && (rest.is_empty() || rest.starts_with(' '))
    })
}

fn is_log_line(line: &str) -> bool {
    let bytes = line.as_bytes();
    let dated = bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && bytes.first().is_some_and(u8::is_ascii_digit);
    let clock = bytes.get(2) == Some(&b':')
        && bytes.get(5) == Some(&b':')
        && bytes.first().is_some_and(u8::is_ascii_digit);
    let head: String = line.chars().take(30).collect::<String>().to_uppercase();
    let level = ["INFO", "WARN", "ERROR", "DEBUG", "TRACE", "FATAL"]
        .iter()
        .any(|l| head.contains(l));
    dated
        || clock
        || (level && (line.starts_with('[') || head.starts_with(|c: char| c.is_ascii_uppercase())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(name: Option<&str>, text: &str) -> ContentKind {
        detect(name, text.as_bytes())
    }

    #[test]
    fn fence_hints_route_to_kinds_and_unknown_words_stay_safe() {
        assert_eq!(
            ContentKind::from_hint("Rust"),
            ContentKind::Code(Lang::Rust)
        );
        assert_eq!(
            ContentKind::from_hint("py"),
            ContentKind::Code(Lang::Python)
        );
        assert_eq!(
            ContentKind::from_hint("json"),
            ContentKind::Data(Lang::Json)
        );
        assert_eq!(ContentKind::from_hint("diff"), ContentKind::Diff);
        assert_eq!(ContentKind::from_hint("tsv"), ContentKind::Delimited('\t'));
        assert_eq!(ContentKind::from_hint(""), ContentKind::PlainText);
        assert_eq!(ContentKind::from_hint("markdown"), ContentKind::PlainText);
        assert_eq!(
            ContentKind::from_hint("brainfuck"),
            ContentKind::Code(Lang::Other)
        );
    }

    #[test]
    fn extensions_decide_when_the_content_is_text() {
        assert_eq!(
            kind(Some("README.md"), "plain words"),
            ContentKind::Markdown
        );
        assert_eq!(
            kind(Some("main.rs"), "fn main() {}"),
            ContentKind::Code(Lang::Rust)
        );
        assert_eq!(kind(Some("a.PY"), "x"), ContentKind::Code(Lang::Python));
        assert_eq!(
            kind(Some("data.csv"), "not,really"),
            ContentKind::Delimited(',')
        );
        assert_eq!(
            kind(Some("notes.txt"), "{\"a\": 1}"),
            ContentKind::PlainText
        );
        assert_eq!(kind(Some("x.rb"), "puts 1"), ContentKind::Code(Lang::Other));
        assert_eq!(kind(Some("x.weird"), "hello there"), ContentKind::Unknown);
    }

    #[test]
    fn binary_content_wins_over_any_name() {
        assert_eq!(
            detect(Some("main.rs"), b"fn main\0() {}"),
            ContentKind::Binary
        );
        assert_eq!(detect(None, &[0xff, 0xfe, 0x00, 0x01]), ContentKind::Binary);
        assert_eq!(detect(None, &[0xc3, 0x28, b'a']), ContentKind::Binary);
        assert_eq!(
            detect(None, &[1, 2, 3, 4, 5, 6, 7, b'a', b'b', b'c']),
            ContentKind::Binary
        );
        assert_eq!(detect(None, "caf\u{e9}".as_bytes()), ContentKind::Unknown);
        let cut = &"caf\u{e9}".as_bytes()[..4];
        assert_eq!(
            detect(None, cut),
            ContentKind::Unknown,
            "a sample cut inside a character is still text"
        );
        assert_eq!(detect(None, b""), ContentKind::Unknown);
    }

    #[test]
    fn content_is_sniffed_without_a_name() {
        assert_eq!(
            kind(None, "#!/usr/bin/env python3\nprint(1)"),
            ContentKind::Code(Lang::Python)
        );
        assert_eq!(
            kind(None, "#!/bin/bash\nls"),
            ContentKind::Code(Lang::Shell)
        );
        assert_eq!(
            kind(None, "diff --git a/x b/x\n--- a/x\n+++ b/x\n"),
            ContentKind::Diff
        );
        assert_eq!(kind(None, "@@ -1,2 +1,2 @@\n-a\n+b\n"), ContentKind::Diff);
        assert_eq!(kind(None, "{\"a\": [1, 2]}"), ContentKind::Data(Lang::Json));
        assert_eq!(kind(None, "{\"a\": [1, 2"), ContentKind::Data(Lang::Json));
        assert_eq!(
            kind(None, "a,b,c\n1,2,3\n4,5,6\n"),
            ContentKind::Delimited(',')
        );
        assert_eq!(kind(None, "a\tb\n1\t2\n"), ContentKind::Delimited('\t'));
        assert_eq!(
            kind(None, "# Title\n\nSome text\n\n## More\n- a\n"),
            ContentKind::Markdown
        );
        assert_eq!(
            kind(None, "[package]\nname = \"x\"\nversion = \"1\"\n"),
            ContentKind::Data(Lang::Toml)
        );
        assert_eq!(
            kind(None, "name: x\nitems:\n  - a: 1\n  - b: 2\n"),
            ContentKind::Data(Lang::Yaml)
        );
        assert_eq!(
            kind(
                None,
                "2024-01-01 10:00:00 INFO a\n2024-01-01 10:00:01 WARN b\n2024-01-01 10:00:02 ERROR c\n"
            ),
            ContentKind::Log
        );
        assert_eq!(
            kind(None, "just a sentence, with a comma.\nanother line here.\n"),
            ContentKind::Unknown
        );
        assert_eq!(
            kind(None, "- one\n- two\n- three\n"),
            ContentKind::Unknown,
            "bare lists are not assumed to be markdown"
        );
    }

    #[test]
    fn only_the_start_of_a_huge_input_is_examined() {
        let mut big = vec![b'a'; SAMPLE_BYTES * 2];
        big.push(0);
        assert_eq!(detect(None, &big), ContentKind::Unknown);
    }
}
