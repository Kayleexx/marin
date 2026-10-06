use unicode_width::UnicodeWidthStr;

use super::stream::MarkdownStream;
use super::theme::Theme;

const COLOR: Theme = Theme {
    color: true,
    unicode: true,
};

fn render_with(
    theme: Theme,
    width: usize,
    margin: usize,
    midline: bool,
    chunk: usize,
    text: &str,
) -> String {
    let mut stream = MarkdownStream::new(theme, width, margin, midline);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    for piece in chars.chunks(chunk.max(1)) {
        stream.feed(&piece.iter().collect::<String>());
        out.push_str(&stream.take());
    }
    stream.finish();
    out.push_str(&stream.take());
    out
}

fn plain(width: usize, text: &str) -> String {
    render_with(Theme::plain(), width, 0, false, usize::MAX, text)
}

fn strip(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Every escape sequence is closed on the line it opened on, so no style can leak into the prompt.
fn styles_balanced(text: &str) -> bool {
    text.split('\n').all(|line| {
        let mut open = false;
        let mut rest = line;
        while let Some(at) = rest.find("\x1b[") {
            let tail = rest.get(at.saturating_add(2)..).unwrap_or("");
            let Some(end) = tail.find('m') else {
                return false;
            };
            open = tail.get(..end) != Some("0");
            rest = tail.get(end.saturating_add(1)..).unwrap_or("");
        }
        !open
    })
}

#[test]
fn headings_emphasis_code_and_links_in_plain_text() {
    let out = plain(
        40,
        "# Title\n\nA **bold** *it* `code` [go](http://x.io) end\n",
    );
    assert_eq!(
        out,
        "Title\n=====\n\nA bold it `code` go (http://x.io) end\n"
    );
}

#[test]
fn second_level_headings_are_underlined_with_dashes_and_deeper_ones_are_plain() {
    assert_eq!(
        plain(40, "## Part\n### Detail\ntext\n"),
        "Part\n----\n\nDetail\n\ntext\n"
    );
}

#[test]
fn nested_lists_indent_and_change_bullets() {
    let out = plain(40, "- a\n  - b\n    - c\n- d\n");
    assert_eq!(out, "- a\n  * b\n    + c\n- d\n");
}

#[test]
fn ordered_items_keep_their_numbers_and_wrapped_lines_hang_under_the_text() {
    let out = plain(
        22,
        "1. first item that is long enough\n   second line of it\n2. two\n",
    );
    assert_eq!(
        out,
        "1. first item that is\n   long enough\n   second line of it\n2. two\n"
    );
}

#[test]
fn a_blank_line_between_lists_is_kept() {
    assert_eq!(plain(40, "- a\n- b\n\n1. x\n"), "- a\n- b\n\n1. x\n");
}

#[test]
fn block_quotes_get_a_bar_and_horizontal_rules_a_line() {
    assert_eq!(
        plain(30, "> one\n> two\n\ntext\n\n---\n\nend\n"),
        "| one\n| two\n\ntext\n\n------------------------------\n\nend\n"
            .replace("------------------------------", &"-".repeat(30))
    );
}

#[test]
fn code_fences_are_framed_highlighted_and_keep_their_indentation() {
    let out = plain(
        40,
        "Run:\n```rust\nfn main() {\n    let x = 42; // answer\n}\n```\nDone\n",
    );
    assert_eq!(
        out,
        "Run:\n\n+- rust\n| fn main() {\n|     let x = 42; // answer\n| }\n+-\n\nDone\n"
    );
    let colored = render_with(
        COLOR,
        40,
        0,
        false,
        usize::MAX,
        "```rust\nlet x = 42;\n```\n",
    );
    assert!(
        colored.contains("\x1b[35mlet\x1b[0m"),
        "keywords are colored: {colored:?}"
    );
    assert!(
        colored.contains("\x1b[33m42\x1b[0m"),
        "numbers are colored: {colored:?}"
    );
}

#[test]
fn a_python_fence_with_a_docstring_and_a_tilde_fence_both_work() {
    let out = plain(
        40,
        "~~~python\ndef f():\n    \"\"\"doc\n    more\"\"\"\n~~~\n",
    );
    assert_eq!(
        out,
        "+- python\n| def f():\n|     \"\"\"doc\n|     more\"\"\"\n+-\n"
    );
}

#[test]
fn a_longer_closing_fence_and_text_that_only_resembles_one_stay_correct() {
    let out = plain(40, "```\na\n``` not a close\n`````\nafter\n");
    assert_eq!(out, "+-\n| a\n| ``` not a close\n+-\n\nafter\n");
}

#[test]
fn tables_align_and_wrap_to_the_width() {
    let doc = "| name | n |\n|:--|--:|\n| alpha | 1 |\n| beta | 22 |\n";
    assert_eq!(
        plain(40, doc),
        "+-------+----+\n| name  |  n |\n+-------+----+\n| alpha |  1 |\n| beta  | 22 |\n+-------+----+\n"
    );
    let narrow = plain(
        24,
        "| topic | notes |\n|---|---|\n| memory | the quick brown fox jumps over the lazy dog |\n",
    );
    assert!(narrow.lines().all(|l| l.width() <= 24), "{narrow}");
    assert!(narrow.contains("quick"), "{narrow}");
    let wide = plain(
        10,
        "| a | b | c | d |\n|---|---|---|---|\n| 1 | 2 | 3 | 4 |\n",
    );
    assert_eq!(
        wide, "a: 1\nb: 2\nc: 3\nd: 4\n",
        "a table that cannot fit is stacked"
    );
}

#[test]
fn a_bar_line_that_is_not_a_table_is_ordinary_text() {
    assert_eq!(
        plain(40, "| just a line |\nnext\n"),
        "| just a line |\nnext\n"
    );
    assert_eq!(
        plain(40, "| a | b |\n| c | d |\n"),
        "| a | b |\n| c | d |\n"
    );
    assert_eq!(plain(40, "| a | b |\n"), "| a | b |\n");
}

#[test]
fn emphasis_markers_that_are_not_emphasis_stay_literal() {
    assert_eq!(
        plain(60, "2 * 3 * 4 and snake_case_name and a_b\n"),
        "2 * 3 * 4 and snake_case_name and a_b\n"
    );
    assert_eq!(
        plain(60, "\\*not italic\\* and 5*3\n"),
        "*not italic* and 53\n"
    );
}

#[test]
fn unfinished_markdown_degrades_to_readable_text() {
    assert_eq!(
        plain(60, "**never closed and `neither\n"),
        "never closed and `neither\n"
    );
    assert_eq!(
        plain(60, "see [broken link and [x](y\n"),
        "see [broken link and [x](y\n"
    );
    assert_eq!(plain(60, "```rust\nfn x() {"), "+- rust\n| fn x() {\n+-\n");
    assert_eq!(plain(60, "#"), "");
    assert_eq!(plain(60, "text\n```"), "text\n\n+-\n+-\n");
}

#[test]
fn the_chat_layout_leads_with_the_margin_and_keeps_lines_inside_the_terminal() {
    let out = render_with(
        Theme::plain(),
        40,
        8,
        true,
        1,
        "# Hi\n\nA fairly long paragraph that has to wrap more than once here.\n\n- item one that is long enough to wrap around\n",
    );
    assert!(out.starts_with("Hi\n        ==\n"), "{out:?}");
    for (i, line) in out.lines().enumerate() {
        let lead = if i == 0 { 8 } else { 0 };
        assert!(line.width() + lead <= 40, "{line:?}");
        assert!(
            line.is_empty() || i == 0 || line.starts_with("        "),
            "aligned under the prefix: {line:?}"
        );
    }
}

#[test]
fn colored_output_closes_every_style_on_its_line() {
    let doc = "# T\n\n**b *i* c** `x` [l](http://u)\n\n- a **b\n- c\n\n```rust\nlet s = \"x\"; // c\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n";
    let out = render_with(COLOR, 30, 8, true, 1, doc);
    assert!(styles_balanced(&out), "{out:?}");
    assert!(out.contains("\x1b[1m"), "bold is used");
}

/// A document set that covers every construct, used to check properties that must hold for all of them.
const CORPUS: [&str; 8] = [
    "# Heading one\n\nParagraph with **bold**, *italic*, `code` and a [link](https://example.com/very/long/path?query=1) that wraps.\n",
    "1. one\n   - nested a\n   - nested b with enough words to need wrapping on narrow screens\n2. two\n\n- x\n- y\n",
    "> quote line one\n> quote line two with many words to wrap around the edge of the terminal window\n\n---\n",
    "```python\ndef f(x):\n    return x * 2  # double\n```\n\n```json\n{\"a\": [1, 2, true]}\n```\n\n```diff\n+added\n-removed\n@@ -1 +1 @@\n```\n",
    "| Name | Description | Count |\n|:---|:---:|---:|\n| alpha | the first letter of the greek alphabet | 1 |\n| beta | second | 22 |\n\ntext after\n",
    "```csv\nname,score\nann,10\nbob,7\n```\n\n```log\n2024-01-01 ERROR boom\n2024-01-01 INFO ok\n```\n",
    "A line with 你好世界 wide characters and émojis 😀😀 mixed in a long enough line to wrap somewhere.\n",
    "Unclosed **bold and `code\n\n```rust\nfn unfinished(",
];

#[test]
fn output_does_not_depend_on_how_the_input_was_split_into_chunks() {
    for doc in CORPUS {
        for theme in [Theme::plain(), COLOR] {
            let whole = render_with(theme, 40, 8, true, usize::MAX, doc);
            for chunk in [1, 2, 3, 5, 8, 13] {
                assert_eq!(
                    render_with(theme, 40, 8, true, chunk, doc),
                    whole,
                    "chunk {chunk}: {doc:?}"
                );
            }
        }
    }
}

#[test]
fn every_line_fits_every_terminal_width_and_the_output_ends_on_a_line_start() {
    for doc in CORPUS {
        for width in [20, 24, 40, 60, 80, 100, 120, 200] {
            let out = render_with(Theme::plain(), width, 8, true, 7, doc);
            assert!(out.is_empty() || out.ends_with('\n'), "{width}: {out:?}");
            assert!(!out.contains('\x1b'), "plain output has no escapes");
            for (i, line) in out.lines().enumerate() {
                let lead = if i == 0 { 8 } else { 0 };
                assert!(
                    line.width() + lead <= width,
                    "width {width}: {line:?} in {doc:?}"
                );
            }
            let colored = render_with(COLOR, width, 8, true, 7, doc);
            assert!(styles_balanced(&colored), "{width}: {colored:?}");
        }
    }
}

/// Small deterministic generator for fuzzing.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0.wrapping_shl(13);
        self.0 ^= self.0.wrapping_shr(7);
        self.0 ^= self.0.wrapping_shl(17);
        self.0
    }
}

#[test]
fn random_markdown_like_input_never_breaks_the_invariants() {
    const ALPHABET: &[char] = &[
        '*', '_', '`', '[', ']', '(', ')', '#', '|', '>', '-', '+', '~', '\\', ' ', ' ', '\n',
        '\n', '\t', '1', '.', ':', 'a', 'B', 'c', '你', 'é', '😀', '!', 'h', 't', 'p', '/',
    ];
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for case in 0..4000_usize {
        let len = usize::try_from(rng.next() % 160).unwrap_or(0);
        let text: String = (0..len)
            .map(|_| {
                ALPHABET
                    .get(usize::try_from(rng.next() % ALPHABET.len() as u64).unwrap_or(0))
                    .copied()
                    .unwrap_or('a')
            })
            .collect();
        let width = 20 + usize::try_from(rng.next() % 40).unwrap_or(0);
        let theme = if case % 2 == 0 { Theme::plain() } else { COLOR };
        let whole = render_with(theme, width, 8, true, usize::MAX, &text);
        let single = render_with(theme, width, 8, true, 1, &text);
        assert_eq!(whole, single, "chunking changed the output for {text:?}");
        assert!(
            whole.is_empty() || whole.ends_with('\n'),
            "{text:?} -> {whole:?}"
        );
        assert!(styles_balanced(&whole), "{text:?} -> {whole:?}");
        for (i, line) in strip(&whole).lines().enumerate() {
            let lead = if i == 0 { 8 } else { 0 };
            assert!(
                line.width() + lead <= width,
                "{text:?} -> {line:?} at {width}"
            );
        }
    }
}

#[test]
fn utf8_split_across_chunks_is_reassembled_and_bad_bytes_are_replaced() {
    use super::Utf8Tail;
    let bytes = "a😀é".as_bytes();
    for cut in 0..=bytes.len() {
        let mut tail = Utf8Tail::default();
        let (head, rest) = bytes.split_at(cut);
        let text = format!("{}{}", tail.push(head), tail.push(rest));
        assert_eq!(text, "a😀é", "cut at {cut}");
        assert_eq!(tail.finish(), "");
    }
    let mut tail = Utf8Tail::default();
    assert_eq!(tail.push(&[b'a', 0xff, b'b', 0xc3]), "a\u{fffd}b");
    assert_eq!(
        tail.finish(),
        "\u{fffd}",
        "a character cut off at the end is replaced"
    );
    let mut all_bad = Utf8Tail::default();
    assert_eq!(
        all_bad.push(&[0xff, 0xfe, 0xfd]),
        "\u{fffd}\u{fffd}\u{fffd}"
    );
}

#[test]
fn the_writer_renders_what_is_written_in_any_byte_chunks() {
    use std::io::Write;

    use super::MarkdownWriter;
    let doc = "# Hi\n\n**bold** 😀 text\n";
    let expected = plain(40, doc);
    for chunk in [1, 2, 5, 100] {
        let mut sink = Vec::new();
        {
            let mut writer =
                MarkdownWriter::new(&mut sink, MarkdownStream::new(Theme::plain(), 40, 0, false));
            for piece in doc.as_bytes().chunks(chunk) {
                writer.write_all(piece).unwrap();
            }
            writer.finish().unwrap();
        }
        assert_eq!(String::from_utf8(sink).unwrap(), expected, "chunk {chunk}");
    }
}
