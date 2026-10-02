use super::super::caps::{Color, strip_ansi};
use super::super::screen::display_width;
use super::*;

fn caps(color: Color, unicode: bool) -> Caps {
    Caps {
        color,
        unicode,
        dumb: false,
    }
}

fn truecolor() -> Caps {
    caps(Color::Truecolor, true)
}

/// (visible left part, 1-based right-column start, visible right text) of a laid-out line.
fn split(line: &str) -> (String, Option<usize>, String) {
    for (at, _) in line.match_indices("\x1b[") {
        let rest = line.get(at.saturating_add(2)..).unwrap_or("");
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let after = rest.get(digits.len()..).unwrap_or("");
        if !digits.is_empty() && after.starts_with('G') {
            let left = strip_ansi(line.get(..at).unwrap_or(""));
            return (
                left,
                digits.parse().ok(),
                strip_ansi(after.get(1..).unwrap_or("")),
            );
        }
    }
    (strip_ansi(line), None, String::new())
}

fn right_text(splash: &Splash) -> String {
    splash
        .lines
        .iter()
        .map(|l| split(l).2)
        .collect::<Vec<_>>()
        .join("\n")
}

fn column_of(width: usize, height: usize) -> Option<usize> {
    render(&truecolor(), width, height).prompt_column
}

#[test]
fn size_picks_the_large_portrait_the_face_crop_or_no_portrait() {
    let xl = Some(1 + 96 + 2);
    let large = Some(1 + 68 + 2);
    let medium = Some(1 + 46 + 2);
    assert_eq!(
        column_of(147, 48),
        xl,
        "exactly big enough for the extra-large portrait"
    );
    assert_eq!(
        column_of(146, 60),
        large,
        "one column short of the extra-large portrait"
    );
    assert_eq!(
        column_of(200, 47),
        large,
        "one row short of the extra-large portrait"
    );
    assert_eq!(
        column_of(120, 35),
        large,
        "the full portrait fits a 120x35 terminal"
    );
    assert_eq!(column_of(119, 35), large, "exactly big enough");
    assert_eq!(
        column_of(118, 35),
        medium,
        "one column short of the full portrait"
    );
    assert_eq!(
        column_of(119, 34),
        medium,
        "one row short of the full portrait"
    );
    assert_eq!(column_of(100, 30), medium);
    assert_eq!(
        column_of(97, 25),
        medium,
        "exactly big enough for the face crop"
    );
    assert_eq!(column_of(96, 40), None, "too narrow for either portrait");
    assert_eq!(column_of(120, 24), None, "too short for either portrait");
    assert_eq!(column_of(80, 24), None);
}

#[test]
fn the_right_column_spans_the_portrait_height_and_the_prompt_row_is_the_last() {
    for (width, height, art_rows) in [(123, 36, 31), (140, 46, 31), (160, 52, 44), (100, 30, 21)] {
        let splash = render(&truecolor(), width, height);
        assert_eq!(splash.lines.len(), art_rows, "{width}x{height}");
        let text: Vec<String> = splash.lines.iter().map(|l| split(l).2).collect();
        let first = text.iter().position(|t| !t.trim().is_empty()).unwrap();
        let last = text.iter().rposition(|t| !t.trim().is_empty()).unwrap();
        assert!(first <= 2, "the wordmark starts near the top: row {first}");
        assert!(
            last + 3 >= art_rows,
            "the closing rule sits just above the prompt row: row {last} of {art_rows}"
        );
        let last_line = splash.lines.last().unwrap();
        assert_eq!(
            split(last_line).1,
            None,
            "the prompt row has no right-column text; the app draws the prompt there"
        );
    }
}

#[test]
fn content_is_the_requested_identity_with_status_only_and_no_typed_commands() {
    let text = right_text(&render(&truecolor(), 123, 36));
    for expected in [
        "private, local-first intelligence agent",
        "built on a hardware-adaptive inference runtime",
        "Status    Ready",
        "Hi there.",
        "What are we working on?",
        "Press / to open actions.",
    ] {
        assert!(text.contains(expected), "missing {expected:?}");
    }
    for dropped in [
        "private assistant",
        "Model",
        "Backend",
        "Context",
        "Mode",
        "/help",
        "/exit",
        "/clear",
        "/model",
    ] {
        assert!(
            !text.contains(dropped),
            "{dropped:?} must not be on the startup screen"
        );
    }
}

#[test]
fn the_right_column_starts_in_the_same_place_whatever_the_width() {
    for (width, height, start, art_columns) in [
        (123, 36, 72, 69),
        (150, 40, 72, 69),
        (160, 52, 100, 97),
        (240, 60, 100, 97),
        (97, 25, 50, 47),
        (130, 30, 50, 47),
    ] {
        let splash = render(&truecolor(), width, height);
        let mut found: Vec<usize> = splash.lines.iter().filter_map(|l| split(l).1).collect();
        found.dedup();
        assert_eq!(found, [start], "{width}x{height}");
        for line in &splash.lines {
            let (left, _, right) = split(line);
            assert_eq!(
                display_width(&left),
                art_columns,
                "the art is exactly its column width: {left:?}"
            );
            assert!(
                display_width(&right) <= width.saturating_sub(start),
                "{width}x{height}: {right:?}"
            );
        }
    }
}

#[test]
fn no_line_is_wider_than_the_terminal_at_any_size_color_level_or_locale() {
    for color in [Color::Truecolor, Color::Ansi256, Color::None] {
        for unicode in [true, false] {
            for (w, h) in [
                (24, 10),
                (30, 30),
                (40, 20),
                (50, 24),
                (60, 30),
                (80, 24),
                (97, 25),
                (100, 30),
                (123, 35),
                (200, 60),
            ] {
                for line in render(&caps(color, unicode), w, h).lines {
                    let (left, column, right) = split(&line);
                    let cells = column.map_or(display_width(&left), |c| {
                        c.saturating_sub(1).saturating_add(display_width(&right))
                    });
                    assert!(
                        cells < w,
                        "{color:?} {unicode} {w}x{h}: {cells} columns in {line:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn compact_terminals_stack_the_same_content_with_the_prompt_below() {
    let splash = render(&truecolor(), 80, 24);
    assert!(splash.prompt_column.is_none());
    let text = splash
        .lines
        .iter()
        .map(|l| strip_ansi(l))
        .collect::<Vec<_>>()
        .join("\n");
    for expected in [
        "private, local-first intelligence agent",
        "Status    Ready",
        "Press / to open actions.",
    ] {
        assert!(text.contains(expected), "missing {expected:?}");
    }
    assert!(
        splash.lines.iter().all(|l| split(l).1.is_none()),
        "no portrait, so no column positioning"
    );
}

#[test]
fn narrow_terminals_get_the_small_wordmark_instead_of_a_cut_one() {
    for width in [30, 40, 46] {
        let splash = render(&truecolor(), width, 24);
        let art: Vec<&String> = splash.lines.iter().take(4).collect();
        assert!(
            art.iter().all(|l| !l.contains('…')),
            "width {width}: the wordmark is never truncated"
        );
        assert!(
            art.iter().all(|l| display_width(&strip_ansi(l)) < width),
            "width {width}"
        );
    }
    let wide = render(&truecolor(), 60, 24);
    let widest = wide
        .lines
        .iter()
        .take(8)
        .map(|l| display_width(&strip_ansi(l)))
        .max()
        .unwrap();
    assert!(
        widest >= 44,
        "with room to spare the wordmark is the full 44 columns: {widest}"
    );
}

#[test]
fn monochrome_uses_no_color_codes_and_non_unicode_uses_ascii_art() {
    for line in render(&caps(Color::None, true), 123, 36).lines {
        assert!(
            !line.contains("\x1b[0m") && !line.contains("\x1b[38") && !line.contains("\x1b[48"),
            "{line:?}"
        );
    }
    let ascii = render(&caps(Color::Truecolor, false), 123, 36);
    assert!(
        ascii.lines.iter().all(|l| split(l).0.is_ascii()),
        "the portrait falls back to the ASCII ramp"
    );
    assert!(!right_text(&ascii).contains('┃') && right_text(&ascii).contains('|'));
}

#[test]
fn dumb_and_tiny_terminals_get_plain_short_text() {
    let dumb = Caps {
        color: Color::None,
        unicode: true,
        dumb: true,
    };
    for splash in [render(&dumb, 120, 40), render(&truecolor(), 20, 40)] {
        assert_eq!(splash.lines.len(), 2);
        assert!(splash.lines.iter().all(|l| !l.contains('\x1b')));
        assert!(splash.prompt_column.is_none());
    }
    assert!(render(&dumb, 120, 40).lines[0].starts_with("Marin: private"));
}

#[test]
fn narrow_terminals_wrap_the_tagline_instead_of_cutting_it() {
    for width in [30, 36, 40, 46] {
        let text = render(&truecolor(), width, 30)
            .lines
            .iter()
            .map(|l| strip_ansi(l))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!text.contains('…'), "width {width}: {text:?}");
        for word in [
            "local-first",
            "intelligence",
            "hardware-adaptive",
            "runtime",
        ] {
            assert!(text.contains(word), "width {width}: missing {word}");
        }
    }
}
