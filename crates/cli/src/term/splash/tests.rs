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

fn text(splash: &Splash) -> String {
    splash
        .lines
        .iter()
        .map(|l| strip_ansi(l))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn content_is_the_requested_identity_with_status_only_and_no_typed_commands() {
    let text = text(&render(&truecolor(), 123));
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
fn no_line_is_wider_than_the_terminal_at_any_size_color_level_or_locale() {
    for color in [Color::Truecolor, Color::Ansi256, Color::None] {
        for unicode in [true, false] {
            for width in [24, 30, 40, 50, 60, 80, 97, 100, 123, 200] {
                for line in render(&caps(color, unicode), width).lines {
                    let cells = display_width(&strip_ansi(&line));
                    assert!(
                        cells < width,
                        "{color:?} {unicode} {width}: {cells} columns in {line:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn narrow_terminals_get_the_small_wordmark_instead_of_a_cut_one() {
    for width in [30, 40, 46] {
        let splash = render(&truecolor(), width);
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
    let wide = render(&truecolor(), 60);
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
    for line in render(&caps(Color::None, true), 123).lines {
        assert!(
            !line.contains("\x1b[0m") && !line.contains("\x1b[38") && !line.contains("\x1b[48"),
            "{line:?}"
        );
    }
    let ascii = render(&caps(Color::Truecolor, false), 123);
    assert!(
        ascii.lines.iter().all(|l| strip_ansi(l).is_ascii()),
        "the wordmark falls back to the ASCII ramp"
    );
    assert!(!text(&ascii).contains('┃') && text(&ascii).contains('|'));
}

#[test]
fn dumb_and_tiny_terminals_get_plain_short_text() {
    let dumb = Caps {
        color: Color::None,
        unicode: true,
        dumb: true,
    };
    for splash in [render(&dumb, 120), render(&truecolor(), 20)] {
        assert_eq!(splash.lines.len(), 2);
        assert!(splash.lines.iter().all(|l| !l.contains('\x1b')));
    }
    assert!(render(&dumb, 120).lines[0].starts_with("Marin: private"));
}

#[test]
fn narrow_terminals_wrap_the_tagline_instead_of_cutting_it() {
    for width in [30, 36, 40, 46] {
        let text = text(&render(&truecolor(), width)).replace('\n', " ");
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
