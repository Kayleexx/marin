use super::{INFO_WIDTH, MARGIN};
use crate::term::art::Art;
use crate::term::caps::Caps;
use crate::term::screen::{display_width, fit};

const TEXT: [u8; 3] = [222, 232, 244];
const MUTED: [u8; 3] = [118, 140, 170];
const LAVENDER: [u8; 3] = [152, 160, 228];
const ACCENT: [u8; 3] = [88, 214, 255];
const WIDE_WORDMARK_MIN_WIDTH: usize = 47;

/// A run of text and its color.
struct Span(String, [u8; 3]);

type Row = Vec<Span>;

fn span(text: &str, color: [u8; 3]) -> Span {
    Span(text.to_string(), color)
}

fn paint(caps: &Caps, row: &Row, max: usize) -> String {
    let mut left = max;
    let mut out = String::new();
    for Span(text, color) in row {
        let shown = fit(text, left);
        left = left.saturating_sub(display_width(&shown));
        out.push_str(&caps.paint(*color, &shown));
        if left == 0 {
            break;
        }
    }
    out
}

fn centered(text: &str, color: [u8; 3], span_width: usize) -> Row {
    let pad = span_width.saturating_sub(display_width(text)) / 2;
    vec![span(&" ".repeat(pad), TEXT), span(text, color)]
}

fn centered_art(caps: &Caps, art: &Art, span_width: usize) -> Vec<String> {
    let pad = " ".repeat(span_width.saturating_sub(art.cols()) / 2);
    art.render(caps)
        .into_iter()
        .map(|row| format!("{pad}{row}"))
        .collect()
}

fn wordmark(caps: &Caps, wide: bool, span_width: usize) -> Vec<String> {
    let art = if wide {
        Art::wordmark_wide()
    } else {
        Art::wordmark_narrow()
    };
    art.map(|art| centered_art(caps, &art, span_width))
        .unwrap_or_default()
}

/// Breaks `text` at spaces so no line is wider than `width`.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ') {
        match lines.last_mut() {
            Some(last)
                if display_width(last)
                    .saturating_add(1)
                    .saturating_add(display_width(word))
                    <= width =>
            {
                last.push(' ');
                last.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

fn rule(caps: &Caps, width: usize) -> String {
    let glyph = if caps.unicode { "─" } else { "-" };
    paint(caps, &vec![span(&glyph.repeat(width), MUTED)], width)
}

fn greeting(caps: &Caps) -> Vec<Row> {
    let bar = if caps.unicode { "┃" } else { "|" };
    vec![
        vec![span(bar, ACCENT), span(" Hi there.", ACCENT)],
        vec![span(bar, ACCENT), span(" What are we working on?", TEXT)],
    ]
}

/// The right column's sections, top to bottom, with `None` marking a flexible gap.
fn sections(caps: &Caps, wide: bool, window: usize) -> Vec<Option<Vec<String>>> {
    let block = window.min(INFO_WIDTH);
    let line = |row: Row| vec![paint(caps, &row, window)];
    let tagline: Vec<String> = [
        ("private, local-first intelligence agent", TEXT),
        ("built on a hardware-adaptive inference runtime", LAVENDER),
    ]
    .into_iter()
    .flat_map(|(text, color)| {
        wrap(text, block)
            .into_iter()
            .map(move |part| paint(caps, &centered(&part, color, block), window))
    })
    .collect();
    let status = line(vec![
        span("Status", MUTED),
        span("    ", TEXT),
        span("Ready", ACCENT),
    ]);
    let hint = line(vec![
        span("Press ", LAVENDER),
        span("/", ACCENT),
        span(" to open actions.", LAVENDER),
    ]);
    let greeting: Vec<String> = greeting(caps)
        .iter()
        .map(|r| paint(caps, r, window))
        .collect();
    vec![
        Some(wordmark(caps, wide, block)),
        None,
        Some(tagline),
        None,
        Some(status),
        Some(vec![rule(caps, INFO_WIDTH.min(window))]),
        None,
        Some(greeting),
        None,
        Some(hint),
        None,
    ]
}

/// The sections stacked under the wordmark, with the prompt on the line below at the left edge.
pub(super) fn compact(caps: &Caps, width: usize) -> super::Splash {
    let window = width.saturating_sub(MARGIN).saturating_sub(1);
    let mut lines = Vec::new();
    for part in sections(caps, width >= WIDE_WORDMARK_MIN_WIDTH, window) {
        match part {
            Some(rows) => lines.extend(
                rows.into_iter()
                    .map(|r| format!("{}{r}", " ".repeat(MARGIN))),
            ),
            None => lines.push(String::new()),
        }
    }
    lines.pop();
    super::Splash { lines }
}
