use super::{INFO_WIDTH, MARGIN};
use crate::term::art::Art;
use crate::term::caps::Caps;
use crate::term::screen::{display_width, fit};

const TEXT: [u8; 3] = [222, 232, 244];
const MUTED: [u8; 3] = [118, 140, 170];
const LAVENDER: [u8; 3] = [152, 160, 228];
const ACCENT: [u8; 3] = [88, 214, 255];
const WIDE_WORDMARK_MIN_WIDTH: usize = 47;
/// Gaps between the right column's sections; spare height is shared out among them.
const SECTION_GAPS: usize = 5;
const MAX_GAP: usize = 4;

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

/// Right-column lines for a portrait of `height` rows: sections spread over the full height
/// like the reference, ending with an empty line where the prompt goes.
fn column(caps: &Caps, window: usize, height: usize) -> Vec<String> {
    let parts = sections(caps, true, window);
    let closing = vec![rule(caps, INFO_WIDTH.min(window))];
    let fixed: usize = parts
        .iter()
        .flatten()
        .map(Vec::len)
        .fold(closing.len().saturating_add(1), usize::saturating_add);
    let spare = height.saturating_sub(fixed);
    let top = usize::from(spare > SECTION_GAPS);
    let inner = spare.saturating_sub(top);
    // Gaps stay modest on tall terminals; whatever is left opens up just above the closing rule.
    let each = (inner / SECTION_GAPS).min(MAX_GAP);
    let mut remainder = inner.saturating_sub(each.saturating_mul(SECTION_GAPS));
    let mut sizes = [each; SECTION_GAPS];
    if each < MAX_GAP {
        for size in sizes.iter_mut() {
            let more = usize::from(remainder > 0);
            remainder = remainder.saturating_sub(more);
            *size = size.saturating_add(more);
        }
        remainder = 0;
    }
    if let Some(last) = sizes.last_mut() {
        *last = last.saturating_add(remainder);
    }
    let mut lines: Vec<String> = vec![String::new(); top];
    let mut gaps = sizes.into_iter();
    for part in parts {
        match part {
            Some(rows) => lines.extend(rows),
            None => lines.extend(std::iter::repeat_n(String::new(), gaps.next().unwrap_or(0))),
        }
    }
    lines.extend(closing);
    lines.push(String::new());
    lines
}

pub(super) fn with_portrait(
    caps: &Caps,
    portrait: &Art,
    column_start: usize,
    width: usize,
) -> super::Splash {
    let art = portrait.render(caps);
    let window = width
        .saturating_sub(column_start)
        .saturating_sub(1)
        .min(INFO_WIDTH);
    let right = column(caps, window, art.len());
    let lines = art
        .iter()
        .enumerate()
        .map(|(y, left)| {
            let text = right.get(y).filter(|t| !t.is_empty());
            let left = format!("{}{left}", " ".repeat(MARGIN));
            match text {
                // Absolute column, so a terminal that measures the art differently cannot shift the text.
                Some(text) => format!("{left}\x1b[{}G{text}", column_start.saturating_add(1)),
                None => left,
            }
        })
        .collect();
    super::Splash {
        lines,
        prompt_column: Some(column_start),
    }
}

/// No portrait: the same sections stacked, with the prompt on the line below at the left edge.
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
    super::Splash {
        lines,
        prompt_column: None,
    }
}
