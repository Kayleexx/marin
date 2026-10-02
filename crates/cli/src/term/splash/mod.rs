mod layout;
#[cfg(test)]
mod tests;

use std::io::{self, Write};

use super::art::Art;
use super::caps::Caps;
use super::screen::{fit, height, width};

const MARGIN: usize = 1;
const GAP: usize = 2;
/// Width of the right column: the longest line of the identity text.
const INFO_WIDTH: usize = 47;
/// Rows under the portrait's last row for the status line and a spare line.
const ROOM_BELOW: usize = 4;
const MINIMAL_WIDTH: usize = 24;

/// The startup view, laid out for one terminal size.
pub struct Splash {
    pub lines: Vec<String>,
    /// With a portrait, the 0-based column of the right-hand column. The last line then holds only the
    /// left part of the portrait's last row, and the prompt belongs on that row, at this column.
    pub prompt_column: Option<usize>,
}

/// The right column starts after the margin, the art and the gap.
fn right_column(portrait: &Art) -> usize {
    MARGIN.saturating_add(portrait.cols()).saturating_add(GAP)
}

fn choose_portrait(width: usize, height: usize) -> Option<Art> {
    let fits = |p: Art| {
        let wide = right_column(&p)
            .saturating_add(INFO_WIDTH)
            .saturating_add(1)
            <= width;
        (wide && p.rows().saturating_add(ROOM_BELOW) <= height).then_some(p)
    };
    [Art::portrait_xl, Art::portrait_large, Art::portrait_medium]
        .into_iter()
        .find_map(|load| load().and_then(fits))
}

pub fn render(caps: &Caps, width: usize, height: usize) -> Splash {
    if caps.dumb || width < MINIMAL_WIDTH {
        let lines = vec![
            fit(
                "Marin: private, local-first intelligence agent",
                width.saturating_sub(1),
            ),
            "Status: Ready".to_string(),
        ];
        return Splash {
            lines,
            prompt_column: None,
        };
    }
    match choose_portrait(width, height) {
        Some(portrait) => layout::with_portrait(caps, &portrait, right_column(&portrait), width),
        None => layout::compact(caps, width),
    }
}

/// Prints the startup view once, as ordinary terminal output that stays in the scrollback above the
/// conversation. Returns the column the first prompt belongs at: the right-hand column when a portrait is
/// shown (the cursor rests at the end of the portrait's last row), otherwise 0 with the prompt on the line below.
pub fn show(caps: &Caps) -> io::Result<usize> {
    let splash = render(caps, width(), height());
    let mut out = io::stdout().lock();
    writeln!(out)?;
    let last = splash.lines.len().saturating_sub(1);
    for (i, line) in splash.lines.iter().enumerate() {
        if splash.prompt_column.is_some() && i == last {
            write!(out, "{line}")?;
        } else {
            writeln!(out, "{line}")?;
        }
    }
    if splash.prompt_column.is_none() {
        writeln!(out)?;
    }
    out.flush()?;
    Ok(splash.prompt_column.unwrap_or(0))
}
