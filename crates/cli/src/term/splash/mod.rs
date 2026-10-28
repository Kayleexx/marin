mod layout;
#[cfg(test)]
mod tests;

use std::io::{self, Write};

use super::caps::Caps;
use super::screen::{fit, width};

const MARGIN: usize = 1;
/// Width of the text column: the longest line of the identity text.
const INFO_WIDTH: usize = 47;
const MINIMAL_WIDTH: usize = 24;

/// The startup view, laid out for one terminal size.
pub struct Splash {
    pub lines: Vec<String>,
}

pub fn render(caps: &Caps, width: usize) -> Splash {
    if caps.dumb || width < MINIMAL_WIDTH {
        let lines = vec![
            fit(
                "Marin: private, local-first intelligence agent",
                width.saturating_sub(1),
            ),
            "Status: Ready".to_string(),
        ];
        return Splash { lines };
    }
    layout::compact(caps, width)
}

/// Prints the startup view once, as ordinary terminal output that stays in the scrollback above the
/// conversation, with the first prompt on the line below.
pub fn show(caps: &Caps) -> io::Result<()> {
    let splash = render(caps, width());
    let mut out = io::stdout().lock();
    writeln!(out)?;
    for line in &splash.lines {
        writeln!(out, "{line}")?;
    }
    writeln!(out)?;
    out.flush()
}
