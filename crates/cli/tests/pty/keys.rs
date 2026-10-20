//! Typing into a program on a pseudo-terminal and waiting for what it draws. Shared by the terminal-UI tests.

use std::process::ExitStatus;

use crate::pty::{Res, Sandbox, Step, have, on_terminal};

/// What to type and what must then be drawn (empty: nothing in particular, the keys only edit). `REPLY` waits for
/// an answer to end and the prompt to be back, since keys typed before that are dropped on purpose.
pub type Keys<'a> = &'a [(&'a str, &'a str)];

pub const REPLY: &str = "\u{0}reply";
/// Waits until an answer is well under way: far more text than the question that was typed.
pub const STREAMING: &str = "\u{0}streaming";

fn drawn(shown: &str, text: &str) -> bool {
    if shown == REPLY {
        return text
            .find("stopped at")
            .and_then(|at| text.get(at..))
            .is_some_and(|rest| rest.contains("you >"));
    }
    if shown == STREAMING {
        let words = text.lines().filter(|line| !line.contains("ctrl+"));
        return words
            .flat_map(str::chars)
            .filter(|c| c.is_alphabetic())
            .count()
            > 200;
    }
    text.contains(shown)
}

/// Runs the command line `command` builds in a fresh sandbox, which is also the config folder; returns what was
/// drawn, how the program ended and the sandbox so a test can look at what was saved.
pub fn run_in(
    name: &str,
    keys: Keys<'_>,
    command: impl FnOnce(&Sandbox) -> Res<String>,
) -> Res<Option<(String, ExitStatus, Sandbox)>> {
    if !have("script") {
        eprintln!("skipping: `script` is not installed");
        return Ok(None);
    }
    let sandbox = Sandbox::new(name)?;
    let command_line = command(&sandbox)?;
    let checks: Vec<_> = keys
        .iter()
        .map(|(_, shown)| {
            let shown = (*shown).to_string();
            (shown.clone(), move |_: &[u8], new: &[u8]| {
                let text = String::from_utf8_lossy(new);
                if drawn(&shown, &text) {
                    Ok(())
                } else {
                    Err(text.escape_debug().to_string())
                }
            })
        })
        .collect();
    let steps: Vec<Step> = keys
        .iter()
        .zip(&checks)
        .map(|((typed, shown), (what, check))| Step {
            keys: typed,
            until: (!shown.is_empty()).then_some((what.as_str(), check)),
        })
        .collect();
    let (drawn, _, status) = on_terminal(&command_line, (110, 36), &sandbox, false, &steps)?;
    Ok(Some((
        String::from_utf8_lossy(&drawn).into_owned(),
        status,
        sandbox,
    )))
}

pub const PROMPT: (&str, &str) = ("", "you >");
