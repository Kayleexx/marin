//! The chat session on a real pseudo-terminal: wrapping, alignment, blank lines and the prompt position are
//! judged on an emulated screen, as a person would see them.

mod pty;
mod vt;

use std::path::{Path, PathBuf};

use pty::{Check, Res, Sandbox, Step, have, on_terminal};
use unicode_width::UnicodeWidthStr;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn screen_after(bytes: &[u8], upto: usize, cols: usize) -> vt::Screen {
    let mut screen = vt::Screen::new(cols);
    screen.feed(bytes.get(..upto.min(bytes.len())).unwrap_or(&[]));
    screen
}

fn chat_command() -> String {
    format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 300",
        env!("CARGO_BIN_EXE_marin"),
        fixture("tiny-chat").display()
    )
}

/// A check that the emulated screen satisfies `want`; if not, the failure message is the screen as it was.
fn screen_is(
    cols: usize,
    want: impl Fn(&[String]) -> bool,
) -> impl Fn(&[u8], &[u8]) -> Result<(), String> {
    move |all, _| {
        let lines = screen_after(all, all.len(), cols).lines();
        if want(&lines) {
            Ok(())
        } else {
            Err(lines.join("\n"))
        }
    }
}

fn seen<'a>(keys: &'a str, what: &'a str, check: Check<'a>) -> Step<'a> {
    Step {
        keys,
        until: Some((what, check)),
    }
}

fn some_row(lines: &[String], matches: impl Fn(&String) -> bool) -> bool {
    lines.iter().any(matches)
}

/// A chat turn followed by the palette, the inspect panel and both closed again; returns the screen after each.
/// Every step waits for the screen it needs instead of for a fixed time, so a slow machine only makes it slower.
fn chat_session(cols: usize, color: bool, sandbox: &Sandbox) -> Res<Vec<vt::Screen>> {
    let prompt = screen_is(cols, |l| some_row(l, |r| r.contains("you >")));
    let reply_started = screen_is(cols, |l| some_row(l, |r| r.starts_with("marin > ")));
    let reply_done = screen_is(cols, |l| {
        l.iter()
            .position(|r| r.starts_with("stopped at"))
            .and_then(|stop| l.get(stop.saturating_add(1)..))
            .is_some_and(|after| some_row(after, |r| r.starts_with("you >")))
    });
    let palette = screen_is(cols, |l| some_row(l, |r| r.contains("Inspect last reply")));
    let palette_gone = screen_is(cols, |l| !some_row(l, |r| r.contains("Inspect last reply")));
    let panel = screen_is(cols, |l| some_row(l, |r| r.contains("[Reply]")));
    let panel_gone = screen_is(cols, |l| !some_row(l, |r| r.contains("[Reply]")));
    let steps = [
        seen("", "the first prompt", &prompt),
        seen("hello there\r", "the reply to start", &reply_started),
        seen("", "the reply to finish", &reply_done),
        seen("/", "the palette to open", &palette),
        seen("\x1b", "the palette to close", &palette_gone),
        seen("\x0f", "the inspect panel to open", &panel),
        seen("\x1b", "the inspect panel to close", &panel_gone),
    ];
    let (bytes, marks, _) = on_terminal(&chat_command(), (cols, 40), sandbox, color, &steps)?;
    Ok(marks
        .iter()
        .map(|m| screen_after(&bytes, *m, cols))
        .collect())
}

fn reply_rows(lines: &[String]) -> Vec<&String> {
    let start = lines
        .iter()
        .position(|l| l.starts_with("marin > "))
        .unwrap_or(0);
    let end = lines
        .iter()
        .position(|l| l.starts_with("stopped at"))
        .unwrap_or(lines.len());
    lines.get(start..end).unwrap_or(&[]).iter().collect()
}

#[test]
fn chat_replies_wrap_under_the_prefix_and_leave_the_prompt_in_place() -> Res<()> {
    if !have("script") {
        return Ok(());
    }
    for cols in [80, 100, 120] {
        let sandbox = Sandbox::new(&format!("chat-{cols}"))?;
        let screens = chat_session(cols, false, &sandbox)?;
        let done = &screens[2];
        let lines = done.lines();
        let reply = reply_rows(&lines);
        assert!(
            reply.len() >= 3,
            "{cols}: the reply wraps over several rows: {lines:?}"
        );
        for (i, row) in reply.iter().enumerate() {
            assert!(row.width() <= cols, "{cols}: row too wide: {row:?}");
            assert!(
                i == 0 || row.starts_with("        "),
                "{cols}: continuation not aligned under the text: {row:?}"
            );
        }
        let prompt = lines.iter().rposition(|l| l.starts_with("you >")).unwrap();
        assert!(
            lines[prompt - 1].is_empty(),
            "{cols}: one blank row separates the turns"
        );
        assert!(
            lines[prompt - 2].starts_with("stopped at"),
            "{cols}: {lines:?}"
        );
        assert_eq!(
            done.cursor(),
            (prompt, 6),
            "{cols}: the cursor waits right after `you > `"
        );
    }
    Ok(())
}

#[test]
fn the_palette_and_inspect_panel_open_and_close_without_disturbing_the_conversation() -> Res<()> {
    if !have("script") {
        return Ok(());
    }
    let sandbox = Sandbox::new("restore")?;
    let screens = chat_session(100, false, &sandbox)?;
    let (after_reply, palette, closed, panel, restored) = (
        &screens[2],
        &screens[3],
        &screens[4],
        &screens[5],
        &screens[6],
    );
    let base = after_reply.lines();
    let conversation = base.iter().rposition(|l| l.starts_with("you >")).unwrap();
    for (name, screen) in [
        ("palette", palette),
        ("closed palette", closed),
        ("panel", panel),
        ("closed panel", restored),
    ] {
        assert_eq!(
            screen.lines()[..conversation],
            base[..conversation],
            "{name} changed the conversation above the prompt"
        );
    }
    assert!(
        palette
            .lines()
            .iter()
            .any(|l| l.contains("Inspect last reply")),
        "the palette opened"
    );
    assert!(
        panel.lines().iter().any(|l| l.contains("[Reply]")),
        "the panel opened"
    );
    assert_eq!(
        closed.lines(),
        base,
        "closing the palette restores the screen exactly"
    );
    assert_eq!(
        restored.lines(),
        base,
        "closing the panel restores the screen exactly"
    );
    assert_eq!(restored.cursor(), after_reply.cursor());
    Ok(())
}

#[test]
fn color_changes_how_text_looks_but_not_where_anything_is() -> Res<()> {
    if !have("script") {
        return Ok(());
    }
    let plain = chat_session(100, false, &Sandbox::new("plain-run")?)?;
    let colored = chat_session(100, true, &Sandbox::new("color-run")?)?;
    let (a, b) = (&plain[2], &colored[2]);
    let strip_banner = |lines: Vec<String>| -> Vec<String> {
        let start = lines.iter().position(|l| l.starts_with("you >")).unwrap();
        lines[start..].to_vec()
    };
    assert_eq!(strip_banner(a.lines()), strip_banner(b.lines()));
    assert_eq!(a.cursor(), b.cursor(), "styling does not move the prompt");
    Ok(())
}
