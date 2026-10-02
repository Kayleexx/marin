mod common;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;

use common::{Sandbox, fixtures, text};

/// Drives the real interactive UI through a pseudo-terminal supplied by util-linux `script`.
#[test]
fn interactive_session_opens_the_palette_runs_an_action_and_exits() {
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("skipping: `script` is not installed");
        return;
    }
    let sandbox = Sandbox::new("pty").unwrap();
    let model = fixtures().join("tiny-chat");
    let command_line = format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 4",
        env!("CARGO_BIN_EXE_marin"),
        model.display()
    );
    let mut child = Command::new("script")
        .args(["-qec", &command_line, "/dev/null"])
        .env("MARIN_HOME", &sandbox.0)
        .env("NO_COLOR", "1")
        .env("TERM", "xterm")
        .env("COLUMNS", "120")
        .env("LINES", "40")
        .env("LANG", "C.UTF-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    for keys in [
        "",
        "/tok",
        "\r",
        "hello world\r",
        "hello world\r",
        "\x0f",
        "\x1b",
        "\x04",
    ] {
        sleep(Duration::from_millis(if keys.is_empty() {
            1500
        } else {
            600
        }));
        input.write_all(keys.as_bytes()).unwrap();
        input.flush().unwrap();
    }
    sleep(Duration::from_millis(800));
    drop(input);
    let out = child.wait_with_output().unwrap();
    let screen = text(&out.stdout);
    for expected in [
        "Status    Ready",
        "private, local-first intelligence agent",
        "built on a hardware-adaptive inference runtime",
        "Press / to open actions.",
        "you > ",
        "/ actions · ctrl+o inspect",
        "Show tokens",
        "Text to split into tokens",
        "6 tokens",
        "you > hello world",
        "[Reply]",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in {screen:?}"
        );
    }
    assert!(
        out.status.success(),
        "ctrl+d at an empty prompt exits cleanly: {:?}",
        out.status
    );
}

/// Keys typed while a reply streams are echoed by the terminal; they must be discarded when the prompt returns.
#[test]
fn keys_typed_during_a_reply_do_not_leak_into_the_next_prompt() {
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("skipping: `script` is not installed");
        return;
    }
    let sandbox = Sandbox::new("typeahead").unwrap();
    let model = fixtures().join("tiny-chat");
    let command_line = format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 3",
        env!("CARGO_BIN_EXE_marin"),
        model.display()
    );
    let mut child = Command::new("script")
        .args(["-qec", &command_line, "/dev/null"])
        .env("MARIN_HOME", &sandbox.0)
        .env("NO_COLOR", "1")
        .env("TERM", "xterm")
        .env("COLUMNS", "100")
        .env("LINES", "30")
        .env("LANG", "C.UTF-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    sleep(Duration::from_millis(1500));
    input.write_all(b"hello world\rjunkjunk\r").unwrap();
    input.flush().unwrap();
    sleep(Duration::from_millis(900));
    input.write_all(b"\x04").unwrap();
    drop(input);
    let screen = text(&child.wait_with_output().unwrap().stdout);
    assert!(screen.contains("you > hello world"), "{screen:?}");
    assert!(
        !screen.contains("junkjunk\u{23ce}") && !screen.contains("you > junkjunk"),
        "stray keys leaked into the prompt: {screen:?}"
    );
}

/// Runs a session at `columns` x `lines`, sends `keys` (pausing between them), then asks it to quit with
/// Ctrl+D until it does. Returns everything drawn, or None when `script` is unavailable.
fn session_output(
    name: &str,
    columns: &str,
    lines: &str,
    color: bool,
    keys: &[&str],
) -> std::io::Result<Option<String>> {
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("skipping: `script` is not installed");
        return Ok(None);
    }
    let sandbox = Sandbox::new(name)?;
    let command_line = format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 3",
        env!("CARGO_BIN_EXE_marin"),
        fixtures().join("tiny-chat").display()
    );
    let mut command = Command::new("script");
    command
        .args(["-qec", &command_line, "/dev/null"])
        .env("MARIN_HOME", &sandbox.0)
        .env("TERM", "xterm-256color")
        .env("COLORTERM", "truecolor")
        .env("COLUMNS", columns)
        .env("LINES", lines)
        .env("LANG", "C.UTF-8");
    if !color {
        command.env("NO_COLOR", "1");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    // Drain the output while keys are sent: the startup art alone fills a pipe's buffer.
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("no stdout"))?;
    let reader = std::thread::spawn(move || {
        let mut drawn = Vec::new();
        let _ = stdout.read_to_end(&mut drawn);
        drawn
    });
    for (i, key) in keys.iter().enumerate() {
        sleep(Duration::from_millis(if i == 0 { 1500 } else { 700 }));
        input.write_all(key.as_bytes())?;
        input.flush()?;
    }
    // Keys that arrive while a reply streams are discarded on purpose, so keep asking to quit.
    for _ in 0..40 {
        if child.try_wait()?.is_some() {
            break;
        }
        let _ = input.write_all(&[4]).and_then(|()| input.flush());
        sleep(Duration::from_millis(500));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
    }
    drop(input);
    child.wait()?;
    let drawn = reader
        .join()
        .map_err(|_| std::io::Error::other("reader thread panicked"))?;
    Ok(Some(text(&drawn)))
}

const CLEAR_SCREEN: &str = "\u{1b}[2J";
const ALTERNATE_SCREEN: &str = "\u{1b}[?1049";
const PURGE_SCROLLBACK: &str = "\u{1b}[3J";

/// The welcome is ordinary terminal output: it stays above the conversation in the scrollback, so nothing
/// clears the screen, switches to the alternate screen or purges history, in any layout.
#[test]
fn the_welcome_stays_in_the_scrollback_above_the_conversation() {
    for (columns, lines, layout) in [
        ("160", "52", "extra-large portrait"),
        ("120", "35", "large portrait"),
        ("100", "30", "face crop"),
        ("80", "24", "compact"),
        ("40", "20", "narrow"),
    ] {
        let Some(screen) =
            session_output("welcome", columns, lines, true, &["", "hello world\r"]).unwrap()
        else {
            return;
        };
        for control in [CLEAR_SCREEN, ALTERNATE_SCREEN, PURGE_SCROLLBACK] {
            assert!(
                !screen.contains(control),
                "{layout}: {control:?} would hide or destroy terminal history"
            );
        }
        let ready = screen.find("Ready").unwrap_or(usize::MAX);
        let said = screen.find("hello world").unwrap_or(0);
        assert!(
            ready < said,
            "{layout}: the welcome is printed first and the conversation follows it"
        );
        assert!(
            screen.contains("you >") && screen.contains("\u{1b}[0m hello world"),
            "{layout}: the first prompt is echoed as a conversation line"
        );
    }
}

/// With a portrait the first prompt belongs in the right-hand column, as in the reference.
#[test]
fn with_a_portrait_the_first_prompt_is_drawn_in_the_right_hand_column() {
    let Some(screen) = session_output("column", "120", "35", true, &[""]).unwrap() else {
        return;
    };
    assert!(
        screen.contains("\u{1b}[72G\u{1b}[J\u{1b}[38;5;147myou >\u{1b}[0m "),
        "the prompt is drawn at column 72, right of the 68-column portrait: {:?}",
        screen
            .chars()
            .rev()
            .take(200)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
}

/// NO_COLOR gets a readable monochrome startup: no color codes, but the same layout.
#[test]
fn no_color_startup_has_no_color_codes_but_the_same_content() {
    let Some(screen) = session_output("mono", "120", "35", false, &[""]).unwrap() else {
        return;
    };
    for code in ["\u{1b}[38;", "\u{1b}[48;", "\u{1b}[0m"] {
        assert!(
            !screen.contains(code),
            "{code:?} must not appear with NO_COLOR"
        );
    }
    assert!(
        screen.contains("private, local-first intelligence agent")
            && screen.contains("Status    Ready")
    );
}
