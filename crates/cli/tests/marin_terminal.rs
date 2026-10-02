//! The terminal must come back usable however Marin ends or fails: after the run a `stty -a` in the same
//! pseudo-terminal must show canonical mode and echo again, and a model switch, which fails on machines without
//! CUDA, must leave a working prompt.

mod common;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;

use common::{Sandbox, fixtures, text};

/// Runs marin on the tiny chat model, sends `keys`, quits with Ctrl+D if still running, then prints the
/// terminal flags. Returns None when `script` is unavailable.
fn run(name: &str, keys: &[&str]) -> std::io::Result<Option<String>> {
    if Command::new("script").arg("--version").output().is_err() {
        eprintln!("skipping: `script` is not installed");
        return Ok(None);
    }
    let sandbox = Sandbox::new(name)?;
    let command_line = format!(
        "{} --model {} --backend cpu --temp 0 --max-tokens 3; stty -a",
        env!("CARGO_BIN_EXE_marin"),
        fixtures().join("tiny-chat").display()
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
        .spawn()?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
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
    for _ in 0..30 {
        if child.try_wait()?.is_some() {
            break;
        }
        let _ = input.write_all(&[4]).and_then(|()| input.flush());
        sleep(Duration::from_millis(400));
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

fn assert_cooked(screen: &str, how: &str) {
    let flags = screen.rsplit("lflags").next().unwrap_or("");
    assert!(
        flags.contains(" icanon ") && !flags.contains("-icanon") && !flags.contains("-echo "),
        "{how}: terminal left raw: {flags:?}"
    );
}

#[test]
fn the_terminal_is_restored_after_ctrl_d_ctrl_c_and_the_quit_action() {
    for (how, keys) in [
        ("ctrl+d", vec![""]),
        ("ctrl+c twice", vec!["", "\x03", "\x03"]),
        ("quit action", vec!["", "/", "quit", "\r"]),
        ("after a reply", vec!["", "hello world\r"]),
    ] {
        let Some(screen) = run("restore", &keys).unwrap() else {
            return;
        };
        assert_cooked(&screen, how);
    }
}

#[test]
fn a_backend_switch_attempt_leaves_a_working_prompt_and_a_restored_terminal() {
    let keys = ["", "/", "backend", "\r", "cuda", "\r", "hello world\r"];
    let Some(screen) = run("switch", &keys).unwrap() else {
        return;
    };
    assert!(
        screen.contains("you > hello world") && screen.contains("marin >"),
        "the conversation continues after the switch attempt: {screen:?}"
    );
    assert!(
        !screen.contains("no model is loaded"),
        "the previous model came back: {screen:?}"
    );
    assert_cooked(&screen, "after the switch");
}
