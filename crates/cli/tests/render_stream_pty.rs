//! Markdown on a real pseudo-terminal, from files and from streams cut at arbitrary points.

mod pty;
mod vt;

use std::io::{self, Write};
use std::process::{Command, ExitStatus};
use std::sync::PoisonError;
use std::thread::sleep;
use std::time::{Duration, Instant};

use pty::{Res, Sandbox, bytes_of, collect, have, on_terminal, spawn_script};
use unicode_width::UnicodeWidthStr;

const SAMPLE: &str = "# Mutexes in Rust\n\nA **mutex** (short for *mutual exclusion*) lets only one thread touch data at a time. See the [std docs](https://doc.rust-lang.org/std/sync/struct.Mutex.html) or call `Mutex::lock()`.\n\n## Why?\n\n- Protects shared state\n  - from data races\n  - without `forget`\n- Keeps code simple\n\n1. Create it\n2. Lock it\n\n> Note: a poisoned mutex means a thread panicked.\n\n---\n\n```rust\nfn main() {\n    let m = Mutex::new(5); // shared value that is long enough to need wrapping on narrow screens\n    println!(\"{:?}\", m);\n}\n```\n\n```python\ndef f(x):\n    return x * 2\n```\n\n| Type | Blocking | Notes |\n|:-----|:--------:|------:|\n| Mutex | yes | simple |\n| RwLock | yes | many readers |\n\nThat is all.\n";

fn view(file: &str, cols: usize, color: bool, sandbox: &Sandbox) -> Res<(Vec<u8>, vt::Screen)> {
    let command = format!("{} show {file}", env!("CARGO_BIN_EXE_marin"));
    let (bytes, _, _) = on_terminal(&command, (cols, 40), sandbox, color, &[])?;
    let mut screen = vt::Screen::new(cols);
    screen.feed(&bytes);
    Ok((bytes, screen))
}

#[test]
fn markdown_is_rendered_natively_at_every_terminal_width() -> Res<()> {
    if !have("script") {
        return Ok(());
    }
    let sandbox = Sandbox::new("show")?;
    let file = sandbox.0.join("sample.md");
    std::fs::write(&file, SAMPLE)?;
    for cols in [40, 80, 100, 120, 200] {
        for color in [false, true] {
            let (bytes, screen) = view(&file.display().to_string(), cols, color, &sandbox)?;
            let lines = screen.lines();
            let text = lines.join("\n");
            let label = format!("{cols} columns, color {color}");
            for row in &lines {
                assert!(row.width() <= cols, "{label}: too wide: {row:?}");
            }
            for literal in ["**", "##", "```", "](", "|:--"] {
                assert!(
                    !text.contains(literal),
                    "{label}: raw markdown {literal:?} shown:\n{text}"
                );
            }
            assert_eq!(
                bytes.contains(&0x1b),
                color,
                "{label}: escape codes only with color"
            );
            assert!(
                text.contains("Mutexes in Rust") && text.contains("That is all."),
                "{label}"
            );
            assert!(
                text.contains("https://doc.rust-lang.org/std/sync/struct.Mutex.html") || cols < 60,
                "{label}: link target"
            );
            assert!(text.contains("Mutex::lock()") || !color, "{label}");
            assert!(
                text.contains("let m = Mutex::new(5)") || cols < 60,
                "{label}: code keeps its text"
            );
            assert!(
                text.contains("Blocking") && text.contains("many readers"),
                "{label}: table content"
            );
            if !color {
                assert!(
                    lines.iter().any(|l| l.starts_with("====")),
                    "{label}: a plain heading is underlined"
                );
            }
            assert!(
                bytes.ends_with(b"\n") || bytes.ends_with(b"\r\n"),
                "{label}: ends on a line start"
            );
        }
    }
    Ok(())
}

/// Whether any escape sequence moves the cursor or clears something: a stream that only appends has none.
fn csi_moves_the_cursor(bytes: &[u8]) -> bool {
    String::from_utf8_lossy(bytes)
        .split("\x1b[")
        .skip(1)
        .filter_map(|part| part.chars().find(char::is_ascii_alphabetic))
        .any(|last| "ABCDEFGHJKSTf".contains(last))
}

/// Feeds `chunks` to `marin show --as markdown` through a FIFO, so they reach it exactly as cut.
fn streamed(
    chunks: &[String],
    cols: usize,
    color: bool,
    sandbox: &Sandbox,
) -> Res<(Vec<u8>, bool, ExitStatus)> {
    let fifo = sandbox.0.join("input.fifo");
    if !Command::new("mkfifo").arg(&fifo).status()?.success() {
        return Err("mkfifo failed".into());
    }
    let line = format!(
        "{} show --as markdown < {}",
        env!("CARGO_BIN_EXE_marin"),
        fifo.display()
    );
    let mut child = spawn_script(&line, (cols, 40), sandbox, color)?;
    let (drawn, reader) = collect(&mut child)?;
    let pieces = chunks.to_vec();
    let writer = std::thread::spawn(move || -> io::Result<Instant> {
        let mut f = std::fs::OpenOptions::new().write(true).open(fifo)?;
        for piece in pieces {
            f.write_all(piece.as_bytes())?;
            f.flush()?;
            sleep(Duration::from_millis(8));
        }
        Ok(Instant::now())
    });
    let finished_writing = writer.join().map_err(|_| "writer panicked")??;
    let status = child.wait()?;
    reader.join().map_err(|_| "reader panicked")?;
    let early = drawn
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter(|(t, _)| *t < finished_writing)
        .count();
    Ok((bytes_of(&drawn), early >= 3, status))
}

/// `doc` cut into pieces of uneven sizes, so cuts land inside words, markers and fences.
fn pieces(doc: &str) -> Vec<String> {
    let mut chars = doc.chars();
    let mut out = Vec::new();
    for size in [1, 3, 2, 7, 5, 1, 4].iter().cycle() {
        let piece: String = chars.by_ref().take(*size).collect();
        if piece.is_empty() {
            break;
        }
        out.push(piece);
    }
    out
}

#[test]
fn a_stream_cut_anywhere_renders_progressively_without_redrawing_and_matches_the_one_shot_view()
-> Res<()> {
    if !have("script") || !have("mkfifo") {
        return Ok(());
    }
    for cols in [80, 120] {
        let sandbox = Sandbox::new(&format!("stream-{cols}"))?;
        let (bytes, progressive, status) = streamed(&pieces(SAMPLE), cols, false, &sandbox)?;
        assert!(status.success());
        assert!(
            progressive,
            "{cols}: output arrived while the input was still being written"
        );
        assert!(
            !csi_moves_the_cursor(&bytes),
            "{cols}: the stream is append-only: no cursor movement or clearing"
        );
        let file = sandbox.0.join("sample.md");
        std::fs::write(&file, SAMPLE)?;
        let (_, whole) = view(&file.display().to_string(), cols, false, &sandbox)?;
        let mut live = vt::Screen::new(cols);
        live.feed(&bytes);
        assert_eq!(
            live.lines(),
            whole.lines(),
            "{cols}: streaming and one-shot rendering agree"
        );
        assert_eq!(
            live.cursor().1,
            0,
            "{cols}: the stream ends at the start of a line"
        );
    }
    Ok(())
}

#[test]
fn malformed_streams_end_on_a_clean_line_with_no_style_left_open() -> Res<()> {
    if !have("script") || !have("mkfifo") {
        return Ok(());
    }
    let docs = [
        "**bold never closed and `code never closed\n",
        "text\n```rust\nfn unfinished() {",
        "| a | b |\n|---|---|\n| 1 |",
        "[broken link](http://x",
        "# \n#\n##### ok\n",
        "1. one\n 2. odd\n  - mixed\n*",
    ];
    for (n, doc) in docs.iter().enumerate() {
        let sandbox = Sandbox::new(&format!("bad-{n}"))?;
        let (bytes, _, status) = streamed(&pieces(doc), 60, true, &sandbox)?;
        assert!(status.success(), "{doc:?}");
        assert!(
            bytes.ends_with(b"\n"),
            "{doc:?} must end on a line start: {bytes:?}"
        );
        let text = String::from_utf8_lossy(&bytes);
        for line in text.split('\n') {
            let last_open = line.rfind("\x1b[").map(|at| &line[at..]);
            assert!(
                last_open.is_none_or(|s| s.starts_with("\x1b[0m")),
                "{doc:?}: a style is left open on {line:?}"
            );
        }
    }
    Ok(())
}
