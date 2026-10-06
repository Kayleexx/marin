//! Helpers for tests that drive a program on a pseudo-terminal supplied by util-linux `script`.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::sleep;
use std::time::{Duration, Instant};

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A scratch folder that is removed when it goes out of scope.
pub struct Sandbox(pub PathBuf);

impl Sandbox {
    pub fn new(name: &str) -> io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("marin-render-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

/// Starts `command_line` on a pseudo-terminal of `cols` columns and `lines` rows.
pub fn spawn_script(
    command_line: &str,
    (cols, lines): (usize, usize),
    sandbox: &Sandbox,
    color: bool,
) -> io::Result<Child> {
    let mut command = Command::new("script");
    command
        .args(["-qec", command_line, "/dev/null"])
        .env("MARIN_HOME", &sandbox.0)
        .env("TERM", "xterm-256color")
        .env("COLUMNS", cols.to_string())
        .env("LINES", lines.to_string())
        .env("LANG", "C.UTF-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if color {
        command.env_remove("NO_COLOR").env("COLORTERM", "truecolor");
    } else {
        command.env("NO_COLOR", "1");
    }
    command.spawn()
}

/// Bytes drawn so far, with the time each read arrived.
pub type Drawn = Arc<Mutex<Vec<(Instant, Vec<u8>)>>>;

pub fn collect(child: &mut Child) -> io::Result<(Drawn, std::thread::JoinHandle<()>)> {
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("no stdout"))?;
    let drawn: Drawn = Arc::default();
    let sink = Arc::clone(&drawn);
    let reader = std::thread::spawn(move || {
        let mut chunk = [0_u8; 4096];
        while let Ok(n) = stdout.read(&mut chunk) {
            if n == 0 {
                break;
            }
            let piece = chunk.get(..n).unwrap_or(&[]).to_vec();
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((Instant::now(), piece));
        }
    });
    Ok((drawn, reader))
}

pub fn bytes_of(drawn: &Drawn) -> Vec<u8> {
    drawn
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .flat_map(|(_, b)| b.clone())
        .collect()
}

/// A wait fails after this long. It is far above what any step takes, so only a program that is stuck or broken
/// reaches it, not a slow machine.
const LIMIT: Duration = Duration::from_secs(60);
/// A screen counts as settled once nothing has been drawn for this long: a redraw arrives as one burst of bytes.
const QUIET: Duration = Duration::from_millis(150);
/// How often the output is looked at while waiting.
const POLL: Duration = Duration::from_millis(10);

/// Accepts what has been drawn (all of it, then only what came after the previous step settled), or describes what
/// was seen instead, which becomes the failure message if the wait runs out.
pub type Check<'a> = &'a dyn Fn(&[u8], &[u8]) -> Result<(), String>;

/// Keys to type, and what to wait for before the next step: the text names the screen waited for and the check
/// recognizes it. `None` waits only for the output to go quiet, which is enough for keys the program queues in
/// order, like typing into a filter, and never for keys that start work during which further keys are dropped.
pub struct Step<'a> {
    pub keys: &'a str,
    pub until: Option<(&'a str, Check<'a>)>,
}

fn drawn_len(drawn: &Drawn) -> usize {
    drawn
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .fold(0_usize, |total, (_, piece)| {
            total.saturating_add(piece.len())
        })
}

/// The end of the output, with control characters made visible, for failure messages.
fn tail(drawn: &Drawn) -> String {
    let bytes = bytes_of(drawn);
    let from = bytes.len().saturating_sub(1500);
    String::from_utf8_lossy(bytes.get(from..).unwrap_or(&[]))
        .escape_debug()
        .to_string()
}

/// Waits as `until` says, then for the output to settle. Returns how many bytes had been drawn by then.
fn wait(drawn: &Drawn, from: usize, step: &Step) -> Res<usize> {
    let (started, keys) = (Instant::now(), step.keys);
    if let Some((what, check)) = &step.until {
        loop {
            let all = bytes_of(drawn);
            let seen = match check(&all, all.get(from..).unwrap_or(&[])) {
                Ok(()) => break,
                Err(seen) => seen,
            };
            if started.elapsed() > LIMIT {
                return Err(format!(
                    "waited {LIMIT:?} for {what} after typing {keys:?}; the terminal showed:\n{seen}"
                )
                .into());
            }
            sleep(POLL);
        }
    }
    let (mut len, mut changed) = (drawn_len(drawn), Instant::now());
    while changed.elapsed() < QUIET {
        if started.elapsed() > LIMIT {
            return Err(format!(
                "the output was still changing {LIMIT:?} after typing {keys:?}; the end of it:\n{}",
                tail(drawn)
            )
            .into());
        }
        sleep(POLL);
        let now = drawn_len(drawn);
        if now != len {
            (len, changed) = (now, Instant::now());
        }
    }
    Ok(len)
}

fn type_steps(input: &mut impl Write, drawn: &Drawn, steps: &[Step]) -> Res<Vec<usize>> {
    let mut marks = Vec::new();
    let mut from = 0;
    for step in steps {
        input.write_all(step.keys.as_bytes())?;
        input.flush()?;
        from = wait(drawn, from, step)?;
        marks.push(from);
    }
    Ok(marks)
}

fn wait_for_exit(child: &mut Child, drawn: &Drawn) -> Res<ExitStatus> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if started.elapsed() > LIMIT {
            return Err(format!(
                "the program was still running {LIMIT:?} after Ctrl+D; the end of the output:\n{}",
                tail(drawn)
            )
            .into());
        }
        sleep(POLL);
    }
}

/// Types each of `steps`, waiting after each for what it names; returns everything drawn and how much had been
/// drawn when each step settled, and how the program ended. Then quits with Ctrl+D. A step that fails or times
/// out ends the program.
pub fn on_terminal(
    command_line: &str,
    size: (usize, usize),
    sandbox: &Sandbox,
    color: bool,
    steps: &[Step],
) -> Res<(Vec<u8>, Vec<usize>, ExitStatus)> {
    let mut child = spawn_script(command_line, size, sandbox, color)?;
    let mut input = child.stdin.take().ok_or("no stdin")?;
    let (drawn, reader) = collect(&mut child)?;
    let marks = type_steps(&mut input, &drawn, steps);
    // If the keys already ended the program there is nobody left to hear this.
    let _ = input.write_all(b"\x04").and_then(|()| input.flush());
    drop(input);
    let outcome =
        marks.and_then(|marks| wait_for_exit(&mut child, &drawn).map(|status| (marks, status)));
    if outcome.is_err() {
        let _ = child.kill();
    }
    let _ = child.wait();
    reader.join().map_err(|_| "reader panicked")?;
    let (marks, status) = outcome?;
    Ok((bytes_of(&drawn), marks, status))
}
