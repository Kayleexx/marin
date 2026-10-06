//! Helpers for tests that drive a program on a pseudo-terminal supplied by util-linux `script`.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
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

pub fn spawn_script(
    command_line: &str,
    cols: usize,
    sandbox: &Sandbox,
    color: bool,
) -> io::Result<Child> {
    let mut command = Command::new("script");
    command
        .args(["-qec", command_line, "/dev/null"])
        .env("MARIN_HOME", &sandbox.0)
        .env("TERM", "xterm-256color")
        .env("COLUMNS", cols.to_string())
        .env("LINES", "40")
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

/// Types each of `steps` after its pause; returns everything drawn and how much had been drawn after each step.
pub fn on_terminal(
    command_line: &str,
    cols: usize,
    sandbox: &Sandbox,
    color: bool,
    steps: &[(&str, u64)],
) -> Res<(Vec<u8>, Vec<usize>)> {
    let mut child = spawn_script(command_line, cols, sandbox, color)?;
    let mut input = child.stdin.take().ok_or("no stdin")?;
    let (drawn, reader) = collect(&mut child)?;
    let mut marks = Vec::new();
    for (keys, pause) in steps {
        sleep(Duration::from_millis(*pause));
        input.write_all(keys.as_bytes())?;
        input.flush()?;
        sleep(Duration::from_millis(400));
        marks.push(bytes_of(&drawn).len());
    }
    input.write_all(b"\x04")?;
    drop(input);
    child.wait()?;
    reader.join().map_err(|_| "reader panicked")?;
    Ok((bytes_of(&drawn), marks))
}
