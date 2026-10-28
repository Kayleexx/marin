use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::thread;

use super::client;
use super::inbox::{Event, TAIL_BYTES};
use super::protocol::Request;
use crate::CliError;

const SENT_BYTES: usize = 2 * TAIL_BYTES;
const INTERRUPTED: [i32; 2] = [130, 143];

/// Copies `from` to `to` as it arrives and returns the last bytes it saw.
fn tee(mut from: impl Read, mut to: impl Write) -> Vec<u8> {
    let mut tail: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; 4096];
    while let Ok(n) = from.read(&mut chunk) {
        let Some(read) = chunk.get(..n).filter(|r| !r.is_empty()) else {
            break;
        };
        let _ = to.write_all(read).and_then(|()| to.flush());
        tail.extend_from_slice(read);
        let extra = tail.len().saturating_sub(SENT_BYTES);
        tail.drain(..extra);
    }
    tail
}

fn exit_code(code: Option<i32>) -> i32 {
    code.unwrap_or(1)
}

/// Runs `command` as it would run anyway. With the assistant off nothing is captured or sent; with it on, a failure
/// is reported to it (the end of the output, which the assistant filters and bounds again).
pub fn run(socket: &Path, command: &[String]) -> Result<ExitCode, CliError> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| CliError::BadOption("give the command to run after `watch --`".into()))?;
    let mut process = Command::new(program);
    process.args(args);
    if !client::running(socket) {
        let status = process.status()?;
        return Ok(code(exit_code(status.code())));
    }
    let mut child = process
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let out = child
        .stdout
        .take()
        .map(|s| thread::spawn(move || tee(s, io::stdout())));
    let err = child
        .stderr
        .take()
        .map(|s| thread::spawn(move || tee(s, io::stderr())));
    let status = child.wait()?;
    let join =
        |t: Option<thread::JoinHandle<Vec<u8>>>| t.and_then(|t| t.join().ok()).unwrap_or_default();
    let mut tail = join(err);
    tail.extend(join(out));
    let exit = exit_code(status.code());
    if exit != 0 && !INTERRUPTED.contains(&exit) {
        let repo = std::env::current_dir()?.to_string_lossy().into_owned();
        let event = Event {
            repo,
            command: command.join(" "),
            exit,
            tail: String::from_utf8_lossy(&tail).into_owned(),
        };
        let _ = client::send(socket, &Request::Event(event));
    }
    Ok(code(exit))
}

fn code(exit: i32) -> ExitCode {
    ExitCode::from(u8::try_from(exit).unwrap_or(1))
}
