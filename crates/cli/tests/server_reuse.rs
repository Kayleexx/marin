//! `marin code` finds a server on port 8000 by itself. It may use one only if it is Marin's, runs the model that
//! would be loaded and has a window code mode works in; anything else must be said and a model loaded here instead,
//! without waiting on a server that is silent or never stops talking. One test holds the port for all of it.

mod common;
#[path = "common/long.rs"]
mod long;

use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use common::{Sandbox, fixtures, text};
use long::long_tiny;

const PORT: u16 = 8000;

fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

fn marin(sandbox: &Sandbox, models: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
    command
        .env("MARIN_HOME", sandbox.0.join("home"))
        .env("MARIN_MODELS", models)
        .env("NO_COLOR", "1");
    command
}

/// A real `marin serve` on the port, stopped when dropped.
struct Server(Child);

impl Server {
    fn start(sandbox: &Sandbox, model: &Path, ctx: Option<usize>) -> io::Result<Self> {
        let mut command = marin(sandbox, model);
        command.args(["--backend", "cpu", "--model"]).arg(model);
        if let Some(ctx) = ctx {
            command.args(["--ctx", &ctx.to_string()]);
        }
        let child = command
            .args(["serve", "--port", &PORT.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let server = Self(child);
        let started = Instant::now();
        while TcpStream::connect(("127.0.0.1", PORT)).is_err() {
            if started.elapsed() > Duration::from_secs(60) {
                return Err(io::Error::other("the server did not start"));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(server)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn code(sandbox: &Sandbox, models: &Path) -> io::Result<(Output, Duration)> {
    let began = Instant::now();
    let mut child = marin(sandbox, models)
        .arg("code")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(b"where is main?\n")?;
    }
    let out = child.wait_with_output()?;
    Ok((out, began.elapsed()))
}

fn refused(out: &Output, why: &str) {
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(
        stderr.contains("not using the Marin server on port 8000") && stderr.contains(why),
        "{stderr}"
    );
    assert!(
        !stderr.contains("using the Marin server on port 8000 (") && !stdout.contains("main"),
        "never answered by that server: {stdout}{stderr}"
    );
    assert!(
        stderr.contains("code mode needs a context window"),
        "a model is loaded here instead: {stderr}"
    );
}

/// Accepts one connection and then says `reply` once, and `drip` for as long as the client stays.
fn impostor(reply: &'static [u8], drip: &'static [u8]) -> io::Result<std::thread::JoinHandle<()>> {
    let listener = TcpListener::bind(("127.0.0.1", PORT))?;
    Ok(std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        let mut buffer = [0_u8; 512];
        let _ = stream.read(&mut buffer);
        if stream.write_all(reply).is_err() {
            return;
        }
        while if drip.is_empty() {
            stream.read(&mut buffer).is_ok_and(|n| n > 0)
        } else {
            stream.write_all(drip).is_ok()
        } {
            std::thread::sleep(Duration::from_millis(50));
        }
    }))
}

#[test]
fn a_server_on_port_8000_is_used_only_when_it_is_the_right_one() -> io::Result<()> {
    if TcpListener::bind(("127.0.0.1", PORT)).is_err() {
        eprintln!("skipping: port {PORT} is in use");
        return Ok(());
    }
    let sandbox = Sandbox::new("reuse")?;
    let models: PathBuf = sandbox.0.join("models");
    let tiny = fixtures().join("tiny-chat");
    copy_dir(&tiny, &models.join("alpha"))?;
    copy_dir(&tiny, &sandbox.0.join("other").join("beta"))?;

    {
        let _server = Server::start(&sandbox, &sandbox.0.join("other").join("beta"), None)?;
        let (out, _) = code(&sandbox, &models)?;
        refused(&out, "it runs beta, not alpha");
    }
    {
        let _server = Server::start(&sandbox, &models.join("alpha"), None)?;
        let (out, _) = code(&sandbox, &models)?;
        refused(&out, "its window is 1024 tokens");
    }
    {
        let fresh = Sandbox::new("reuse-long")?;
        let long = fresh.0.join("long");
        long_tiny(&fixtures().join("tiny-chat"), &long.join("gamma"))?;
        let _server = Server::start(&fresh, &long.join("gamma"), Some(2048))?;
        let (out, _) = code(&fresh, &long)?;
        let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
        assert!(out.status.success(), "{stderr}");
        assert!(
            stderr.contains("using the Marin server on port 8000 (gamma)"),
            "the matching server is used: {stderr}"
        );
        assert!(
            stdout.contains("context") && stdout.contains("2,048"),
            "the answer came from the server's 2048 window, not a model loaded here: {stdout}"
        );
    }
    for (reply, drip) in [
        (&b""[..], &b""[..]),
        (&b"HTTP/1.1 200 OK\r\nX-Pad: "[..], &b"x"[..]),
        (
            &b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nnot json!"[..],
            &b""[..],
        ),
    ] {
        let server = impostor(reply, drip)?;
        let (out, took) = code(&sandbox, &models)?;
        let stderr = text(&out.stderr);
        assert!(
            !stderr.contains("using the Marin server on port 8000 ("),
            "{stderr}"
        );
        assert!(
            stderr.contains("code mode needs a context window"),
            "{stderr}"
        );
        assert!(took < Duration::from_secs(10), "took {took:?}");
        let _ = server.join();
    }
    Ok(())
}
