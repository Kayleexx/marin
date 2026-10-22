//! Conversations with a test server: a talker keeps its messages and its session id, and helpers for the real model and
//! for a second program that takes GPU memory.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::{Value, json};

use super::qwen35::Res;
use super::server::Server;

/// The real Qwen3.5-4B Q8 folder, with a window of 8,192 tokens.
pub fn start_real(model: &std::path::Path, home: &std::path::Path, serve: &[&str]) -> Res<Server> {
    let name = model
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("no model name")?;
    Server::spawn(model, home, &["--ctx", "8192"], serve, name)
}

/// Sentences per message of a conversation long enough for its cache to need many pages, within the 1,024-token window.
pub const LONG: usize = 4;
/// The same for the real model, whose tokens are longer: about 1,500 tokens a message.
pub const REAL: usize = 100;

pub struct Talker<'a> {
    pub name: &'a str,
    pub repeat: usize,
    pub messages: Vec<Value>,
    pub session: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Turn {
    pub answer: String,
    pub reused: u64,
    pub replayed: u64,
}

impl<'a> Talker<'a> {
    pub fn new(name: &'a str) -> Self {
        Self::long(name, 2)
    }

    /// A conversation whose messages are `repeat` sentences long.
    pub fn long(name: &'a str, repeat: usize) -> Self {
        Self {
            name,
            repeat,
            messages: Vec::new(),
            session: "new".into(),
        }
    }

    pub fn say(&mut self, server: &Server, n: usize) -> Res<Turn> {
        self.try_say(server, n)?
            .map_err(|status| format!("status {status}").into())
    }

    /// One turn; a refusal leaves the conversation as it was and returns its status.
    pub fn try_say(&mut self, server: &Server, n: usize) -> Res<Result<Turn, u16>> {
        let said = format!(
            "{} says the quick brown fox number {n} jumps over the lazy dog ",
            self.name
        )
        .repeat(self.repeat);
        self.messages.push(json!({"role": "user", "content": said}));
        let (status, reply) = server.post(&Value::Array(self.messages.clone()), &self.session)?;
        if status != 200 {
            self.messages.pop();
            return Ok(Err(status));
        }
        let answer = reply
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or("no answer")?
            .to_owned();
        self.session = reply
            .get("marin_session_id")
            .and_then(Value::as_str)
            .ok_or("no session")?
            .to_owned();
        self.messages
            .push(json!({"role": "assistant", "content": answer}));
        let count = |key: &str| reply.get(key).and_then(Value::as_u64).ok_or(key.to_owned());
        Ok(Ok(Turn {
            answer,
            reused: count("marin_reused_tokens")?,
            replayed: count("marin_replayed_tokens")?,
        }))
    }
}

/// Another program holds all of the GPU's free memory but `leave` bytes until the value is dropped.
pub fn take_gpu(leave: usize) -> Res<impl Sized> {
    let gpu = model::Gpu::new()?;
    let free = gpu.info()?.free_bytes;
    Ok((gpu.zeros(free.saturating_sub(leave) / 4)?, gpu))
}

/// The real 4B folder, when the test environment names it (as the other real-checkpoint tests do).
pub fn real_model() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(std::env::var_os("MARIN_QWEN35_Q8_DIR")?);
    dir.is_dir().then_some(dir)
}

pub fn text(reply: &Value) -> Option<Value> {
    reply.pointer("/choices/0/message/content").cloned()
}

/// A streamed request that is running: returns once the server has begun to answer, so it holds its session.
pub fn start_stream(
    server: &Server,
    messages: &Value,
    session: &str,
    max_tokens: usize,
) -> Res<BufReader<TcpStream>> {
    let mut socket = TcpStream::connect(&server.address)?;
    socket.set_read_timeout(Some(Duration::from_secs(300)))?;
    socket.write_all(
        server
            .request(messages, session, max_tokens, true)
            .as_bytes(),
    )?;
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    while !line.starts_with("data:") {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("the stream ended before it began".into());
        }
    }
    Ok(reader)
}

/// Sends a request and hangs up at once, as a client that gives up.
pub fn send_and_hang_up(server: &Server, messages: &Value, session: &str) -> Res {
    let mut socket = TcpStream::connect(&server.address)?;
    socket.write_all(server.request(messages, session, 6, false).as_bytes())?;
    Ok(())
}
