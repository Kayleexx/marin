use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use thiserror::Error;

const CONNECT: Duration = Duration::from_secs(3);
const TICK: Duration = Duration::from_millis(100);
/// A long prompt is read before the first token appears, so the server may be quiet for a while.
const SILENCE: Duration = Duration::from_secs(240);
/// How long the quick check for a server and the request for its model may take in all.
const PROBE: Duration = Duration::from_secs(1);
const INFO: Duration = Duration::from_secs(5);
const MAX_CHUNK: usize = 1 << 20;
/// More than this without the end of an event is not a reply from Marin.
const MAX_EVENT: usize = 4 << 20;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("no Marin server answers at {0}")]
    Unreachable(SocketAddr),
    #[error("the server refused the request ({status}): {message}")]
    Rejected { status: u16, message: String },
    #[error("the server has been silent for {} s", .0.as_secs())]
    Silent(Duration),
    #[error("the connection broke: {0}")]
    Broken(String),
    #[error("unexpected answer from the server: {0}")]
    Protocol(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Local(String),
}

impl From<io::Error> for ClientError {
    fn from(e: io::Error) -> Self {
        Self::Broken(e.to_string())
    }
}

pub struct ServerInfo {
    pub model: String,
    pub context: usize,
}

#[derive(Clone, Copy, Default)]
pub struct Usage {
    pub prompt: usize,
    pub completion: usize,
}

#[derive(Default)]
pub struct Reply {
    pub text: String,
    pub usage: Usage,
    pub session: Option<String>,
    pub finish: String,
}

pub struct Ask<'a> {
    pub model: &'a str,
    pub messages: Vec<Value>,
    pub session: &'a str,
    pub max_tokens: usize,
}

pub struct Client {
    addr: SocketAddr,
    silence: Duration,
}

enum Event {
    Text(String),
    Done(Reply),
    Failed(ClientError),
}

struct Head {
    status: u16,
    length: usize,
}

/// Reads from a stream for `within` in all, however slowly the other side sends, so a peer that keeps talking cannot hold
/// a caller that wants a short answer.
struct Deadline {
    stream: TcpStream,
    started: Instant,
    within: Duration,
}

impl Read for Deadline {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let left = self.within.saturating_sub(self.started.elapsed());
        if left.is_zero() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        self.stream.set_read_timeout(Some(left))?;
        self.stream.read(buf)
    }
}

/// Closes the connection when the request ends, whichever way, so the server stops generating for a client that left.
struct Hangup(TcpStream);

impl Drop for Hangup {
    fn drop(&mut self) {
        let _ = self.0.shutdown(Shutdown::Both);
    }
}

impl Client {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            silence: SILENCE,
        }
    }

    #[cfg(test)]
    pub fn with_silence(self, silence: Duration) -> Self {
        Self { silence, ..self }
    }

    fn connect(&self) -> Result<TcpStream, ClientError> {
        let stream = TcpStream::connect_timeout(&self.addr, CONNECT)
            .map_err(|_| ClientError::Unreachable(self.addr))?;
        stream.set_nodelay(true)?;
        Ok(stream)
    }

    fn send(
        &self,
        stream: &mut TcpStream,
        method: &str,
        path: &str,
        body: &[u8],
    ) -> io::Result<()> {
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.addr,
            body.len()
        )?;
        stream.write_all(body)?;
        stream.flush()
    }

    /// Whether a Marin server answers, quickly: for deciding between using one and loading a model here.
    pub fn probe(&self) -> Option<ServerInfo> {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_millis(300)).ok()?;
        self.send(&mut stream, "GET", "/v1/models", b"").ok()?;
        parse_info(stream, PROBE).ok()
    }

    pub fn info(&self) -> Result<ServerInfo, ClientError> {
        let mut stream = self.connect()?;
        self.send(&mut stream, "GET", "/v1/models", b"")?;
        parse_info(stream, INFO)
    }

    /// Streams one reply, passing each piece of text to `on_text` as it arrives. Setting `cancel` hangs up, and so
    /// does `on_text` returning false, which ends the reply early with the text so far.
    pub fn chat(
        &self,
        ask: &Ask<'_>,
        cancel: &AtomicBool,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<Reply, ClientError> {
        let body = json!({
            "model": ask.model,
            "messages": ask.messages,
            "stream": true,
            "temperature": 0,
            "max_tokens": ask.max_tokens,
            "marin_session_id": ask.session,
        });
        let body = serde_json::to_vec(&body).map_err(|e| ClientError::Protocol(e.to_string()))?;
        let mut stream = self.connect()?;
        self.send(&mut stream, "POST", "/v1/chat/completions", &body)?;
        let (tx, rx) = mpsc::channel();
        let reader = stream.try_clone()?;
        let _hangup = Hangup(stream);
        thread::spawn(move || {
            if let Err(e) = pump(reader, &tx) {
                let _ = tx.send(Event::Failed(e));
            }
        });
        let mut heard = Instant::now();
        let mut reply_so_far = String::new();
        loop {
            match rx.recv_timeout(TICK) {
                Ok(Event::Text(piece)) => {
                    heard = Instant::now();
                    reply_so_far.push_str(&piece);
                    if !on_text(&piece) {
                        return Ok(Reply {
                            text: reply_so_far,
                            finish: "client_stop".into(),
                            ..Reply::default()
                        });
                    }
                }
                Ok(Event::Done(reply)) => return Ok(reply),
                Ok(Event::Failed(e)) => return Err(e),
                Err(RecvTimeoutError::Timeout) => {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(ClientError::Cancelled);
                    }
                    if heard.elapsed() > self.silence {
                        return Err(ClientError::Silent(self.silence));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ClientError::Broken("the reader stopped".into()));
                }
            }
        }
    }
}

fn parse_info(stream: TcpStream, within: Duration) -> Result<ServerInfo, ClientError> {
    let stream = Deadline {
        stream,
        started: Instant::now(),
        within,
    };
    let mut reader = BufReader::new(stream).take(MAX_EVENT as u64);
    let head = read_head(&mut reader)?;
    let body = read_body(&mut reader, head.length)?;
    if head.status != 200 {
        return Err(rejected(head.status, &body));
    }
    let json: Value = serde_json::from_slice(&body)
        .map_err(|e| ClientError::Protocol(format!("/v1/models: {e}")))?;
    let model = json
        .pointer("/data/0/id")
        .and_then(Value::as_str)
        .ok_or_else(|| ClientError::Protocol("/v1/models lists no model".into()))?;
    let context = json
        .pointer("/data/0/context_window")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            ClientError::Protocol("the server does not report its context window".into())
        })?;
    Ok(ServerInfo {
        model: model.to_owned(),
        context,
    })
}

fn rejected(status: u16, body: &[u8]) -> ClientError {
    let message = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(body).chars().take(200).collect());
    ClientError::Rejected { status, message }
}

fn read_head(reader: &mut impl BufRead) -> Result<Head, ClientError> {
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let status = line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| ClientError::Protocol(format!("bad status line {:?}", line.trim())))?;
    let mut length = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    Ok(Head { status, length })
}

fn read_body(reader: &mut impl Read, length: usize) -> Result<Vec<u8>, ClientError> {
    let mut body = Vec::new();
    reader
        .take(length.min(MAX_CHUNK) as u64)
        .read_to_end(&mut body)?;
    Ok(body)
}

/// The next piece of a chunked body; `None` after the last one.
fn read_chunk(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>, ClientError> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(ClientError::Broken(
            "the reply ended before it was complete".into(),
        ));
    }
    let size = line.trim().split(';').next().unwrap_or("");
    let size = usize::from_str_radix(size, 16)
        .map_err(|_| ClientError::Protocol(format!("bad chunk size {size:?}")))?;
    if size == 0 {
        return Ok(None);
    }
    if size > MAX_CHUNK {
        return Err(ClientError::Protocol("oversized chunk".into()));
    }
    let mut data = vec![0_u8; size];
    reader.read_exact(&mut data)?;
    let mut end = [0_u8; 2];
    reader.read_exact(&mut end)?;
    Ok(Some(data))
}

/// `pending` held no complete event before `from`, so only the new bytes need scanning.
fn event_end(pending: &[u8], from: usize) -> Option<usize> {
    let tail = pending.get(from..)?;
    tail.windows(2)
        .position(|w| w == b"\n\n")
        .and_then(|at| at.checked_add(from))
}

fn pump(stream: TcpStream, tx: &Sender<Event>) -> Result<(), ClientError> {
    let mut reader = BufReader::new(stream);
    let head = read_head(&mut reader)?;
    if head.status != 200 {
        let body = read_body(&mut reader, head.length)?;
        return Err(rejected(head.status, &body));
    }
    let mut reply = Reply::default();
    let mut pending = Vec::new();
    while let Some(chunk) = read_chunk(&mut reader)? {
        let fresh = pending.len().saturating_sub(1);
        pending.extend_from_slice(&chunk);
        if pending.len() > MAX_EVENT && event_end(&pending, fresh).is_none() {
            return Err(ClientError::Protocol("an event that never ends".into()));
        }
        while let Some(end) = event_end(&pending, 0) {
            let event: Vec<u8> = pending.drain(..end.saturating_add(2)).collect();
            if event_done(&event, &mut reply, tx)? {
                let _ = tx.send(Event::Done(reply));
                return Ok(());
            }
        }
    }
    Err(ClientError::Broken(
        "the reply ended before it was complete".into(),
    ))
}

/// Applies one server-sent event; true once the stream is over.
fn event_done(event: &[u8], reply: &mut Reply, tx: &Sender<Event>) -> Result<bool, ClientError> {
    let text = std::str::from_utf8(event).map_err(|e| ClientError::Protocol(e.to_string()))?;
    for payload in text.lines().filter_map(|l| l.strip_prefix("data:")) {
        let payload = payload.trim();
        if payload == "[DONE]" {
            return Ok(true);
        }
        let json: Value = serde_json::from_str(payload)
            .map_err(|e| ClientError::Protocol(format!("unreadable event: {e}")))?;
        if let Some(message) = json.pointer("/error/message").and_then(Value::as_str) {
            return Err(ClientError::Rejected {
                status: 500,
                message: message.to_owned(),
            });
        }
        if let Some(piece) = json
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            reply.text.push_str(piece);
            let _ = tx.send(Event::Text(piece.to_owned()));
        }
        if let Some(finish) = json
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            reply.finish = finish.to_owned();
        }
        if let Some(id) = json.get("marin_session_id").and_then(Value::as_str) {
            reply.session = Some(id.to_owned());
        }
        let count = |key: &str| {
            json.pointer(&format!("/usage/{key}"))
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
        };
        if let (Some(prompt), Some(completion)) =
            (count("prompt_tokens"), count("completion_tokens"))
        {
            reply.usage = Usage { prompt, completion };
        }
    }
    Ok(false)
}
