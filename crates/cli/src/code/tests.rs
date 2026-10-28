//! A scripted stand-in for `marin serve` and a scratch project, shared by the tests of the client and the tool loop.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

use super::client::{Ask, Client, ClientError};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A project folder that is removed when it goes out of scope.
pub(super) struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("marin-code-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn file(&self, name: &str, text: impl AsRef<[u8]>) -> &Self {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
        self
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What the fake server does with one request.
#[derive(Clone)]
pub(super) enum Step {
    /// Streams this reply in a few pieces, then the final event with usage.
    Reply(String),
    /// Refuses with this status and error message.
    Refuse(u16, String),
    /// Streams part of a reply and drops the connection.
    Cut(String),
    /// Accepts the request and never answers.
    Hang,
    /// Streams data that never forms an event.
    Flood,
}

pub(super) struct Fake {
    pub addr: SocketAddr,
    pub bodies: Arc<Mutex<Vec<Value>>>,
}

impl Fake {
    /// Serves one scripted step per request, in order; the last step repeats if the script runs out.
    pub fn start(script: Vec<Step>) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let bodies: Arc<Mutex<Vec<Value>>> = Arc::default();
        let seen = Arc::clone(&bodies);
        thread::spawn(move || {
            for (n, stream) in listener.incoming().enumerate() {
                let Ok(stream) = stream else { return };
                let step = script.get(n).or(script.last()).cloned();
                let seen = Arc::clone(&seen);
                thread::spawn(move || {
                    if let Some(step) = step {
                        let _ = serve(stream, &step, &seen);
                    }
                });
            }
        });
        Self { addr, bodies }
    }

    pub fn client(&self) -> Client {
        Client::new(self.addr).with_silence(Duration::from_millis(600))
    }

    pub fn requests(&self) -> Vec<Value> {
        self.bodies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

fn serve(stream: TcpStream, step: &Step, seen: &Mutex<Vec<Value>>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let is_get = line.starts_with("GET");
    let mut length = 0;
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0_u8; length];
    reader.read_exact(&mut body)?;
    let mut out = stream;
    if is_get {
        let json =
            json!({"object": "list", "data": [{"id": "fake-model", "context_window": 4096}]})
                .to_string();
        write!(
            out,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
            json.len()
        )?;
        return Ok(());
    }
    if let Ok(value) = serde_json::from_slice(&body) {
        seen.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(value);
    }
    match step {
        Step::Refuse(status, message) => {
            let json = json!({"error": {"message": message}}).to_string();
            write!(
                out,
                "HTTP/1.1 {status} Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                json.len()
            )
        }
        Step::Hang => {
            thread::sleep(Duration::from_secs(5));
            Ok(())
        }
        Step::Flood => {
            out.write_all(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )?;
            let block = "x".repeat(1 << 19);
            for _ in 0..16 {
                chunk(&mut out, &block)?;
            }
            Ok(())
        }
        Step::Reply(text) => stream_reply(&mut out, text, Some(body.len() / 3)),
        Step::Cut(text) => {
            stream_reply(&mut out, text, None)?;
            out.shutdown(Shutdown::Both)
        }
    }
}

fn chunk(out: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(out, "{:x}\r\n{data}\r\n", data.len())?;
    out.flush()
}

/// `prompt_tokens` is `None` for a reply that is cut off.
fn stream_reply(
    out: &mut TcpStream,
    text: &str,
    prompt_tokens: Option<usize>,
) -> std::io::Result<()> {
    out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
    let chars: Vec<char> = text.chars().collect();
    for piece in chars.chunks(7) {
        let piece: String = piece.iter().collect();
        let event =
            json!({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]});
        chunk(out, &format!("data: {event}\n\n"))?;
    }
    if let Some(prompt) = prompt_tokens {
        let last = json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "marin_session_id": "s1",
            "usage": {"prompt_tokens": prompt, "completion_tokens": 9, "total_tokens": prompt.saturating_add(9)}
        });
        chunk(out, &format!("data: {last}\n\n"))?;
        chunk(out, "data: [DONE]\n\n")?;
        out.write_all(b"0\r\n\r\n")?;
    }
    Ok(())
}

fn ask() -> Ask<'static> {
    Ask {
        model: "fake-model",
        messages: vec![json!({"role": "user", "content": "hi"})],
        session: "new",
        max_tokens: 64,
    }
}

fn never() -> AtomicBool {
    AtomicBool::new(false)
}

#[test]
fn a_streamed_reply_arrives_in_pieces_with_usage_and_session() {
    let fake = Fake::start(vec![Step::Reply("Hello from the fake server.".into())]);
    let mut pieces = 0;
    let reply = fake.client().chat(&ask(), &never(), |_| {
        pieces += 1;
        true
    });
    let reply = reply.unwrap();
    assert_eq!(reply.text, "Hello from the fake server.");
    assert!(pieces > 1);
    assert!(reply.usage.prompt > 0 && reply.usage.completion == 9);
    assert_eq!(reply.session.as_deref(), Some("s1"));
    let request = &fake.requests()[0];
    assert_eq!(request["stream"], true);
    assert_eq!(request["marin_session_id"], "new");
}

#[test]
fn the_server_context_window_is_read_from_the_model_list() {
    let fake = Fake::start(vec![Step::Hang]);
    let info = fake.client().info().unwrap();
    assert_eq!((info.model.as_str(), info.context), ("fake-model", 4096));
}

#[test]
fn a_refusal_carries_the_servers_message() {
    let fake = Fake::start(vec![Step::Refuse(503, "queue is full".into())]);
    let err = fake
        .client()
        .chat(&ask(), &never(), |_| true)
        .err()
        .unwrap();
    assert!(
        matches!(&err, ClientError::Rejected { status: 503, message } if message == "queue is full"),
        "{err}"
    );
}

#[test]
fn a_dropped_connection_is_an_error_not_a_short_answer() {
    let fake = Fake::start(vec![Step::Cut("half of an ans".into())]);
    let err = fake
        .client()
        .chat(&ask(), &never(), |_| true)
        .err()
        .unwrap();
    assert!(matches!(err, ClientError::Broken(_)), "{err}");
}

#[test]
fn a_server_that_goes_quiet_times_out() {
    let fake = Fake::start(vec![Step::Hang]);
    let err = fake
        .client()
        .chat(&ask(), &never(), |_| true)
        .err()
        .unwrap();
    assert!(matches!(err, ClientError::Silent(_)), "{err}");
}

#[test]
fn cancelling_hangs_up_promptly() {
    let fake = Fake::start(vec![Step::Hang]);
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(150));
        flag.store(true, Ordering::Relaxed);
    });
    let began = std::time::Instant::now();
    let err = Client::new(fake.addr)
        .chat(&ask(), &cancel, |_| true)
        .err()
        .unwrap();
    assert!(matches!(err, ClientError::Cancelled));
    assert!(began.elapsed() < Duration::from_secs(2));
}

#[test]
fn the_caller_can_end_a_reply_early() {
    let fake = Fake::start(vec![Step::Reply(
        "a very long runaway answer that goes on".into(),
    )]);
    let reply = fake.client().chat(&ask(), &never(), |_| false).unwrap();
    assert_eq!(reply.finish, "client_stop");
    assert!(reply.text.len() < 40);
}

#[test]
fn nothing_listening_is_reported_as_unreachable() {
    // A port that was just released can be taken by another program before the connect; any of a few free ports will do.
    let unreachable = (0..5).any(|_| {
        let addr = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            listener.local_addr().unwrap()
        };
        matches!(
            Client::new(addr).info().err(),
            Some(ClientError::Unreachable(_))
        )
    });
    assert!(unreachable);
}

#[test]
fn an_event_that_never_ends_is_refused_instead_of_buffered_forever() {
    let fake = Fake::start(vec![Step::Flood]);
    let err = fake
        .client()
        .chat(&ask(), &never(), |_| true)
        .err()
        .unwrap();
    assert!(matches!(err, ClientError::Protocol(_)), "{err}");
}
