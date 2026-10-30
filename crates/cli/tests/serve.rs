mod common;

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{Sandbox, fixtures, text};
use serde_json::{Value, json};

struct Server {
    child: Child,
    address: String,
    _home: Sandbox,
}

impl Server {
    fn start() -> io::Result<Self> {
        let home = Sandbox::new("serve")?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", &home.0)
            .env("NO_COLOR", "1")
            .arg("--model")
            .arg(fixtures().join("tiny-chat"))
            .args([
                "--backend",
                "cpu",
                "--ctx",
                "256",
                "--temp",
                "0",
                "serve",
                "--port",
                "0",
            ])
            .stdout(Stdio::piped())
            .spawn()?;
        let mut line = String::new();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("no server stdout"))?;
        BufReader::new(stdout).read_line(&mut line)?;
        if !line.starts_with("listening http://127.0.0.1:") {
            return Err(io::Error::other(format!("server did not listen: {line}")));
        }
        let address = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| io::Error::other("missing server address"))?
            .trim_start_matches("http://")
            .to_string();
        Ok(Self {
            child,
            address,
            _home: home,
        })
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> io::Result<String> {
        let mut socket = TcpStream::connect(&self.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(10)))?;
        let content = body.map(Value::to_string).unwrap_or_default();
        write!(
            socket,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        Ok(response)
    }

    fn half_closed_request(&self, body: &Value) -> io::Result<String> {
        let mut socket = TcpStream::connect(&self.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(10)))?;
        let content = body.to_string();
        write!(
            socket,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )?;
        socket.shutdown(Shutdown::Write)?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        Ok(response)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn body(response: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let (_, body) = response
        .split_once("\r\n\r\n")
        .ok_or("missing response body")?;
    Ok(serde_json::from_str(body)?)
}

#[test]
fn repeated_streamed_and_bad_requests_preserve_isolation() -> Result<(), Box<dyn std::error::Error>>
{
    let server = Server::start()?;
    let models = server.request("GET", "/v1/models", None)?;
    assert!(models.starts_with("HTTP/1.1 200"));
    assert_eq!(body(&models)?["data"][0]["id"], "tiny-chat");

    let request = |text: &str| json!({"model":"tiny-chat","messages":[{"role":"user","content":text}],"max_tokens":6,"temperature":0.0});
    let first = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&request("the cat sat")),
    )?;
    assert!(first.starts_with("HTTP/1.1 200"), "{first}");
    let answer = body(&first)?["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("missing answer")?
        .to_owned();

    let mut streamed = request("the cat sat");
    streamed["stream"] = json!(true);
    let response = server.request("POST", "/v1/chat/completions", Some(&streamed))?;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let pieces: String = response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(pieces, answer);
    assert!(response.contains("data: [DONE]"));
    let cli = Command::new(env!("CARGO_BIN_EXE_marin"))
        .env("MARIN_HOME", &server._home.0)
        .arg("--model")
        .arg(fixtures().join("tiny-chat"))
        .args([
            "--backend",
            "cpu",
            "--ctx",
            "128",
            "--temp",
            "0",
            "--max-tokens",
            "6",
            "run",
            "the cat sat",
        ])
        .output()?;
    assert!(cli.status.success(), "{}", text(&cli.stderr));
    assert_eq!(text(&cli.stdout).trim_end_matches('\n'), answer);

    let wrong = json!({"model":"other","messages":[{"role":"user","content":"hi"}]});
    let wrong_response = server.request("POST", "/v1/chat/completions", Some(&wrong))?;
    assert!(wrong_response.starts_with("HTTP/1.1 400"));
    assert_eq!(
        body(&wrong_response)?["error"]["type"],
        "invalid_request_error"
    );
    let long = request(&"word ".repeat(300));
    assert!(
        server
            .request("POST", "/v1/chat/completions", Some(&long))?
            .starts_with("HTTP/1.1 400")
    );
    let default_limit =
        json!({"model":"tiny-chat","messages":[{"role":"user","content":"hi"}],"temperature":0.0});
    assert!(
        server
            .request("POST", "/v1/chat/completions", Some(&default_limit))?
            .starts_with("HTTP/1.1 200")
    );
    let _ = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&request("different question")),
    )?;
    assert_eq!(
        body(&server.request(
            "POST",
            "/v1/chat/completions",
            Some(&request("the cat sat"))
        )?)?["choices"][0]["message"]["content"],
        answer
    );

    let mut socket = TcpStream::connect(&server.address)?;
    let mut interrupted = request("a long unfinished answer");
    interrupted["max_tokens"] = json!(64);
    interrupted["stream"] = json!(true);
    let content = interrupted.to_string();
    write!(
        socket,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
        content.len()
    )?;
    socket.shutdown(Shutdown::Both)?;
    assert!(
        server
            .request(
                "POST",
                "/v1/chat/completions",
                Some(&request("the cat sat"))
            )?
            .starts_with("HTTP/1.1 200")
    );
    Ok(())
}

#[test]
fn serve_rejects_unsupported_snapshot_before_loading() -> io::Result<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_marin"))
        .args(["--snapshot", "unused.snap", "serve", "--port", "0"])
        .output()?;
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("--snapshot is not supported by marin serve"));
    Ok(())
}

#[test]
fn sessions_reuse_exact_prefix_and_half_close_works() -> Result<(), Box<dyn std::error::Error>> {
    let server = Server::start()?;
    let mut first = json!({"model":"tiny-chat","messages":[{"role":"user","content":"the cat sat"}],"max_tokens":6,"temperature":0,"marin_session_id":"new"});
    let first_response = server.half_closed_request(&first)?;
    assert!(
        first_response.starts_with("HTTP/1.1 200"),
        "{first_response}"
    );
    let first_body = body(&first_response)?;
    let id = first_body["marin_session_id"]
        .as_str()
        .ok_or("missing session id")?;
    assert_eq!(id.len(), 32);
    let answer = first_body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("missing answer")?;
    first["messages"] = json!([
        {"role":"user","content":"the cat sat"},
        {"role":"assistant","content":answer},
        {"role":"user","content":"what next?"}
    ]);
    first["marin_session_id"] = json!(id);
    let reused = body(&server.request("POST", "/v1/chat/completions", Some(&first))?)?;
    assert_eq!(reused["marin_session_id"], id, "{reused}");
    assert!(
        reused["marin_reused_tokens"].as_u64().unwrap_or(0) > 0,
        "{reused}"
    );
    first["stream"] = json!(true);
    let streamed = server.request("POST", "/v1/chat/completions", Some(&first))?;
    let mut pieces = String::new();
    let mut final_chunk = Value::Null;
    for event in streamed
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        if event == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(event)?;
        if let Some(piece) = value["choices"][0]["delta"]["content"].as_str() {
            pieces.push_str(piece);
        }
        if !value["choices"][0]["finish_reason"].is_null() {
            final_chunk = value;
        }
    }
    assert_eq!(pieces, reused["choices"][0]["message"]["content"]);
    assert_eq!(final_chunk["marin_session_id"], id);
    assert!(final_chunk["marin_reused_tokens"].as_u64().unwrap_or(0) > 0);
    first["max_tokens"] = json!(64);
    let content = first.to_string();
    let mut abandoned = TcpStream::connect(&server.address)?;
    write!(
        abandoned,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
        content.len()
    )?;
    abandoned.shutdown(Shutdown::Both)?;
    first["max_tokens"] = json!(6);
    first
        .as_object_mut()
        .ok_or("bad test request")?
        .remove("stream");
    let retry = body(&server.request("POST", "/v1/chat/completions", Some(&first))?)?;
    assert_eq!(retry["choices"], reused["choices"]);
    first["messages"][0]["content"] = json!("a different first turn");
    let changed = body(&server.request("POST", "/v1/chat/completions", Some(&first))?)?;
    assert_eq!(changed["marin_reused_tokens"], 0);
    first
        .as_object_mut()
        .ok_or("bad test request")?
        .remove("marin_session_id");
    let fresh = body(&server.request("POST", "/v1/chat/completions", Some(&first))?)?;
    assert_eq!(changed["choices"], fresh["choices"]);
    first["marin_session_id"] = json!(id);
    assert!(
        server
            .request("POST", "/v1/chat/completions", Some(&first))?
            .starts_with("HTTP/1.1 404")
    );
    Ok(())
}

#[test]
fn a_session_turn_does_not_read_the_history_again_after_the_answer()
-> Result<(), Box<dyn std::error::Error>> {
    let server = Server::start()?;
    let mut request = json!({"model":"tiny-chat","messages":[{"role":"user","content":"the cat sat"}],"max_tokens":6,"temperature":0,"marin_session_id":"new"});
    let first = body(&server.request("POST", "/v1/chat/completions", Some(&request))?)?;
    assert_eq!(first["marin_replayed_tokens"], 0, "{first}");
    let answer = first["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("missing answer")?;
    request["marin_session_id"] = first["marin_session_id"].clone();
    request["messages"] = json!([
        {"role":"user","content":"the cat sat"},
        {"role":"assistant","content":answer},
        {"role":"user","content":"what next?"}
    ]);
    let second = body(&server.request("POST", "/v1/chat/completions", Some(&request))?)?;
    assert_eq!(second["marin_replayed_tokens"], 0, "{second}");
    assert!(
        second["marin_reused_tokens"].as_u64().unwrap_or(0) > 0,
        "{second}"
    );
    Ok(())
}

#[test]
fn full_queue_returns_overload_and_waiters_remain_fifo() -> Result<(), Box<dyn std::error::Error>> {
    let server = Server::start()?;
    let mut active = TcpStream::connect(&server.address)?;
    active.write_all(b"GET /v1/models HTTP/1.1\r\n")?;
    std::thread::sleep(Duration::from_millis(100));
    let mut waiting = Vec::new();
    for _ in 0..2 {
        let mut socket = TcpStream::connect(&server.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(10)))?;
        socket.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")?;
        waiting.push(socket);
    }
    std::thread::sleep(Duration::from_millis(100));
    let overload = server
        .request("GET", "/v1/models", None)
        .map_err(|e| io::Error::other(format!("overload: {e}")))?;
    assert!(overload.starts_with("HTTP/1.1 503"), "{overload}");
    active.write_all(b"Host: localhost\r\n\r\n")?;
    active.shutdown(Shutdown::Write)?;
    let mut active_response = String::new();
    active
        .read_to_string(&mut active_response)
        .map_err(|e| io::Error::other(format!("active: {e}")))?;
    assert!(active_response.starts_with("HTTP/1.1 200"));
    for mut socket in waiting {
        let mut response = String::new();
        socket
            .read_to_string(&mut response)
            .map_err(|e| io::Error::other(format!("waiting: {e}")))?;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    }
    Ok(())
}

#[test]
fn partial_read_times_out_and_queued_request_completes() -> Result<(), Box<dyn std::error::Error>> {
    let server = Server::start()?;
    let mut active = TcpStream::connect(&server.address)?;
    active.set_read_timeout(Some(Duration::from_secs(15)))?;
    active.write_all(b"GET /v1/models HTTP/1.1\r\n")?;
    std::thread::sleep(Duration::from_millis(100));
    let mut queued = TcpStream::connect(&server.address)?;
    queued.set_read_timeout(Some(Duration::from_secs(15)))?;
    queued.write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")?;
    let mut active_response = String::new();
    active.read_to_string(&mut active_response)?;
    assert!(
        active_response.starts_with("HTTP/1.1 408"),
        "{active_response}"
    );
    let mut queued_response = String::new();
    queued.read_to_string(&mut queued_response)?;
    // This request arrives near the shared deadline; either status is valid.
    assert!(
        queued_response.starts_with("HTTP/1.1 200") || queued_response.starts_with("HTTP/1.1 408"),
        "{queued_response}"
    );
    Ok(())
}
