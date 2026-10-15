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
                "128",
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
