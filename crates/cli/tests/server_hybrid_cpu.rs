//! A session turn on a Qwen3.5 CPU server must not read the history again after the answer, and must answer exactly
//! as a full replay does.

use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

type Res<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct Server {
    child: Child,
    address: String,
    home: PathBuf,
}

impl Server {
    fn start() -> io::Result<Self> {
        let home = std::env::temp_dir().join(format!("marin-hybrid-cpu-{}", std::process::id()));
        fs::create_dir_all(&home)?;
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let model = home.join("model");
        fs::create_dir_all(&model)?;
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            fs::copy(fixtures.join("tiny-qwen35").join(name), model.join(name))?;
        }
        fs::copy(
            fixtures.join("tiny-chat/tokenizer_config.json"),
            model.join("tokenizer_config.json"),
        )?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", &home)
            .env("NO_COLOR", "1")
            .arg("--model")
            .arg(&model)
            .args(["--backend", "cpu", "--ctx", "512", "--temp", "0"])
            .args(["serve", "--port", "0"])
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("no server stdout"))?;
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line)?;
        let address = line
            .strip_prefix("listening http://")
            .and_then(|rest| rest.split_whitespace().next())
            .ok_or_else(|| io::Error::other(format!("server did not listen: {line}")))?
            .to_owned();
        Ok(Self {
            child,
            address,
            home,
        })
    }

    /// Sends a request and closes the connection without reading the answer.
    fn abandon(&self, body: &Value) -> Res {
        let mut socket = TcpStream::connect(&self.address)?;
        let content = body.to_string();
        write!(
            socket,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )?;
        socket.shutdown(Shutdown::Both)?;
        Ok(())
    }

    fn post(&self, body: &Value) -> Res<Value> {
        let mut socket = TcpStream::connect(&self.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(60)))?;
        let content = body.to_string();
        write!(
            socket,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        let (_, body) = response.split_once("\r\n\r\n").ok_or("no response body")?;
        Ok(serde_json::from_str(body)?)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.home);
    }
}

#[test]
fn a_hybrid_cpu_session_turn_is_not_read_again_and_matches_a_full_replay() -> Res {
    let server = Server::start()?;
    let mut request = json!({"model":"model","messages":[{"role":"user","content":"the cat sat on the mat"}],"max_tokens":6,"temperature":0,"marin_session_id":"new"});
    let first = server.post(&request)?;
    assert_eq!(first["marin_replayed_tokens"], 0, "{first}");
    let answer = first["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("missing answer")?;
    request["marin_session_id"] = first["marin_session_id"].clone();
    request["messages"] = json!([
        {"role":"user","content":"the cat sat on the mat"},
        {"role":"assistant","content":answer},
        {"role":"user","content":"what next?"}
    ]);
    let second = server.post(&request)?;
    assert_eq!(second["marin_replayed_tokens"], 0, "{second}");
    assert!(
        second["marin_reused_tokens"].as_u64().unwrap_or(0) > 0,
        "{second}"
    );
    let mut abandoned = request.clone();
    abandoned["max_tokens"] = json!(400);
    abandoned["stream"] = json!(true);
    server.abandon(&abandoned)?;
    let third = server.post(&request)?;
    request
        .as_object_mut()
        .ok_or("bad test request")?
        .remove("marin_session_id");
    let fresh = server.post(&request)?;
    assert_eq!(second["choices"], fresh["choices"], "reuse equals replay");
    assert_eq!(third["choices"], fresh["choices"], "a repeat equals replay");
    Ok(())
}
