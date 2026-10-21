//! A session on the server after another program has taken nearly all the free GPU memory: the answers are the same
//! as without the pressure and the finished turn is put back at the end of the conversation without reading it again.
//! Needs a GPU; the only test in this binary, so nothing else is using the GPU while it holds the memory.

#[path = "common/qwen35.rs"]
mod qwen35;

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use serde_json::{Value, json};

const LEFT_FREE: usize = 150 << 20;

struct Server {
    child: Child,
    address: String,
}

impl Server {
    fn start(model: &std::path::Path, home: &std::path::Path) -> Res<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", home)
            .env("NO_COLOR", "1")
            .arg("--model")
            .arg(model)
            .args(["--quantize", "int8", "--quant-group", "32"])
            .args(["--backend", "cuda", "--ctx", "1024", "--temp", "0"])
            .args(["serve", "--port", "0"])
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("no server stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while !line.starts_with("listening http://") {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err("the server did not start".into());
            }
        }
        let address = line
            .split_whitespace()
            .nth(1)
            .ok_or("missing server address")?
            .trim_start_matches("http://")
            .to_owned();
        Ok(Self { child, address })
    }

    fn ask(&self, messages: &Value, session: &str) -> Res<Value> {
        let content = json!({"model": "packed", "messages": messages, "max_tokens": 6, "temperature": 0, "marin_session_id": session}).to_string();
        let mut socket = TcpStream::connect(&self.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(60)))?;
        write!(
            socket,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        let (head, body) = response.split_once("\r\n\r\n").ok_or("no response body")?;
        if !head.starts_with("HTTP/1.1 200") {
            return Err(format!("{head}\n{body}").into());
        }
        Ok(serde_json::from_str(body)?)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn conversation(server: &Server, pressure: bool) -> Res<Vec<(String, u64, u64)>> {
    let mut messages = Vec::new();
    let mut session = "new".to_owned();
    let mut taken = None;
    let mut turns = Vec::new();
    for n in 0..4 {
        if n == 1 && pressure {
            let gpu = model::Gpu::new()?;
            let free = gpu.info()?.free_bytes;
            taken = Some((gpu.zeros(free.saturating_sub(LEFT_FREE) / 4)?, gpu));
        }
        let said = format!("the quick brown fox number {n} jumps over the lazy dog ").repeat(2);
        messages.push(json!({"role": "user", "content": said}));
        let reply = server.ask(&Value::Array(messages.clone()), &session)?;
        let answer = reply
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or("no answer")?
            .to_owned();
        session = reply
            .get("marin_session_id")
            .and_then(Value::as_str)
            .ok_or("no session")?
            .to_owned();
        messages.push(json!({"role": "assistant", "content": answer}));
        let count = |key: &str| reply.get(key).and_then(Value::as_u64).ok_or(key.to_owned());
        turns.push((
            answer,
            count("marin_reused_tokens")?,
            count("marin_replayed_tokens")?,
        ));
    }
    drop(taken);
    Ok(turns)
}

#[test]
fn session_turns_under_gpu_memory_pressure_answer_the_same_and_read_nothing_again() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let scratch = Scratch::new("serve-pressure")?;
    let packed = packed_tiny(&scratch.0)?;
    let calm = {
        let server = Server::start(&packed, &scratch.0)?;
        conversation(&server, false)?
    };
    let pressed = {
        let server = Server::start(&packed, &scratch.0)?;
        conversation(&server, true)?
    };
    assert_eq!(
        calm.iter().map(|t| &t.0).collect::<Vec<_>>(),
        pressed.iter().map(|t| &t.0).collect::<Vec<_>>(),
        "the same answers"
    );
    for (n, (_, reused, replayed)) in pressed.iter().enumerate().skip(1) {
        assert_eq!(*replayed, 0, "turn {n} was not read again");
        assert!(*reused > 0, "turn {n} kept the conversation");
    }
    let _ = io::stdout().flush();
    Ok(())
}
