//! A real `marin serve` on a packed tiny Qwen3.5 model on the GPU, stopped when dropped.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};

use super::qwen35::Res;

pub struct Server {
    child: Child,
    pub address: String,
    pub name: String,
    log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    pub fn start(model: &Path, home: &Path) -> Res<Self> {
        let tiny = ["--quant-group", "32", "--ctx", "1024"];
        Self::spawn(model, home, &tiny, &[], "packed")
    }

    /// Starts a server. The GPU of a server that was just stopped is given back a moment after its process ends, so a
    /// refusal for lack of free memory is tried again for up to half a minute before it is reported.
    pub fn spawn(
        model: &Path,
        home: &Path,
        extra: &[&str],
        serve: &[&str],
        name: &str,
    ) -> Res<Self> {
        for _ in 0..30 {
            match Self::try_spawn(model, home, extra, serve, name) {
                Err(e) if e.to_string().contains("the GPU has") => {
                    std::thread::sleep(Duration::from_secs(1));
                }
                started => return started,
            }
        }
        Self::try_spawn(model, home, extra, serve, name)
    }

    fn try_spawn(
        model: &Path,
        home: &Path,
        extra: &[&str],
        serve: &[&str],
        name: &str,
    ) -> Res<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", home)
            .env("NO_COLOR", "1")
            .arg("--model")
            .arg(model)
            .args(["--quantize", "int8"])
            .args(["--backend", "cuda", "--temp", "0"])
            .args(extra)
            .args(["serve", "--port", "0"])
            .args(serve)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("no server stdout")?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        while !line.starts_with("listening http://") {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                let mut why = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    stderr.read_to_string(&mut why)?;
                }
                return Err(format!("the server did not start: {}", why.trim()).into());
            }
        }
        let address = line
            .split_whitespace()
            .nth(1)
            .ok_or("missing server address")?
            .trim_start_matches("http://")
            .to_owned();
        let log = Arc::new(Mutex::new(Vec::new()));
        let stderr = child.stderr.take().ok_or("no server stderr")?;
        let sink = Arc::clone(&log);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                sink.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(line);
            }
        });
        Ok(Self {
            child,
            address,
            name: name.to_owned(),
            log,
        })
    }

    pub fn request(
        &self,
        messages: &Value,
        session: &str,
        max_tokens: usize,
        stream: bool,
    ) -> String {
        let content = json!({"model": self.name, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "marin_session_id": session, "stream": stream}).to_string();
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{content}",
            content.len()
        )
    }

    /// The status and body of one request.
    pub fn post(&self, messages: &Value, session: &str) -> Res<(u16, Value)> {
        let mut socket = TcpStream::connect(&self.address)?;
        socket.set_read_timeout(Some(Duration::from_secs(300)))?;
        socket.write_all(self.request(messages, session, 6, false).as_bytes())?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        let (head, body) = response.split_once("\r\n\r\n").ok_or("no response body")?;
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .ok_or("no status")?;
        Ok((status, serde_json::from_str(body)?))
    }

    /// How many lines the server has written to its standard error that contain `needle`.
    pub fn logged(&self, needle: &str) -> usize {
        let log = self.log.lock().unwrap_or_else(PoisonError::into_inner);
        log.iter().filter(|line| line.contains(needle)).count()
    }

    /// A request that must succeed.
    pub fn ask(&self, messages: &Value, session: &str) -> Res<Value> {
        match self.post(messages, session)? {
            (200, body) => Ok(body),
            (status, body) => Err(format!("{status}: {body}").into()),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
