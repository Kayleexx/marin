use std::io::{self, Read, Write};
use std::net::TcpStream;

use serde_json::Value;

const MAX_HEADER: usize = 16 * 1024;
const MAX_BODY: usize = 1024 * 1024;

pub(super) struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

pub(super) enum RequestError {
    Bad(u16, &'static str),
    Io(io::Error),
}

impl From<io::Error> for RequestError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub(super) fn read(stream: &mut TcpStream) -> Result<Option<Request>, RequestError> {
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return if raw.is_empty() {
                Ok(None)
            } else {
                Err(RequestError::Bad(400, "incomplete HTTP headers"))
            };
        }
        raw.extend_from_slice(chunk.get(..count).unwrap_or(&[]));
        if let Some(start) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let end = start.saturating_add(4);
            if end > MAX_HEADER {
                return Err(RequestError::Bad(431, "HTTP headers exceed 16 KiB"));
            }
            break end;
        }
        if raw.len() > MAX_HEADER {
            return Err(RequestError::Bad(431, "HTTP headers exceed 16 KiB"));
        }
    };
    let header = std::str::from_utf8(raw.get(..header_end).unwrap_or(&[]))
        .map_err(|_| RequestError::Bad(400, "HTTP headers must be UTF-8"))?
        .to_owned();
    let mut lines = header.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split_ascii_whitespace();
    let method = request_line.next().unwrap_or_default();
    let path = request_line.next().unwrap_or_default();
    let version = request_line.next().unwrap_or_default();
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || request_line.next().is_some()
        || method.is_empty()
        || !path.starts_with('/')
    {
        return Err(RequestError::Bad(400, "invalid HTTP request line"));
    }
    let mut length = None;
    let mut json = false;
    for line in lines.take_while(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or(RequestError::Bad(400, "invalid HTTP header"))?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err(RequestError::Bad(400, "duplicate content length"));
            }
            length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| RequestError::Bad(400, "invalid content length"))?,
            );
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(RequestError::Bad(
                400,
                "chunked request bodies are unsupported",
            ));
        } else if name.eq_ignore_ascii_case("content-type") {
            json = value
                .split(';')
                .next()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("application/json"));
        }
    }
    let length = length.unwrap_or(0);
    if length > MAX_BODY {
        return Err(RequestError::Bad(413, "request body exceeds 1 MiB"));
    }
    if method == "POST" && (!json || length == 0) {
        return Err(RequestError::Bad(415, "POST requires a JSON body"));
    }
    let target = header_end.saturating_add(length);
    while raw.len() < target {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return Err(RequestError::Bad(400, "incomplete HTTP body"));
        }
        raw.extend_from_slice(chunk.get(..count).unwrap_or(&[]));
    }
    let body = raw.get(header_end..target).unwrap_or(&[]).to_vec();
    Ok(Some(Request {
        method: method.to_string(),
        path: path.to_string(),
        body,
    }))
}

pub(super) fn json(stream: &mut TcpStream, status: u16, body: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(body).map_err(io::Error::other)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
}

pub(super) fn sse_start(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
    stream.flush()
}

fn chunk(stream: &mut TcpStream, data: &[u8]) -> io::Result<()> {
    write!(stream, "{:x}\r\n", data.len())?;
    stream.write_all(data)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

pub(super) fn sse(stream: &mut TcpStream, body: &Value) -> io::Result<()> {
    let json = serde_json::to_string(body).map_err(io::Error::other)?;
    chunk(stream, format!("data: {json}\n\n").as_bytes())
}

pub(super) fn sse_end(stream: &mut TcpStream) -> io::Result<()> {
    chunk(stream, b"data: [DONE]\n\n")?;
    stream.write_all(b"0\r\n\r\n")
}
