mod http;

use std::io::{self, Read};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chat::{Message, Role};
use runtime::{GenerateParams, StopReason, generate};
use sampler::Sampler;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::CliError;
use crate::load::Loaded;
use crate::session::Settings;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    model: String,
    messages: Vec<InputMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputMessage {
    role: InputRole,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum InputRole {
    System,
    User,
    Assistant,
}

fn api_error(status: u16, message: impl ToString) -> Value {
    let kind = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json!({"error": {"message": message.to_string(), "type": kind}})
}

fn send_error(stream: &mut TcpStream, status: u16, message: impl ToString) -> io::Result<()> {
    http::json(stream, status, &api_error(status, message))
}

fn finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfText => "stop",
        StopReason::MaxTokens | StopReason::ContextFull => "length",
        StopReason::Cancelled => "cancelled",
    }
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
    })
}

fn watch_disconnect(
    stream: &TcpStream,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
) -> io::Result<thread::JoinHandle<()>> {
    let mut peer = stream.try_clone()?;
    peer.set_read_timeout(Some(Duration::from_millis(100)))?;
    Ok(thread::spawn(move || {
        let mut byte = [0_u8; 1];
        while !done.load(Ordering::Relaxed) {
            match peer.read(&mut byte) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) => {}
                _ => {
                    if !done.load(Ordering::Relaxed) {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    break;
                }
            }
        }
    }))
}

fn validated_prompt(
    loaded: &Loaded,
    input: &Completion,
    defaults: Settings,
) -> Result<(Vec<u32>, Sampler, usize), String> {
    if input.model != loaded.name {
        return Err(format!("model must be `{}` (see /v1/models)", loaded.name));
    }
    if input.messages.is_empty()
        || !matches!(
            input.messages.last().map(|m| &m.role),
            Some(InputRole::User)
        )
        || input
            .messages
            .last()
            .is_some_and(|m| m.content.trim().is_empty())
    {
        return Err("messages must end with a non-empty user message".into());
    }
    let mut messages: Vec<Message> = input
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                InputRole::System => Role::System,
                InputRole::User => Role::User,
                InputRole::Assistant => Role::Assistant,
            };
            Message::new(role, &m.content)
        })
        .collect();
    if !messages.iter().any(|m| m.role == Role::System) {
        messages.insert(0, Message::new(Role::System, chat::DEFAULT_SYSTEM));
    }
    let text = chat::render(&messages, &loaded.dir.tokenizer.special_tokens())
        .map_err(|e| e.to_string())?;
    let prompt = loaded
        .dir
        .tokenizer
        .encode(&text)
        .map_err(|e| e.to_string())?;
    if prompt.len() > loaded.ctx {
        return Err(format!(
            "prompt has {} tokens but context holds {}",
            prompt.len(),
            loaded.ctx
        ));
    }
    let max_tokens = input
        .max_tokens
        .unwrap_or(defaults.max_tokens.min(loaded.ctx));
    if max_tokens == 0 || max_tokens > loaded.ctx {
        return Err(format!("max_tokens must be between 1 and {}", loaded.ctx));
    }
    let settings = Settings {
        max_tokens,
        temperature: input.temperature.unwrap_or(defaults.temperature),
        top_p: input.top_p.unwrap_or(defaults.top_p),
        top_k: input.top_k.unwrap_or(defaults.top_k),
        seed: input.seed.or(defaults.seed),
    };
    let seed = settings.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let sampler = Sampler::new(settings.sampler_config(seed)).map_err(|e| e.to_string())?;
    Ok((prompt, sampler, max_tokens))
}

fn completion(
    stream: &mut TcpStream,
    loaded: &Loaded,
    defaults: Settings,
    request_id: u64,
    body: &[u8],
) -> io::Result<()> {
    let input: Completion = match serde_json::from_slice(body) {
        Ok(value) => value,
        Err(e) => return send_error(stream, 400, format!("invalid JSON request: {e}")),
    };
    let (prompt, mut sampler, max_tokens) = match validated_prompt(loaded, &input, defaults) {
        Ok(value) => value,
        Err(e) => return send_error(stream, 400, e),
    };
    let mut cache = match loaded.model.new_cache(loaded.ctx) {
        Ok(cache) => cache,
        Err(e) => return send_error(stream, 500, e),
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let id = format!("chatcmpl-{created}-{request_id}");
    if input.stream {
        http::sse_start(stream)?;
        http::sse(
            stream,
            &chunk(
                &id,
                created,
                &loaded.name,
                json!({"role": "assistant"}),
                None,
            ),
        )?;
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let watcher = watch_disconnect(stream, Arc::clone(&cancel), Arc::clone(&done))?;
    let params = GenerateParams {
        max_tokens,
        stop_tokens: loaded.stop_tokens(),
        usable_vocab: loaded.dir.tokenizer.vocab_size(),
        speculation: None,
    };
    let mut decoder = loaded.dir.tokenizer.stream();
    let mut answer = String::new();
    let mut failure = None;
    let result = generate(
        &loaded.model,
        &mut cache,
        &mut sampler,
        &prompt,
        &params,
        &cancel,
        |token| {
            if failure.is_some() {
                return;
            }
            match decoder.push(token) {
                Ok(Some(piece)) => {
                    answer.push_str(&piece);
                    if input.stream
                        && let Err(e) = http::sse(
                            stream,
                            &chunk(&id, created, &loaded.name, json!({"content": piece}), None),
                        )
                    {
                        failure = Some(e.to_string());
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    failure = Some(e.to_string());
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        },
    );
    done.store(true, Ordering::Relaxed);
    let _ = stream.shutdown(Shutdown::Read);
    let _ = watcher.join();
    if let Some(message) = failure {
        if input.stream {
            let _ = http::sse(stream, &api_error(500, message));
            let _ = http::sse_end(stream);
            return Ok(());
        }
        return send_error(stream, 500, message);
    }
    let outcome = match result {
        Ok(value) => value,
        Err(e) => {
            if input.stream {
                let _ = http::sse(stream, &api_error(500, e));
                let _ = http::sse_end(stream);
                return Ok(());
            }
            return send_error(stream, 500, e);
        }
    };
    if outcome.stop == StopReason::Cancelled {
        return Ok(());
    }
    if input.stream {
        http::sse(
            stream,
            &chunk(
                &id,
                created,
                &loaded.name,
                json!({}),
                Some(finish_reason(outcome.stop)),
            ),
        )?;
        http::sse_end(stream)
    } else {
        http::json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": loaded.name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": finish_reason(outcome.stop)
                }],
                "usage": {
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.generated,
                    "total_tokens": outcome.prompt_tokens.saturating_add(outcome.generated)
                }
            }),
        )
    }
}

fn connection(
    stream: &mut TcpStream,
    loaded: &Loaded,
    defaults: Settings,
    id: u64,
) -> io::Result<()> {
    let request = match http::read(stream) {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(http::RequestError::Bad(status, message)) => {
            return send_error(stream, status, message);
        }
        Err(http::RequestError::Io(e)) => return Err(e),
    };
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => http::json(
            stream,
            200,
            &json!({"object":"list","data":[{"id":loaded.name,"object":"model"}]}),
        ),
        ("POST", "/v1/chat/completions") => completion(stream, loaded, defaults, id, &request.body),
        (_, "/v1/models" | "/v1/chat/completions") => send_error(stream, 405, "method not allowed"),
        _ => send_error(stream, 404, "unknown endpoint"),
    }
}

pub fn run(loaded: Loaded, defaults: Settings, port: u16) -> Result<(), CliError> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let address = listener.local_addr()?;
    println!("listening http://{address} (model: {})", loaded.name);
    for (id, incoming) in listener.incoming().enumerate() {
        let mut stream = incoming?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        stream.set_nodelay(true)?;
        if let Err(e) = connection(&mut stream, &loaded, defaults, id as u64) {
            eprintln!("request failed: {e}");
        }
    }
    Ok(())
}
