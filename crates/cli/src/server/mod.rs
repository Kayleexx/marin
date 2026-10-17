mod http;
mod queue;
mod request;
mod session;
mod watch;

use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use runtime::{
    CausalLm, GenerateParams, Kept, Retained, SequenceCache, StopReason, generate_reusing,
};
use serde_json::{Value, json};

use crate::CliError;
use crate::load::Loaded;
use crate::session::Settings;
use request::{Completion, validated_prompt};
use session::Stored;
use watch::watch_disconnect;

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

fn completion(
    stream: &mut TcpStream,
    loaded: &Loaded,
    defaults: Settings,
    request_id: u64,
    body: &[u8],
    stored: &mut Option<Stored>,
) -> io::Result<()> {
    let input: Completion = match serde_json::from_slice(body) {
        Ok(value) => value,
        Err(e) => return send_error(stream, 400, format!("invalid JSON request: {e}")),
    };
    let (prompt, history, mut sampler, max_tokens) =
        match validated_prompt(loaded, &input, defaults) {
            Ok(value) => value,
            Err(e) => return send_error(stream, 400, e),
        };
    let persistent = input.marin_session_id.is_some();
    let session_id = match input.marin_session_id.as_deref() {
        Some("new") => match session::new_id() {
            Ok(id) => id,
            Err(e) => return send_error(stream, 500, e),
        },
        Some(id) if session::valid_id(id) => id.to_owned(),
        Some(_) => return send_error(stream, 400, "invalid marin_session_id"),
        None => String::new(),
    };
    let existing = persistent && input.marin_session_id.as_deref() != Some("new");
    if existing && stored.as_ref().map(|state| state.id.as_str()) != Some(session_id.as_str()) {
        return send_error(stream, 404, "session not found or evicted");
    }
    let previous = if existing {
        stored
            .as_ref()
            .map(|state| state.retained.tokens().to_vec())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let reusable = if existing {
        stored.take().filter(|state| state.can_reuse(&prompt))
    } else {
        None
    };
    *stored = None;
    let mut active = match reusable
        .map(Ok)
        .unwrap_or_else(|| Stored::new(loaded, session_id))
    {
        Ok(state) => state,
        Err(e) => return send_error(stream, 500, e),
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let id = format!("chatcmpl-{created}-{request_id}");
    if input.stream {
        let opened = http::sse_start(stream).and_then(|()| {
            http::sse(
                stream,
                &chunk(
                    &id,
                    created,
                    &loaded.name,
                    json!({"role": "assistant"}),
                    None,
                ),
            )
        });
        if let Err(e) = opened {
            session::recover(stored, active, loaded, &previous, existing);
            return Err(e);
        }
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let watcher = match watch_disconnect(
        stream,
        Arc::clone(&cancel),
        Arc::clone(&done),
        Arc::clone(&timed_out),
        queue::RUN_LIMIT,
    ) {
        Ok(watcher) => watcher,
        Err(e) => {
            session::recover(stored, active, loaded, &previous, existing);
            return Err(e);
        }
    };
    let (checkpoint, skipped) = if persistent
        && !history.is_empty()
        && prompt.starts_with(&history)
        && matches!(&loaded.model, model::Engine::Qwen35Cuda(_))
    {
        match session::checkpoint(&mut active, loaded, &history) {
            Ok((checkpoint, keep)) => (checkpoint, Some(keep)),
            Err(e) => {
                done.store(true, Ordering::Relaxed);
                let _ = stream.shutdown(Shutdown::Read);
                let _ = watcher.join();
                session::recover(stored, active, loaded, &previous, existing);
                if input.stream {
                    let _ = http::sse(stream, &api_error(500, e));
                    let _ = http::sse_end(stream);
                    return Ok(());
                }
                return send_error(stream, 500, e);
            }
        }
    } else {
        (None, None)
    };
    if cancel.load(Ordering::Relaxed) {
        done.store(true, Ordering::Relaxed);
        let _ = stream.shutdown(Shutdown::Read);
        let _ = watcher.join();
        session::recover(stored, active, loaded, &previous, existing);
        if timed_out.load(Ordering::Relaxed) {
            if input.stream {
                let _ = http::sse(stream, &api_error(504, "generation timed out"));
                let _ = http::sse_end(stream);
                return Ok(());
            }
            return send_error(stream, 504, "generation timed out");
        }
        return Ok(());
    }
    let params = GenerateParams {
        max_tokens,
        stop_tokens: loaded.stop_tokens(),
        usable_vocab: loaded.dir.tokenizer.vocab_size(),
        speculation: None,
    };
    let mut decoder = loaded.dir.tokenizer.stream();
    let mut answer = String::new();
    let mut failure = None;
    let result = generate_reusing(
        &loaded.model,
        Kept {
            cache: &mut active.cache,
            retained: &mut active.retained,
        },
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
    if timed_out.load(Ordering::Relaxed) {
        session::recover(stored, active, loaded, &previous, existing);
        if input.stream {
            let _ = http::sse(stream, &api_error(504, "generation timed out"));
            let _ = http::sse_end(stream);
            return Ok(());
        }
        return send_error(stream, 504, "generation timed out");
    }
    if let Some(message) = failure {
        session::recover(stored, active, loaded, &previous, existing);
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
            session::recover(stored, active, loaded, &previous, existing);
            if input.stream {
                let _ = http::sse(stream, &api_error(500, e));
                let _ = http::sse_end(stream);
                return Ok(());
            }
            return send_error(stream, 500, e);
        }
    };
    let reused = skipped.unwrap_or(outcome.reused);
    if outcome.stop == StopReason::Cancelled {
        session::recover(stored, active, loaded, &previous, existing);
        return Ok(());
    }
    if let Some(checkpoint) = checkpoint {
        active.cache = checkpoint;
        active.retained = Retained::restored(history);
    } else if persistent && !history.is_empty() && prompt.starts_with(&history) {
        active.cache.clear();
        if loaded.model.forward(&history, &mut active.cache).is_ok() {
            active.retained = Retained::restored(history);
        } else {
            // A failed refresh must not publish a partially rebuilt session.
            session::recover(stored, active, loaded, &previous, existing);
            if input.stream {
                let _ = http::sse(stream, &api_error(500, "failed to refresh session state"));
                let _ = http::sse_end(stream);
                return Ok(());
            }
            return send_error(stream, 500, "failed to refresh session state");
        }
    }
    let response = if input.stream {
        let mut final_chunk = chunk(
            &id,
            created,
            &loaded.name,
            json!({}),
            Some(finish_reason(outcome.stop)),
        );
        if let Some(fields) = final_chunk.as_object_mut() {
            fields.insert("marin_reused_tokens".into(), json!(reused));
            if persistent {
                fields.insert("marin_session_id".into(), json!(active.id.as_str()));
            }
        }
        http::sse(stream, &final_chunk).and_then(|()| http::sse_end(stream))
    } else {
        http::json(
            stream,
            200,
            &json!({
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": loaded.name,
                "marin_session_id": persistent.then_some(active.id.as_str()),
                "marin_reused_tokens": reused,
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
    };
    if response.is_ok() {
        if persistent {
            *stored = Some(active);
        }
    } else {
        session::recover(stored, active, loaded, &previous, existing);
    }
    response
}

fn connection(
    stream: &mut TcpStream,
    loaded: &Loaded,
    defaults: Settings,
    id: u64,
    stored: &mut Option<Stored>,
) -> io::Result<()> {
    let request = match http::read(stream, Instant::now()) {
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
        ("POST", "/v1/chat/completions") => {
            completion(stream, loaded, defaults, id, &request.body, stored)
        }
        (_, "/v1/models" | "/v1/chat/completions") => send_error(stream, 405, "method not allowed"),
        _ => send_error(stream, 404, "unknown endpoint"),
    }
}

pub fn run(loaded: Loaded, defaults: Settings, port: u16) -> Result<(), CliError> {
    let (address, receiver) = queue::listen(port)?;
    println!("listening http://{address} (model: {})", loaded.name);
    let mut stored = None;
    for pending in receiver {
        let mut stream = pending.stream;
        if let Err(e) = stream.set_write_timeout(Some(Duration::from_secs(10))) {
            eprintln!("request setup failed: {e}");
            continue;
        }
        if let Err(e) = stream.set_nodelay(true) {
            eprintln!("request setup failed: {e}");
            continue;
        }
        if queue::expired(pending.accepted) {
            queue::drain(&mut stream);
            let _ = send_error(&mut stream, 408, "request waited too long");
            continue;
        }
        if let Err(e) = connection(&mut stream, &loaded, defaults, pending.id, &mut stored) {
            eprintln!("request failed: {e}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
