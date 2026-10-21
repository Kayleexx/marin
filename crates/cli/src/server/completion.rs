use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use runtime::{CausalLm, generate_reusing};
use runtime::{GenerateError, GenerateParams, Kept, Retained, SequenceCache, StopReason};
use serde_json::{Value, json};

use super::gate::Gate;
use super::prefix::Prefix;
use super::request::{Completion, validated_prompt};
use super::sched::{ScheduleError, Scheduled};
use super::session::{self, Store, Stored};
use super::watch::watch_disconnect;
use super::{api_error, finish_reason, http, queue, send_error};
use crate::load::Loaded;
use crate::session::Settings;

#[derive(Clone, Copy)]
pub(super) struct Context<'a> {
    pub loaded: &'a Loaded,
    pub defaults: Settings,
    pub gate: &'a Gate,
    pub store: &'a Store,
    pub prefix: Option<&'a Prefix>,
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

/// Reports a failure on a response that may already be streaming.
fn fail(
    stream: &mut TcpStream,
    streaming: bool,
    status: u16,
    message: impl ToString,
) -> io::Result<()> {
    if streaming {
        let _ = http::sse(stream, &api_error(status, message));
        let _ = http::sse_end(stream);
        Ok(())
    } else {
        send_error(stream, status, message)
    }
}

/// A GPU out of memory is a condition of the moment: the request can be sent again, unlike a failure of the model.
fn failure_status(out_of_memory: bool) -> u16 {
    if out_of_memory { 503 } else { 500 }
}

fn stop_watching(stream: &TcpStream, done: &AtomicBool, watcher: JoinHandle<()>) {
    done.store(true, Ordering::Relaxed);
    let _ = stream.shutdown(Shutdown::Read);
    let _ = watcher.join();
}

pub(super) fn completion(
    ctx: Context<'_>,
    stream: &mut TcpStream,
    request_id: u64,
    body: &[u8],
) -> io::Result<()> {
    let Context {
        loaded,
        defaults,
        gate,
        store,
        prefix,
    } = ctx;
    let plain = Scheduled::new(loaded, gate, request_id);
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
    let taken = if existing {
        match store.take(&session_id) {
            Some(state) => Some(state),
            None => return send_error(stream, 404, "session not found, evicted or in use"),
        }
    } else {
        store.evict();
        None
    };
    let previous = taken
        .as_ref()
        .map(|state| state.retained.tokens().to_vec())
        .unwrap_or_default();
    let reusable = taken.filter(|state| state.can_reuse(&prompt));
    let mut active = match reusable
        .or_else(|| prefix.and_then(|p| p.start(plain, &prompt, &session_id)))
        .map(Ok)
        .unwrap_or_else(|| Stored::new(loaded, session_id))
    {
        Ok(state) => state,
        Err(e) => return send_error(stream, failure_status(e.is_out_of_memory()), e),
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
            store.recover(active, plain, &previous, existing);
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
            store.recover(active, plain, &previous, existing);
            return Err(e);
        }
    };
    let (checkpoint, skipped) = if persistent
        && !history.is_empty()
        && prompt.starts_with(&history)
        && matches!(&loaded.model, model::Engine::Qwen35Cuda(_))
    {
        match session::checkpoint(&mut active, plain, &history) {
            Ok((checkpoint, keep)) => (checkpoint, Some(keep)),
            Err(e) => {
                stop_watching(stream, &done, watcher);
                store.recover(active, plain, &previous, existing);
                return fail(
                    stream,
                    input.stream,
                    failure_status(e.is_out_of_memory()),
                    e,
                );
            }
        }
    } else {
        (None, None)
    };
    if cancel.load(Ordering::Relaxed) {
        stop_watching(stream, &done, watcher);
        store.recover(active, plain, &previous, existing);
        if timed_out.load(Ordering::Relaxed) {
            return fail(stream, input.stream, 504, "generation timed out");
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
        &plain.cancellable(&cancel),
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
    stop_watching(stream, &done, watcher);
    if timed_out.load(Ordering::Relaxed) {
        store.recover(active, plain, &previous, existing);
        return fail(stream, input.stream, 504, "generation timed out");
    }
    if let Some(message) = failure {
        store.recover(active, plain, &previous, existing);
        return fail(stream, input.stream, 500, message);
    }
    let outcome = match result {
        Ok(value) => value,
        Err(GenerateError::Model(ScheduleError::Cancelled)) => {
            store.recover(active, plain, &previous, existing);
            return Ok(());
        }
        Err(e) => {
            store.recover(active, plain, &previous, existing);
            let status = match &e {
                GenerateError::Model(model) => failure_status(model.is_out_of_memory()),
                _ => 500,
            };
            return fail(stream, input.stream, status, e);
        }
    };
    let reused = skipped.unwrap_or(outcome.reused);
    if outcome.stop == StopReason::Cancelled {
        store.recover(active, plain, &previous, existing);
        return Ok(());
    }
    let mut replayed = 0;
    let rewound = checkpoint.is_some_and(|mark| plain.rewind(&mut active.cache, mark).is_ok());
    if rewound {
        active.retained = Retained::restored(history);
    } else if persistent && !history.is_empty() && prompt.starts_with(&history) {
        active.cache.clear();
        if plain.forward(&history, &mut active.cache).is_ok() {
            replayed = history.len();
            active.retained = Retained::restored(history);
        } else {
            // A failed refresh must not publish a partially rebuilt session.
            store.recover(active, plain, &previous, existing);
            return fail(stream, input.stream, 500, "failed to refresh session state");
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
            fields.insert("marin_replayed_tokens".into(), json!(replayed));
            fields.insert(
                "usage".into(),
                json!({
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.generated,
                    "total_tokens": outcome.prompt_tokens.saturating_add(outcome.generated)
                }),
            );
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
                "marin_replayed_tokens": replayed,
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
            store.put(active);
        }
    } else {
        store.recover(active, plain, &previous, existing);
    }
    response
}
