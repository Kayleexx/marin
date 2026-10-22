mod cold;
mod completion;
mod gate;
mod http;
mod lifecycle;
mod prefix;
mod queue;
mod request;
mod sched;
mod session;
mod watch;

use std::io;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use runtime::StopReason;
use serde_json::{Value, json};

use crate::CliError;
use crate::load::Loaded;
use crate::session::Settings;
use completion::{Context, completion};
use gate::Gate;
use lifecycle::{Lifecycle, Reload};
use session::Store;

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

/// What every request thread shares. The model itself is behind the lifecycle, which may have it asleep.
#[derive(Clone, Copy)]
struct Shared<'a> {
    life: &'a Lifecycle<'a>,
    defaults: Settings,
    gate: &'a Gate,
    store: &'a Store,
}

fn connection(shared: Shared<'_>, stream: &mut TcpStream, id: u64) -> io::Result<()> {
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
            &json!({"object":"list","data":[{"id":shared.life.info.name,"object":"model","context_window":shared.life.info.ctx}]}),
        ),
        ("POST", "/v1/chat/completions") => {
            let guard = match shared.life.acquire() {
                Ok(guard) => guard,
                Err(message) => return send_error(stream, 503, message),
            };
            let loaded = &guard.awake.loaded;
            if let Err(e) = loaded.model.bind_to_thread() {
                return send_error(stream, 500, e);
            }
            let ctx = Context {
                loaded,
                defaults: shared.defaults,
                gate: shared.gate,
                store: shared.store,
                prefix: guard.awake.prefix.as_ref(),
            };
            completion(ctx, stream, id, &request.body)
        }
        (_, "/v1/models" | "/v1/chat/completions") => send_error(stream, 405, "method not allowed"),
        _ => send_error(stream, 404, "unknown endpoint"),
    }
}

fn serve(shared: Shared<'_>, pending: queue::Pending) {
    let mut stream = pending.stream;
    if let Err(e) = stream.set_write_timeout(Some(Duration::from_secs(10))) {
        eprintln!("request setup failed: {e}");
        return;
    }
    if let Err(e) = stream.set_nodelay(true) {
        eprintln!("request setup failed: {e}");
        return;
    }
    if queue::expired(pending.accepted) {
        queue::drain(&mut stream);
        let _ = send_error(&mut stream, 408, "request waited too long");
        return;
    }
    if let Err(e) = connection(shared, &mut stream, pending.id) {
        eprintln!("request failed: {e}");
    }
}

pub fn run(
    loaded: Loaded,
    reload: Reload<'_>,
    defaults: Settings,
    port: u16,
    idle: Option<Duration>,
) -> Result<(), CliError> {
    let (address, receiver) = queue::listen(port)?;
    let started = Instant::now();
    let gate = Gate::new(sched::max_active(&loaded), loaded.model.batch_limit());
    let store = Store::default();
    let life = Lifecycle::new(loaded, reload, idle, &gate, &store);
    if let Ok(guard) = life.acquire()
        && let Some(prefix) = &guard.awake.prefix
    {
        println!(
            "shared prompt prefix: {} tokens, {} MiB, built in {:.0} ms",
            prefix.len(),
            prefix.bytes() >> 20,
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    println!(
        "listening http://{address} (model: {}, up to {} at once{})",
        life.info.name,
        gate.max_active(),
        idle.map_or(String::new(), |t| format!(
            ", unloads after {} s idle",
            t.as_secs()
        ))
    );
    let shared = Shared {
        life: &life,
        defaults,
        gate: &gate,
        store: &store,
    };
    let stop = AtomicBool::new(false);
    thread::scope(|scope| {
        if life.idle().is_some() {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(500));
                    life.sleep_if_idle();
                }
            });
        }
        loop {
            gate.wait_for_slot();
            let Ok(pending) = receiver.recv() else {
                break;
            };
            let slot = gate.enter(pending.id);
            let worker = thread::Builder::new()
                .name("marin-request".into())
                .spawn_scoped(scope, move || {
                    let _slot = slot;
                    serve(shared, pending);
                });
            if let Err(e) = worker {
                eprintln!("request setup failed: {e}");
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
    Ok(())
}

#[cfg(test)]
mod tests;
