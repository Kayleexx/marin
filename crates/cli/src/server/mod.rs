mod completion;
mod gate;
mod http;
mod prefix;
mod queue;
mod request;
mod sched;
mod session;
mod watch;

use std::io;
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

use runtime::StopReason;
use serde_json::{Value, json};

use crate::CliError;
use crate::load::Loaded;
use crate::session::Settings;
use completion::{Context, completion};
use gate::Gate;
use prefix::Prefix;
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

fn connection(ctx: Context<'_>, stream: &mut TcpStream, id: u64) -> io::Result<()> {
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
            &json!({"object":"list","data":[{"id":ctx.loaded.name,"object":"model"}]}),
        ),
        ("POST", "/v1/chat/completions") => completion(ctx, stream, id, &request.body),
        (_, "/v1/models" | "/v1/chat/completions") => send_error(stream, 405, "method not allowed"),
        _ => send_error(stream, 404, "unknown endpoint"),
    }
}

fn serve(ctx: Context<'_>, pending: queue::Pending) {
    let mut stream = pending.stream;
    if let Err(e) = ctx.loaded.model.bind_to_thread() {
        eprintln!("request setup failed: {e}");
        let _ = send_error(&mut stream, 500, e);
        return;
    }
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
    if let Err(e) = connection(ctx, &mut stream, pending.id) {
        eprintln!("request failed: {e}");
    }
}

pub fn run(loaded: Loaded, defaults: Settings, port: u16) -> Result<(), CliError> {
    let (address, receiver) = queue::listen(port)?;
    let started = Instant::now();
    let prefix = Prefix::build(&loaded);
    if let Some(prefix) = &prefix {
        println!(
            "shared prompt prefix: {} tokens, {} MiB, built in {:.0} ms",
            prefix.len(),
            prefix.bytes() >> 20,
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    let gate = Gate::new(sched::max_active(&loaded), loaded.model.batch_limit());
    let store = Store::default();
    println!(
        "listening http://{address} (model: {}, up to {} at once)",
        loaded.name,
        gate.max_active()
    );
    let ctx = Context {
        loaded: &loaded,
        defaults,
        gate: &gate,
        store: &store,
        prefix: prefix.as_ref(),
    };
    thread::scope(|scope| {
        loop {
            gate.wait_for_slot();
            let Ok(pending) = receiver.recv() else {
                break;
            };
            let slot = gate.enter();
            let worker = thread::Builder::new()
                .name("marin-request".into())
                .spawn_scoped(scope, move || {
                    let _slot = slot;
                    serve(ctx, pending);
                });
            if let Err(e) = worker {
                eprintln!("request setup failed: {e}");
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests;
