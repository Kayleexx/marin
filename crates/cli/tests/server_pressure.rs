//! A session on the server after another program has taken nearly all the free GPU memory: the answers are the same
//! as without the pressure and the finished turn is put back at the end of the conversation without reading it again.
//! Needs a GPU; the only test in this binary, so nothing else is using the GPU while it holds the memory.

#[path = "common/qwen35.rs"]
mod qwen35;
#[path = "common/server.rs"]
mod server;

use std::io::{self, Write};

use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use serde_json::{Value, json};
use server::Server;

const LEFT_FREE: usize = 150 << 20;

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
        let turns = conversation(&server, true)?;
        assert_eq!(
            server.logged("panicked"),
            0,
            "no thread of the server panicked"
        );
        turns
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
