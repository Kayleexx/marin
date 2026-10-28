//! A server started with `--idle-unload` releases its model when it has been quiet, and loads it again for the next
//! request: every conversation, hot or cold, comes back with the answers it would have given without the sleep; any
//! number of requests arriving while it sleeps wake it once; a model that cannot load stays asleep with its sessions
//! kept; files changed while it slept drop the sessions made with the old ones. Needs a GPU.

#[path = "common/qwen35.rs"]
mod qwen35;
#[path = "common/server.rs"]
mod server;
#[path = "common/talk.rs"]
mod talk;

use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use serde_json::{Value, json};
use server::Server;
use talk::{
    LONG, REAL, Talker, Turn, exclusive_gpu, real_model, send_and_hang_up, start_real,
    start_stream, take_gpu, text,
};

fn start_idle(model: &Path, home: &Path) -> Res<Server> {
    let tiny = ["--quant-group", "32", "--ctx", "1024"];
    Server::spawn(model, home, &tiny, &["--idle-unload", "1"], "packed")
}

fn wait_logged(server: &Server, needle: &str, times: usize) -> Res {
    let started = Instant::now();
    while server.logged(needle) < times {
        if started.elapsed() > Duration::from_secs(30) {
            return Err(format!("the server never logged {needle:?} {times} times").into());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Ok(())
}

fn asleep(server: &Server, times: usize) -> Res {
    wait_logged(server, "model asleep", times)
}

fn alone(packed: &Path, home: &Path, name: &str, turns: usize) -> Res<Vec<Turn>> {
    let server = Server::start(packed, home)?;
    let mut talker = Talker::long(name, LONG);
    (0..turns).map(|n| talker.say(&server, n)).collect()
}

#[test]
fn every_conversation_survives_repeated_sleep_and_wake() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-cycles")?;
    let packed = packed_tiny(&scratch.0)?;
    let (control_a, control_b) = (
        alone(&packed, &scratch.0, "alpha", 4)?,
        alone(&packed, &scratch.0, "beta", 4)?,
    );
    let server = start_idle(&packed, &scratch.0)?;
    let (mut a, mut b) = (Talker::long("alpha", LONG), Talker::long("beta", LONG));
    let (mut turns_a, mut turns_b) = (vec![a.say(&server, 0)?], vec![b.say(&server, 0)?]);
    for n in 1..4 {
        asleep(&server, n)?;
        turns_a.push(a.say(&server, n)?);
        turns_b.push(b.say(&server, n)?);
        assert_eq!(server.logged("model awake in"), n, "one wake per sleep");
    }
    let answers = |turns: &[Turn]| turns.iter().map(|t| t.answer.clone()).collect::<Vec<_>>();
    assert_eq!(answers(&turns_a), answers(&control_a));
    assert_eq!(answers(&turns_b), answers(&control_b));
    for turn in turns_a.iter().chain(&turns_b).skip(1) {
        assert_eq!(turn.replayed, 0);
        assert!(
            turn.reused > 0,
            "a conversation came back from a sleep without being read again"
        );
    }
    assert_eq!(server.logged("panicked"), 0);
    Ok(())
}

#[test]
fn requests_arriving_while_it_sleeps_wake_it_once() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-wake-once")?;
    let packed = packed_tiny(&scratch.0)?;
    let server = start_idle(&packed, &scratch.0)?;
    let mut a = Talker::long("alpha", LONG);
    a.say(&server, 0)?;
    asleep(&server, 1)?;
    let hello = json!([{"role": "user", "content": "hello there"}]);
    let replies: Vec<(u16, String)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..3)
            .map(|_| {
                let worker = scope.spawn(|| {
                    server
                        .post(&hello, "new")
                        .map(|(status, body)| (status, body.to_string()))
                        .map_err(|e| e.to_string())
                });
                thread::sleep(Duration::from_millis(50));
                worker
            })
            .collect();
        workers
            .into_iter()
            .map(|w| {
                w.join()
                    .map_err(|_| "a request thread panicked".to_owned())?
            })
            .collect::<Result<_, String>>()
    })?;
    assert!(
        replies.iter().all(|r| r.0 == 200),
        "every request that arrived while it slept was served: {replies:?}"
    );
    assert_eq!(server.logged("model awake in"), 1, "the model loaded once");
    let (again, first) = (server.ask(&hello, "new")?, server.ask(&hello, "new")?);
    assert_eq!(text(&again), text(&first), "and answers the same twice");
    let turn = a.say(&server, 1)?;
    assert!(
        turn.reused > 0,
        "and the conversation made before the sleep is still there"
    );
    Ok(())
}

#[test]
fn hanging_up_while_the_model_wakes_loses_nothing() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-hangup")?;
    let packed = packed_tiny(&scratch.0)?;
    let control = alone(&packed, &scratch.0, "alpha", 2)?;
    let server = start_idle(&packed, &scratch.0)?;
    let mut a = Talker::long("alpha", LONG);
    a.say(&server, 0)?;
    asleep(&server, 1)?;
    let mut next = a.messages.clone();
    next.push(json!({"role": "user", "content": "alpha says hello"}));
    send_and_hang_up(&server, &Value::Array(next), &a.session)?;
    wait_logged(&server, "model awake in", 1)?;
    let turn = loop {
        match a.try_say(&server, 1)? {
            Ok(turn) => break turn,
            Err(409) => thread::sleep(Duration::from_millis(50)),
            Err(status) => return Err(format!("the session was lost: {status}").into()),
        }
    };
    assert_eq!(turn.answer, control[1].answer);
    Ok(())
}

#[test]
fn files_changed_while_it_sleeps_drop_the_sessions_made_with_the_old_ones() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-stale")?;
    let packed = packed_tiny(&scratch.0)?;
    let server = start_idle(&packed, &scratch.0)?;
    let mut a = Talker::new("alpha");
    a.say(&server, 0)?;
    asleep(&server, 1)?;
    let changed = std::fs::OpenOptions::new()
        .write(true)
        .open(packed.join("config.json"))?;
    changed.set_modified(
        std::time::SystemTime::now()
            .checked_add(Duration::from_secs(5))
            .ok_or("time overflow")?,
    )?;
    let hello = json!([{"role": "user", "content": "hello there"}]);
    assert_eq!(server.post(&hello, "new")?.0, 200, "the model loads again");
    assert_eq!(server.logged("model files changed"), 1);
    assert_eq!(
        a.try_say(&server, 1)?.err(),
        Some(404),
        "the old session is not trusted"
    );
    Ok(())
}

/// The real 4B: a wake that finds no GPU memory is refused as retryable and the sessions are kept; a model is never
/// released under a running request.
#[test]
fn a_model_that_cannot_load_stays_asleep_and_a_running_request_keeps_it() -> Res {
    let (true, Some(model)) = (have_gpu()?, real_model()) else {
        return Ok(());
    };
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-real")?;
    let server = start_real(&model, &scratch.0, &["--idle-unload", "1"])?;
    let mut a = Talker::long("alpha", REAL);
    a.say(&server, 0)?;
    asleep(&server, 1)?;
    let hello = json!([{"role": "user", "content": "hello there"}]);
    let refused = {
        let _held = take_gpu(1 << 30)?;
        server.post(&hello, "new")?
    };
    assert_eq!(refused.0, 503, "{}", refused.1);
    assert_eq!(server.logged("model awake in"), 0, "it is still asleep");
    let turn = a.say(&server, 1)?;
    assert_eq!(turn.replayed, 0);
    assert!(
        turn.reused > 1000,
        "the session survived the failed wake: {}",
        turn.reused
    );

    let asleep_before = server.logged("model asleep");
    let story =
        json!([{"role": "user", "content": "Write a long story about a lighthouse keeper."}]);
    let mut running = start_stream(&server, &story, "new", 250)?;
    thread::sleep(Duration::from_secs(3));
    assert_eq!(
        server.logged("model asleep"),
        asleep_before,
        "not released under a running request"
    );
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut running, &mut rest)?;
    assert!(rest.contains("[DONE]"));
    asleep(&server, asleep_before + 1)?;
    Ok(())
}

/// The real 4B: requests that arrive together while the model cannot load are each refused as retryable, none hangs
/// and none loses a session, and the first request after the memory returns wakes the model.
#[test]
fn requests_arriving_while_the_model_cannot_load_are_all_refused_and_the_next_wakes_it() -> Res {
    let (true, Some(model)) = (have_gpu()?, real_model()) else {
        return Ok(());
    };
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("sleep-stampede")?;
    let server = start_real(&model, &scratch.0, &["--idle-unload", "1"])?;
    let mut a = Talker::long("alpha", REAL);
    a.say(&server, 0)?;
    asleep(&server, 1)?;
    let hello = json!([{"role": "user", "content": "hello there"}]);
    let started = Instant::now();
    let codes: Vec<u16> = {
        let _held = take_gpu(1 << 30)?;
        thread::scope(|scope| {
            let workers: Vec<_> = (0..3)
                .map(|_| {
                    let worker = scope.spawn(|| {
                        server
                            .post(&hello, "new")
                            .map(|(status, _)| status)
                            .map_err(|e| e.to_string())
                    });
                    thread::sleep(Duration::from_millis(50));
                    worker
                })
                .collect();
            workers
                .into_iter()
                .map(|w| {
                    w.join()
                        .map_err(|_| "a request thread panicked".to_owned())?
                })
                .collect::<Result<_, String>>()
        })?
    };
    assert_eq!(codes, [503, 503, 503]);
    assert!(
        started.elapsed() < Duration::from_secs(120),
        "refusals took {:?}",
        started.elapsed()
    );
    assert_eq!(server.logged("model awake in"), 0);
    let turn = a.say(&server, 1)?;
    assert_eq!(turn.replayed, 0);
    assert!(turn.reused > 1000, "the session survived: {}", turn.reused);
    assert_eq!(server.logged("model awake in"), 1);
    assert_eq!(server.logged("panicked"), 0);
    Ok(())
}
