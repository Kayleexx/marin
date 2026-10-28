//! Several independent conversations on one server, which keeps one session hot on the GPU and the others cold in host
//! memory: a conversation that another displaced comes back with the same answers and without being read again; ids
//! the server does not hold (never made, or lost with a restart) are refused; a history that does not continue the
//! kept one is read again and answers as a new conversation would. Needs a GPU.

#[path = "common/qwen35.rs"]
mod qwen35;
#[path = "common/server.rs"]
mod server;
#[path = "common/talk.rs"]
mod talk;

use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use serde_json::{Value, json};
use server::Server;
use talk::{
    LONG, REAL, Talker, Turn, exclusive_gpu, real_model, send_and_hang_up, start_real,
    start_stream, take_gpu, text,
};

/// The next user message added to the conversation, and the whole conversation to send.
fn ask_next(talker: &mut Talker<'_>, n: usize) -> Value {
    let said = format!("{} asks number {n} again and again ", talker.name).repeat(talker.repeat);
    talker
        .messages
        .push(json!({"role": "user", "content": said}));
    Value::Array(talker.messages.clone())
}

#[test]
fn displaced_conversations_come_back_from_host_memory_with_the_same_answers() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-cold")?;
    let packed = packed_tiny(&scratch.0)?;
    let alone = |name: &str| -> Res<Vec<Turn>> {
        let server = Server::start(&packed, &scratch.0)?;
        let mut talker = Talker::new(name);
        (0..4).map(|n| talker.say(&server, n)).collect()
    };
    let (control_a, control_b) = (alone("alpha")?, alone("beta")?);

    let server = Server::start(&packed, &scratch.0)?;
    let (mut a, mut b) = (Talker::new("alpha"), Talker::new("beta"));
    let (mut turns_a, mut turns_b) = (Vec::new(), Vec::new());
    for n in 0..4 {
        turns_a.push(a.say(&server, n)?);
        turns_b.push(b.say(&server, n)?);
    }
    let answers = |turns: &[Turn]| turns.iter().map(|t| t.answer.clone()).collect::<Vec<_>>();
    assert_eq!(
        answers(&turns_a),
        answers(&control_a),
        "alpha answers as if alone"
    );
    assert_eq!(
        answers(&turns_b),
        answers(&control_b),
        "beta answers as if alone"
    );
    for (who, turns) in [("alpha", &turns_a), ("beta", &turns_b)] {
        for (n, turn) in turns.iter().enumerate().skip(1) {
            assert_eq!(turn.replayed, 0, "{who} turn {n} was not read again");
            assert!(turn.reused > 0, "{who} turn {n} kept its conversation");
        }
    }
    assert_eq!(
        server.logged("panicked"),
        0,
        "no thread of the server panicked"
    );
    Ok(())
}

#[test]
fn unknown_and_lost_ids_are_refused_and_a_different_history_is_read_again() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-cold-refused")?;
    let packed = packed_tiny(&scratch.0)?;
    let server = Server::start(&packed, &scratch.0)?;
    let ask = |messages: &Value, session: &str| server.post(messages, session);
    let hello = json!([{"role": "user", "content": "hello there"}]);
    assert_eq!(
        ask(&hello, &"0".repeat(32))?.0,
        404,
        "an id the server never made"
    );
    assert_eq!(ask(&hello, "not-an-id")?.0, 400, "a malformed id");

    let (mut a, mut b) = (Talker::new("alpha"), Talker::new("beta"));
    a.say(&server, 0)?;
    b.say(&server, 0)?;
    let mut changed = a.messages.clone();
    changed.push(json!({"role": "user", "content": "alpha says something else entirely"}));
    changed[0] = json!({"role": "user", "content": "a different first message"});
    let rebuilt = server.ask(&Value::Array(changed.clone()), &a.session)?;
    let fresh = server.ask(&Value::Array(changed), "new")?;
    let text = |reply: &Value| reply.pointer("/choices/0/message/content").cloned();
    assert_eq!(
        text(&rebuilt),
        text(&fresh),
        "a different history answers as a new conversation"
    );

    let old = a.session.clone();
    let after_restart = Server::start(&packed, &scratch.0)?;
    let (status, body) = after_restart.post(&hello, &old)?;
    assert_eq!(
        status, 404,
        "sessions are not kept across a restart: {body}"
    );
    Ok(())
}

#[test]
fn a_session_asked_for_twice_at_once_runs_once_and_stays_intact() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-twice")?;
    let packed = packed_tiny(&scratch.0)?;
    let server = Server::start(&packed, &scratch.0)?;
    let mut a = Talker::new("alpha");
    a.say(&server, 0)?;
    let next = ask_next(&mut a, 1);
    let mut running = start_stream(&server, &next, &a.session, 800)?;
    let (status, body) = server.post(&next, &a.session)?;
    assert_eq!(status, 409, "the session is running: {body}");
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut running, &mut rest)?;
    assert!(rest.contains("[DONE]"), "the first request ran to its end");

    let fresh = server.ask(&next, "new")?;
    let again = server.ask(&next, &a.session)?;
    assert_eq!(
        text(&again),
        text(&fresh),
        "the session answers as a new conversation"
    );
    assert_eq!(again["marin_replayed_tokens"], 0);
    assert!(
        again["marin_reused_tokens"].as_u64().unwrap_or(0) > 0,
        "and it was not lost"
    );
    Ok(())
}

/// A request to a cold session after another program took the GPU memory: refused as retryable, the session is kept and
/// answers as it would have without the pressure once the memory is back. The real model: a tiny one is served from
/// memory its allocator already holds.
#[test]
fn a_cold_session_that_finds_no_gpu_memory_is_kept_and_resumes_after_a_retry() -> Res {
    let (true, Some(model)) = (have_gpu()?, real_model()) else {
        return Ok(());
    };
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-cold-oom")?;
    let control = {
        let server = start_real(&model, &scratch.0, &[])?;
        let mut a = Talker::long("alpha", REAL);
        (0..3).map(|n| a.say(&server, n)).collect::<Res<Vec<_>>>()?
    };
    let server = start_real(&model, &scratch.0, &[])?;
    let (mut a, mut b) = (Talker::long("alpha", REAL), Talker::new("beta"));
    a.say(&server, 0)?;
    a.say(&server, 1)?;
    b.say(&server, 0)?;
    let refused = {
        let _held = take_gpu(100 << 20)?;
        a.try_say(&server, 2)?
    };
    assert_eq!(refused.err(), Some(503), "no memory is a retryable refusal");
    let turn = a.say(&server, 2)?;
    assert_eq!(
        turn.answer, control[2].answer,
        "the same answer as without the pressure"
    );
    assert_eq!(turn.replayed, 0);
    assert!(
        turn.reused > 2000,
        "the session was kept, not read again: {}",
        turn.reused
    );
    Ok(())
}

/// The history sent does not continue the kept one, and the GPU has no memory for its pass: refused as retryable, and
/// the kept session is still there afterwards, brought back from its host copy and not read again. The kept history is
/// longer than the request, so wherever the request's pass runs out of memory, reading the kept history again would
/// too. How much memory the pass needs depends on what other programs hold, so the card is squeezed in steps and every
/// refusal is checked; if no step is refused the test fails, because then nothing was proved.
#[test]
fn a_different_history_that_finds_no_gpu_memory_does_not_lose_the_session() -> Res {
    let (true, Some(model)) = (have_gpu()?, real_model()) else {
        return Ok(());
    };
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-diverge-oom")?;
    let control = {
        let server = start_real(&model, &scratch.0, &[])?;
        let mut a = Talker::long("alpha", REAL);
        (0..4).map(|n| a.say(&server, n)).collect::<Res<Vec<_>>>()?
    };
    let server = start_real(&model, &scratch.0, &[])?;
    // Longer than the whole kept session, so the memory freed by releasing it cannot cover the new pass.
    let different = "a different first message that shares nothing with the kept one ".repeat(300);
    let mut refusals = 0;
    let mut steps = Vec::new();
    for left in [600, 300, 200, 150, 120, 100, 80, 60, 40] {
        let mut a = Talker::long("alpha", REAL);
        a.say(&server, 0)?;
        a.say(&server, 1)?;
        a.say(&server, 2)?;
        let mut other = a.messages.clone();
        *other.first_mut().ok_or("no first message")? =
            json!({"role": "user", "content": different});
        other.push(json!({"role": "user", "content": "and a question"}));
        let status = {
            let _held = take_gpu(left << 20)?;
            server.post(&Value::Array(other), &a.session)?.0
        };
        steps.push((left, status));
        if status == 200 {
            continue;
        }
        assert_eq!(status, 503, "no memory is a retryable refusal");
        let turn = a.say(&server, 3)?;
        assert_eq!(
            turn.answer,
            control.get(3).ok_or("no control turn")?.answer,
            "the same answer as without the pressure"
        );
        assert_eq!(turn.replayed, 0);
        assert!(
            turn.reused > 3000,
            "with {left} MiB left the kept session was kept, not read again: {}",
            turn.reused
        );
        refusals += 1;
    }
    assert!(
        refusals > 0,
        "every step left the request room: no memory pressure was reached (MiB left, status): {steps:?}"
    );
    Ok(())
}

/// A client that hangs up while its cold session is being brought back: the session is still there and answers.
#[test]
fn hanging_up_during_a_cold_resume_loses_nothing() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let _alone = exclusive_gpu();
    let scratch = Scratch::new("serve-cold-cancel")?;
    let packed = packed_tiny(&scratch.0)?;
    let control = {
        let server = Server::start(&packed, &scratch.0)?;
        let mut a = Talker::long("alpha", LONG);
        (0..3).map(|n| a.say(&server, n)).collect::<Res<Vec<_>>>()?
    };
    let server = Server::start(&packed, &scratch.0)?;
    let (mut a, mut b) = (Talker::long("alpha", LONG), Talker::new("beta"));
    a.say(&server, 0)?;
    a.say(&server, 1)?;
    for round in 0..6 {
        b.say(&server, round)?;
        let next = ask_next(&mut a, 2);
        send_and_hang_up(&server, &next, &a.session)?;
        a.messages.pop();
    }
    let turn = loop {
        match a.try_say(&server, 2)? {
            Ok(turn) => break turn,
            Err(409) => std::thread::sleep(std::time::Duration::from_millis(50)),
            Err(status) => return Err(format!("the session was lost: {status}").into()),
        }
    };
    assert_eq!(
        turn.answer, control[2].answer,
        "the same answer as if nobody hung up"
    );
    Ok(())
}
