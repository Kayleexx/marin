use std::io::{self, Write};

use cli::chat::Chat;
use cli::load::{ChatPrecision, load, load_chat};
use cli::session::{Session, Settings};
use cli::setup::precision_for_pick;
use cli::signal::Interrupt;
use cli::style::Style;
use model::BackendChoice;
use runtime::SequenceCache;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn session(ctx: usize, temperature: f32, reuse: bool) -> Result<Session> {
    let model =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let loaded = load(
        &model,
        Some(ctx),
        ctx,
        BackendChoice::Cpu,
        model::Precision::default(),
        &Style::plain(),
        false,
    )?;
    let settings = Settings {
        max_tokens: 12,
        temperature,
        top_k: 20,
        top_p: 0.8,
        seed: Some(3),
    };
    let mut session = Session::new(loaded, settings, Style::plain())?;
    session.reuse = reuse;
    Ok(session)
}

struct Side {
    session: Session,
    chat: Chat,
}

impl Side {
    fn new(ctx: usize, temperature: f32, reuse: bool) -> Result<Self> {
        let session = session(ctx, temperature, reuse)?;
        let chat = Chat::new(&session);
        Ok(Self { session, chat })
    }

    fn ask(&mut self, text: &str, interrupt: &Interrupt) -> Result<(Vec<u32>, usize)> {
        let reply = self
            .chat
            .ask(&mut self.session, text, interrupt, &mut Vec::new())?;
        assert_eq!(
            self.session.retained().len(),
            self.session.kv().used(),
            "the record covers the cache"
        );
        Ok((reply.tokens, reply.outcome.reused))
    }
}

fn same_reply(
    keep: &mut Side,
    replay: &mut Side,
    text: &str,
    interrupt: &Interrupt,
) -> Result<usize> {
    let before = keep.session.retained().tokens().to_vec();
    let (kept_tokens, reused) = keep.ask(text, interrupt)?;
    let (replayed_tokens, replay_reused) = replay.ask(text, interrupt)?;
    assert_eq!(kept_tokens, replayed_tokens, "{text:?}");
    assert_eq!(keep.session.last_prompt, replay.session.last_prompt);
    assert_eq!(replay_reused, 0, "full replay reads everything");
    let prompt = &keep.session.last_prompt;
    let shared = before
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(
        reused,
        shared.min(prompt.len().saturating_sub(1)),
        "{text:?}: exactly the shared start"
    );
    Ok(reused)
}

fn a_conversation_reuses_its_history_and_answers_exactly_as_a_replay(
    interrupt: &Interrupt,
) -> Result<()> {
    for temperature in [0.0, 0.7] {
        let mut keep = Side::new(1024, temperature, true)?;
        let mut replay = Side::new(1024, temperature, false)?;
        let first = same_reply(&mut keep, &mut replay, "hi there", interrupt)?;
        assert_eq!(first, 0, "a new session holds nothing");
        let mut previous_prompt = keep.session.last_prompt.len();
        for text in [
            "tell me more",
            "and then?",
            "why",
            "the cat sat on the mat",
            "ok",
        ] {
            let reused = same_reply(&mut keep, &mut replay, text, interrupt)?;
            // The template drops the last answer's empty reasoning block, so that answer is read again.
            assert!(
                reused.saturating_add(8) >= previous_prompt,
                "{text:?}: kept {reused} of the {previous_prompt}-token previous prompt"
            );
            previous_prompt = keep.session.last_prompt.len();
        }
        keep.chat.clear();
        replay.chat.clear();
        let reused = same_reply(&mut keep, &mut replay, "something new", interrupt)?;
        assert!(reused > 0, "the system message is kept");
    }
    Ok(())
}

fn tools_that_touch_the_cache_make_the_next_turn_read_everything(
    interrupt: &Interrupt,
) -> Result<()> {
    let mut keep = Side::new(1024, 0.7, true)?;
    let mut replay = Side::new(1024, 0.7, false)?;
    same_reply(&mut keep, &mut replay, "hi", interrupt)?;
    let probe = keep.session.encode("hello")?;
    keep.session.next_token_lines(&probe)?;
    assert!(keep.session.retained().is_empty());
    assert_eq!(same_reply(&mut keep, &mut replay, "more", interrupt)?, 0);
    keep.session.engine_and_kv().1.clear();
    assert!(keep.session.retained().is_empty());
    assert_eq!(same_reply(&mut keep, &mut replay, "again", interrupt)?, 0);
    Ok(())
}

fn forgetting_old_turns_to_fit_the_window_stays_exact(interrupt: &Interrupt) -> Result<()> {
    let mut keep = Side::new(200, 0.7, true)?;
    let mut replay = Side::new(200, 0.7, false)?;
    let mut smaller = false;
    let mut last = 0;
    for text in [
        "one two three",
        "four five six",
        "seven eight",
        "nine ten",
        "eleven",
    ] {
        let reused = same_reply(&mut keep, &mut replay, text, interrupt)?;
        smaller |= reused < last;
        last = keep.session.last_prompt.len();
    }
    assert!(smaller, "the window filled up, so an old turn was dropped");
    Ok(())
}

struct Breaks {
    room: usize,
}

impl Write for Breaks {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.room < buf.len() {
            return Err(io::Error::other("closed"));
        }
        self.room = self.room.saturating_sub(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn an_interrupted_reply_leaves_the_next_turn_exact(interrupt: &Interrupt) -> Result<()> {
    let mut keep = Side::new(1024, 0.7, true)?;
    let mut replay = Side::new(1024, 0.7, false)?;
    same_reply(&mut keep, &mut replay, "hello", interrupt)?;
    for side in [&mut keep, &mut replay] {
        let broken = side.chat.ask(
            &mut side.session,
            "go on",
            interrupt,
            &mut Breaks { room: 3 },
        );
        assert!(broken.is_err(), "the reply was cut");
    }
    assert_eq!(
        keep.session.retained().len(),
        keep.session.kv().used(),
        "a cancelled reply leaves an exact record"
    );
    let reused = same_reply(&mut keep, &mut replay, "go on", interrupt)?;
    assert!(reused > 0);
    Ok(())
}

/// Asks `text` and cancels it as Ctrl-C would: all the while (so it lands while the prompt is read, wherever that
/// is), or on the first piece of the reply.
fn cancelled_ask(
    side: &mut Side,
    text: &str,
    interrupt: &Interrupt,
    in_prefill: bool,
) -> Result<()> {
    struct Stops<'a>(&'a Interrupt);
    impl Write for Stops<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .cancel_flag()
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let done = std::sync::atomic::AtomicBool::new(false);
    let asked = std::thread::scope(|scope| {
        if in_prefill {
            scope.spawn(|| {
                while !done.load(std::sync::atomic::Ordering::SeqCst) {
                    interrupt
                        .cancel_flag()
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }
        let result = side
            .chat
            .ask(&mut side.session, text, interrupt, &mut Stops(interrupt));
        done.store(true, std::sync::atomic::Ordering::SeqCst);
        result
    });
    let cancelled = match asked {
        Err(e) if e.to_string().contains("cancelled") => true,
        Ok(reply) => reply.outcome.stop == runtime::StopReason::Cancelled,
        Err(_) => false,
    };
    assert!(cancelled, "the turn was cancelled");
    assert_eq!(
        side.session.retained().len(),
        side.session.kv().used(),
        "a cancelled turn leaves an exact record"
    );
    Ok(())
}

fn a_cancel_while_reading_or_replying_leaves_the_next_turn_exact(
    interrupt: &Interrupt,
) -> Result<()> {
    for in_prefill in [true, false] {
        let mut keep = Side::new(1024, 0.0, true)?;
        let mut replay = Side::new(1024, 0.0, false)?;
        same_reply(&mut keep, &mut replay, "hello there", interrupt)?;
        for side in [&mut keep, &mut replay] {
            cancelled_ask(side, "tell me about cats", interrupt, in_prefill)?;
        }
        let reused = same_reply(&mut keep, &mut replay, "go on", interrupt)?;
        assert!(
            reused > 0 || in_prefill,
            "what was read before the cancel is kept"
        );
    }
    Ok(())
}

/// A Qwen3.5 model on the GPU, from `MARIN_QWEN35_Q8_DIR`: a long question cancelled while its prompt is read leaves
/// a cache that answers the next question as a replay does. Skipped without the model unless CUDA is required.
fn a_qwen35_cuda_cancel_during_prefill_recovers_exactly(interrupt: &Interrupt) -> Result<()> {
    let Some(dir) = std::env::var_os("MARIN_QWEN35_Q8_DIR").map(std::path::PathBuf::from) else {
        assert!(
            std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
            "MARIN_QWEN35_Q8_DIR is not set"
        );
        eprintln!("skipping: set MARIN_QWEN35_Q8_DIR to a Qwen3.5 folder for the GPU cancel check");
        return Ok(());
    };
    let long = "Describe the history of the Roman Empire in great detail. ".repeat(30);
    let run = |reuse: bool| -> Result<(Vec<u32>, usize)> {
        let asked = ChatPrecision {
            weights: None,
            kv: None,
        };
        let loaded = load_chat(
            &dir,
            (Some(2048), 2048),
            BackendChoice::Cuda,
            precision_for_pick(asked, &dir),
            &Style::plain(),
            (false, false),
        )?;
        let settings = Settings {
            max_tokens: 24,
            temperature: 0.0,
            top_k: 20,
            top_p: 0.8,
            seed: Some(1),
        };
        let mut session = Session::new(loaded, settings, Style::plain())?;
        session.reuse = reuse;
        let chat = Chat::new(&session);
        let mut side = Side { session, chat };
        side.ask("My name is Ada.", interrupt)?;
        cancelled_ask(&mut side, &long, interrupt, true)?;
        side.ask("What is my name?", interrupt)
    };
    let (kept, reused) = run(true)?;
    let (replayed, _) = run(false)?;
    assert_eq!(kept, replayed, "the cancelled question left no trace");
    assert!(reused > 0, "the conversation before it was kept");
    Ok(())
}

/// The signal handler can be installed once per process, so all checks share one test.
#[test]
fn chat_turns_reuse_the_cache_exactly() -> Result<()> {
    let interrupt = Interrupt::install()?;
    a_conversation_reuses_its_history_and_answers_exactly_as_a_replay(&interrupt)?;
    tools_that_touch_the_cache_make_the_next_turn_read_everything(&interrupt)?;
    forgetting_old_turns_to_fit_the_window_stays_exact(&interrupt)?;
    an_interrupted_reply_leaves_the_next_turn_exact(&interrupt)?;
    a_cancel_while_reading_or_replying_leaves_the_next_turn_exact(&interrupt)?;
    a_qwen35_cuda_cancel_during_prefill_recovers_exactly(&interrupt)
}
