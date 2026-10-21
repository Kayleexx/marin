//! Chat on a Qwen3.5 model on the GPU: each turn reads only what is new because the cache is marked at the end of the
//! conversation and returned to it, no second copy of the keys and values is made, and a start that cannot be kept is
//! reported as replayed instead of reused. Needs a GPU.

#[path = "common/qwen35.rs"]
mod qwen35;

use cli::chat::Chat;
use cli::load::{ChatPrecision, load_chat};
use cli::session::{Session, Settings};
use cli::setup::precision_for_pick;
use cli::signal::Interrupt;
use cli::style::Style;
use model::BackendChoice;
use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use runtime::SequenceCache;

fn session(dir: &std::path::Path) -> Res<Session> {
    let asked = ChatPrecision {
        weights: None,
        kv: None,
    };
    let loaded = load_chat(
        dir,
        (Some(1024), 1024),
        BackendChoice::Cuda,
        precision_for_pick(asked, dir),
        &Style::plain(),
        (false, false),
    )?;
    let settings = Settings {
        max_tokens: 6,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(3),
    };
    Ok(Session::new(loaded, settings, Style::plain())?)
}

#[test]
fn turns_read_only_what_is_new_and_a_lost_start_is_reported_as_replayed() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let scratch = Scratch::new("marks")?;
    let packed = packed_tiny(&scratch.0)?;
    let mut session = session(&packed)?;
    let mut chat = Chat::new(&session);
    let interrupt = Interrupt::unhooked();
    let words =
        |n: usize| format!("the quick brown fox number {n} jumps over the lazy dog ").repeat(2);
    let mut previous = 0_usize;
    for n in 0..5 {
        let reply = chat.ask(&mut session, &words(n), &interrupt, &mut Vec::new())?;
        let o = &reply.outcome;
        assert_eq!(o.replayed, 0, "turn {n}");
        if n == 0 {
            assert_eq!(o.reused, 0);
        } else {
            assert!(
                o.reused >= previous.saturating_sub(8),
                "turn {n}: kept {} of {previous}",
                o.reused
            );
            assert!(
                o.prompt_tokens - o.reused < o.prompt_tokens / 2,
                "turn {n} read only what is new"
            );
        }
        assert_eq!(
            session.retained().len(),
            session.kv().used(),
            "the record covers the cache"
        );
        previous = session.kv().used();
    }

    chat.clear();
    let restarted = chat.ask(&mut session, &words(9), &interrupt, &mut Vec::new())?;
    assert!(
        restarted.outcome.replayed > 0,
        "the old conversation could not be cut back: {:?}",
        restarted.outcome.replayed
    );
    assert_eq!(
        restarted.outcome.reused, 0,
        "nothing is called kept that was read again"
    );
    let next = chat.ask(&mut session, &words(10), &interrupt, &mut Vec::new())?;
    assert_eq!(
        next.outcome.replayed, 0,
        "after a replay, turns are marked again"
    );
    assert!(next.outcome.reused > 0);
    Ok(())
}
