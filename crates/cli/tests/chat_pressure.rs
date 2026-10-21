//! Chat on a Qwen3.5 model on the GPU after another program has taken nearly all the free GPU memory: the mark of the
//! conversation goes to host memory, so turns still read only what is new instead of the whole history again. Needs a
//! GPU. This is the only test in this binary, so nothing else is using the GPU while it holds the memory.

#[path = "common/qwen35.rs"]
mod qwen35;

use cli::chat::Chat;
use cli::load::{ChatPrecision, load_chat};
use cli::session::{Session, Settings};
use cli::setup::precision_for_pick;
use cli::signal::Interrupt;
use cli::style::Style;
use model::{BackendChoice, GPU_SLACK_BYTES};
use qwen35::{Res, Scratch, have_gpu, packed_tiny};

const LEFT_FREE: usize = 150 << 20;

#[test]
fn turns_on_a_gpu_with_almost_no_free_memory_still_read_only_what_is_new() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let scratch = Scratch::new("pressure")?;
    let packed = packed_tiny(&scratch.0)?;
    let asked = ChatPrecision {
        weights: None,
        kv: None,
    };
    let loaded = load_chat(
        &packed,
        (Some(1024), 1024),
        BackendChoice::Cuda,
        precision_for_pick(asked, &packed),
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
    let mut session = Session::new(loaded, settings, Style::plain())?;
    let mut chat = Chat::new(&session);
    let interrupt = Interrupt::unhooked();
    let words =
        |n: usize| format!("the quick brown fox number {n} jumps over the lazy dog ").repeat(2);

    let other_program = model::Gpu::new()?;
    let free = other_program.info()?.free_bytes;
    let _taken = other_program.zeros(free.saturating_sub(LEFT_FREE) / 4)?;
    assert!(
        other_program.info()?.free_bytes < usize::try_from(GPU_SLACK_BYTES)?,
        "the memory is taken"
    );

    chat.ask(&mut session, &words(0), &interrupt, &mut Vec::new())?;
    for n in 1..4 {
        let reply = chat.ask(&mut session, &words(n), &interrupt, &mut Vec::new())?;
        let o = &reply.outcome;
        assert_eq!(o.replayed, 0, "turn {n} was not read again");
        assert!(o.reused > 0, "turn {n} kept the conversation");
    }
    Ok(())
}
