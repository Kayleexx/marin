//! Chat and code mode in the real terminal on models with a window long enough for both: switching modes and
//! models, and `/clear`, as a person would type them.

#[path = "pty/keys.rs"]
mod keys;
#[path = "common/long.rs"]
mod long;
mod pty;

use keys::{PROMPT, REPLY, run_in};
use pty::{Res, Sandbox};

fn two_long_models(sandbox: &Sandbox) -> Res<String> {
    let models = sandbox.0.join("models");
    let tiny =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    for name in ["alpha", "beta"] {
        long::long_tiny(&tiny, &models.join(name))?;
    }
    Ok(format!(
        "MARIN_MODELS={} {} --model {} --backend cpu --temp 0 --max-tokens 3",
        models.display(),
        env!("CARGO_BIN_EXE_marin"),
        models.join("alpha").display()
    ))
}

#[test]
fn modes_clear_and_model_switch_work_together_in_the_terminal() -> Res<()> {
    let keys = [
        PROMPT,
        ("hello there\r", REPLY),
        ("/", "Mode: chat"),
        ("mode code", ""),
        ("\r", "code mode: reading"),
        ("where is main?\r", "tokens written"),
        ("/", "Mode: chat"),
        ("mode chat", ""),
        ("\r", "chat mode. Code history is kept"),
        ("/", "Mode: chat"),
        ("clear", ""),
        ("\r", "started a new chat conversation"),
        ("/", "Mode: chat"),
        ("model", ""),
        ("\r", "beta"),
        ("beta", ""),
        ("\r", "new model: both conversations start over"),
        ("/", "Mode: chat"),
        ("mode code", ""),
        ("\r", "code mode: reading"),
        ("where is main?\r", "tokens written"),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  code"),
    ];
    let Some((screen, status, sandbox)) = run_in("long-modes", &keys, two_long_models)? else {
        return Ok(());
    };
    assert!(status.success(), "{screen:?}");
    assert!(
        !screen.contains("code mode needs a context window"),
        "the long window is enough for code mode: {screen:?}"
    );
    let saved = std::fs::read_to_string(sandbox.0.join("config.json"))?;
    assert!(saved.contains("beta"), "{saved}");
    Ok(())
}

#[test]
fn save_sessions_and_resume_work_from_the_palette() -> Res<()> {
    let keys = [
        PROMPT,
        ("hello there\r", REPLY),
        ("/", "Mode: chat"),
        ("save", ""),
        ("\r", "Save as"),
        ("mine\r", "saved `mine`"),
        ("/", "Mode: chat"),
        ("clear", ""),
        ("\r", "started a new chat conversation"),
        ("/", "Mode: chat"),
        ("resume", ""),
        ("\r", "Resume which"),
        ("mine\r", "resumed `mine` · 1 chat and 0 code questions"),
        ("/", "Mode: chat"),
        ("sessions", ""),
        ("\r", "mine · hello there"),
    ];
    let Some((screen, status, sandbox)) = run_in("long-save", &keys, two_long_models)? else {
        return Ok(());
    };
    assert!(status.success(), "{screen:?}");
    let saved = std::fs::read_to_string(sandbox.0.join("sessions/mine.json"))?;
    assert!(saved.contains("hello there"), "{saved}");
    Ok(())
}
