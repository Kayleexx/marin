//! The one terminal UI on a pseudo-terminal: modes and commands typed after `/`, `marin run` without a question,
//! and a terminal that comes back usable. The tiny chat model has a window too small for code mode, which is also
//! how a too-small `--ctx` shows up to a user.

#[path = "pty/keys.rs"]
mod keys;
mod pty;

use std::path::Path;
use std::process::ExitStatus;

use keys::{Keys, PROMPT, REPLY, STREAMING, run_in};
use pty::{Res, Sandbox};

fn run(name: &str, tail: &str, keys: Keys<'_>) -> Res<Option<(String, ExitStatus)>> {
    run_in(name, keys, |_| {
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        Ok(format!(
            "{} --model {} --backend cpu --temp 0 --max-tokens 3 {tail}",
            env!("CARGO_BIN_EXE_marin"),
            model.display()
        ))
    })
    .map(|ran| ran.map(|(screen, status, _)| (screen, status)))
}

#[test]
fn slash_commands_switch_modes_show_status_and_explain_a_window_that_is_too_small() -> Res<()> {
    let keys = [
        PROMPT,
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
        ("/", "Mode: code"),
        ("mode chat", ""),
        ("\r", "already in chat mode"),
        ("/", "Mode: code"),
        ("mode code", ""),
        ("\r", "code mode needs a context window of at least 2048"),
        ("hello world\r", REPLY),
    ];
    let Some((screen, status)) = run("modes", "", &keys)? else {
        return Ok(());
    };
    assert!(
        screen.contains("current · talk with the model"),
        "the palette marks the current mode: {screen:?}"
    );
    assert!(
        screen.contains("larger window") || screen.contains("--ctx"),
        "the error says what to do: {screen:?}"
    );
    assert!(
        !screen.contains("code mode ·"),
        "a mode that could not start is not shown as active: {screen:?}"
    );
    assert!(status.success(), "ctrl+d leaves cleanly: {status:?}");
    Ok(())
}

#[test]
fn slash_help_and_settings_and_clear_work_from_the_prompt() -> Res<()> {
    let keys = [
        PROMPT,
        ("/", "Mode: chat"),
        ("help", ""),
        ("\r", "/status"),
        ("hello world\r", REPLY),
        ("/", "Mode: chat"),
        ("clear", ""),
        ("\r", "started a new chat conversation"),
        ("/", "Mode: chat"),
        ("settings", ""),
        ("\r", "temperature"),
        ("\x1b", ""),
    ];
    let Some((screen, status)) = run("help", "", &keys)? else {
        return Ok(());
    };
    assert!(
        screen.contains("/status") && screen.contains("/exit"),
        "{screen:?}"
    );
    assert!(status.success());
    Ok(())
}

#[test]
fn run_without_a_question_on_a_terminal_opens_the_same_chat() -> Res<()> {
    let keys = [PROMPT, ("hello world\r", REPLY)];
    let Some((screen, status)) = run("run", "run", &keys)? else {
        return Ok(());
    };
    assert!(screen.contains("you > hello world"), "{screen:?}");
    assert!(status.success());
    Ok(())
}

#[test]
fn ctrl_c_clears_a_line_then_asks_twice_and_the_terminal_comes_back() -> Res<()> {
    let keys = [
        PROMPT,
        ("some text", ""),
        ("\x03", ""),
        ("\x03", "press ctrl+c again to exit"),
        ("\x03", ""),
    ];
    let Some((screen, _)) = run("ctrlc", "; stty -a", &keys)? else {
        return Ok(());
    };
    let flags = screen.rsplit("lflags").next().unwrap_or("");
    assert!(
        flags.contains(" icanon ") && !flags.contains("-icanon"),
        "terminal left raw: {flags:?}"
    );
    Ok(())
}

#[test]
fn an_upgrading_user_with_a_saved_packed_model_is_asked_once_and_not_again() -> Res<()> {
    let setup = |sandbox: &Sandbox| -> Res<String> {
        let packed = sandbox.0.join("packed");
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        let export = std::process::Command::new(env!("CARGO_BIN_EXE_marin-run"))
            .arg("export-int8")
            .arg(&source)
            .arg(&packed)
            .args(["--quant-group", "32"])
            .output()?;
        assert!(
            export.status.success(),
            "{}",
            String::from_utf8_lossy(&export.stderr)
        );
        std::fs::write(
            sandbox.0.join("config.json"),
            format!("{{\"model\": \"{}\"}}\n", packed.display()),
        )?;
        Ok(format!(
            "{} --backend cpu --temp 0 --max-tokens 3",
            env!("CARGO_BIN_EXE_marin")
        ))
    };
    let first = [("", "Load it?"), ("\r", "you >")];
    let Some((screen, status, sandbox)) = run_in("upgrade", &first, setup)? else {
        return Ok(());
    };
    assert!(status.success(), "{screen:?}");
    let saved = std::fs::read_to_string(sandbox.0.join("config.json"))?;
    assert!(
        saved.contains("\"quantize\": \"int8\"") && saved.contains("\"quant_group\": 32"),
        "{saved}"
    );
    // The same folder and config again: no question this time.
    let again = [("", "you >")];
    let Some((screen, status, _)) = run_in("upgrade-again", &again, |next| {
        std::fs::create_dir_all(&next.0)?;
        std::fs::copy(sandbox.0.join("config.json"), next.0.join("config.json"))?;
        std::os::unix::fs::symlink(sandbox.0.join("packed"), next.0.join("packed"))?;
        let line = std::fs::read_to_string(next.0.join("config.json"))?.replace(
            &sandbox.0.join("packed").display().to_string(),
            &next.0.join("packed").display().to_string(),
        );
        std::fs::write(next.0.join("config.json"), line)?;
        Ok(format!(
            "{} --backend cpu --temp 0 --max-tokens 3",
            env!("CARGO_BIN_EXE_marin")
        ))
    })?
    else {
        return Ok(());
    };
    assert!(
        !screen.contains("Load it?") && status.success(),
        "{screen:?}"
    );
    Ok(())
}

fn two_models(sandbox: &Sandbox) -> Res<String> {
    let models = sandbox.0.join("models");
    for name in ["alpha", "beta"] {
        let to = models.join(name);
        std::fs::create_dir_all(&to)?;
        let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            std::fs::copy(entry.path(), to.join(entry.file_name()))?;
        }
    }
    Ok(format!(
        "MARIN_MODELS={} {} --model {} --backend cpu --temp 0 --max-tokens 3",
        models.display(),
        env!("CARGO_BIN_EXE_marin"),
        models.join("alpha").display()
    ))
}

#[test]
fn switching_model_starts_both_conversations_over_and_remembers_the_new_one() -> Res<()> {
    let keys = [
        PROMPT,
        ("hello world\r", REPLY),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
        ("/", "Mode: chat"),
        ("model", ""),
        ("\r", "beta"),
        ("beta", ""),
        ("\r", "new model: both conversations start over"),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
    ];
    let Some((screen, status, sandbox)) = run_in("switch", &keys, two_models)? else {
        return Ok(());
    };
    assert!(status.success(), "{screen:?}");
    let (before, after) = screen
        .split_once("new model: both conversations start over")
        .ok_or("no switch notice")?;
    assert!(
        !before.contains("context 0 of")
            && after.contains("beta")
            && after.contains("context 0 of"),
        "the window is empty again on the new model: {after:?}"
    );
    let saved = std::fs::read_to_string(sandbox.0.join("config.json"))?;
    assert!(
        saved.contains("beta") && !saved.contains("alpha"),
        "{saved}"
    );
    Ok(())
}

#[test]
fn ctrl_c_stops_an_answer_and_leaves_the_prompt_and_the_conversation_usable() -> Res<()> {
    let keys = [
        PROMPT,
        (
            "Write a detailed 1000 word essay on the history of Rome.\r",
            STREAMING,
        ),
        ("\x03", "cancelled"),
        ("/", "Mode: chat"),
        ("status", ""),
        ("\r", "mode  chat"),
    ];
    let Some(model) = std::env::var_os("MARIN_QWEN3_CHAT_DIR") else {
        eprintln!("skipping: the tiny model answers too fast to stop; set MARIN_QWEN3_CHAT_DIR");
        return Ok(());
    };
    let Some((screen, status, _)) = run_in("cancel", &keys, |_| {
        Ok(format!(
            "{} --model {} --backend cpu --temp 0 --max-tokens 400",
            env!("CARGO_BIN_EXE_marin"),
            Path::new(&model).display()
        ))
    })?
    else {
        return Ok(());
    };
    assert!(
        status.success(),
        "the terminal is back and Ctrl-D leaves: {screen:?}"
    );
    assert!(
        !screen.contains("stopped at max tokens"),
        "it was stopped, not run to the end: {screen:?}"
    );
    Ok(())
}
