//! The one terminal UI on a pseudo-terminal: modes and commands typed after `/`, `marin run` without a question,
//! and a terminal that comes back usable. The tiny chat model has a window too small for code mode, which is also
//! how a too-small `--ctx` shows up to a user.

mod pty;

use std::path::Path;
use std::process::ExitStatus;

use pty::{Res, Sandbox, Step, have, on_terminal};

/// What to type and what must then be drawn (empty: nothing in particular, the keys only edit). `REPLY` waits for
/// an answer to end and the prompt to be back, since keys typed before that are dropped on purpose.
type Keys<'a> = &'a [(&'a str, &'a str)];

const REPLY: &str = "\u{0}reply";

fn drawn(shown: &str, text: &str) -> bool {
    if shown == REPLY {
        return text
            .find("stopped at")
            .and_then(|at| text.get(at..))
            .is_some_and(|rest| rest.contains("you >"));
    }
    text.contains(shown)
}

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

/// Runs the command line `command` builds in a fresh sandbox, which is also the config folder; returns what was
/// drawn, how the program ended and the sandbox so a test can look at what was saved.
fn run_in(
    name: &str,
    keys: Keys<'_>,
    command: impl FnOnce(&Sandbox) -> Res<String>,
) -> Res<Option<(String, ExitStatus, Sandbox)>> {
    if !have("script") {
        eprintln!("skipping: `script` is not installed");
        return Ok(None);
    }
    let sandbox = Sandbox::new(name)?;
    let command_line = command(&sandbox)?;
    let checks: Vec<_> = keys
        .iter()
        .map(|(_, shown)| {
            let shown = (*shown).to_string();
            (shown.clone(), move |_: &[u8], new: &[u8]| {
                let text = String::from_utf8_lossy(new);
                if drawn(&shown, &text) {
                    Ok(())
                } else {
                    Err(text.escape_debug().to_string())
                }
            })
        })
        .collect();
    let steps: Vec<Step> = keys
        .iter()
        .zip(&checks)
        .map(|((typed, shown), (what, check))| Step {
            keys: typed,
            until: (!shown.is_empty()).then_some((what.as_str(), check)),
        })
        .collect();
    let (drawn, _, status) = on_terminal(&command_line, (110, 36), &sandbox, false, &steps)?;
    Ok(Some((
        String::from_utf8_lossy(&drawn).into_owned(),
        status,
        sandbox,
    )))
}

const PROMPT: (&str, &str) = ("", "you >");

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
