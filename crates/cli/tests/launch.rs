//! What a user meets when `marin` starts: no model yet, one, several, a saved choice that went missing, a damaged
//! config, flags against saved settings, commands on a pipe, and servers that are absent, busy or not Marin's.

mod common;

use std::fs;
use std::io::{self, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use common::{Sandbox, fixtures, text};

fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

/// A scratch home plus a models folder holding a copy of the tiny chat model under each of `names`.
struct Setup {
    sandbox: Sandbox,
    models: PathBuf,
}

impl Setup {
    fn new(name: &str, names: &[&str]) -> io::Result<Self> {
        let sandbox = Sandbox::new(name)?;
        let models = sandbox.0.join("models");
        fs::create_dir_all(&models)?;
        for model in names {
            copy_dir(&fixtures().join("tiny-chat"), &models.join(model))?;
        }
        Ok(Self { sandbox, models })
    }

    fn home(&self) -> PathBuf {
        self.sandbox.0.join("home")
    }

    fn config(&self) -> String {
        fs::read_to_string(self.home().join("config.json")).unwrap_or_default()
    }

    fn write_config(&self, text: &str) -> io::Result<()> {
        fs::create_dir_all(self.home())?;
        fs::write(self.home().join("config.json"), text)
    }

    fn marin(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
        command
            .env("MARIN_HOME", self.home())
            .env("MARIN_MODELS", &self.models)
            .env("NO_COLOR", "1");
        command
    }
}

fn go(command: &mut Command, stdin: &str) -> io::Result<Output> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(stdin.as_bytes())?;
    }
    child.wait_with_output()
}

#[test]
fn with_no_model_anywhere_a_pipe_gets_an_error_that_says_what_to_do() -> io::Result<()> {
    let setup = Setup::new("none", &[])?;
    let out = go(&mut setup.marin(), "hello\n")?;
    assert!(!out.status.success());
    let stderr = text(&out.stderr);
    assert!(stderr.contains("no chat model found"), "{stderr}");
    assert!(
        stderr.contains("hint:") && stderr.contains("huggingface-cli"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn the_only_model_is_used_and_remembered_without_asking() -> io::Result<()> {
    let setup = Setup::new("only", &["alpha"])?;
    let out = go(setup.marin().args(["--backend", "cpu"]), "/status\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    assert!(
        stdout.contains("alpha") && stdout.contains("mode  chat"),
        "{stdout}"
    );
    assert!(setup.config().contains("alpha"), "{}", setup.config());
    let again = go(setup.marin().args(["--backend", "cpu"]), "/model\n")?;
    assert!(text(&again.stdout).contains("alpha"));
    Ok(())
}

#[test]
fn several_models_on_a_pipe_ask_for_a_choice_instead_of_guessing() -> io::Result<()> {
    let setup = Setup::new("several", &["alpha", "beta"])?;
    let out = go(setup.marin().args(["--backend", "cpu"]), "/status\n")?;
    assert!(!out.status.success());
    let stderr = text(&out.stderr);
    assert!(
        stderr.contains("several models found: alpha, beta"),
        "{stderr}"
    );
    assert!(stderr.contains("--model"), "{stderr}");
    let chosen = go(
        setup
            .marin()
            .args(["--backend", "cpu", "--model"])
            .arg(setup.models.join("beta")),
        "/model\n",
    )?;
    assert!(
        text(&chosen.stdout).contains("beta"),
        "{}",
        text(&chosen.stderr)
    );
    assert!(
        setup.config().contains("beta"),
        "an explicit --model becomes the default"
    );
    Ok(())
}

#[test]
fn a_saved_model_that_went_missing_is_noted_and_replaced() -> io::Result<()> {
    let setup = Setup::new("missing", &["alpha"])?;
    setup.write_config(r#"{"model": "/nowhere/gone-model"}"#)?;
    let out = go(setup.marin().args(["--backend", "cpu"]), "/model\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        text(&out.stderr).contains("no longer there"),
        "{}",
        text(&out.stderr)
    );
    assert!(text(&out.stdout).contains("alpha"));
    assert!(setup.config().contains("alpha") && !setup.config().contains("gone-model"));
    Ok(())
}

#[test]
fn a_damaged_config_is_reported_and_marin_still_starts() -> io::Result<()> {
    let setup = Setup::new("damaged", &["alpha"])?;
    setup.write_config("{this is not json")?;
    let out = go(setup.marin().args(["--backend", "cpu"]), "/model\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        text(&out.stderr).contains("ignoring the saved settings"),
        "{}",
        text(&out.stderr)
    );
    assert!(
        serde_json::from_str::<serde_json::Value>(&setup.config()).is_ok(),
        "{}",
        setup.config()
    );
    Ok(())
}

#[test]
fn an_old_config_with_only_a_model_keeps_working() -> io::Result<()> {
    let setup = Setup::new("old", &["alpha", "beta"])?;
    setup.write_config(&format!(
        "{{\"model\": \"{}\"}}\n",
        setup.models.join("beta").display()
    ))?;
    let out = go(setup.marin().args(["--backend", "cpu"]), "/model\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("beta"));
    assert!(text(&out.stderr).is_empty(), "{}", text(&out.stderr));
    Ok(())
}

#[test]
fn flags_beat_saved_settings_and_saved_settings_beat_the_defaults() -> io::Result<()> {
    let setup = Setup::new("precedence", &["alpha"])?;
    setup.write_config(r#"{"backend": "cpu", "temp": 0.2, "max_tokens": 3}"#)?;
    let saved = go(&mut setup.marin(), "/status\n")?;
    let stdout = text(&saved.stdout);
    assert!(
        stdout.contains("temp 0.2") && stdout.contains("max 3 tokens"),
        "{stdout}{}",
        text(&saved.stderr)
    );
    assert!(stdout.contains("cpu"), "{stdout}");
    let flagged = go(
        setup.marin().args(["--temp", "0.9", "--max-tokens", "7"]),
        "/status\n",
    )?;
    let stdout = text(&flagged.stdout);
    assert!(
        stdout.contains("temp 0.9") && stdout.contains("max 7 tokens"),
        "{stdout}"
    );
    assert!(
        !setup.config().contains("0.9"),
        "flags are not saved: {}",
        setup.config()
    );
    let none = Setup::new("defaults", &["alpha"])?;
    let stdout = text(&go(none.marin().args(["--backend", "cpu"]), "/status\n")?.stdout);
    assert!(
        stdout.contains("temp 0.7") && stdout.contains("max 512 tokens"),
        "{stdout}"
    );
    Ok(())
}

#[test]
fn commands_on_a_pipe_switch_modes_clear_and_leave_text_alone() -> io::Result<()> {
    let setup = Setup::new("commands", &["alpha"])?;
    let script = "/help\n/model\n/mode\n/mode code\n/mode chat\n/clear\n/usr/bin is a path\n/exit\nnever answered\n";
    let out = go(
        setup
            .marin()
            .args(["--backend", "cpu", "--temp", "0", "--max-tokens", "4"]),
        script,
    )?;
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert!(stdout.contains("/mode chat|code"), "{stdout}");
    assert!(stdout.contains("alpha"), "{stdout}");
    assert!(stdout.contains("use /mode chat or /mode code"), "{stdout}");
    assert!(
        stderr.contains("code mode needs a context window of at least 2048"),
        "the tiny model's window is too small: {stderr}"
    );
    assert!(
        stdout.contains("started a new chat conversation"),
        "{stdout}"
    );
    assert!(
        !stdout.contains("unknown command"),
        "an unknown /word is just text: {stdout}"
    );
    assert!(!stdout.contains("never answered"));
    Ok(())
}

#[test]
fn run_still_answers_once_and_prints_only_the_reply() -> io::Result<()> {
    let setup = Setup::new("run", &["alpha"])?;
    let out = go(
        setup.marin().args([
            "--backend",
            "cpu",
            "--temp",
            "0",
            "--max-tokens",
            "4",
            "run",
            "the",
            "cat",
        ]),
        "",
    )?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        !out.stdout.is_empty() && out.stderr.is_empty(),
        "{}",
        text(&out.stderr)
    );
    let piped = go(
        setup.marin().args([
            "--backend",
            "cpu",
            "--temp",
            "0",
            "--max-tokens",
            "4",
            "run",
        ]),
        "the cat sat\n",
    )?;
    assert!(piped.status.success(), "{}", text(&piped.stderr));
    assert!(!piped.stdout.is_empty());
    Ok(())
}

#[test]
fn code_without_a_server_loads_a_model_of_its_own_and_says_what_is_missing() -> io::Result<()> {
    let setup = Setup::new("code-none", &[])?;
    let out = go(setup.marin().arg("code"), "where is main?\n")?;
    let stderr = text(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("no chat model found") && !stderr.contains("no Marin server"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn a_port_that_answers_with_something_else_is_a_clear_error_not_a_hang() -> io::Result<()> {
    let setup = Setup::new("stale", &["alpha"])?;
    for reply in [
        &b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nnot json!"[..],
        &b""[..],
    ] {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let reply = reply.to_vec();
        let answerer = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0_u8; 512];
                let _ = stream.read(&mut buffer);
                let _ = stream.write_all(&reply);
            }
        });
        let began = std::time::Instant::now();
        let out = go(
            setup.marin().args(["code", "--port", &port.to_string()]),
            "",
        )?;
        let _ = answerer.join();
        let stderr = text(&out.stderr);
        assert!(!out.status.success(), "{stderr}");
        assert!(
            stderr.contains("error:") && stderr.contains("hint:"),
            "{stderr}"
        );
        assert!(began.elapsed() < std::time::Duration::from_secs(10));
    }
    Ok(())
}

#[test]
fn serve_on_a_busy_port_says_so_before_loading_anything() -> io::Result<()> {
    let setup = Setup::new("busy", &["alpha"])?;
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let began = std::time::Instant::now();
    let out = go(
        setup
            .marin()
            .args(["--backend", "cpu", "serve", "--port", &port.to_string()]),
        "",
    )?;
    let stderr = text(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains(&format!("port {port} is already in use")) && stderr.contains("hint:"),
        "{stderr}"
    );
    assert!(began.elapsed() < std::time::Duration::from_secs(3));
    drop(listener);
    Ok(())
}
