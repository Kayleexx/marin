//! Saving a conversation, leaving, and resuming it, through the real `marin` on a pipe: the history comes back exactly,
//! a run that dies or is cancelled leaves nothing half-written, and anything damaged or unsuitable is refused.

mod common;
#[path = "common/long.rs"]
mod long;

use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use common::{Sandbox, fixtures, text};
use long::long_tiny;
use serde_json::Value;

struct Rig {
    sandbox: Sandbox,
    models: PathBuf,
    project: PathBuf,
}

impl Rig {
    fn new(name: &str, models: &[&str]) -> io::Result<Self> {
        let sandbox = Sandbox::new(name)?;
        let root = sandbox.0.clone();
        let tiny = fixtures().join("tiny-chat");
        for model in models {
            long_tiny(&tiny, &root.join("models").join(model))?;
        }
        let project = root.join("project");
        fs::create_dir_all(project.join("src"))?;
        fs::write(project.join("src/main.rs"), "fn main() {}\n")?;
        Ok(Self {
            models: root.join("models"),
            project,
            sandbox,
        })
    }

    fn home(&self) -> PathBuf {
        self.sandbox.0.join("home")
    }

    fn file(&self, name: &str) -> PathBuf {
        self.home().join("sessions").join(format!("{name}.json"))
    }

    fn saved(&self, name: &str) -> io::Result<Value> {
        serde_json::from_slice(&fs::read(self.file(name))?).map_err(io::Error::other)
    }

    fn command(&self, model: &str, cwd: &Path, max_tokens: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_marin"));
        command
            .current_dir(cwd)
            .env("MARIN_HOME", self.home())
            .env("MARIN_MODELS", &self.models)
            .env("NO_COLOR", "1")
            .args([
                "--backend",
                "cpu",
                "--temp",
                "0",
                "--max-tokens",
                max_tokens,
            ])
            .arg("--model")
            .arg(self.models.join(model))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    fn run_in(&self, model: &str, cwd: &Path, script: &str) -> io::Result<Output> {
        let mut child = self.command(model, cwd, "6").spawn()?;
        if let Some(mut input) = child.stdin.take() {
            input.write_all(script.as_bytes())?;
        }
        child.wait_with_output()
    }

    fn run(&self, script: &str) -> io::Result<Output> {
        self.run_in("alpha", &self.project, script)
    }
}

fn conversation(saved: &Value) -> (Value, Value) {
    (saved["chat"].clone(), saved["code"].clone())
}

#[test]
fn a_resumed_conversation_is_the_saved_one_and_continues_exactly_as_if_it_never_stopped()
-> io::Result<()> {
    let rig = Rig::new("resume", &["alpha"])?;
    let first = rig.run("my name is Ada\nwhat is it\n/save mine\n")?;
    assert!(first.status.success(), "{}", text(&first.stderr));
    assert!(
        text(&first.stdout).contains("saved `mine`"),
        "{}",
        text(&first.stdout)
    );

    let again = rig.run("/resume mine\n/save copy\n")?;
    assert!(text(&again.stdout).contains("resumed `mine` · 2 chat and 0 code questions"));
    assert_eq!(
        conversation(&rig.saved("mine")?),
        conversation(&rig.saved("copy")?)
    );

    let straight = rig.run("my name is Ada\nwhat is it\nand now\n/save straight\n")?;
    assert!(straight.status.success(), "{}", text(&straight.stderr));
    let resumed = rig.run("/resume mine\nand now\n/save resumed\n")?;
    assert!(resumed.status.success(), "{}", text(&resumed.stderr));
    assert_eq!(
        conversation(&rig.saved("straight")?),
        conversation(&rig.saved("resumed")?),
        "the same third answer after a restart"
    );
    let listing = rig.run("/sessions\n")?;
    let shown = text(&listing.stdout);
    assert!(
        shown.contains("mine") && shown.contains("my name is Ada"),
        "{shown}"
    );
    Ok(())
}

#[test]
fn code_and_chat_histories_are_saved_apart_and_resume_in_the_mode_they_were_left_in()
-> io::Result<()> {
    let rig = Rig::new("modes", &["alpha"])?;
    let out = rig.run("hello\n/mode code\nwhere is main?\n/save both\n")?;
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stderr}");
    assert!(
        stdout.contains("source code and tool output"),
        "the privacy note is shown: {stdout}"
    );
    let saved = rig.saved("both")?;
    assert_eq!(saved["mode"], "code");
    assert_eq!(saved["chat"].as_array().map(Vec::len), Some(2));
    let code = saved["code"]["items"].as_array().map(Vec::len);
    assert!(code.is_some_and(|n| n >= 2), "{saved}");
    assert_eq!(
        Path::new(saved["code"]["root"].as_str().unwrap_or("")),
        rig.project.canonicalize()?,
    );

    let elsewhere = rig.sandbox.0.join("elsewhere");
    fs::create_dir_all(&elsewhere)?;
    let back = rig.run_in(
        "alpha",
        &elsewhere,
        "/resume both\n/status\n/mode chat\n/clear\n/save both\n",
    )?;
    let shown = text(&back.stdout);
    assert!(shown.contains("code mode"), "{shown}");
    assert!(
        shown.contains(&format!(
            "mode  code · {}",
            rig.project.canonicalize()?.display()
        )),
        "{shown}"
    );
    let after = rig.saved("both")?;
    assert_eq!(
        after["code"], saved["code"],
        "clearing the chat left the code history alone"
    );
    assert_eq!(after["chat"].as_array().map(Vec::len), Some(0));
    Ok(())
}

#[test]
fn nothing_is_written_unless_asked_and_a_cancelled_question_is_never_saved() -> io::Result<()> {
    let rig = Rig::new("optin", &["alpha"])?;
    let out = rig.run("hello\n")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(!rig.home().join("sessions").exists(), "saving is opt-in");

    for (mode, ask) in [
        ("", "tell me a long story"),
        ("/mode code\n", "where is main?"),
    ] {
        let mut child = rig.command("alpha", &rig.project, "3000").spawn()?;
        child
            .stdin
            .take()
            .map(|mut i| i.write_all(format!("{mode}{ask}\n/save lost\n").as_bytes()))
            .transpose()?;
        std::thread::sleep(Duration::from_secs(3));
        Command::new("kill")
            .args(["-INT", &child.id().to_string()])
            .status()?;
        let out = child.wait_with_output()?;
        let shown = format!("{}{}", text(&out.stdout), text(&out.stderr));
        assert!(shown.contains("nothing to save yet"), "{mode:?}: {shown}");
        assert!(!rig.file("lost").exists(), "{mode:?}");
    }
    Ok(())
}

#[test]
fn a_run_killed_while_saving_never_leaves_a_damaged_session() -> io::Result<()> {
    let rig = Rig::new("killed", &["alpha"])?;
    let script = format!("hello\n{}", "/save k\n".repeat(4000));
    for delay in [0, 1, 3, 7, 15, 30, 60, 120] {
        let mut child = rig.command("alpha", &rig.project, "6").spawn()?;
        child
            .stdin
            .take()
            .map(|mut i| i.write_all(script.as_bytes()))
            .transpose()?;
        while !rig.file("k").exists() {
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(delay));
        child.kill()?;
        child.wait()?;
        let saved = rig.saved("k")?;
        assert_eq!(
            saved["chat"].as_array().map(Vec::len),
            Some(2),
            "delay {delay}: {saved}"
        );
    }
    Ok(())
}

#[test]
fn damaged_oversized_outdated_and_foreign_sessions_are_refused_with_the_reason() -> io::Result<()> {
    let rig = Rig::new("bad", &["alpha", "beta"])?;
    rig.run("hello\n/save good\n")?;
    let good = fs::read_to_string(rig.file("good"))?;
    let write = |name: &str, bytes: &[u8]| fs::write(rig.file(name), bytes);
    write("cut", &good.as_bytes()[..good.len() / 2])?;
    write(
        "future",
        good.replace("\"version\":1", "\"version\":9").as_bytes(),
    )?;
    write("huge", &vec![b' '; 3 << 20])?;
    write("foreign", b"{\"hello\": 1}")?;
    let out = rig.run("/sessions\n/resume cut\n/resume future\n/resume huge\n/resume foreign\n/resume nothere\n/resume ../good\n")?;
    let shown = format!("{}{}", text(&out.stdout), text(&out.stderr));
    for needle in [
        "cut · damaged",
        "future · damaged",
        "huge · damaged",
        "is damaged",
        "session format 9",
        "sessions larger than",
        "no saved session called `nothere`",
        "not a usable session name",
    ] {
        assert!(shown.contains(needle), "{needle}: {shown}");
    }
    assert!(out.status.success(), "the run goes on after each refusal");

    let other = rig.run_in("beta", &rig.project, "/resume good\n")?;
    assert!(
        text(&other.stderr).contains("was saved with alpha but beta is loaded"),
        "{}",
        text(&other.stderr)
    );
    let ok = rig.run("/resume good\n")?;
    assert!(
        text(&ok.stdout).contains("resumed `good`"),
        "an unrelated bad file does not stop a good one"
    );
    Ok(())
}

#[test]
fn a_changed_tokenizer_or_a_missing_project_stops_a_resume() -> io::Result<()> {
    let rig = Rig::new("changed", &["alpha"])?;
    rig.run("hello\n/mode code\nwhere is main?\n/save s\n")?;
    let tokenizer = rig.models.join("alpha/tokenizer.json");
    let text_before = fs::read_to_string(&tokenizer)?;
    fs::write(&tokenizer, format!("{text_before} "))?;
    let out = rig.run("/resume s\n")?;
    assert!(
        text(&out.stderr).contains("configuration or tokenizer has since changed"),
        "{}",
        text(&out.stderr)
    );
    fs::write(&tokenizer, text_before)?;

    let elsewhere = rig.sandbox.0.join("elsewhere");
    fs::create_dir_all(&elsewhere)?;
    fs::remove_dir_all(&rig.project)?;
    let gone = rig.run_in("alpha", &elsewhere, "/resume s\n/status\n")?;
    let shown = format!("{}{}", text(&gone.stdout), text(&gone.stderr));
    assert!(shown.contains("is no longer there"), "{shown}");
    assert!(
        shown.contains("mode  chat"),
        "the refused resume changed nothing: {shown}"
    );
    Ok(())
}

#[test]
fn a_loose_or_linked_session_folder_is_refused_and_nothing_follows_a_link() -> io::Result<()> {
    let rig = Rig::new("loose", &["alpha"])?;
    let sessions = rig.home().join("sessions");
    fs::create_dir_all(&sessions)?;
    fs::set_permissions(&sessions, fs::Permissions::from_mode(0o755))?;
    let loose = rig.run("hello\n/save a\n")?;
    assert!(
        text(&loose.stderr).contains("other users can enter it"),
        "{}",
        text(&loose.stderr)
    );
    assert!(!rig.file("a").exists());
    fs::remove_dir(&sessions)?;

    let target = rig.sandbox.0.join("target");
    fs::create_dir_all(&target)?;
    fs::set_permissions(&target, fs::Permissions::from_mode(0o700))?;
    symlink(&target, &sessions)?;
    let linked = rig.run("hello\n/save a\n/sessions\n")?;
    assert!(
        text(&linked.stderr).contains("a link or a file"),
        "{}",
        text(&linked.stderr)
    );
    assert_eq!(
        fs::read_dir(&target)?.count(),
        0,
        "nothing was written through the link"
    );
    Ok(())
}

#[test]
fn two_runs_cannot_both_save_over_one_session_and_a_stranger_cannot_replace_it() -> io::Result<()> {
    use std::io::{BufRead, BufReader};
    let rig = Rig::new("conflict", &["alpha"])?;
    rig.run("hello\n/save mine\n")?;
    let mut slow = rig.command("alpha", &rig.project, "6").spawn()?;
    let mut input = slow
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("no stdin"))?;
    input.write_all(b"/resume mine\n")?;
    let mut shown = BufReader::new(
        slow.stdout
            .take()
            .ok_or_else(|| io::Error::other("no stdout"))?,
    );
    let mut line = String::new();
    while !line.contains("resumed `mine`") {
        line.clear();
        if shown.read_line(&mut line)? == 0 {
            return Err(io::Error::other("the first run never resumed"));
        }
    }
    let quick = rig.run("/resume mine\nsecond\n/save mine\n")?;
    assert!(
        text(&quick.stdout).contains("updated `mine`"),
        "{}",
        text(&quick.stdout)
    );
    input.write_all(b"other\n/save mine\n")?;
    drop(input);
    let rest = slow.wait_with_output()?;
    let stderr = text(&rest.stderr);
    assert!(
        stderr.contains("was saved by another run since this conversation was opened"),
        "{stderr}"
    );
    let kept = rig.saved("mine")?.to_string();
    assert!(
        kept.contains("second") && !kept.contains("other"),
        "the first save to land was kept whole: {kept}"
    );
    assert_eq!(rig.saved("mine")?["revision"], 2);

    let stranger = rig.run("hi again\n/save mine\n")?;
    assert!(
        text(&stranger.stderr).contains("did not come from it"),
        "{}",
        text(&stranger.stderr)
    );
    assert!(!rig.saved("mine")?.to_string().contains("hi again"));
    Ok(())
}
