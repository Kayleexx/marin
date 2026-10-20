//! `/checkpoint` and `/resume` through the real `marin` on the tiny Qwen3.5 model on the GPU: a checkpoint speeds up a
//! resume without changing a word of it, is made only when asked and only for the session as saved, and anything
//! wrong with it is a note while the history is read again. Needs a GPU.

#[path = "common/qwen35.rs"]
mod qwen35;

use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

use qwen35::{Res, Scratch, have_gpu, packed_tiny};

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

struct Rig {
    sandbox: Scratch,
    packed: PathBuf,
}

impl Rig {
    fn new(name: &str) -> Res<Option<Self>> {
        if !have_gpu()? {
            return Ok(None);
        }
        let sandbox = Scratch::new(name)?;
        let packed = packed_tiny(&sandbox.0)?;
        Ok(Some(Self { sandbox, packed }))
    }

    fn file(&self, name: &str) -> PathBuf {
        self.sandbox.0.join("home/sessions").join(name)
    }

    fn run_on(&self, backend: &str, script: &str) -> Res<Output> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", self.sandbox.0.join("home"))
            .env("NO_COLOR", "1")
            .args([
                "--backend",
                backend,
                "--quantize",
                "int8",
                "--quant-group",
                "32",
            ])
            .args([
                "--ctx",
                "1024",
                "--temp",
                "1",
                "--seed",
                "3",
                "--max-tokens",
                "8",
                "--model",
            ])
            .arg(&self.packed)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            input.write_all(script.as_bytes())?;
        }
        Ok(child.wait_with_output()?)
    }

    fn run(&self, script: &str) -> Res<Output> {
        self.run_on("cuda", script)
    }

    /// `/save`s a long conversation as `s` and `/checkpoint`s it; returns what the checkpoint said.
    fn make(&self) -> Res<String> {
        let long = "the quick brown fox jumps over the lazy dog ".repeat(12);
        let out = self.run(&format!("{long}\n{long} again\n/save s\n/checkpoint s\n"))?;
        let shown = format!("{}{}", text(&out.stdout), text(&out.stderr));
        assert!(shown.contains("kept the model's state for `s`"), "{shown}");
        Ok(shown)
    }

    fn plain_copy(&self) -> Res {
        let json =
            fs::read_to_string(self.file("s.json"))?.replace("\"name\":\"s\"", "\"name\":\"p\"");
        Ok(fs::write(self.file("p.json"), json)?)
    }
}

const FOLLOW_UP: &str = "what did the fox do\n";

fn last_line(out: &Output) -> String {
    text(&out.stdout).lines().last().unwrap_or("").to_owned()
}

#[test]
fn a_checkpoint_makes_the_resume_fast_and_changes_nothing_else() -> Res {
    let Some(rig) = Rig::new("ckpt-use")? else {
        return Ok(());
    };
    let made = rig.make()?;
    assert!(rig.file("s.state").is_file());
    assert_eq!(
        fs::metadata(rig.file("s.state"))?.permissions().mode() & 0o777,
        0o600
    );
    assert!(made.contains("only you can read it"), "{made}");
    rig.plain_copy()?;

    let plain = rig.run(&format!("/resume p\n{FOLLOW_UP}"))?;
    let kept = rig.run(&format!("/resume s\n{FOLLOW_UP}"))?;
    let (plain_shown, kept_shown) = (text(&plain.stdout), text(&kept.stdout));
    assert!(!plain_shown.contains("model's state"), "{plain_shown}");
    assert!(
        kept_shown.contains("loaded the model's state for the first"),
        "{kept_shown}"
    );
    assert_eq!(
        last_line(&plain),
        last_line(&kept),
        "the same answer either way"
    );
    assert!(
        rig.run("/sessions\n")
            .is_ok_and(|o| text(&o.stdout).contains("checkpoint"))
    );
    Ok(())
}

#[test]
fn a_damaged_or_outdated_checkpoint_is_a_note_and_the_answer_is_unchanged() -> Res {
    let Some(rig) = Rig::new("ckpt-bad")? else {
        return Ok(());
    };
    rig.make()?;
    rig.plain_copy()?;
    let plain = rig.run(&format!("/resume p\n{FOLLOW_UP}"))?;
    let original = fs::read(rig.file("s.state"))?;

    let mut damaged = original.clone();
    let middle = damaged.len() / 2;
    if let Some(byte) = damaged.get_mut(middle) {
        *byte ^= 1;
    }
    fs::write(rig.file("s.state"), &damaged)?;
    let out = rig.run(&format!("/resume s\n{FOLLOW_UP}"))?;
    assert!(
        text(&out.stdout).contains("checkpoint ignored"),
        "{}",
        text(&out.stdout)
    );
    assert_eq!(last_line(&out), last_line(&plain));

    fs::write(rig.file("s.state"), &original[..original.len() / 2])?;
    let cut = rig.run(&format!("/resume s\n{FOLLOW_UP}"))?;
    assert!(
        text(&cut.stdout).contains("checkpoint ignored"),
        "{}",
        text(&cut.stdout)
    );
    assert_eq!(last_line(&cut), last_line(&plain));

    fs::write(rig.file("s.state"), &original)?;
    let resaved = rig.run("/resume s\nanother thing to say\n/save s\n")?;
    assert!(
        text(&resaved.stdout).contains("updated `s`"),
        "{}",
        text(&resaved.stdout)
    );
    assert!(
        !rig.file("s.state").exists(),
        "saving the session again removes the checkpoint it outdated"
    );
    fs::write(rig.file("s.state"), &original)?;
    let old = rig.run(&format!("/resume s\n{FOLLOW_UP}"))?;
    assert!(
        text(&old.stdout).contains("another version of the session"),
        "{}",
        text(&old.stdout)
    );
    Ok(())
}

#[test]
fn a_checkpoint_is_made_only_on_request_and_only_for_what_was_saved() -> Res {
    let Some(rig) = Rig::new("ckpt-ask")? else {
        return Ok(());
    };
    let long = "the quick brown fox jumps over the lazy dog ".repeat(12);
    let out = rig.run(&format!("{long}\n/save s\n"))?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        !rig.file("s.state").exists(),
        "/save never keeps the model's state"
    );

    let refusals = rig.run(&format!(
        "/checkpoint\n/checkpoint nothere\n{long} more\n/checkpoint s\n/save t\nhi\n/checkpoint t\n"
    ))?;
    let shown = format!("{}{}", text(&refusals.stdout), text(&refusals.stderr));
    for needle in [
        "name the session, or /save one first",
        "no saved session called `nothere`",
        "is not what `s` holds",
    ] {
        assert!(shown.contains(needle), "{needle}: {shown}");
    }
    assert!(!rig.file("s.state").exists() && !rig.file("t.state").exists());

    let short = rig.run("hi\n/save u\n/checkpoint u\n")?;
    let shown = format!("{}{}", text(&short.stdout), text(&short.stderr));
    assert!(shown.contains("under 256 tokens"), "{shown}");

    let Some(rig) = Rig::new("ckpt-ask2")? else {
        return Ok(());
    };
    rig.make()?;
    let cpu = rig.run_on("cpu", "/resume s\n/checkpoint s\n")?;
    let shown = format!("{}{}", text(&cpu.stdout), text(&cpu.stderr));
    assert!(
        shown.contains("checkpoints need Qwen3.5 on the GPU"),
        "{shown}"
    );
    assert!(
        shown.contains("checkpoint ignored"),
        "a checkpoint it cannot use is ignored: {shown}"
    );
    Ok(())
}
