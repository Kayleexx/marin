//! Chat and code mode on a model whose window is long enough for both: each keeps its own history, `/clear` forgets
//! only the mode it is typed in, and a new model starts both over. The answers are noise; the sizes of the prompts
//! the model was given show what each conversation held.

mod common;
#[path = "common/long.rs"]
mod long;

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use common::{Sandbox, fixtures, text};
use long::long_tiny;

struct Setup {
    sandbox: Sandbox,
    models: PathBuf,
}

impl Setup {
    fn new(name: &str) -> io::Result<Self> {
        let sandbox = Sandbox::new(name)?;
        let models = sandbox.0.join("models");
        long_tiny(&fixtures().join("tiny-chat"), &models.join("alpha"))?;
        Ok(Self { sandbox, models })
    }

    /// The `context N/M` figure after each code answer, in order, for the lines typed.
    fn contexts(&self, script: &str) -> io::Result<Vec<usize>> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
            .env("MARIN_HOME", self.sandbox.0.join("home"))
            .env("MARIN_MODELS", &self.models)
            .env("NO_COLOR", "1")
            .args(["--backend", "cpu", "--temp", "0", "--max-tokens", "4"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            input.write_all(script.as_bytes())?;
        }
        let out = child.wait_with_output()?;
        assert!(out.status.success(), "{}", text(&out.stderr));
        let shown = text(&out.stdout);
        Ok(shown
            .split("context ")
            .skip(1)
            .filter_map(|rest| {
                let digits: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == ',')
                    .filter(char::is_ascii_digit)
                    .collect();
                rest.get(digits.len()..).filter(|r| r.contains('/'))?;
                digits.parse().ok()
            })
            .collect())
    }
}

#[test]
fn each_mode_keeps_its_own_history_and_clear_forgets_only_the_current_one() -> io::Result<()> {
    let setup = Setup::new("isolation")?;
    let alone = setup.contexts("/mode code\nwhere is main?\n")?;
    let mixed = setup.contexts(
        "tell me about cats\n/mode code\nwhere is main?\n/clear\nwhere is main?\n/mode chat\n/clear\n",
    )?;
    let first = *alone.first().ok_or_else(|| io::Error::other("no answer"))?;
    assert_eq!(
        mixed.first(),
        Some(&first),
        "a chat question does not reach the code prompt: {alone:?} {mixed:?}"
    );
    assert_eq!(
        mixed.get(1),
        Some(&first),
        "/clear in code mode left the code prompt as new: {mixed:?}"
    );
    Ok(())
}

#[test]
fn code_history_survives_a_trip_to_chat_and_back() -> io::Result<()> {
    let setup = Setup::new("roundtrip")?;
    let seen = setup
        .contexts("/mode code\nwhere is main?\n/mode chat\nhello\n/mode code\nwhere is main?\n")?;
    let (first, second) = (seen.first().copied(), seen.get(1).copied());
    assert!(
        first.zip(second).is_some_and(|(a, b)| b > a),
        "the second code question still sees the first: {seen:?}"
    );
    Ok(())
}
