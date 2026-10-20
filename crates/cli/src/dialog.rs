//! The conversation `marin` holds: chat or read-only code questions, each mode with its own history, on one loaded
//! model. The terminal UI and piped input both go through it.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::CliError;
use crate::chat::Chat;
use crate::code::{CodeChat, Ui};
use crate::config::Home;
use crate::perf;
use crate::session::Session;
use crate::signal::Interrupt;
use crate::style::Style;
use crate::telemetry;
use crate::units;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Talk {
    Chat,
    Code,
}

impl Talk {
    pub fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Code => "code",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim() {
            "chat" => Some(Self::Chat),
            "code" => Some(Self::Code),
            _ => None,
        }
    }
}

mod persist;
mod state;

pub struct Dialog {
    pub chat: Chat,
    talk: Talk,
    root: PathBuf,
    log: Option<PathBuf>,
    code: Option<CodeChat>,
    /// The saved session this conversation came from or was last saved as, and the revision it was then at; `/save`
    /// updates it, but only if no one else has since.
    origin: Option<Origin>,
}

/// A saved session a conversation came from. After `/clear` it no longer holds that session's conversation, so a
/// plain `/save` does not default to replacing it, though saving to its name on purpose still can.
struct Origin {
    name: String,
    revision: u64,
    same: bool,
}

impl Dialog {
    /// Starts in `talk`. `root` is the project code mode reads; `log` collects its requests and tool calls.
    pub fn new(
        session: &Session,
        talk: Talk,
        root: PathBuf,
        log: Option<PathBuf>,
    ) -> Result<Self, CliError> {
        let mut dialog = Self {
            chat: Chat::new(session),
            talk: Talk::Chat,
            root,
            log,
            code: None,
            origin: None,
        };
        dialog.switch(talk, session)?;
        Ok(dialog)
    }

    pub fn talk(&self) -> Talk {
        self.talk
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Changes mode and says what became of the conversations. Each mode keeps its own until `clear` or a new model.
    pub fn switch(&mut self, talk: Talk, session: &Session) -> Result<String, CliError> {
        if talk == Talk::Code && self.code.is_none() {
            self.code = Some(CodeChat::new(&self.root, session, self.log.as_deref())?);
        }
        self.talk = talk;
        Ok(match talk {
            Talk::Code => format!(
                "code mode: reading {} (read-only). Chat history is kept; /clear forgets code history",
                self.root.display()
            ),
            Talk::Chat => {
                "chat mode. Code history is kept; /clear forgets chat history".to_string()
            }
        })
    }

    /// Forgets the conversation of the current mode.
    pub fn clear(&mut self) {
        if let Some(origin) = self.origin.as_mut() {
            origin.same = false;
        }
        match (self.talk, self.code.as_mut()) {
            (Talk::Code, Some(code)) => code.clear(),
            _ => self.chat.clear(),
        }
    }

    /// A different model has been loaded: both conversations restart, since their text was written for another. If
    /// code mode cannot start on the new model (its window is too small) the chat carries on without it.
    pub fn reset(&mut self, session: &Session) -> Result<(), CliError> {
        self.chat = Chat::new(session);
        self.origin = None;
        if self.code.take().is_some() {
            match CodeChat::new(&self.root, session, self.log.as_deref()) {
                Ok(code) => self.code = Some(code),
                Err(e) => {
                    self.talk = Talk::Chat;
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }

    /// Answers a code question, showing the tool use as it happens; true if the user cancelled.
    pub fn ask_code(
        &mut self,
        session: &mut Session,
        interrupt: &Interrupt,
        ui: Ui,
        question: &str,
    ) -> Result<bool, CliError> {
        let Some(code) = self.code.as_mut() else {
            return Err(CliError::BadOption("code mode is not active".into()));
        };
        Ok(code.ask(session, interrupt, ui, question)?)
    }

    /// Model, hardware and window in a few lines.
    pub fn status(&self, session: &Session) -> Vec<String> {
        let weights = session
            .loaded
            .model
            .quantization()
            .map_or_else(|| model::PRECISION.to_string(), |q| q.label());
        let mut lines = vec![
            format!(
                "{} · {} · {} · context {} of {} used",
                session.loaded.name,
                session.loaded.model.backend_name(),
                weights,
                units::count(session.kv_used as u64),
                units::count(session.ctx as u64)
            ),
            match self.talk {
                Talk::Chat => "mode  chat".to_string(),
                Talk::Code => format!("mode  code · {} · read-only", self.root.display()),
            },
        ];
        let s = session.settings;
        lines.push(format!(
            "sampling  temp {} · top-k {} · top-p {} · max {} tokens",
            s.temperature, s.top_k, s.top_p, s.max_tokens
        ));
        lines.extend(telemetry::resource_lines(session).into_iter().take(2));
        lines
    }
}

/// The commands every input path understands, with whether they need a model loaded in this process.
pub const COMMAND_HELP: [(&str, &str, bool); 11] = [
    (
        "/mode chat|code",
        "switch between chatting and asking about the project",
        true,
    ),
    (
        "/clear",
        "forget the conversation of the current mode",
        false,
    ),
    ("/status", "model, hardware, window and mode", true),
    (
        "/bench [N [M]] [runs=R]",
        "time the loaded model: load, prefill, decode, first token, memory",
        true,
    ),
    (
        "/verify [FILE]",
        "check the loaded model against a float32 reference",
        true,
    ),
    (
        "/save [name]",
        "keep both conversations to resume later (private to you)",
        true,
    ),
    (
        "/resume [name]",
        "continue a saved conversation; the newest by default",
        true,
    ),
    (
        "/checkpoint [name]",
        "also keep the model's state of a saved session, so /resume is fast (large file, Qwen3.5 GPU only)",
        true,
    ),
    ("/sessions", "list the saved conversations", true),
    ("/model", "which model is loaded", true),
    ("/exit", "leave (also /quit)", false),
];

/// A line that is one of the commands (`/help` too), as its name and arguments; anything else is a question.
pub fn slash(line: &str) -> Option<(&str, &str)> {
    let body = line.trim().strip_prefix('/')?;
    let (name, rest) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
    let known = name == "help"
        || name == "quit"
        || COMMAND_HELP
            .iter()
            .any(|(command, _, _)| command.trim_start_matches('/').split(' ').next() == Some(name));
    known.then_some((name, rest.trim()))
}

pub(crate) fn here() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_default()
}

/// Prints what `/bench` or `/verify` found; a cancelled run says so and nothing else changed.
pub(crate) fn show_perf(done: Result<perf::Output, CliError>, style: Style) {
    match done {
        Ok(out) => {
            for line in &out.lines {
                if out.failed && (line.starts_with("result: FAIL") || line.contains(" FAIL ")) {
                    println!("{}", style.red(line));
                } else {
                    println!("{line}");
                }
            }
        }
        Err(CliError::Cancelled) => println!("{}", style.dim("cancelled; nothing was changed")),
        Err(e) => crate::report(&e, &style),
    }
}

fn show(done: Result<Vec<String>, CliError>, style: Style) {
    match done {
        Ok(lines) => lines.iter().for_each(|l| println!("{}", style.dim(l))),
        Err(e) => crate::report(&e, &style),
    }
}

/// Handles `line` if it is one of the commands. `Some(true)` asks to leave.
fn command(
    dialog: &mut Dialog,
    (session, home): (&mut Session, Option<&Home>),
    interrupt: &Interrupt,
    line: &str,
    style: Style,
) -> Option<bool> {
    let (name, rest) = slash(line)?;
    match name {
        "exit" | "quit" => return Some(true),
        "clear" => {
            dialog.clear();
            println!(
                "{}",
                style.dim(&format!(
                    "started a new {} conversation",
                    dialog.talk().name()
                ))
            );
        }
        "help" => {
            for (command, what, _) in COMMAND_HELP {
                println!("  {command:<18}{what}");
            }
        }
        "model" => println!("{}", session.loaded.name),
        "bench" => show_perf(perf::bench(session, interrupt, rest), style),
        "verify" => show_perf(perf::verify(session, interrupt, rest, &here()), style),
        "save" => show(dialog.save(session, home, Some(rest)), style),
        "resume" => show(dialog.resume(session, home, Some(rest)), style),
        "checkpoint" => show(dialog.checkpoint(session, home, Some(rest)), style),
        "sessions" => show(dialog.sessions(session, home), style),
        "status" => {
            for line in dialog.status(session) {
                println!("{line}");
            }
        }
        _ => match Talk::parse(rest) {
            Some(talk) => match dialog.switch(talk, session) {
                Ok(note) => println!("{}", style.dim(&note)),
                Err(e) => crate::report(&e, &style),
            },
            None => println!(
                "{} is the {} mode; use /mode chat or /mode code",
                dialog.talk().name(),
                dialog.talk().name()
            ),
        },
    }
    Some(false)
}

/// Piped input: one question or command per line, answered in the current mode. A failed question is reported and
/// the next line still runs, as it always has.
pub fn run_piped(
    session: &mut Session,
    dialog: &mut Dialog,
    interrupt: &Interrupt,
    (style, home): (Style, Option<&Home>),
    lines: impl Iterator<Item = io::Result<String>>,
) -> Result<(), CliError> {
    for line in lines {
        let line = line?;
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        match command(dialog, (session, home), interrupt, line.trim(), style) {
            Some(true) => break,
            Some(false) => continue,
            None => {}
        }
        let answered = match dialog.talk() {
            Talk::Chat => dialog
                .chat
                .ask(session, line, interrupt, &mut io::stdout().lock())
                .map(drop),
            Talk::Code => dialog
                .ask_code(session, interrupt, Ui::piped(style), line)
                .map(drop),
        };
        if let Err(e) = answered {
            if e.is_broken_pipe() {
                return Err(e);
            }
            crate::report(&e, &style);
        }
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse_by_name_only() {
        assert_eq!(Talk::parse("code"), Some(Talk::Code));
        assert_eq!(Talk::parse(" chat "), Some(Talk::Chat));
        assert_eq!(Talk::parse("coding"), None);
        assert_eq!(Talk::parse(""), None);
    }
}
