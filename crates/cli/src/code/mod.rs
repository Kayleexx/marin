//! Code mode: a read-only coding assistant. In the chat it answers with the loaded model (`CodeChat`); `run` is the
//! stand-alone client of a running `marin serve`.

mod agent;
mod brain;
mod call;
mod client;
mod convo;
mod guide;
mod local;
mod prompt;
mod repo;
mod tools;
mod ui;

#[cfg(test)]
mod agent_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tools_tests;

use std::io::{self, BufRead};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;

use thiserror::Error;

pub use local::CodeChat;
pub use ui::Ui;

use agent::{Agent, AgentError, Log, Outcome};
use client::{Client, ClientError};
use convo::Conversation;
use prompt::Input;
use repo::Repo;

use crate::config::Home;
use crate::dialog;
use crate::error::CliError;
use crate::signal::{INTERRUPTED_EXIT_CODE, Interrupt};
use crate::term::editor::Editor;

pub struct Options {
    pub path: Option<PathBuf>,
    pub port: u16,
    pub log: Option<PathBuf>,
    pub no_color: bool,
}

#[derive(Debug, Error)]
pub enum CodeError {
    #[error(transparent)]
    Server(#[from] ClientError),
    #[error(transparent)]
    Agent(#[from] AgentError),
    #[error("cannot open {path} as a project: {reason}")]
    Project { path: String, reason: String },
    #[error("cannot write the log {path}: {reason}")]
    Log { path: String, reason: String },
    #[error(
        "code mode needs a context window of at least {MIN_CONTEXT} tokens (the tool instructions alone take about 650) and this one has {0}"
    )]
    Window(usize),
}

impl CodeError {
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::Server(ClientError::Unreachable(_)) => Some(
                "start one in another terminal: marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8 --ctx 8192 serve, or run `marin code` without --port to use a model of its own",
            ),
            Self::Server(ClientError::Rejected { status: 503, .. })
            | Self::Agent(AgentError::Client(ClientError::Rejected { status: 503, .. })) => {
                Some("the server is busy with other requests; try again in a moment")
            }
            Self::Agent(AgentError::Client(
                ClientError::Unreachable(_) | ClientError::Broken(_),
            )) => Some("the server went away; start it again and repeat the question"),
            Self::Server(ClientError::Protocol(_) | ClientError::Broken(_)) => {
                Some("something else may be listening on that port; check --port")
            }
            Self::Agent(AgentError::TooLarge(_)) => {
                Some("restart with a larger --ctx, or use /clear")
            }
            Self::Project { .. } => Some("pass the folder you want to ask about"),
            Self::Window(_) => {
                Some("start with a larger window, for example `marin code --ctx 8192`")
            }
            _ => None,
        }
    }
}

/// The model and context window of a Marin server answering on `port`, if one does.
pub fn server_at(port: u16) -> Option<(String, usize)> {
    Client::new(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
        .probe()
        .map(|info| (info.model, info.context))
}

/// The smallest context window code mode works in.
pub const MIN_CONTEXT: usize = 2048;

enum Next {
    Ask(String),
    Quit,
}

fn next(
    ui: &Ui,
    editor: &mut Editor,
    stdin: &mut impl Iterator<Item = io::Result<String>>,
) -> io::Result<Next> {
    if ui.interactive() {
        return Ok(match prompt::read(editor, &ui.style())? {
            Input::Line(text) => Next::Ask(text),
            Input::Quit => Next::Quit,
        });
    }
    Ok(stdin.next().transpose()?.map_or(Next::Quit, Next::Ask))
}

pub fn run(options: Options) -> Result<ExitCode, CliError> {
    let root = options.path.unwrap_or_else(|| PathBuf::from("."));
    let client = Client::new(SocketAddr::from((Ipv4Addr::LOCALHOST, options.port)));
    let info = client.info().map_err(CodeError::from)?;
    if info.context < MIN_CONTEXT {
        return Err(CodeError::Window(info.context).into());
    }
    let repo = Repo::open(&root, info.context).map_err(|e| CodeError::Project {
        path: root.display().to_string(),
        reason: e.to_string(),
    })?;
    let log = Log::open(options.log.as_deref())?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let ui = Ui::new(options.no_color);
    ui.banner(&info.model, info.context, repo.root());
    let mut client = client;
    let mut convo = Conversation::new(
        guide::system_prompt(&repo.name(), &repo.overview()),
        info.context,
    );
    let (mut seen, mut log) = (Vec::new(), log);
    let home = Home::detect();
    let mut editor =
        Editor::with_history(home.as_ref().map(Home::load_history).unwrap_or_default());
    let mut lines = io::stdin().lock().lines();
    let mut interrupted = false;
    let mut failed = false;
    loop {
        interrupt.busy();
        let text = match next(&ui, &mut editor, &mut lines)? {
            Next::Quit => break,
            Next::Ask(text) => text.trim().to_owned(),
        };
        if text.is_empty() {
            continue;
        }
        if let Some((name, _)) = dialog::slash(&text) {
            match name {
                "quit" | "exit" => break,
                "clear" => {
                    convo.clear();
                    ui.notice("started a new conversation");
                }
                "help" => ui.help(),
                other => ui.warn(&format!(
                    "/{other} needs a model loaded by marin itself: run `marin code` without --port"
                )),
            }
            continue;
        }
        editor.remember(&text);
        if let Some(home) = &home {
            let _ = home.save_history(editor.history());
        }
        ui.you(&text);
        println!();
        interrupt.generating();
        let mut agent = Agent {
            brain: &mut client,
            info: &info,
            repo: &repo,
            ui,
            convo: &mut convo,
            cancel: interrupt.cancel_flag(),
            log: &mut log,
            seen: &mut seen,
        };
        match agent.ask(&text) {
            Ok((Outcome::Cancelled, _)) => {
                interrupted = true;
                ui.notice("cancelled; that question was not kept");
            }
            Ok(_) => {}
            Err(e) => {
                failed = true;
                let e = CodeError::from(e);
                ui.error(&e.to_string(), e.hint());
            }
        }
    }
    Ok(if ui.interactive() {
        ExitCode::SUCCESS
    } else if interrupted {
        ExitCode::from(u8::try_from(INTERRUPTED_EXIT_CODE).unwrap_or(130))
    } else if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}
