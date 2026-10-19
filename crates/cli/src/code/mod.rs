//! `marin code`: a read-only coding assistant that talks to a running `marin serve`.

mod agent;
mod call;
mod client;
mod convo;
mod guide;
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

use std::fs::OpenOptions;
use std::io::{self, BufRead};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;

use thiserror::Error;

use agent::{Agent, AgentError, Log, Outcome};
use client::{Client, ClientError};
use convo::Conversation;
use prompt::Input;
use repo::Repo;
use ui::Ui;

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
    #[error("cannot open {path} as a project: {reason}")]
    Project { path: String, reason: String },
    #[error("cannot write the log {path}: {reason}")]
    Log { path: String, reason: String },
}

impl CodeError {
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::Server(ClientError::Unreachable(_)) => Some(
                "start one in another terminal: marin --model ~/models/Qwen3.5-4B-Q8 --quantize int8 --ctx 8192 serve",
            ),
            Self::Server(ClientError::Rejected { status: 503, .. }) => {
                Some("the server is busy with other requests; try again in a moment")
            }
            Self::Project { .. } => Some("pass the folder you want to ask about"),
            _ => None,
        }
    }
}

fn explain(error: &AgentError) -> (String, Option<&'static str>) {
    let hint = match error {
        AgentError::Client(ClientError::Unreachable(_) | ClientError::Broken(_)) => {
            Some("the server went away; start it again and repeat the question")
        }
        AgentError::Client(ClientError::Rejected { status: 503, .. }) => {
            Some("the server is busy with other requests; try again in a moment")
        }
        AgentError::TooLarge(_) => Some("restart the server with a larger --ctx, or use /clear"),
        _ => None,
    };
    (error.to_string(), hint)
}

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
    let repo = Repo::open(&root, info.context).map_err(|e| CodeError::Project {
        path: root.display().to_string(),
        reason: e.to_string(),
    })?;
    let log = match &options.log {
        Some(path) => Log::to(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| CodeError::Log {
                    path: path.display().to_string(),
                    reason: e.to_string(),
                })?,
        ),
        None => Log::default(),
    };
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let ui = Ui::new(options.no_color);
    ui.banner(&info.model, info.context, repo.root());
    let mut agent = Agent {
        client: &client,
        info: &info,
        repo: &repo,
        ui,
        convo: Conversation::new(
            guide::system_prompt(&repo.name(), &repo.overview()),
            info.context,
        ),
        cancel: interrupt.cancel_flag(),
        log,
        seen: Vec::new(),
    };
    let mut editor = Editor::default();
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
        if let Some(command) = text.strip_prefix('/') {
            match command {
                "quit" | "exit" => break,
                "clear" => {
                    agent.convo.clear();
                    ui.notice("started a new conversation");
                }
                "help" => ui.help(),
                other => ui.warn(&format!("unknown command /{other}; try /help")),
            }
            continue;
        }
        editor.remember(&text);
        ui.you(&text);
        println!();
        interrupt.generating();
        match agent.ask(&text) {
            Ok((Outcome::Cancelled, _)) => {
                interrupted = true;
                ui.notice("cancelled; that question was not kept");
            }
            Ok(_) => {}
            Err(e) => {
                failed = true;
                let (message, hint) = explain(&e);
                ui.error(&message, hint);
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
