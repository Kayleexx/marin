//! Code mode inside the terminal chat: the same tool loop, answered by the model this process has loaded.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use chat::{Message, Role};
use runtime::StopReason;
use serde_json::Value;

use super::CodeError;
use super::agent::{Agent, Log, Outcome};
use super::brain::Brain;
use super::call::Call;
use super::client::{Ask, ClientError, Reply, ServerInfo, Usage};
use super::convo::{Conversation, Item};
use super::guide;
use super::repo::Repo;
use super::ui::Ui;
use crate::CliError;
use crate::session::Session;
use crate::signal::Interrupt;

/// Reads and answers with the loaded model. Code mode decodes greedily: a sampled call is a broken call more often.
struct LocalBrain<'a> {
    session: &'a mut Session,
    interrupt: &'a Interrupt,
}

struct Sink<'a> {
    on_text: &'a mut dyn FnMut(&str) -> bool,
    cancel: &'a AtomicBool,
    stopped: bool,
}

impl Write for Sink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.stopped && !(self.on_text)(&String::from_utf8_lossy(bytes)) {
            self.stopped = true;
            self.cancel.store(true, Ordering::SeqCst);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn role(name: &str) -> Role {
    match name {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

fn messages_of(values: &[Value]) -> Vec<Message> {
    values
        .iter()
        .map(|m| {
            let text = m.get("content").and_then(Value::as_str).unwrap_or("");
            Message::new(
                role(m.get("role").and_then(Value::as_str).unwrap_or("")),
                text,
            )
        })
        .collect()
}

/// The tokens of `messages` as the start of a later prompt: every message, none of the reply to come.
fn history_tokens(session: &Session, messages: &[Message]) -> Result<Vec<u32>, String> {
    let tokenizer = &session.loaded.dir.tokenizer;
    let text =
        chat::render_history(messages, &tokenizer.special_tokens()).map_err(|e| e.to_string())?;
    tokenizer.encode(&text).map_err(|e| e.to_string())
}

impl LocalBrain<'_> {
    /// The prompt's tokens and how many of them end the conversation so far, before the start of the reply.
    fn prompt(&self, ask: &Ask<'_>) -> Result<(Vec<u32>, usize), ClientError> {
        let messages = messages_of(&ask.messages);
        let tokenizer = &self.session.loaded.dir.tokenizer;
        let local = |e: &dyn std::fmt::Display| ClientError::Local(e.to_string());
        let ids = tokenizer
            .encode(&chat::render(&messages, &tokenizer.special_tokens()).map_err(|e| local(&e))?)
            .map_err(|e| local(&e))?;
        let history = history_tokens(self.session, &messages).map_err(ClientError::Local)?;
        let shared = history.iter().zip(&ids).take_while(|(a, b)| a == b).count();
        Ok((ids, shared))
    }
}

impl Brain for LocalBrain<'_> {
    fn reply(
        &mut self,
        ask: &Ask<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Reply, ClientError> {
        let (ids, upto) = self.prompt(ask)?;
        let saved = self.session.settings;
        self.session.settings.max_tokens = ask.max_tokens;
        self.session.settings.temperature = 0.0;
        let mut sink = Sink {
            on_text,
            cancel,
            stopped: false,
        };
        let result = self
            .session
            .complete_after(ids, upto, self.interrupt, &mut sink);
        self.session.settings = saved;
        let stopped = sink.stopped;
        let Some(done) = result.map_err(|e| ClientError::Local(e.to_string()))? else {
            return Err(ClientError::Cancelled);
        };
        if done.outcome.stop == StopReason::Cancelled && !stopped {
            return Err(ClientError::Cancelled);
        }
        // The next stage of the question (a tool, another reply) can still be cancelled.
        self.interrupt.generating();
        Ok(Reply {
            text: done.text,
            usage: Usage {
                prompt: done.outcome.prompt_tokens,
                completion: done.outcome.generated,
            },
            session: None,
            finish: match (stopped, done.outcome.stop) {
                (true, _) => "client_stop",
                (_, StopReason::EndOfText) => "stop",
                _ => "length",
            }
            .into(),
        })
    }
}

/// A conversation about a project, kept between questions.
pub struct CodeChat {
    repo: Repo,
    convo: Conversation,
    seen: Vec<Call>,
    log: Log,
    info: ServerInfo,
}

impl CodeChat {
    pub fn new(root: &Path, session: &Session, log: Option<&Path>) -> Result<Self, CodeError> {
        if session.ctx < super::MIN_CONTEXT {
            return Err(CodeError::Window(session.ctx));
        }
        let repo = Repo::open(root, session.ctx).map_err(|e| CodeError::Project {
            path: root.display().to_string(),
            reason: e.to_string(),
        })?;
        let log = Log::open(log)?;
        let convo = Conversation::new(
            guide::system_prompt(&repo.name(), &repo.overview()),
            session.ctx,
        );
        let info = ServerInfo {
            model: session.loaded.name.clone(),
            context: session.ctx,
        };
        Ok(Self {
            repo,
            convo,
            seen: Vec::new(),
            log,
            info,
        })
    }

    pub fn root(&self) -> &Path {
        self.repo.root()
    }

    pub fn items(&self) -> &[Item] {
        self.convo.items()
    }

    /// The tokens a prompt after this conversation begins with: everything up to the last question or tool result.
    pub fn history_ids(&self, session: &Session) -> Result<Vec<u32>, CliError> {
        let mut messages = messages_of(&self.convo.messages());
        if messages.last().is_some_and(|m| m.role == Role::Assistant) {
            messages.pop();
        }
        history_tokens(session, &messages).map_err(CliError::BadOption)
    }

    /// Continues from questions and answers saved earlier.
    pub fn restore(&mut self, items: Vec<Item>) {
        self.convo.restore(items);
    }

    pub fn clear(&mut self) {
        self.convo.clear();
    }

    /// Answers one question with the tool loop, showing everything through `ui`; true if the user cancelled. Cancelled
    /// or failed questions leave the conversation as it was.
    pub fn ask(
        &mut self,
        session: &mut Session,
        interrupt: &Interrupt,
        ui: Ui,
        question: &str,
    ) -> Result<bool, CodeError> {
        let mut brain = LocalBrain { session, interrupt };
        interrupt.generating();
        let result = Agent {
            brain: &mut brain,
            info: &self.info,
            repo: &self.repo,
            ui,
            convo: &mut self.convo,
            cancel: interrupt.cancel_flag(),
            log: &mut self.log,
            seen: &mut self.seen,
        }
        .ask(question);
        interrupt.busy();
        Ok(matches!(result?.0, Outcome::Cancelled))
    }
}
