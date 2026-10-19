use std::fs::File;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde_json::{Value, json};
use thiserror::Error;

use super::call::{self, CLOSE, INVENTED, OPEN};
use super::client::{Ask, Client, ClientError, Reply, ServerInfo};
use super::convo::{Conversation, Done};
use super::repo::Repo;
use super::ui::{Stats, Ui};

/// Rounds of tool use one question may take before the model is asked to answer with what it has.
const MAX_ROUNDS: usize = 8;
const MAX_CALLS: usize = 4;
const MAX_CALL_CHARS: usize = 1000;
const BAD_ROUNDS: usize = 2;
const LIMIT_NOTE: &str = "No more tool calls will be run. Answer now with what you have found so far, and say what you could not check.";
const EXAMPLE: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>";

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error("this question needs about {0} tokens, more than the context window holds")]
    TooLarge(usize),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub enum Outcome {
    Answered,
    Cancelled,
}

/// Writes one JSON line per model request and tool call, for studying how a run went.
#[derive(Default)]
pub struct Log(Option<File>);

impl Log {
    pub fn to(file: File) -> Self {
        Self(Some(file))
    }

    fn record(&mut self, value: &Value) {
        if let Some(file) = &mut self.0 {
            let _ = writeln!(file, "{value}");
        }
    }
}

pub struct Agent<'a> {
    pub client: &'a Client,
    pub info: &'a ServerInfo,
    pub repo: &'a Repo,
    pub ui: Ui,
    pub convo: Conversation,
    pub cancel: &'a AtomicBool,
    pub log: Log,
    pub seen: Vec<call::Call>,
}

/// Passes on the model's words until it starts a tool call (or a result of its own); the markup never reaches the screen.
#[derive(Default)]
struct Cutter {
    held: String,
    stopped: bool,
    /// What the model wrote from the start of its call (or invented result) on, never shown.
    after: String,
    invented: bool,
}

impl Cutter {
    fn push(&mut self, piece: &str) -> String {
        if self.stopped {
            self.after.push_str(piece);
            return String::new();
        }
        self.held.push_str(piece);
        let marker = [OPEN, INVENTED]
            .iter()
            .filter_map(|m| self.held.find(m))
            .min();
        if let Some(at) = marker {
            self.stopped = true;
            self.invented = self
                .held
                .get(at..)
                .is_some_and(|rest| rest.starts_with(INVENTED));
            let shown = self.held.get(..at).unwrap_or("").to_owned();
            self.after = self.held.split_off(at);
            self.held.clear();
            return shown;
        }
        let keep = (1..=OPEN.len().max(INVENTED.len()))
            .rev()
            .find(|n| {
                self.held.len() >= *n
                    && self
                        .held
                        .is_char_boundary(self.held.len().saturating_sub(*n))
                    && self
                        .held
                        .get(self.held.len().saturating_sub(*n)..)
                        .is_some_and(|tail| OPEN.starts_with(tail) || INVENTED.starts_with(tail))
            })
            .unwrap_or(0);
        let cut = self.held.len().saturating_sub(keep);
        let shown = self.held.get(..cut).unwrap_or("").to_owned();
        self.held.drain(..cut);
        shown
    }

    /// True once the rest of the reply cannot be useful: a result the model made up, or a call that never ends.
    fn runaway(&self) -> bool {
        self.invented || (self.after.len() > MAX_CALL_CHARS && !self.after.contains(CLOSE))
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            String::new()
        } else {
            std::mem::take(&mut self.held)
        }
    }
}

impl Agent<'_> {
    fn max_tokens(&self) -> usize {
        (self.info.context / 4).clamp(256, 1024)
    }

    /// Answers one question, using tools as needed. Whatever happens, a failed or cancelled question leaves no trace
    /// in the conversation.
    pub fn ask(&mut self, question: &str) -> Result<(Outcome, Stats), AgentError> {
        let mark = self.convo.mark();
        let result = self.turn(question);
        if !matches!(result, Ok((Outcome::Answered, _))) {
            self.convo.truncate(mark);
        }
        result
    }

    fn turn(&mut self, question: &str) -> Result<(Outcome, Stats), AgentError> {
        self.convo.push_user(question.to_owned());
        self.seen.clear();
        let began = Instant::now();
        let mut stats = Stats {
            tools: 0,
            generated: 0,
            context_used: 0,
            elapsed: began.elapsed(),
        };
        let mut bad_rounds = 0_usize;
        let mut last = false;
        for round in 0.. {
            let Some(reply) = self.request(&mut stats)? else {
                return Ok((Outcome::Cancelled, stats));
            };
            let parsed = call::parse(&reply.text);
            if parsed.invented {
                self.ui
                    .notice("the model wrote a tool result of its own; it was ignored");
            }
            if parsed.repaired > 0 {
                self.log
                    .record(&json!({"event": "repaired", "calls": parsed.repaired}));
            }
            if reply.finish == "length" {
                self.ui.warn("the reply was cut at the token limit");
            }
            self.convo.push_assistant(parsed.kept);
            if parsed.calls.is_empty() || last {
                if parsed.calls.is_empty() && parsed.shown.trim().is_empty() {
                    self.ui.warn("the model gave an empty reply");
                }
                if !parsed.calls.is_empty() {
                    self.ui
                        .warn("the model kept asking for tools; stopped here");
                }
                break;
            }
            let (done, valid) = self.run_calls(parsed.calls, round)?;
            let Some(done) = done else {
                return Ok((Outcome::Cancelled, stats));
            };
            stats.tools = stats.tools.saturating_add(valid);
            bad_rounds = if valid == 0 {
                bad_rounds.saturating_add(1)
            } else {
                0
            };
            last = round.saturating_add(1) >= MAX_ROUNDS || bad_rounds >= BAD_ROUNDS;
            self.convo
                .push_results(done, if last { LIMIT_NOTE } else { "" });
        }
        stats.elapsed = began.elapsed();
        self.ui.footer(&stats, self.info.context);
        Ok((Outcome::Answered, stats))
    }

    /// Runs the calls of one reply. `None` when the user cancelled meanwhile. Also counts the calls that were valid.
    fn run_calls(
        &mut self,
        calls: Vec<Result<call::Call, String>>,
        round: usize,
    ) -> Result<(Option<Vec<Done>>, usize), AgentError> {
        let mut done = Vec::new();
        let mut valid = 0;
        let extra = calls.len().saturating_sub(MAX_CALLS);
        for parsed in calls.into_iter().take(MAX_CALLS) {
            let began = Instant::now();
            match parsed {
                Ok(call) if self.seen.contains(&call) => {
                    self.ui
                        .event("repeated call", "already answered above", false);
                    self.log.record(
                        &json!({"event": "repeated", "round": round, "call": format!("{call:?}")}),
                    );
                    done.push(Done {
                        label: "repeated call".into(),
                        text: "error: you already made this exact call; its result is above. Use what it showed, or try a different path or tool.".into(),
                    });
                }
                Ok(call) => {
                    self.seen.push(call.clone());
                    let out = self.repo.run(&call, self.cancel);
                    if self.cancel.load(Ordering::Relaxed) {
                        return Ok((None, valid));
                    }
                    valid = valid.saturating_add(1);
                    self.ui.event(&out.label, &out.note, out.ok);
                    self.log.record(&json!({
                        "event": "tool", "round": round, "call": format!("{call:?}"), "ok": out.ok,
                        "note": out.note, "chars": out.text.chars().count(), "ms": began.elapsed().as_millis() as u64,
                        "result": out.text,
                    }));
                    done.push(Done {
                        label: out.label,
                        text: out.text,
                    });
                }
                Err(reason) => {
                    self.ui.event("invalid tool call", &reason, false);
                    self.log
                        .record(&json!({"event": "invalid", "round": round, "reason": reason}));
                    done.push(Done {
                        label: "invalid tool call".into(),
                        text: format!("error: {reason}. A call looks like this:\n{EXAMPLE}"),
                    });
                }
            }
        }
        if extra > 0 {
            done.push(Done {
                label: "extra calls".into(),
                text: format!("error: only the first {MAX_CALLS} calls of a reply are run; {extra} were skipped"),
            });
        }
        Ok((Some(done), valid))
    }

    /// Fits the conversation into the window, asks the model and shows what it says. `None` if cancelled.
    fn request(&mut self, stats: &mut Stats) -> Result<Option<Reply>, AgentError> {
        let reserve = self.max_tokens().saturating_add(64);
        let mut share = 1.0;
        let mut renewed = false;
        loop {
            let trimmed = self
                .convo
                .make_room(reserve, share)
                .map_err(AgentError::TooLarge)?;
            if trimmed.dropped > 0 {
                self.ui
                    .notice("older questions were dropped to fit the context window");
            } else if trimmed.elided > 0 {
                self.ui
                    .notice("older tool output was shortened to fit the context window");
            }
            let session = self.convo.session.clone();
            let ask = Ask {
                model: &self.info.model,
                messages: self.convo.messages(),
                session: &session,
                max_tokens: self.max_tokens(),
            };
            let began = Instant::now();
            let mut out = self.ui.reply();
            let mut cutter = Cutter::default();
            let mut broken = None;
            let result = self.client.chat(&ask, self.cancel, |piece| {
                let text = cutter.push(piece);
                if broken.is_none()
                    && let Err(e) = out.push(&text)
                {
                    broken = Some(e);
                }
                !cutter.runaway() && broken.is_none()
            });
            let tail = cutter.finish();
            out.push(&tail)?;
            out.end()?;
            if let Some(e) = broken {
                return Err(e.into());
            }
            match result {
                Ok(reply) => {
                    self.log.record(&json!({
                        "event": "reply", "prompt_tokens": reply.usage.prompt, "completion_tokens": reply.usage.completion,
                        "ms": began.elapsed().as_millis() as u64, "finish": reply.finish, "text": reply.text,
                    }));
                    if reply.usage.prompt > 0 {
                        self.convo.calibrate(reply.usage.prompt);
                    }
                    if let Some(id) = &reply.session {
                        self.convo.session.clone_from(id);
                    }
                    stats.generated = stats.generated.saturating_add(reply.usage.completion);
                    stats.context_used = match reply.usage.prompt {
                        0 => self.convo.estimate(),
                        prompt => prompt.saturating_add(reply.usage.completion),
                    };
                    return Ok(Some(reply));
                }
                Err(ClientError::Cancelled) => return Ok(None),
                Err(ClientError::Rejected { status: 404, .. }) if !renewed => {
                    renewed = true;
                    self.convo.session = "new".into();
                }
                Err(ClientError::Rejected {
                    status: 400,
                    message,
                }) if message.contains("context holds") && share > 0.5 => {
                    share *= 0.6;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str]) -> (String, String) {
        let mut cutter = Cutter::default();
        let mut shown: String = pieces.iter().map(|p| cutter.push(p)).collect();
        shown.push_str(&cutter.finish());
        (shown, cutter.held)
    }

    #[test]
    fn words_before_a_call_show_and_the_markup_never_does() {
        let (shown, _) = run(&["Let me ", "look.\n<tool", "_call>\n<function=x>"]);
        assert_eq!(shown, "Let me look.\n");
    }

    #[test]
    fn a_lookalike_that_is_not_a_marker_is_released() {
        let (shown, _) = run(&["a < b and <to", "ols> too"]);
        assert_eq!(shown, "a < b and <tools> too");
    }

    #[test]
    fn a_call_that_never_ends_is_a_runaway() {
        let mut cutter = Cutter::default();
        cutter.push("<tool_call>\n<function=read_file>\n");
        assert!(!cutter.runaway());
        for _ in 0..200 {
            cutter.push("\n</parameter>");
        }
        assert!(cutter.runaway());
    }

    #[test]
    fn an_invented_result_is_cut_too() {
        let (shown, _) = run(&["Done. <tool_response>fake"]);
        assert_eq!(shown, "Done. ");
    }
}
