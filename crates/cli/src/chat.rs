use std::io::Write;

use chat::{Conversation, FitError, Role, fit};
use runtime::StopReason;

use crate::CliError;
use crate::code::Item;
use crate::session::{Reply, Session};
use crate::signal::Interrupt;

/// One conversation with a chat model: remembers the exchange and formats every turn
/// with the model's official chat template.
pub struct Chat {
    conversation: Conversation,
    specials: Vec<String>,
}

impl Chat {
    pub fn new(session: &Session) -> Self {
        Self {
            conversation: Conversation::default(),
            specials: session.loaded.dir.tokenizer.special_tokens(),
        }
    }

    pub fn clear(&mut self) {
        self.conversation.clear();
    }

    /// The finished exchanges, oldest first.
    pub fn items(&self) -> Vec<Item> {
        self.conversation
            .messages()
            .into_iter()
            .filter_map(|m| match m.role {
                Role::User => Some(Item::User(m.content)),
                Role::Assistant => Some(Item::Assistant(m.content)),
                Role::System => None,
            })
            .collect()
    }

    /// Continues from exchanges saved earlier, instead of the current ones.
    pub fn restore(&mut self, items: &[Item]) {
        self.conversation.clear();
        for item in items {
            match item {
                Item::User(text) => self.conversation.push_user(text),
                Item::Assistant(text) => self.conversation.push_assistant(text),
                Item::Results { .. } => {}
            }
        }
    }

    /// Streams the answer to `text` into `out`. A reply that ran to its end (or its token limit) is remembered; a
    /// question that was cancelled, failed or got no answer is forgotten, as in code mode.
    pub fn ask(
        &mut self,
        session: &mut Session,
        text: &str,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Reply, CliError> {
        if text.trim().is_empty() {
            return Err(CliError::EmptyPrompt);
        }
        self.conversation.push_user(text);
        let result = self.answer(session, interrupt, out);
        match &result {
            Ok(reply)
                if !reply.text.trim().is_empty() && reply.outcome.stop != StopReason::Cancelled =>
            {
                self.conversation.push_assistant(&reply.text)
            }
            _ => self.conversation.pop(),
        }
        result
    }

    /// The tokens a prompt after this conversation begins with: everything up to the last question, which is where
    /// the model's state stands after the last answer was asked for.
    pub fn history_ids(&self, session: &Session) -> Result<Vec<u32>, CliError> {
        let mut messages = self.conversation.messages();
        if messages.last().is_some_and(|m| m.role == Role::Assistant) {
            messages.pop();
        }
        let text = chat::render_history(&messages, &self.specials)
            .map_err(|e| CliError::BadOption(e.to_string()))?;
        Ok(session.loaded.dir.tokenizer.encode(&text)?)
    }

    /// The tokens of the prompt that asking `text` would send, without asking.
    pub fn prompt_tokens(&mut self, session: &Session, text: &str) -> Result<Vec<u32>, CliError> {
        self.conversation.push_user(text);
        let ids = self.render(session);
        self.conversation.pop();
        ids.map(|(ids, _)| ids)
    }

    /// The prompt's tokens and how many of them end the conversation so far, before the start of the reply.
    fn render(&mut self, session: &Session) -> Result<(Vec<u32>, usize), CliError> {
        let reserved = session.settings.max_tokens.min(session.ctx / 2);
        let budget = session.ctx.saturating_sub(reserved);
        let tokenizer = &session.loaded.dir.tokenizer;
        let (prompt, _) = fit(&mut self.conversation, &self.specials, budget, |t| {
            tokenizer.encode(t).map(|ids| ids.len())
        })
        .map_err(|e| match e {
            FitError::TooLong { tokens, .. } => CliError::PromptTooLong {
                tokens,
                ctx: session.ctx,
            },
            FitError::Chat(e) => CliError::BadOption(e.to_string()),
            FitError::Count(e) => CliError::Loader(e),
        })?;
        let ids = tokenizer.encode(&prompt)?;
        let history = chat::render_history(&self.conversation.messages(), &self.specials)
            .map_err(|e| CliError::BadOption(e.to_string()))?;
        let history = tokenizer.encode(&history)?;
        let shared = history.iter().zip(&ids).take_while(|(a, b)| a == b).count();
        Ok((ids, shared))
    }

    fn answer(
        &mut self,
        session: &mut Session,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Reply, CliError> {
        let (ids, upto) = self.render(session)?;
        session
            .complete_after(ids, upto, interrupt, out)?
            .ok_or(CliError::Cancelled)
    }
}
