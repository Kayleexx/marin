use std::io::Write;

use chat::{Conversation, FitError, fit};

use crate::CliError;
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

    /// Streams the answer to `text` into `out`. A reply that started is remembered, even if cut short;
    /// a question that got no answer is forgotten.
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
            Ok(reply) if !reply.text.trim().is_empty() => {
                self.conversation.push_assistant(&reply.text)
            }
            _ => self.conversation.pop(),
        }
        result
    }

    /// The tokens of the prompt that asking `text` would send, without asking.
    pub fn prompt_tokens(&mut self, session: &Session, text: &str) -> Result<Vec<u32>, CliError> {
        self.conversation.push_user(text);
        let ids = self.render(session);
        self.conversation.pop();
        ids
    }

    fn render(&mut self, session: &Session) -> Result<Vec<u32>, CliError> {
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
        Ok(tokenizer.encode(&prompt)?)
    }

    fn answer(
        &mut self,
        session: &mut Session,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Reply, CliError> {
        let ids = self.render(session)?;
        session.complete_tokens(ids, interrupt, out)
    }
}
