use crate::qwen3::{ChatError, render};
use crate::{DEFAULT_SYSTEM, Message, Role};

/// The messages of one chat, oldest first, plus the system message that frames it.
#[derive(Debug, Clone)]
pub struct Conversation {
    system: Option<String>,
    turns: Vec<Message>,
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new(Some(DEFAULT_SYSTEM.to_string()))
    }
}

impl Conversation {
    pub fn new(system: Option<String>) -> Self {
        Self {
            system,
            turns: Vec::new(),
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.turns.push(Message::new(Role::User, text));
    }

    pub fn push_assistant(&mut self, text: &str) {
        self.turns.push(Message::new(Role::Assistant, text));
    }

    /// Forgets the newest message, e.g. a question whose answer never started.
    pub fn pop(&mut self) {
        self.turns.pop();
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn messages(&self) -> Vec<Message> {
        let system = self
            .system
            .iter()
            .map(|s| Message::new(Role::System, s.clone()));
        system.chain(self.turns.iter().cloned()).collect()
    }

    /// Drops the oldest question and its answer, never the newest message.
    pub fn drop_oldest_exchange(&mut self) -> bool {
        if self.turns.len() < 2 {
            return false;
        }
        self.turns.remove(0);
        if self
            .turns
            .first()
            .is_some_and(|m| m.role == Role::Assistant)
            && self.turns.len() > 1
        {
            self.turns.remove(0);
        }
        true
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum FitError<E> {
    Chat(ChatError),
    /// Even the newest message alone does not fit.
    TooLong {
        tokens: usize,
        budget: usize,
    },
    Count(E),
}

/// Renders `conversation`, forgetting the oldest exchanges until it fits in `budget` tokens.
/// Returns the prompt text and its token count.
pub fn fit<E>(
    conversation: &mut Conversation,
    specials: &[String],
    budget: usize,
    mut count_tokens: impl FnMut(&str) -> Result<usize, E>,
) -> Result<(String, usize), FitError<E>> {
    loop {
        let text = render(&conversation.messages(), specials).map_err(FitError::Chat)?;
        let tokens = count_tokens(&text).map_err(FitError::Count)?;
        if tokens <= budget {
            return Ok((text, tokens));
        }
        if !conversation.drop_oldest_exchange() {
            return Err(FitError::TooLong { tokens, budget });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Result<usize, ()> {
        Ok(text.split_whitespace().count())
    }

    fn chat() -> Conversation {
        let mut c = Conversation::new(Some("sys".into()));
        c.push_user("one");
        c.push_assistant("two");
        c.push_user("three");
        c
    }

    #[test]
    fn messages_start_with_the_system_message() {
        let messages = chat().messages();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages.first().map(|m| m.role), Some(Role::System));
        assert_eq!(Conversation::new(None).messages(), []);
    }

    #[test]
    fn default_conversation_carries_the_marin_system_message() {
        let mut c = Conversation::default();
        c.push_user("hi");
        assert!(
            c.messages()
                .first()
                .is_some_and(|m| m.role == Role::System && m.content.contains("Marin"))
        );
    }

    #[test]
    fn oldest_exchange_goes_first_and_the_newest_message_is_never_dropped() {
        let mut c = chat();
        assert!(c.drop_oldest_exchange());
        assert_eq!(
            c.messages()
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            ["sys", "three"]
        );
        assert!(!c.drop_oldest_exchange());
        assert!(!c.is_empty());
    }

    #[test]
    fn fit_forgets_old_turns_until_the_prompt_fits() {
        let mut c = chat();
        let (all, all_tokens) = fit(&mut c, &[], usize::MAX, words).unwrap();
        assert!(all.contains("one") && all.contains("two"));
        let (text, tokens) = fit(&mut c, &[], all_tokens.saturating_sub(1), words).unwrap();
        assert!(!text.contains("one") && !text.contains("two") && text.contains("three"));
        assert!(tokens < all_tokens);
    }

    #[test]
    fn fit_reports_a_message_that_can_never_fit_and_propagates_errors() {
        let mut c = chat();
        assert!(matches!(
            fit(&mut c, &[], 3, words),
            Err(FitError::TooLong { budget: 3, .. })
        ));
        let mut again = chat();
        assert_eq!(
            fit(&mut again, &[], 100, |_| Err::<usize, _>("boom")),
            Err(FitError::Count("boom"))
        );
        let mut no_turns = Conversation::default();
        assert_eq!(
            fit(&mut no_turns, &[], 100, words),
            Err(FitError::Chat(ChatError::MustEndWithUser))
        );
        let mut nothing = Conversation::new(None);
        assert_eq!(
            fit(&mut nothing, &[], 100, words),
            Err(FitError::Chat(ChatError::Empty))
        );
    }

    #[test]
    fn pop_and_clear_forget_messages() {
        let mut c = chat();
        c.pop();
        assert_eq!(c.messages().len(), 3);
        c.clear();
        assert!(c.is_empty());
    }
}
