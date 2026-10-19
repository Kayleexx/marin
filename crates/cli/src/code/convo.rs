use serde_json::{Value, json};

/// Tokens per character before the server has told us the real ratio.
const GUESS: f64 = 0.34;

pub struct Done {
    pub label: String,
    pub text: String,
}

enum Item {
    User(String),
    Assistant(String),
    Results {
        done: Vec<Done>,
        note: String,
        elided: bool,
    },
}

#[derive(Default)]
pub struct Trimmed {
    pub elided: usize,
    pub dropped: usize,
}

/// The messages sent to the model, kept within its context window by dropping the least useful parts first.
pub struct Conversation {
    system: String,
    items: Vec<Item>,
    /// The server-side session that holds the processed start of this conversation; "new" asks for one.
    pub session: String,
    context: usize,
    /// Tokens per character, from the last request the server counted.
    ratio: Option<f64>,
}

fn tagged(done: &[Done], note: &str, elided: bool) -> String {
    let blocks = done
        .iter()
        .map(|d| {
            let body = if elided {
                format!("[output omitted to save space: {}]", d.label)
            } else {
                d.text.clone()
            };
            format!("<tool_response>\n{body}\n</tool_response>")
        })
        .collect::<Vec<_>>()
        .join("\n");
    if note.is_empty() || elided {
        blocks
    } else {
        format!("{blocks}\n{note}")
    }
}

impl Item {
    fn text(&self) -> String {
        match self {
            Self::User(text) | Self::Assistant(text) => text.clone(),
            Self::Results { done, note, elided } => tagged(done, note, *elided),
        }
    }

    fn role(&self) -> &'static str {
        match self {
            Self::Assistant(_) => "assistant",
            Self::User(_) | Self::Results { .. } => "user",
        }
    }
}

impl Conversation {
    pub fn new(system: String, context: usize) -> Self {
        Self {
            system,
            items: Vec::new(),
            session: "new".into(),
            context,
            ratio: None,
        }
    }

    pub fn mark(&self) -> usize {
        self.items.len()
    }

    pub fn truncate(&mut self, len: usize) {
        self.items.truncate(len);
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.session = "new".into();
    }

    pub fn push_user(&mut self, text: String) {
        self.items.push(Item::User(text));
    }

    pub fn push_assistant(&mut self, text: String) {
        self.items.push(Item::Assistant(text));
    }

    pub fn push_results(&mut self, done: Vec<Done>, note: &str) {
        self.items.push(Item::Results {
            done,
            note: note.to_owned(),
            elided: false,
        });
    }

    pub fn messages(&self) -> Vec<Value> {
        let system = json!({"role": "system", "content": self.system});
        std::iter::once(system)
            .chain(
                self.items
                    .iter()
                    .map(|i| json!({"role": i.role(), "content": i.text()})),
            )
            .collect()
    }

    fn chars(&self) -> usize {
        self.items
            .iter()
            .map(|i| i.text().chars().count())
            .fold(self.system.chars().count(), usize::saturating_add)
    }

    /// Learns how many tokens a character costs from a request the server has counted.
    pub fn calibrate(&mut self, prompt_tokens: usize) {
        let chars = self.chars();
        if chars > 0 && prompt_tokens > 0 {
            self.ratio = Some(prompt_tokens as f64 / chars as f64);
        }
    }

    pub fn estimate(&self) -> usize {
        (self.chars() as f64 * self.ratio.unwrap_or(GUESS)).ceil() as usize
    }

    /// Makes the next request fit with `reserve` tokens to spare: old tool output goes first, then whole old
    /// questions with their answers. The newest question is never dropped.
    pub fn make_room(&mut self, reserve: usize, share: f64) -> Result<Trimmed, usize> {
        let limit = (self.context.saturating_sub(reserve) as f64 * share) as usize;
        let mut trimmed = Trimmed::default();
        while self.estimate() > limit {
            if let Some(old) = self.oldest_results() {
                if let Some(Item::Results { elided, .. }) = self.items.get_mut(old) {
                    *elided = true;
                    trimmed.elided = trimmed.elided.saturating_add(1);
                }
            } else if let Some(next) = self.second_question() {
                self.items.drain(..next);
                trimmed.dropped = trimmed.dropped.saturating_add(1);
            } else {
                return Err(self.estimate());
            }
        }
        Ok(trimmed)
    }

    /// The oldest tool output still in full, if a newer one exists to keep.
    fn oldest_results(&self) -> Option<usize> {
        let mut full = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, i)| matches!(i, Item::Results { elided: false, .. }));
        let first = full.next()?.0;
        full.next().map(|_| first)
    }

    fn second_question(&self) -> Option<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, i)| matches!(i, Item::User(_)))
            .nth(1)
            .map(|(at, _)| at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(text: &str) -> Vec<Done> {
        vec![Done {
            label: "read a.rs".into(),
            text: text.into(),
        }]
    }

    fn talk(convo: &mut Conversation, question: &str, output: &str) {
        convo.push_user(question.into());
        convo.push_assistant("<tool_call>x</tool_call>".into());
        convo.push_results(done(output), "");
        convo.push_assistant("answer".into());
    }

    #[test]
    fn tool_results_become_tagged_user_messages() {
        let mut convo = Conversation::new("sys".into(), 1000);
        talk(&mut convo, "q", "file text");
        let messages = convo.messages();
        assert_eq!(messages.len(), 5);
        let result = messages
            .get(3)
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str);
        assert_eq!(result, Some("<tool_response>\nfile text\n</tool_response>"));
        assert_eq!(
            messages
                .get(3)
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn old_tool_output_is_elided_before_anything_is_dropped() {
        let mut convo = Conversation::new("s".into(), 400);
        talk(&mut convo, "one", &"a".repeat(500));
        talk(&mut convo, "two", &"b".repeat(500));
        let trimmed = convo.make_room(20, 1.0).unwrap_or_default();
        assert_eq!((trimmed.elided, trimmed.dropped), (1, 0));
        let text = convo
            .messages()
            .iter()
            .map(Value::to_string)
            .collect::<String>();
        assert!(text.contains("output omitted") && text.contains(&"b".repeat(500)));
    }

    #[test]
    fn whole_old_questions_go_next_and_the_newest_stays() {
        let mut convo = Conversation::new("s".into(), 400);
        for n in 0..12 {
            talk(&mut convo, &format!("question {n}"), &"x".repeat(300));
        }
        let trimmed = convo.make_room(20, 1.0).unwrap_or_default();
        assert!(trimmed.dropped >= 1);
        let text = convo
            .messages()
            .iter()
            .map(Value::to_string)
            .collect::<String>();
        assert!(text.contains("question 11") && !text.contains("question 0"));
        assert!(matches!(convo.items.first(), Some(Item::User(_))));
    }

    #[test]
    fn a_single_oversized_question_is_reported_not_cut() {
        let mut convo = Conversation::new("s".into(), 100);
        convo.push_user("y".repeat(1000));
        assert!(convo.make_room(10, 1.0).is_err());
        assert_eq!(convo.mark(), 1);
    }

    #[test]
    fn the_estimate_follows_the_servers_count() {
        let mut convo = Conversation::new("s".into(), 1000);
        convo.push_user("x".repeat(100));
        let guess = convo.estimate();
        convo.calibrate(10);
        assert!(convo.estimate() < guess);
    }
}
