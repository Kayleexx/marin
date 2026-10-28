use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

pub const MAX_ITEMS: usize = 50;
pub const TTL: u64 = 24 * 3600;
pub const TAIL_BYTES: usize = 4096;
const REPO_COOLDOWN: u64 = 300;
const ACCEPTED_QUIET: u64 = 3600;
const HOUR: u64 = 3600;
const MAX_PER_HOUR: usize = 6;

/// A command that failed in an allowed repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub repo: String,
    pub command: String,
    pub exit: i32,
    pub tail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    New,
    Accepted,
    Dismissed,
    Snoozed { until: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub id: u32,
    fingerprint: u64,
    pub event: Event,
    pub first_seen: u64,
    pub last_seen: u64,
    pub count: u32,
    pub state: State,
}

#[derive(Debug, PartialEq)]
pub enum Offer {
    Surfaced(u32),
    Repeated(u32),
    Quiet(&'static str),
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Inbox {
    next_id: u32,
    pub items: Vec<Suggestion>,
    surfaced: Vec<(String, u64)>,
}

impl Suggestion {
    pub fn visible(&self, now: u64) -> bool {
        match self.state {
            State::New => true,
            State::Snoozed { until } => until <= now,
            _ => false,
        }
    }

    pub fn first_error(&self) -> String {
        let key = |l: &&str| {
            let l = l.to_ascii_lowercase();
            ["error", "failed", "panicked", "fail"]
                .iter()
                .any(|w| l.contains(w))
        };
        let tail = &self.event.tail;
        let line = tail.lines().find(key).or_else(|| tail.lines().last());
        line.unwrap_or("").trim().chars().take(140).collect()
    }

    pub fn headline(&self) -> String {
        let repo = self.event.repo.rsplit('/').next().unwrap_or("");
        let again = if self.count > 1 {
            format!(" (seen {} times)", self.count)
        } else {
            String::new()
        };
        format!(
            "`{}` failed (exit {}) in {repo}{again}",
            self.event.command, self.event.exit
        )
    }

    pub fn why(&self) -> Vec<String> {
        vec![
            format!(
                "trigger: `{}` exited with {} in {}",
                self.event.command, self.event.exit, self.event.repo
            ),
            "rule: a command you ran with `marin assist watch` failed in a repo you allowed".into(),
            format!("seen {} time(s) since {}", self.count, ago(self.first_seen)),
            format!(
                "context kept: the last {} bytes of its output, secrets filtered",
                self.event.tail.len()
            ),
            "model: none ran; it is built by rules and only reaches a model if you accept it"
                .into(),
            "data left the device: no".into(),
        ]
    }
}

fn ago(since: u64) -> String {
    match crate::saved::now().saturating_sub(since) {
        s if s < 90 => format!("{s} s ago"),
        s => format!("{} min ago", s.saturating_div(60)),
    }
}

fn fingerprint(event: &Event, first_error: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    event.repo.hash(&mut hasher);
    event.command.hash(&mut hasher);
    first_error
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .collect::<String>()
        .hash(&mut hasher);
    hasher.finish()
}

pub fn bound(text: &str, bytes: usize) -> String {
    let cut = text.len().saturating_sub(bytes);
    let start = (cut..=text.len())
        .find(|i| text.is_char_boundary(*i))
        .unwrap_or(0);
    text.get(start..).unwrap_or("").to_string()
}

impl Inbox {
    fn expire(&mut self, now: u64) {
        self.items.retain(|i| now.saturating_sub(i.last_seen) < TTL);
        self.surfaced.retain(|(_, t)| now.saturating_sub(*t) < HOUR);
    }

    pub fn offer(&mut self, event: Event, now: u64) -> Offer {
        self.expire(now);
        let mut suggestion = Suggestion {
            id: 0,
            fingerprint: 0,
            event,
            first_seen: now,
            last_seen: now,
            count: 1,
            state: State::New,
        };
        suggestion.fingerprint = fingerprint(&suggestion.event, &suggestion.first_error());
        let fp = suggestion.fingerprint;
        if let Some(old) = self.items.iter_mut().find(|i| i.fingerprint == fp) {
            match old.state {
                State::New | State::Snoozed { .. } => {
                    old.count = old.count.saturating_add(1);
                    old.last_seen = now;
                    old.event.tail = suggestion.event.tail;
                    return Offer::Repeated(old.id);
                }
                State::Dismissed => return Offer::Quiet("you dismissed this one"),
                State::Accepted if now.saturating_sub(old.last_seen) < ACCEPTED_QUIET => {
                    return Offer::Quiet("you opened this one a moment ago");
                }
                State::Accepted => self.items.retain(|i| i.fingerprint != fp),
            }
        }
        let repo = suggestion.event.repo.clone();
        if self
            .surfaced
            .iter()
            .any(|(r, t)| *r == repo && now.saturating_sub(*t) < REPO_COOLDOWN)
        {
            return Offer::Quiet("cooling down after the last suggestion for this repo");
        }
        if self.surfaced.len() >= MAX_PER_HOUR {
            return Offer::Quiet("hourly limit reached");
        }
        self.next_id = self.next_id.saturating_add(1);
        suggestion.id = self.next_id;
        self.surfaced.push((repo, now));
        self.items.push(suggestion);
        if self.items.len() > MAX_ITEMS {
            let drop = self
                .items
                .iter()
                .position(|i| i.state != State::New)
                .unwrap_or(0);
            self.items.remove(drop);
        }
        Offer::Surfaced(self.next_id)
    }

    pub fn find(&self, id: u32) -> Option<&Suggestion> {
        self.items.iter().find(|i| i.id == id)
    }

    pub fn mark(&mut self, id: u32, state: State) -> bool {
        self.items
            .iter_mut()
            .find(|i| i.id == id)
            .map(|i| i.state = state)
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure(repo: &str, line: &str) -> Event {
        Event {
            repo: repo.into(),
            command: "cargo test".into(),
            exit: 101,
            tail: format!("running\n{line}\n"),
        }
    }

    #[test]
    fn a_repeat_is_counted_not_repeated_and_line_numbers_do_not_matter() {
        let mut inbox = Inbox::default();
        assert_eq!(
            inbox.offer(failure("/r/a", "error[E0382] at line 12"), 100),
            Offer::Surfaced(1)
        );
        assert_eq!(
            inbox.offer(failure("/r/a", "error[E0382] at line 15"), 160),
            Offer::Repeated(1)
        );
        assert_eq!(inbox.items.len(), 1);
        assert_eq!(inbox.find(1).map(|s| s.count), Some(2));
    }

    #[test]
    fn a_dismissed_failure_stays_silent_until_it_expires() {
        let mut inbox = Inbox::default();
        inbox.offer(failure("/r/a", "error: x"), 0);
        assert!(inbox.mark(1, State::Dismissed));
        assert!(matches!(
            inbox.offer(failure("/r/a", "error: x"), 600),
            Offer::Quiet(_)
        ));
        assert_eq!(
            inbox.offer(failure("/r/a", "error: x"), TTL + 10),
            Offer::Surfaced(2)
        );
    }

    #[test]
    fn cooldown_and_the_hourly_limit_hold() {
        let mut inbox = Inbox::default();
        inbox.offer(failure("/r/a", "error: one"), 0);
        assert!(matches!(
            inbox.offer(failure("/r/a", "error: two"), 100),
            Offer::Quiet(_)
        ));
        assert_eq!(
            inbox.offer(failure("/r/a", "error: two"), 400),
            Offer::Surfaced(2)
        );
        for n in 0..10 {
            inbox.offer(failure(&format!("/r/{n}"), "error: y"), 500);
        }
        assert_eq!(inbox.items.len(), MAX_PER_HOUR);
    }

    #[test]
    fn a_snoozed_suggestion_is_hidden_until_due_and_text_is_cut_on_a_character() {
        let mut inbox = Inbox::default();
        inbox.offer(failure("/r/a", "error: one"), 0);
        inbox.mark(1, State::Snoozed { until: 900 });
        assert!(
            inbox
                .find(1)
                .is_some_and(|s| !s.visible(800) && s.visible(900))
        );
        assert_eq!(bound("héllo", 4), "llo");
    }
}
