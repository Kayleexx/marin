//! Conversations kept between runs, at the user's request: the portable part of a chat and a code session (what was
//! asked and answered, and what the tools returned), never the cache. Resuming rebuilds the cache by reading the
//! history again, so a file written by one run can be used by any run of the same model.

mod check;
mod disk;
mod error;
mod state_file;
mod trust;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::code::Item;

pub use disk::{Listed, Store, Summary, Written, now};
pub use error::SavedError;
pub use trust::Trust;

/// The newest layout this Marin reads and writes. A change that old readers could misread must raise it.
pub const VERSION: u32 = 1;
const FORMAT: &str = "marin-session";
/// A session file larger than this is refused, when saved and when read.
pub const MAX_FILE: u64 = 2 << 20;
/// A checkpoint larger than this is not written or read; all checkpoints together may take `MAX_STATES`.
pub const MAX_STATE: u64 = 1 << 30;
pub const MAX_STATES: u64 = 4 << 30;
/// At most this many sessions are kept; saving a new one beyond it asks the user to remove some.
pub const MAX_SESSIONS: usize = 100;

/// What decides whether a history still makes sense to the model that reads it next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelId {
    pub name: String,
    /// Hash of the model's configuration and tokenizer files, in hex.
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Code {
    pub root: PathBuf,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    format: String,
    version: u32,
    pub name: String,
    /// Seconds since the Unix epoch.
    pub saved_at: u64,
    /// How many times this session has been written: what a save must have started from.
    #[serde(default)]
    pub revision: u64,
    pub model: ModelId,
    /// The mode to continue in: `chat` or `code`.
    pub mode: String,
    pub chat: Vec<Item>,
    pub code: Option<Code>,
}

impl Saved {
    pub fn new(
        name: String,
        (saved_at, mode): (u64, &str),
        model: ModelId,
        chat: Vec<Item>,
        code: Option<Code>,
    ) -> Self {
        Self {
            format: FORMAT.into(),
            version: VERSION,
            name,
            saved_at,
            revision: 0,
            model,
            mode: mode.into(),
            chat,
            code,
        }
    }

    /// The questions asked in the chat and, separately, in code mode.
    pub fn exchanges(&self) -> (usize, usize) {
        let asked = |items: &[Item]| items.iter().filter(|i| matches!(i, Item::User(_))).count();
        (
            asked(&self.chat),
            self.code.as_ref().map_or(0, |c| asked(&c.items)),
        )
    }

    /// The first question, for telling sessions apart in a list.
    pub fn title(&self) -> String {
        let first = |items: &[Item]| {
            items.iter().find_map(|i| match i {
                Item::User(text) => Some(text.lines().next().unwrap_or("").trim().to_owned()),
                _ => None,
            })
        };
        let text = first(&self.chat)
            .or_else(|| self.code.as_ref().and_then(|c| first(&c.items)))
            .unwrap_or_default();
        match text.char_indices().nth(60) {
            Some((at, _)) => format!("{}…", text.get(..at).unwrap_or(&text)),
            None => text,
        }
    }
}

/// How long ago `then` was, in the largest sensible unit.
pub fn ago(now: u64, then: u64) -> String {
    let secs = now.saturating_sub(then);
    match secs {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", secs / 60),
        3600..=86399 => format!("{} h ago", secs / 3600),
        _ => format!("{} d ago", secs / 86400),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_safety;
