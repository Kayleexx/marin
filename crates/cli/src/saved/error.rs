use std::path::PathBuf;

use thiserror::Error;

use super::{MAX_FILE, MAX_SESSIONS, MAX_STATES, VERSION};

#[derive(Debug, Error)]
pub enum SavedError {
    #[error("nothing to save yet: no question has been answered")]
    Nothing,
    #[error("`{0}` is not a usable session name")]
    BadName(String),
    #[error("no saved session called `{0}`")]
    NotFound(String),
    #[error("no sessions are saved")]
    NoneSaved,
    #[error("{path}: {reason}")]
    Io { path: PathBuf, reason: String },
    #[error("{path} is not a usable folder for sessions: {reason}")]
    UnsafeDir { path: PathBuf, reason: String },
    #[error("{path} is {size} bytes; sessions larger than {MAX_FILE} bytes are not kept or read")]
    TooBig { path: PathBuf, size: u64 },
    #[error("{path} is damaged: {reason}")]
    Corrupt { path: PathBuf, reason: String },
    #[error("{path} was written by Marin's session format {found}; this Marin reads {VERSION}")]
    Version { path: PathBuf, found: String },
    #[error("{path} exists and is not a session Marin can read ({reason}), so it was not replaced")]
    NotOurs { path: PathBuf, reason: String },
    #[error("{}", conflict(.name, *.found, *.seen))]
    Conflict {
        name: String,
        found: u64,
        seen: Option<u64>,
    },
    #[error("checkpoints in {dir} already take {used} bytes; all together may take {MAX_STATES}")]
    StateSpace { dir: PathBuf, used: u64 },
    #[error("another Marin is saving in {dir}")]
    Busy { dir: PathBuf },
    #[error("{MAX_SESSIONS} sessions are already saved in {dir}")]
    TooMany { dir: PathBuf },
    #[error("`{name}` was saved with {saved} but {current} is loaded")]
    OtherModel {
        name: String,
        saved: String,
        current: String,
    },
    #[error("`{name}` was saved with a model whose configuration or tokenizer has since changed")]
    ChangedModel { name: String },
    #[error("the project of `{name}`, {root}, is no longer there")]
    MissingProject { name: String, root: PathBuf },
}

fn conflict(name: &str, found: u64, seen: Option<u64>) -> String {
    match seen {
        Some(seen) => format!(
            "`{name}` was saved by another run since this conversation was opened (it is at revision {found}, this one started from {seen}), so it was not replaced"
        ),
        None => format!(
            "`{name}` already exists and this conversation did not come from it, so it was not replaced"
        ),
    }
}

impl SavedError {
    pub fn hint(&self) -> Option<&'static str> {
        Some(match self {
            Self::Nothing => "ask something first, then /save",
            Self::BadName(_) => "use letters, digits, `.`, `-` and `_`, up to 64 characters",
            Self::NotFound(_) | Self::NoneSaved => "/sessions lists what is saved",
            Self::UnsafeDir { .. } => {
                "make it a real folder only you can use (chmod 700), or remove it"
            }
            Self::TooBig { .. } | Self::Corrupt { .. } | Self::Version { .. } => {
                "other sessions are unaffected; remove the file to get rid of it"
            }
            Self::NotOurs { .. } => "choose another name, or remove that file yourself",
            Self::Conflict { .. } => {
                "save under another name, or /resume it and repeat your changes"
            }
            Self::Busy { .. } => "try again in a moment",
            Self::StateSpace { .. } => "delete the .state files you no longer need",
            Self::TooMany { .. } => "delete the session files you no longer need",
            Self::OtherModel { .. } => "switch to that model with /model, then /resume",
            Self::ChangedModel { .. } => {
                "the history is text, but it was written for different model files"
            }
            Self::MissingProject { .. } => "restore the folder, or remove the session",
            Self::Io { .. } => return None,
        })
    }
}
