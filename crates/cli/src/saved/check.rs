use std::path::Path;

use serde_json::Value;

use super::{FORMAT, ModelId, Saved, SavedError, VERSION};
use crate::code::Item;

/// Letters, digits, `.`, `-` and `_`, not starting with `.` or `-`: a name is always a plain file name.
pub fn valid_name(name: &str) -> bool {
    let plain = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_');
    (1..=64).contains(&name.len()) && name.chars().all(plain) && !name.starts_with(['.', '-'])
}

impl Saved {
    /// Reads a session file's bytes. Anything that is not a complete, current, well-formed session is an error that
    /// names `path`: nothing partial is ever returned.
    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self, SavedError> {
        let corrupt = |reason: String| SavedError::Corrupt {
            path: path.to_path_buf(),
            reason,
        };
        let value: Value = serde_json::from_slice(bytes).map_err(|e| corrupt(e.to_string()))?;
        if value.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Err(corrupt("it is not a Marin session".into()));
        }
        match value.get("version").and_then(Value::as_u64) {
            Some(found) if found == u64::from(VERSION) => {}
            Some(found) => {
                return Err(SavedError::Version {
                    path: path.to_path_buf(),
                    found: found.to_string(),
                });
            }
            None => return Err(corrupt("it has no version".into())),
        }
        let saved: Self = serde_json::from_value(value).map_err(|e| corrupt(e.to_string()))?;
        saved.check().map_err(corrupt)?;
        Ok(saved)
    }

    /// Whether the history is something a conversation could have produced: whole exchanges, in order.
    pub fn check(&self) -> Result<(), String> {
        if !valid_name(&self.name) {
            return Err(format!("`{}` is not a session name", self.name));
        }
        match (self.mode.as_str(), &self.code) {
            ("chat", _) | ("code", Some(_)) => {}
            (mode, _) => return Err(format!("mode `{mode}` has no conversation to continue")),
        }
        if self.model.name.is_empty() || self.model.fingerprint.is_empty() {
            return Err("it does not say which model wrote it".into());
        }
        if self.chat.is_empty() && self.code.is_none() {
            return Err("it holds no conversation".into());
        }
        chat_is_whole(&self.chat)?;
        self.code
            .as_ref()
            .map_or(Ok(()), |c| code_is_whole(&c.items))
    }

    /// Whether this history was written for the model that is loaded now.
    pub fn fits(&self, model: &ModelId) -> Result<(), SavedError> {
        if self.model.name != model.name {
            return Err(SavedError::OtherModel {
                name: self.name.clone(),
                saved: self.model.name.clone(),
                current: model.name.clone(),
            });
        }
        if self.model.fingerprint != model.fingerprint {
            return Err(SavedError::ChangedModel {
                name: self.name.clone(),
            });
        }
        Ok(())
    }
}

fn chat_is_whole(items: &[Item]) -> Result<(), String> {
    let well_formed = items.len().is_multiple_of(2)
        && items
            .chunks(2)
            .all(|pair| matches!(pair, [Item::User(_), Item::Assistant(_)]));
    if well_formed {
        Ok(())
    } else {
        Err("the chat is not made of whole question and answer pairs".into())
    }
}

fn code_is_whole(items: &[Item]) -> Result<(), String> {
    let starts = matches!(items.first(), Some(Item::User(_)));
    let ends = matches!(items.last(), Some(Item::Assistant(_)));
    let ordered = items.windows(2).all(|pair| {
        !matches!(
            pair,
            [
                Item::User(_) | Item::Results { .. },
                Item::User(_) | Item::Results { .. }
            ]
        )
    });
    if starts && ends && ordered {
        Ok(())
    } else {
        Err("the code conversation does not start with a question and end with an answer".into())
    }
}
