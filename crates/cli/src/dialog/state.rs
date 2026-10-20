//! Keeping the model's processed state of a saved conversation (`/checkpoint`) and using it when the conversation is
//! resumed. The text of the session stays the source of truth: a checkpoint that does not match it, the model or
//! itself is dropped and the history is read again.

use super::Dialog;
use super::persist::store;
use crate::CliError;
use crate::checkpoint::{self, MIN_TOKENS, Model, Source};
use crate::code::Item;
use crate::config::Home;
use crate::saved::{Saved, Store};
use crate::session::Session;
use crate::units;

fn model_of(session: &Session) -> Result<Model, CliError> {
    let weights = session
        .loaded
        .dir
        .weights
        .spawn_content_hash()
        .join()
        .map_err(|_| CliError::BadOption("hashing the weights failed".into()))??;
    let (weight_bits, weight_group) = session.loaded.model.quantization().map_or((0, 0), |q| {
        (
            match q.bits {
                model::QuantBits::Int8 => 8,
                model::QuantBits::Int4 => 4,
            },
            u32::try_from(q.group).unwrap_or(u32::MAX),
        )
    });
    Ok(Model {
        name: session.loaded.name.clone(),
        config: format!("{:016x}", session.loaded.dir.config_hash()?),
        weights: format!("{weights:016x}"),
        weight_bits,
        weight_group,
    })
}

impl Dialog {
    fn history_ids(&self, session: &Session, mode: &str) -> Result<Vec<u32>, CliError> {
        match (mode, self.code.as_ref()) {
            ("code", Some(code)) => code.history_ids(session),
            _ => self.chat.history_ids(session),
        }
    }

    fn code_items(&self) -> Option<&[Item]> {
        self.code
            .as_ref()
            .map(|c| c.items())
            .filter(|items| !items.is_empty())
    }

    /// Keeps the model's state for the saved session `name` (else the one this conversation came from). The
    /// conversation must be exactly what was saved, and the model's state must be at the end of it.
    pub fn checkpoint(
        &self,
        session: &Session,
        home: Option<&Home>,
        name: Option<&str>,
    ) -> Result<Vec<String>, CliError> {
        let store = store(home)?;
        let name = name
            .filter(|n| !n.is_empty())
            .map(str::to_owned)
            .or_else(|| self.origin.as_ref().map(|o| o.name.clone()))
            .ok_or_else(|| CliError::BadOption("name the session, or /save one first".into()))?;
        let saved = store.load(&name)?;
        let same = saved.chat == self.chat.items()
            && saved.code.as_ref().map(|c| c.items.as_slice()) == self.code_items();
        if !same {
            return Err(CliError::BadOption(format!(
                "the conversation is not what `{name}` holds: /save it again first"
            )));
        }
        let engine = &session.loaded.model;
        let layout = engine
            .state_layout()
            .ok_or_else(|| CliError::BadOption("checkpoints need Qwen3.5 on the GPU".into()))?;
        let tokens = engine.kept_tokens(session.kv())?;
        if tokens.len() < MIN_TOKENS || !self.history_ids(session, &saved.mode)?.starts_with(tokens)
        {
            return Err(CliError::BadOption(format!(
                "the model's state is not at the end of this {} conversation, or is under {MIN_TOKENS} tokens (reading that again takes about as long as checking and loading a file)",
                saved.mode
            )));
        }
        let started = std::time::Instant::now();
        let source = Source {
            session: name.clone(),
            revision: saved.revision,
            mode: saved.mode.clone(),
            model: model_of(session)?,
        };
        let bytes = checkpoint::planned_bytes(&source, &layout, tokens)?;
        let path = store.save_state(&name, saved.revision, bytes, |file| {
            checkpoint::write(session, &source, file)
        })?;
        Ok(vec![
            format!(
                "kept the model's state for `{name}` · {} tokens · {} in {:.2} s · {}",
                tokens.len(),
                units::bytes(bytes),
                started.elapsed().as_secs_f64(),
                path.display()
            ),
            "the file reflects your conversation, code included; only you can read it. /resume uses it only if the session, model and weights still match; saving the session again removes it".into(),
        ])
    }

    /// Loads the checkpoint of `saved`, if it has one, into the session. Says what happened, or nothing if there is
    /// none; a checkpoint that cannot be used is a note, not an error.
    pub(super) fn restore_state(
        &self,
        session: &mut Session,
        store: &Store,
        saved: &Saved,
    ) -> Option<String> {
        let started = std::time::Instant::now();
        let (file, len) = match store.open_state(&saved.name) {
            Ok(Some(found)) => found,
            Ok(None) => return None,
            Err(e) => {
                return Some(format!(
                    "checkpoint ignored ({e}); reading the history again"
                ));
            }
        };
        let ignored = |why: &dyn std::fmt::Display| {
            Some(format!(
                "checkpoint ignored ({why}); reading the history again"
            ))
        };
        let model = match model_of(session) {
            Ok(model) => model,
            Err(e) => return ignored(&e),
        };
        let source = Source {
            session: saved.name.clone(),
            revision: saved.revision,
            mode: saved.mode.clone(),
            model,
        };
        let tokens = match checkpoint::restore(session, file, len, &source) {
            Ok(tokens) => tokens,
            Err(reject) => return ignored(&checkpoint::why(&reject)),
        };
        let matches = self
            .history_ids(session, &saved.mode)
            .is_ok_and(|history| history.starts_with(&tokens));
        if matches {
            Some(format!(
                "loaded the model's state for the first {} tokens from the checkpoint in {:.2} s",
                tokens.len(),
                started.elapsed().as_secs_f64()
            ))
        } else {
            let _ = session.release_cache();
            ignored(&"it does not match the conversation")
        }
    }
}
