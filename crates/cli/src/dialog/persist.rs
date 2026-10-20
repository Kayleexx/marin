//! Keeping the conversations of a `Dialog` between runs, and continuing them: the text of both modes, nothing of the
//! cache. The next prompt after a resume is read through the same paths as any other.

use super::{Dialog, Origin, Talk};
use crate::CliError;
use crate::chat::Chat;
use crate::code::CodeChat;
use crate::config::Home;
use crate::saved::{Code, Listed, ModelId, Saved, SavedError, Store, ago, now};
use crate::session::Session;

pub(super) fn store(home: Option<&Home>) -> Result<Store, CliError> {
    home.map(|h| Store::new(h.sessions_dir())).ok_or_else(|| {
        CliError::BadOption("there is nowhere to keep sessions: set HOME or MARIN_HOME".into())
    })
}

fn model_id(session: &Session) -> Result<ModelId, CliError> {
    Ok(ModelId {
        name: session.loaded.name.clone(),
        fingerprint: format!("{:016x}", session.loaded.dir.config_hash()?),
    })
}

impl Dialog {
    /// Keeps both conversations as session `name` (else the one this came from, else a new one). Cancelled and
    /// failed questions are not in them, so they are not in the file.
    pub fn save(
        &mut self,
        session: &Session,
        home: Option<&Home>,
        name: Option<&str>,
    ) -> Result<Vec<String>, CliError> {
        let store = store(home)?;
        let chat = self.chat.items();
        let code = self
            .code
            .as_ref()
            .filter(|c| !c.items().is_empty())
            .map(|c| Code {
                root: c.root().to_path_buf(),
                items: c.items().to_vec(),
            });
        if chat.is_empty() && code.is_none() {
            return Err(SavedError::Nothing.into());
        }
        let name = name
            .filter(|n| !n.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                self.origin
                    .as_ref()
                    .filter(|o| o.same)
                    .map(|o| o.name.clone())
            })
            .unwrap_or_else(|| format!("session-{}", now()));
        let mode = if self.talk == Talk::Code && code.is_some() {
            Talk::Code
        } else {
            Talk::Chat
        };
        let saved = Saved::new(
            name.clone(),
            (now(), mode.name()),
            model_id(session)?,
            chat,
            code,
        );
        let (asked, coded) = saved.exchanges();
        let from = self
            .origin
            .as_ref()
            .filter(|o| o.name == name)
            .map(|o| o.revision);
        let written = store.save(&saved, from)?;
        let verb = if written.replaced {
            format!("updated `{name}` in {}", written.path.display())
        } else {
            format!("saved `{name}` to {}", written.path.display())
        };
        self.origin = Some(Origin {
            name,
            revision: written.revision,
            same: true,
        });
        let mut lines = vec![format!("{verb} · {asked} chat and {coded} code questions")];
        if let Some(code) = &saved.code {
            lines.push(format!(
                "it holds the source code and tool output the model read from {}; only you can read the file",
                code.root.display()
            ));
        }
        Ok(lines)
    }

    /// Replaces both conversations with those of session `name` (else the newest), if it was written for the model
    /// that is loaded and its project is still there. Nothing changes unless all of that holds.
    pub fn resume(
        &mut self,
        session: &mut Session,
        home: Option<&Home>,
        name: Option<&str>,
    ) -> Result<Vec<String>, CliError> {
        let store = store(home)?;
        let name = match name.filter(|n| !n.is_empty()) {
            Some(name) => name.to_owned(),
            None => newest(&store)?,
        };
        let saved = store.load(&name)?;
        saved.fits(&model_id(session)?)?;
        let mut chat = Chat::new(session);
        chat.restore(&saved.chat);
        let code = match &saved.code {
            Some(kept) if !kept.root.is_dir() => {
                return Err(SavedError::MissingProject {
                    name,
                    root: kept.root.clone(),
                }
                .into());
            }
            Some(kept) => {
                let mut code = CodeChat::new(&kept.root, session, self.log.as_deref())?;
                code.restore(kept.items.clone());
                Some(code)
            }
            None => None,
        };
        let (asked, coded) = saved.exchanges();
        if let Some(kept) = &saved.code {
            self.root = kept.root.clone();
        }
        self.talk = if saved.mode == "code" && code.is_some() {
            Talk::Code
        } else {
            Talk::Chat
        };
        self.chat = chat;
        self.code = code;
        self.origin = Some(Origin {
            name: name.clone(),
            revision: saved.revision,
            same: true,
        });
        let mut lines = vec![format!(
            "resumed `{name}` · {asked} chat and {coded} code questions · {} mode",
            self.talk.name()
        )];
        lines.extend(self.restore_state(session, &store, &saved));
        Ok(lines)
    }

    /// The saved sessions, newest first, with what each is about.
    pub fn sessions(
        &self,
        session: &Session,
        home: Option<&Home>,
    ) -> Result<Vec<String>, CliError> {
        let store = store(home)?;
        let listed = store.list()?;
        if listed.is_empty() {
            return Ok(vec![format!(
                "no sessions saved in {}; /save keeps this conversation",
                store.dir().display()
            )]);
        }
        let current = now();
        let mut lines = vec![format!("saved in {}", store.dir().display())];
        for entry in listed {
            lines.push(match entry {
                Listed::Ok(s) => format!(
                    "  {} · {} · {}+{} questions{} · {} · {}",
                    s.name,
                    s.title,
                    s.exchanges.0,
                    s.exchanges.1,
                    if s.model == session.loaded.name {
                        String::new()
                    } else {
                        format!(" · for {}", s.model)
                    },
                    s.state_bytes.map_or_else(
                        || if s.code {
                            "with code".to_owned()
                        } else {
                            "chat".to_owned()
                        },
                        |bytes| format!(
                            "{} · checkpoint {}",
                            if s.code { "with code" } else { "chat" },
                            crate::units::bytes(bytes)
                        )
                    ),
                    ago(current, s.saved_at)
                ),
                Listed::Damaged { name, why } => format!("  {name} · damaged: {why}"),
            });
        }
        Ok(lines)
    }
}

fn newest(store: &Store) -> Result<String, CliError> {
    store
        .list()?
        .into_iter()
        .find_map(|entry| match entry {
            Listed::Ok(s) => Some(s.name),
            Listed::Damaged { .. } => None,
        })
        .ok_or_else(|| SavedError::NoneSaved.into())
}
