use std::path::{Path, PathBuf};

use model::{BackendChoice, Precision};

use super::app::{App, PanelCache, Parked};
use super::raw::RawMode;
use crate::chat::Chat;
use crate::config::Config;
use crate::session::Session;
use crate::{CliError, load};

impl App {
    /// The loaded session, or the error that tells the user how to get one.
    pub(super) fn live(&mut self) -> Result<&mut Session, CliError> {
        self.session.as_mut().ok_or(CliError::ModelNotLoaded)
    }

    /// Frees the model and its KV cache, remembering what a reload needs. The GPU memory is returned
    /// before anything else is loaded, so two models never have to fit at once.
    fn unload(&mut self) {
        if let Some(session) = self.session.take() {
            self.parked = Some(Parked {
                model: session.loaded.dir.path.clone(),
                ctx: session.ctx,
                settings: session.settings,
            });
        }
    }

    fn load_session(
        &self,
        path: &Path,
        backend: BackendChoice,
        parked: &Parked,
    ) -> Result<Session, CliError> {
        let loaded = load::load(
            path,
            Some(parked.ctx),
            parked.ctx,
            backend,
            Precision::default(),
            &self.style,
            true,
        )?;
        Session::new(loaded, parked.settings, self.style)
    }

    fn install(&mut self, session: Session, backend: BackendChoice, notice: String) {
        self.backend = backend;
        self.cache = PanelCache::default();
        self.notice = Some(notice);
        self.session = Some(session);
        self.parked = None;
    }

    /// Switches model and/or backend: unload, load, and on failure put the previous model back.
    /// Marin stays usable whichever way it ends; if even the previous model cannot return, the prompt
    /// stays open without a model and says so.
    pub(super) fn switch(
        &mut self,
        model: Option<PathBuf>,
        backend: BackendChoice,
        raw: &mut RawMode,
    ) -> Result<(), CliError> {
        self.unload();
        let Some(parked) = self.parked.clone() else {
            return Ok(());
        };
        let target = model.unwrap_or_else(|| parked.model.clone());
        let previous_backend = self.backend;
        raw.suspend()?;
        self.screen.print(&[])?;
        match self.load_session(&target, backend, &parked) {
            Ok(session) => {
                if target != parked.model {
                    self.chat = Chat::new(&session);
                }
                let name = format!(
                    "{} · {}",
                    session.loaded.name,
                    session.loaded.model.backend_name()
                );
                self.install(session, backend, format!("now using {name}"));
                if let Some(home) = &self.home {
                    let _ = home.save_config(&Config {
                        model: Some(target),
                    });
                }
            }
            Err(failure) => {
                self.show_error(&failure)?;
                match self.load_session(&parked.model, previous_backend, &parked) {
                    Ok(session) => {
                        let kept = session.loaded.name.clone();
                        self.install(
                            session,
                            previous_backend,
                            format!("kept {kept}: the switch failed"),
                        );
                    }
                    Err(again) => {
                        self.show_error(&again)?;
                        self.notice = Some("no model is loaded".to_string());
                    }
                }
            }
        }
        raw.resume()?;
        Ok(())
    }
}
