use std::path::{Path, PathBuf};

use model::BackendChoice;

use super::app::{App, PanelCache, Parked};
use super::raw::RawMode;
use super::screen::{Kind, Line};
use crate::session::Session;
use crate::setup;
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
        let loaded = load::load_chat(
            path,
            (Some(parked.ctx), parked.ctx),
            backend,
            setup::precision_for_pick(self.precision, path),
            &self.style,
            (true, false),
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
                let weights = session.loaded.precision.weights;
                if target != parked.model {
                    match self.dialog.reset(&session) {
                        Ok(()) => self.screen.print(&[Line::new(
                            Kind::Dim,
                            "new model: both conversations start over".to_string(),
                        )])?,
                        Err(e) => self.show_error(&e)?,
                    }
                }
                let name = format!(
                    "{} · {}",
                    session.loaded.name,
                    session.loaded.model.backend_name()
                );
                self.install(session, backend, format!("now using {name}"));
                if let Some(home) = &self.home {
                    setup::remember(Some(home), &target, weights);
                    // Only a backend the user chose here is remembered, not one a flag gave this run.
                    if backend != previous_backend {
                        let name = format!("{backend:?}").to_lowercase();
                        let _ = home.update_config(|c| c.backend = Some(name));
                    }
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
