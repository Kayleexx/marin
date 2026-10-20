use std::io::{self, Write};
use std::path::PathBuf;

use std::time::Duration;

use crossterm::cursor::MoveTo;
use crossterm::event;
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal::{Clear, ClearType};
use model::BackendChoice;
use runtime::StopReason;

use super::actions::{self, Action, SHORTCUTS};
use super::app::{App, Flow, Mode};
use super::editor::Editor;
use super::menu::{Item, Menu};
use super::panel::Tab;
use super::raw::RawMode;
use super::reply::ReplyWriter;
use super::screen::{Kind, Line, width};
use super::view::{PROMPT, input_line};
use crate::CliError;
use crate::code::Ui;
use crate::config;
use crate::dialog::Talk;
use crate::setup;

/// Keys typed while a reply streamed were echoed by the terminal and must not become part of the next prompt.
fn discard_pending_input() -> io::Result<()> {
    while event::poll(Duration::ZERO)? {
        event::read()?;
    }
    Ok(())
}

impl App {
    pub(super) fn open_tab(&mut self, tab: Tab) -> Result<(), CliError> {
        if matches!(self.mode, Mode::Input) {
            self.cache.tokens = self
                .session
                .as_ref()
                .and_then(|s| s.token_lines(&s.last_prompt).ok())
                .unwrap_or_default();
            self.cache.next = None;
        }
        self.mode = Mode::Panel(tab);
        let prompt = self
            .session
            .as_ref()
            .map(|s| s.last_prompt.clone())
            .unwrap_or_default();
        if tab == Tab::Next && self.cache.next.is_none() && !prompt.is_empty() {
            let (lines, column) = self.view();
            self.screen.draw(&lines, column)?;
            self.cache.next = Some(match self.live()?.next_token_lines(&prompt) {
                Ok(lines) => lines,
                Err(e) => vec![e.to_string()],
            });
        }
        Ok(())
    }

    pub(super) fn run_action(
        &mut self,
        action: Action,
        raw: &mut RawMode,
    ) -> Result<Flow, CliError> {
        if let Some(question) = action.question() {
            let Some(session) = self.session.as_ref() else {
                self.show_error(&CliError::ModelNotLoaded)?;
                return Ok(Flow::Continue);
            };
            let mut editor = Editor::default();
            editor.set(&actions::current_value(&session.settings, action));
            self.mode = Mode::Ask {
                action,
                question,
                editor,
            };
            return Ok(Flow::Continue);
        }
        match action {
            Action::Inspect => self.open_tab(Tab::Reply)?,
            Action::SwitchModel => self.open_model_picker(),
            Action::SwitchBackend => self.open_backend_picker(),
            Action::Clear => self.clear_screen()?,
            Action::NewChat => {
                self.dialog.clear();
                self.clear_screen()?;
                self.notice = Some(format!(
                    "started a new {} conversation",
                    self.dialog.talk().name()
                ));
            }
            Action::Talk(talk) => self.set_talk(talk)?,
            Action::Status => self.show_status()?,
            Action::Sessions => self.show_sessions()?,
            Action::Settings => self.open_tab(Tab::Settings)?,
            Action::Shortcuts => {
                let lines: Vec<Line> = SHORTCUTS.iter().map(|l| Line::new(Kind::Dim, *l)).collect();
                self.screen.print(&lines)?;
            }
            Action::Quit => return Ok(Flow::Quit(0)),
            _ => {}
        }
        let _ = raw;
        Ok(Flow::Continue)
    }

    fn set_talk(&mut self, talk: Talk) -> Result<(), CliError> {
        let Some(session) = self.session.as_ref() else {
            return Ok(self.show_error(&CliError::ModelNotLoaded)?);
        };
        if talk == self.dialog.talk() {
            self.notice = Some(format!("already in {} mode", talk.name()));
            return Ok(());
        }
        match self.dialog.switch(talk, session) {
            Ok(note) => self.screen.print(&[Line::new(Kind::Dim, note)])?,
            Err(e) => self.show_error(&e)?,
        }
        Ok(())
    }

    fn show_sessions(&self) -> Result<(), CliError> {
        let Some(session) = self.session.as_ref() else {
            return Ok(self.show_error(&CliError::ModelNotLoaded)?);
        };
        match self.dialog.sessions(session, self.home.as_ref()) {
            Ok(lines) => self.print_dim(lines)?,
            Err(e) => self.show_error(&e)?,
        }
        Ok(())
    }

    fn print_dim(&self, lines: Vec<String>) -> io::Result<()> {
        let lines: Vec<Line> = lines
            .into_iter()
            .map(|l| Line::new(Kind::Dim, format!("  {l}")))
            .collect();
        self.screen.print(&lines)
    }

    fn show_status(&self) -> Result<(), CliError> {
        let Some(session) = self.session.as_ref() else {
            return Ok(self.show_error(&CliError::ModelNotLoaded)?);
        };
        let lines: Vec<Line> = self
            .dialog
            .status(session)
            .into_iter()
            .map(|l| Line::new(Kind::Dim, format!("  {l}")))
            .collect();
        self.screen.print(&lines)?;
        Ok(())
    }

    fn open_model_picker(&mut self) {
        let current = self
            .session
            .as_ref()
            .map(|s| s.loaded.dir.path.clone())
            .or_else(|| self.parked.as_ref().map(|p| p.model.clone()))
            .unwrap_or_default();
        let mut models: Vec<PathBuf> = config::discover(&config::model_roots())
            .into_iter()
            .filter(|p| config::model_kind(p) == chat::ModelKind::Chat)
            .collect();
        if !models.contains(&current) {
            models.insert(0, current.clone());
        }
        let items = models
            .into_iter()
            .map(|path| Item {
                label: config::display_name(&path),
                detail: {
                    let tag = if setup::packed_group(&path).is_some() {
                        "packed 8-bit · "
                    } else {
                        ""
                    };
                    if path == current {
                        format!("{tag}current")
                    } else {
                        format!("{tag}{}", path.display())
                    }
                },
                value: path,
            })
            .collect();
        self.mode = Mode::PickModel(Menu::new(items));
    }

    fn open_backend_picker(&mut self) {
        let item = |label: &str, detail: &str, value| Item {
            label: label.to_string(),
            detail: detail.to_string(),
            value,
        };
        self.mode = Mode::PickBackend(Menu::new(vec![
            item(
                "auto",
                "GPU when CUDA works and the model fits, else CPU",
                BackendChoice::Auto,
            ),
            item("cpu", "always the CPU", BackendChoice::Cpu),
            item(
                "cuda",
                "always the GPU (fails if CUDA is unavailable)",
                BackendChoice::Cuda,
            ),
        ]));
    }

    pub(super) fn apply_ask(&mut self, action: Action, text: &str) -> Result<(), CliError> {
        let Some(session) = self.session.as_mut() else {
            return Ok(self.show_error(&CliError::ModelNotLoaded)?);
        };
        let outcome = match action {
            Action::Tokens => session.encode(text).and_then(|ids| {
                let mut lines = vec![format!("{} tokens", ids.len())];
                lines.extend(session.token_lines(&ids)?);
                Ok(lines)
            }),
            Action::NextTokens => session
                .encode(text)
                .and_then(|ids| session.next_token_lines(&ids)),
            Action::Save => self
                .dialog
                .save(session, self.home.as_ref(), Some(text.trim())),
            Action::Resume => self
                .dialog
                .resume(session, self.home.as_ref(), Some(text.trim())),
            _ => actions::apply_setting(&mut session.settings, action, text).map(|note| {
                self.notice = Some(note);
                if let Some(home) = &self.home {
                    let (temp, max_tokens) =
                        (session.settings.temperature, session.settings.max_tokens);
                    let _ = home.update_config(|c| match action {
                        Action::Temperature => c.temp = Some(temp),
                        Action::MaxTokens => c.max_tokens = Some(max_tokens),
                        _ => {}
                    });
                }
                Vec::new()
            }),
        };
        match outcome {
            Ok(lines) if lines.is_empty() => {}
            Ok(lines) => {
                let lines: Vec<Line> = lines
                    .into_iter()
                    .map(|l| Line::new(Kind::Dim, format!("  {l}")))
                    .collect();
                self.screen.print(&lines)?;
            }
            Err(e) => self.show_error(&e)?,
        }
        Ok(())
    }

    pub(super) fn show_error(&self, error: &CliError) -> io::Result<()> {
        let mut lines = vec![Line::new(Kind::Error, format!("error: {error}"))];
        if let Some(hint) = error.hint() {
            lines.push(Line::new(Kind::Dim, format!("hint: {hint}")));
        }
        self.screen.print(&lines)
    }

    /// Wipes the visible screen (not the scrollback) and returns the cursor to the top.
    pub(super) fn clear_screen(&self) -> io::Result<()> {
        let mut out = io::stdout().lock();
        queue!(out, Clear(ClearType::All), MoveTo(0, 0))?;
        out.flush()
    }

    /// The first prompt ends the startup view: its input line beside the portrait is erased, and the
    /// conversation continues at the left edge below the welcome, which stays in the scrollback.
    fn leave_startup_view(&mut self) -> io::Result<()> {
        if let Some(column) = self.splash.take() {
            self.screen.print(&[])?;
            if column > 0 {
                let mut out = io::stdout().lock();
                queue!(out, Print("\r\n\r\n"))?;
                out.flush()?;
            }
            self.screen.set_indent(0);
        }
        Ok(())
    }

    pub(super) fn submit(&mut self, text: &str, raw: &mut RawMode) -> Result<(), CliError> {
        self.editor.remember(text);
        if let Some(home) = &self.home {
            let _ = home.save_history(self.editor.history());
        }
        self.editor.clear();
        self.leave_startup_view()?;
        let (committed, _) = input_line(&Editor::with_text(text), PROMPT, width());
        self.screen.print(&[Line::new(Kind::Prompt, committed)])?;
        let Some(session) = self.session.as_mut() else {
            self.show_error(&CliError::ModelNotLoaded)?;
            return Ok(());
        };
        raw.suspend()?;
        match self.dialog.talk() {
            Talk::Chat => {
                let mut out = ReplyWriter::new(&self.style, &self.caps, true)?;
                let result = self
                    .dialog
                    .chat
                    .ask(session, text, &self.interrupt, &mut out);
                out.finish_markdown()?;
                out.finish_placeholder(result.is_ok())?;
                match result {
                    Ok(reply) => {
                        let note = match reply.outcome.stop {
                            StopReason::EndOfText => None,
                            StopReason::MaxTokens => {
                                Some("stopped at max tokens (change it in / actions)")
                            }
                            StopReason::ContextFull => Some("context window full"),
                            StopReason::Cancelled => Some("cancelled; that question was not kept"),
                        };
                        if let Some(note) = note {
                            println!("{}", self.style.dim(note));
                        }
                    }
                    Err(CliError::Cancelled) => println!(
                        "{}",
                        self.style.dim("cancelled; that question was not kept")
                    ),
                    Err(e) if e.is_broken_pipe() => return Err(e),
                    Err(e) => crate::report(&e, &self.style),
                }
            }
            Talk::Code => {
                let ui = Ui::embedded(self.style, self.caps);
                match self.dialog.ask_code(session, &self.interrupt, ui, text) {
                    Ok(true) => println!(
                        "{}",
                        self.style.dim("cancelled; that question was not kept")
                    ),
                    Ok(false) => {}
                    Err(e) => crate::report(&e, &self.style),
                }
            }
        }
        println!();
        raw.resume()?;
        discard_pending_input()?;
        Ok(())
    }
}
