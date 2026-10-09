use std::path::PathBuf;

use crossterm::event::{self, Event, KeyEventKind};
use model::BackendChoice;

use super::actions::Action;
use super::caps::Caps;
use super::editor::Editor;
use super::menu::Menu;
use super::panel::Tab;
use super::raw::RawMode;
use super::screen::Screen;
use crate::CliError;
use crate::chat::Chat;
use crate::config::Home;
use crate::load::ChatPrecision;
use crate::session::Session;
use crate::signal::Interrupt;
use crate::style::Style;

pub struct Launch {
    pub session: Session,
    pub chat: Chat,
    pub interrupt: Interrupt,
    pub home: Option<Home>,
    pub backend: BackendChoice,
    /// What the user asked for; a model switch keeps it.
    pub precision: ChatPrecision,
    pub style: Style,
    pub caps: Caps,
    /// The startup view is on screen, with the first prompt at this column (0 without a portrait).
    /// It stays in the scrollback; the first submitted prompt starts the conversation below it.
    pub splash: Option<usize>,
}

pub(super) enum Mode {
    Input,
    Palette(Menu<Action>),
    PickModel(Menu<PathBuf>),
    PickBackend(Menu<BackendChoice>),
    Ask {
        action: Action,
        question: &'static str,
        editor: Editor,
    },
    Panel(Tab),
}

/// The state of the unloaded model, kept so a failed switch can restore it.
#[derive(Clone)]
pub(super) struct Parked {
    pub(super) model: PathBuf,
    pub(super) ctx: usize,
    pub(super) settings: crate::session::Settings,
}

pub(super) enum Flow {
    Continue,
    Quit(i32),
}

#[derive(Default)]
pub(super) struct PanelCache {
    pub(super) tokens: Vec<String>,
    pub(super) next: Option<Vec<String>>,
}

pub(super) struct App {
    pub(super) session: Option<Session>,
    /// What a model switch needs while no model is resident: the model to fall back to, its context and the settings.
    pub(super) parked: Option<Parked>,
    pub(super) chat: Chat,
    pub(super) splash: Option<usize>,
    pub(super) interrupt: Interrupt,
    pub(super) home: Option<Home>,
    pub(super) backend: BackendChoice,
    pub(super) precision: ChatPrecision,
    pub(super) style: Style,
    pub(super) caps: Caps,
    pub(super) screen: Screen,
    pub(super) editor: Editor,
    pub(super) mode: Mode,
    pub(super) notice: Option<String>,
    pub(super) exit_armed: bool,
    pub(super) cache: PanelCache,
}

/// Runs the interactive prompt until the user leaves; returns the exit code.
pub fn run(launch: Launch) -> Result<i32, CliError> {
    let history = launch
        .home
        .as_ref()
        .map(Home::load_history)
        .unwrap_or_default();
    let mut screen = Screen::new(launch.style);
    screen.set_indent(launch.splash.unwrap_or(0));
    let mut app = App {
        screen,
        session: Some(launch.session),
        parked: None,
        chat: launch.chat,
        splash: launch.splash,
        interrupt: launch.interrupt,
        home: launch.home,
        backend: launch.backend,
        precision: launch.precision,
        style: launch.style,
        caps: launch.caps,
        editor: Editor::with_history(history),
        mode: Mode::Input,
        notice: None,
        exit_armed: false,
        cache: PanelCache::default(),
    };
    let mut raw = RawMode::enable()?;
    app.event_loop(&mut raw)
}

impl App {
    fn event_loop(&mut self, raw: &mut RawMode) -> Result<i32, CliError> {
        loop {
            let (lines, column) = self.view();
            self.screen.draw(&lines, column)?;
            let flow = match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(key, raw)?,
                Event::Paste(text) => {
                    self.on_paste(&text);
                    Flow::Continue
                }
                _ => Flow::Continue,
            };
            if let Flow::Quit(code) = flow {
                if let Some(session) = self.session.as_mut() {
                    let _ = session.save_snapshot();
                }
                self.screen.print(&[])?;
                return Ok(code);
            }
        }
    }
}
