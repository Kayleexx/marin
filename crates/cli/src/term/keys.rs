use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::actions;
use super::app::{App, Flow, Mode};
use super::panel::Tab;
use super::raw::RawMode;
use crate::CliError;
use crate::signal::INTERRUPTED_EXIT_CODE;

impl App {
    pub(super) fn on_paste(&mut self, text: &str) {
        match &mut self.mode {
            Mode::Input | Mode::Panel(_) => self.editor.insert(text),
            Mode::Ask { editor, .. } => editor.insert(&text.replace(['\r', '\n'], " ")),
            _ => {}
        }
    }

    pub(super) fn on_key(&mut self, key: KeyEvent, raw: &mut RawMode) -> Result<Flow, CliError> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let ctrl_c = ctrl && key.code == KeyCode::Char('c');
        if !ctrl_c {
            self.exit_armed = false;
        }
        self.notice = None;
        match &mut self.mode {
            Mode::Input => self.input_key(key, ctrl, raw),
            Mode::Palette(menu) => {
                match key.code {
                    KeyCode::Esc => self.mode = Mode::Input,
                    KeyCode::Char('c') if ctrl => self.mode = Mode::Input,
                    KeyCode::Backspace if !menu.pop() => self.mode = Mode::Input,
                    KeyCode::Up => menu.up(),
                    KeyCode::Down => menu.down(),
                    KeyCode::Char(c) if !ctrl => menu.push(c),
                    KeyCode::Enter => {
                        if let Some(action) = menu.current().copied() {
                            self.mode = Mode::Input;
                            return self.run_action(action, raw);
                        }
                    }
                    _ => {}
                }
                Ok(Flow::Continue)
            }
            Mode::PickModel(menu) => {
                match key.code {
                    KeyCode::Esc => self.mode = Mode::Input,
                    KeyCode::Char('c') if ctrl => self.mode = Mode::Input,
                    KeyCode::Backspace => {
                        menu.pop();
                    }
                    KeyCode::Up => menu.up(),
                    KeyCode::Down => menu.down(),
                    KeyCode::Char(c) if !ctrl => menu.push(c),
                    KeyCode::Enter => {
                        if let Some(path) = menu.current().cloned() {
                            self.mode = Mode::Input;
                            self.switch(Some(path), self.backend, raw)?;
                        }
                    }
                    _ => {}
                }
                Ok(Flow::Continue)
            }
            Mode::PickBackend(menu) => {
                match key.code {
                    KeyCode::Esc => self.mode = Mode::Input,
                    KeyCode::Char('c') if ctrl => self.mode = Mode::Input,
                    KeyCode::Backspace => {
                        menu.pop();
                    }
                    KeyCode::Up => menu.up(),
                    KeyCode::Down => menu.down(),
                    KeyCode::Char(c) if !ctrl => menu.push(c),
                    KeyCode::Enter => {
                        if let Some(choice) = menu.current().copied() {
                            self.mode = Mode::Input;
                            self.switch(None, choice, raw)?;
                        }
                    }
                    _ => {}
                }
                Ok(Flow::Continue)
            }
            Mode::Ask { action, editor, .. } => {
                match key.code {
                    KeyCode::Esc => self.mode = Mode::Input,
                    KeyCode::Char('c') if ctrl => self.mode = Mode::Input,
                    KeyCode::Enter => {
                        let (action, text) = (*action, editor.text());
                        self.mode = Mode::Input;
                        self.apply_ask(action, &text)?;
                    }
                    _ => {
                        editor.handle(key);
                    }
                }
                Ok(Flow::Continue)
            }
            Mode::Panel(tab) => {
                let tab = *tab;
                match key.code {
                    KeyCode::Esc => self.mode = Mode::Input,
                    KeyCode::Char('o' | 'c') if ctrl => self.mode = Mode::Input,
                    KeyCode::Left => self.open_tab(tab.step(false))?,
                    KeyCode::Right | KeyCode::Tab => self.open_tab(tab.step(true))?,
                    KeyCode::Char(c @ '1'..='6') => {
                        let index =
                            usize::from(u8::try_from(c).unwrap_or(b'1').saturating_sub(b'1'));
                        self.open_tab(Tab::ALL.get(index).copied().unwrap_or(tab))?;
                    }
                    _ => {}
                }
                Ok(Flow::Continue)
            }
        }
    }

    fn input_key(
        &mut self,
        key: KeyEvent,
        ctrl: bool,
        raw: &mut RawMode,
    ) -> Result<Flow, CliError> {
        let plain = key.modifiers.is_empty();
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if !self.editor.is_empty() {
                    self.editor.clear();
                } else if self.exit_armed {
                    return Ok(Flow::Quit(INTERRUPTED_EXIT_CODE));
                } else {
                    self.exit_armed = true;
                    self.notice = Some("press ctrl+c again to exit".to_string());
                }
            }
            KeyCode::Char('d') if ctrl && self.editor.is_empty() => return Ok(Flow::Quit(0)),
            KeyCode::Char('o') if ctrl => self.open_tab(Tab::Reply)?,
            KeyCode::Char('/') if plain && self.editor.is_empty() => {
                self.mode = Mode::Palette(actions::palette(self.dialog.talk()))
            }
            KeyCode::Enter if plain => {
                let text = self.editor.text();
                if !text.trim().is_empty() {
                    self.submit(&text, raw)?;
                }
            }
            _ => {
                self.editor.handle(key);
            }
        }
        Ok(Flow::Continue)
    }
}
