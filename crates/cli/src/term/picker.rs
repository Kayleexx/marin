use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use super::menu::{Item, Menu};
use super::raw::RawMode;
use super::screen::{Kind, Line, Screen, display_width, menu_lines};
use crate::style::Style;

const MAX_ROWS: usize = 6;

/// A one-off searchable list. Returns None when the user cancels.
pub fn pick<T>(style: Style, title: &str, items: Vec<Item<T>>) -> io::Result<Option<T>> {
    let screen = Screen::new(style);
    let mut menu = Menu::new(items);
    let _raw = RawMode::enable()?;
    let choice = loop {
        let mut lines = vec![Line::new(Kind::Strong, format!("{title} {}", menu.query()))];
        lines.extend(menu_lines(&menu, MAX_ROWS));
        lines.push(Line::new(
            Kind::Dim,
            "type to search · ↑/↓ choose · enter select · esc cancel",
        ));
        let column = display_width(title)
            .saturating_add(display_width(menu.query()))
            .saturating_add(1);
        screen.draw(&lines, column)?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => break None,
            KeyCode::Char('c' | 'd') if ctrl => break None,
            KeyCode::Enter => match menu.current() {
                Some(_) => break Some(()),
                None => continue,
            },
            KeyCode::Up => menu.up(),
            KeyCode::Down => menu.down(),
            KeyCode::Backspace => {
                menu.pop();
            }
            KeyCode::Char(c) if !ctrl => menu.push(c),
            _ => {}
        }
    };
    screen.print(&[])?;
    Ok(choice.and_then(|()| menu.into_current()))
}
