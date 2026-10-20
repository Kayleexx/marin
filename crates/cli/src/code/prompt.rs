use std::io::{self, Write};
use std::time::Duration;

use crossterm::cursor::MoveToColumn;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal::{Clear, ClearType};

use crate::style::Style;
use crate::term::editor::Editor;
use crate::term::raw::RawMode;
use crate::term::screen::width;
use crate::term::view::{PROMPT, input_line};

pub enum Input {
    Line(String),
    Quit,
}

fn draw(editor: &Editor, style: &Style) -> io::Result<()> {
    let (line, column) = input_line(editor, PROMPT, width());
    let line = line.replacen("you >", &style.periwinkle("you >"), 1);
    let mut out = io::stdout().lock();
    queue!(
        out,
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        Print(line),
        MoveToColumn(u16::try_from(column).unwrap_or(0))
    )?;
    out.flush()
}

/// One question typed at the prompt, with the usual editing keys and history. Ctrl-C clears the line, or leaves when
/// the line is already empty.
pub fn read(editor: &mut Editor, style: &Style) -> io::Result<Input> {
    let _raw = RawMode::enable()?;
    // Keys typed while an answer streamed were echoed by the terminal and are not part of this question.
    while event::poll(Duration::ZERO)? {
        event::read()?;
    }
    loop {
        draw(editor, style)?;
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Char('c') if ctrl && editor.is_empty() => return finish(Input::Quit),
                    KeyCode::Char('c') if ctrl => editor.clear(),
                    KeyCode::Char('d') if ctrl && editor.is_empty() => return finish(Input::Quit),
                    KeyCode::Enter if !key.modifiers.contains(KeyModifiers::ALT) => {
                        let text = editor.text();
                        editor.clear();
                        return finish(Input::Line(text));
                    }
                    _ => {
                        editor.handle(key);
                    }
                }
            }
            Event::Paste(text) => editor.insert(&text),
            _ => {}
        }
    }
}

fn finish(input: Input) -> io::Result<Input> {
    let mut out = io::stdout().lock();
    write!(out, "\r\n")?;
    out.flush()?;
    Ok(input)
}
