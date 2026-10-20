use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::time::Duration;

use crate::dialog;
use crate::style::Style;
use crate::term::caps::Caps;
use crate::term::reply::ReplyWriter;
use crate::term::screen::{display_width, fit, width};
use crate::units;

pub struct Stats {
    pub tools: usize,
    pub generated: usize,
    /// Tokens the model read in its last request: how much of the window the conversation fills.
    pub context_used: usize,
    pub elapsed: Duration,
}

/// What the terminal shows. All of it is plain text when output is piped.
#[derive(Clone, Copy)]
pub struct Ui {
    style: Style,
    caps: Caps,
    tty: bool,
}

impl Ui {
    pub fn new(no_color: bool) -> Self {
        let tty = io::stdout().is_terminal();
        Self {
            style: if tty {
                Style::detect(no_color)
            } else {
                Style::plain()
            },
            caps: Caps::detect(no_color || !tty),
            tty,
        }
    }

    /// For the chat's own terminal, which has already settled its colors.
    pub fn embedded(style: Style, caps: Caps) -> Self {
        Self {
            style,
            caps,
            tty: true,
        }
    }

    pub fn piped(style: Style) -> Self {
        Self {
            style,
            caps: Caps::detect(true),
            tty: false,
        }
    }

    pub fn style(&self) -> Style {
        self.style
    }

    pub fn interactive(&self) -> bool {
        self.tty
    }

    pub fn banner(&self, model: &str, context: usize, root: &Path) {
        let s = &self.style;
        let home = std::env::var("HOME").unwrap_or_default();
        let shown = match root.strip_prefix(&home) {
            Ok(rest) if !home.is_empty() => format!("~/{}", rest.display()),
            _ => root.display().to_string(),
        };
        println!(
            "\n  {}  {}\n  {}\n  {}\n",
            s.bold(&s.cyan("marin code")),
            s.cyan(model),
            s.dim(&format!(
                "{shown} · read-only · context {}",
                units::count(context as u64)
            )),
            s.dim(
                "Ask where something lives, how it works or why it fails. /help · Ctrl-D to leave."
            )
        );
    }

    pub fn help(&self) {
        let s = &self.style;
        let row = |keys: &str, what: &str| {
            println!("  {}  {}", s.cyan(&format!("{keys:<18}")), s.dim(what))
        };
        println!();
        for (command, what, local_only) in dialog::COMMAND_HELP {
            if !local_only {
                row(command, what);
            }
        }
        row("/help", "show this list");
        row("Alt+Enter", "new line in a question");
        row("↑ ↓", "earlier questions");
        row("Ctrl-C", "stop an answer in progress");
        println!();
    }

    pub fn you(&self, text: &str) {
        if !self.tty {
            println!("> {text}");
        }
    }

    /// A tool the model used, as one quiet line.
    pub fn event(&self, label: &str, note: &str, ok: bool) {
        let s = &self.style;
        let room = width()
            .saturating_sub(display_width(note))
            .saturating_sub(8);
        let label = fit(label, room.max(20));
        let note = if ok { s.dim(note) } else { s.yellow(note) };
        println!("  {} {}  {}", s.cyan("›"), label, note);
    }

    pub fn notice(&self, text: &str) {
        println!("  {}", self.style.dim(&format!("· {text}")));
    }

    pub fn warn(&self, text: &str) {
        println!("  {} {text}", self.style.yellow("!"));
    }

    pub fn error(&self, text: &str, hint: Option<&str>) {
        println!("  {} {text}", self.style.red("error:"));
        if let Some(hint) = hint {
            println!("  {} {hint}", self.style.yellow("hint:"));
        }
    }

    pub fn footer(&self, stats: &Stats, context: usize) {
        let tools = match stats.tools {
            0 => "no tools".to_owned(),
            1 => "1 tool".to_owned(),
            n => format!("{n} tools"),
        };
        let line = format!(
            "{tools} · {} tokens written · context {}/{} · {}",
            units::count(stats.generated as u64),
            units::count(stats.context_used as u64),
            units::count(context as u64),
            units::seconds(stats.elapsed)
        );
        println!("\n  {}\n", self.style.dim(&line));
    }

    /// Starts a model reply: on a terminal a quiet marker shows until the first words arrive.
    pub fn reply(&self) -> io::Result<Reply> {
        Ok(Reply {
            writer: ReplyWriter::new(&self.style, &self.caps, self.tty)?,
            started: false,
        })
    }
}

pub struct Reply {
    writer: ReplyWriter,
    started: bool,
}

impl Reply {
    pub fn push(&mut self, piece: &str) -> io::Result<()> {
        if piece.is_empty() || (!self.started && piece.trim().is_empty()) {
            return Ok(());
        }
        self.started = true;
        self.writer.write_all(piece.as_bytes())
    }

    /// The reply is over: whatever is held is written and the cursor rests at a line start.
    pub fn end(mut self) -> io::Result<()> {
        if self.started {
            self.writer.finish_markdown()?;
        }
        self.writer.finish_placeholder(false)
    }
}
