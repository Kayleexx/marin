use std::io::{self, BufRead, Write};

use crate::CliError;
use crate::commands::{self, Next};
use crate::inspect::is_base_model;
use crate::session::Session;
use crate::signal::Interrupt;
use crate::style::stdin_is_terminal;
use crate::units;

pub fn banner(s: &Session) -> String {
    let cfg = s.loaded.model.config();
    let title = format!(
        "{}  {}   {} · bf16 weights · f32 math · ctx {}",
        s.style.bold("marin"),
        s.style.cyan(&s.loaded.name),
        s.loaded.model.backend_name(),
        units::count(s.ctx as u64)
    );
    let note = if is_base_model(&s.loaded.name) {
        "This is a base model: it continues text, it does not answer questions.\n Try:  The three laws of thermodynamics are".to_string()
    } else {
        "Your text is continued as plain text; chat templates are not supported yet.".to_string()
    };
    let loaded = format!(
        "{} layers · loaded in {}",
        cfg.layers,
        units::seconds(s.loaded.took)
    );
    format!(
        "\n {title}\n {}\n {note}\n {}\n",
        s.style.dim(&loaded),
        s.style.dim("/help for commands, Ctrl-D to leave")
    )
}

pub fn run(session: &mut Session, interrupt: &Interrupt) -> Result<(), CliError> {
    let interactive = stdin_is_terminal();
    let stdin = io::stdin();
    let mut input = stdin.lock();
    loop {
        if interactive {
            print!("{} ", session.style.green(">"));
            io::stdout().flush()?;
        }
        interrupt.waiting_for_input();
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            if interactive {
                println!();
            }
            return Ok(());
        }
        interrupt.busy();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match handle(session, interrupt, line) {
            Ok(Next::Quit) => return Ok(()),
            Ok(Next::Continue) => {}
            Err(e) if e.is_broken_pipe() => return Err(e),
            Err(e) => crate::report(&e, &session.style),
        }
    }
}

fn handle(session: &mut Session, interrupt: &Interrupt, line: &str) -> Result<Next, CliError> {
    if line.starts_with('/') {
        return commands::run(session, line);
    }
    let reply = session.complete(line, interrupt, &mut io::stdout().lock())?;
    eprintln!("{}", session.footer(&reply));
    Ok(Next::Continue)
}
