use std::fs::File;
use std::io::{self, ErrorKind, Read, Write};
use std::path::Path;

use crate::CliError;
use crate::render::{ContentKind, Renderer, SAMPLE_BYTES, Theme, Utf8Tail, detect};
use crate::style::{stdin_is_terminal, terminal_width};
use crate::term::caps::Caps;
use crate::term::screen;

fn read_some(input: &mut dyn Read, buf: &mut [u8]) -> io::Result<usize> {
    loop {
        match input.read(buf) {
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

fn write_out(text: &str) -> Result<(), CliError> {
    if text.is_empty() {
        return Ok(());
    }
    let mut out = io::stdout().lock();
    out.write_all(text.as_bytes())?;
    out.flush()?;
    Ok(())
}

/// Shows `path` (or standard input) with the formatting its kind calls for. Text is rendered as it arrives, so a
/// stream can be piped in. Content that is not text is reported, never printed.
pub fn run(path: Option<&Path>, kind: Option<&str>, no_color: bool) -> Result<(), CliError> {
    let tty = std::io::IsTerminal::is_terminal(&io::stdout());
    let theme = if tty {
        Theme::new(&Caps::detect(no_color))
    } else {
        Theme::plain()
    };
    let width = if tty {
        screen::width().saturating_sub(1)
    } else {
        terminal_width().saturating_sub(1)
    };
    let explicit_stdin = path.is_some_and(|p| p.as_os_str() == "-");
    let file = path.filter(|p| p.as_os_str() != "-");
    let mut input: Box<dyn Read> = match file {
        Some(p) => Box::new(File::open(p).map_err(|e| CliError::Input {
            path: p.display().to_string(),
            reason: e.to_string(),
        })?),
        None if stdin_is_terminal() && !explicit_stdin => return Err(CliError::NoInput),
        None => Box::new(io::stdin().lock()),
    };
    let describe = |e: io::Error| CliError::Input {
        path: file.map_or_else(|| "standard input".into(), |p| p.display().to_string()),
        reason: e.to_string(),
    };
    let mut buf = vec![0_u8; SAMPLE_BYTES];
    let count = read_some(&mut *input, &mut buf).map_err(describe)?;
    let first = buf.get(..count).unwrap_or(&[]);
    if first.is_empty() {
        return Ok(());
    }
    let name = file.and_then(Path::file_name).and_then(|n| n.to_str());
    let kind = kind.map_or_else(|| detect(name, first), ContentKind::from_name);
    if kind == ContentKind::Binary {
        let label = name.unwrap_or("input");
        return write_out(&Renderer::notice(
            theme,
            &format!("{label}: binary content, not shown"),
        ));
    }
    let mut renderer = Renderer::new(kind, theme, width);
    let mut tail = Utf8Tail::default();
    renderer.feed(&tail.push(first));
    write_out(&renderer.take())?;
    loop {
        let count = read_some(&mut *input, &mut buf).map_err(describe)?;
        if count == 0 {
            break;
        }
        renderer.feed(&tail.push(buf.get(..count).unwrap_or(&[])));
        write_out(&renderer.take())?;
    }
    renderer.feed(&tail.finish());
    renderer.finish();
    write_out(&renderer.take())
}
