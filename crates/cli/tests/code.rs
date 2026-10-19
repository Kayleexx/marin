//! `marin code` as a user meets it when the setup is wrong: the message says what to do.

use std::io;
use std::net::TcpListener;
use std::process::{Command, Stdio};

fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map_or(1, |a| a.port())
}

fn code(args: &[&str]) -> io::Result<(bool, String, String)> {
    let out = Command::new(env!("CARGO_BIN_EXE_marin"))
        .arg("code")
        .args(args)
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

#[test]
fn without_a_server_the_error_says_how_to_start_one() -> io::Result<()> {
    let (ok, _, stderr) = code(&["--port", &free_port().to_string()])?;
    assert!(!ok);
    assert!(
        stderr.contains("no Marin server answers at 127.0.0.1:"),
        "{stderr}"
    );
    assert!(
        stderr.contains("hint:") && stderr.contains("marin --model") && stderr.contains("serve"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn the_help_describes_the_command() -> io::Result<()> {
    let (_, text, _) = code(&["--help"])?;
    assert!(
        text.contains("--port") && text.contains("--log") && text.contains("[PATH]"),
        "{text}"
    );
    Ok(())
}
