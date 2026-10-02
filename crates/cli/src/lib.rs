#![forbid(unsafe_code)]

pub mod args;
pub mod bench;
pub mod chat;
pub mod commands;
pub mod config;
pub mod error;
pub mod inspect;
pub mod load;
pub mod marin_args;
pub mod memory;
pub mod progress;
pub mod repl;
pub mod session;
pub mod signal;
pub mod style;
pub mod telemetry;
pub mod term;
pub mod units;
pub mod verify;

use error::CliError;
use style::Style;

pub fn report(error: &CliError, style: &Style) {
    eprintln!("{} {error}", style.red("error:"));
    if let Some(hint) = error.hint() {
        eprintln!("{} {hint}", style.yellow("hint:"));
    }
}
