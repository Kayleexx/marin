#![forbid(unsafe_code)]

pub mod arg_types;
pub mod args;
pub mod bench;
pub mod chat;
pub mod checkpoint;
pub mod choose;
pub mod code;
pub mod commands;
pub mod config;
pub mod dialog;
pub mod error;
pub mod guess;
pub mod inspect;
pub mod json_out;
pub mod load;
pub mod marin_args;
pub mod memory;
pub mod perf;
pub mod plan;
pub mod probe;
pub mod progress;
pub mod quality;
pub mod quality_stats;
pub mod render;
pub mod repl;
pub mod restart;
pub mod saved;
pub mod server;
pub mod session;
pub mod setup;
pub mod show;
pub mod signal;
pub mod snapshot;
pub mod style;
pub mod telemetry;
pub mod term;
pub mod trace;
pub mod turns;
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

mod verify_hybrid;
mod verify_reference;
