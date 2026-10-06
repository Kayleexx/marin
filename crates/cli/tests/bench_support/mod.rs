use std::io;
use std::process::{Command, Output};

use crate::common::fixtures;

/// `marin-run bench` on the tiny test model, on the CPU, plus `args`.
pub fn bench(args: &[&str]) -> io::Result<Output> {
    let model = fixtures().join("tiny-qwen3");
    Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .arg("bench")
        .arg(model)
        .args(["--backend", "cpu"])
        .args(args)
        .env("NO_COLOR", "1")
        .output()
}
