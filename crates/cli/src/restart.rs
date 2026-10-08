//! Times the first token of a prompt after a restart: restoring a saved prefix against reading it again. Run it in
//! separate processes (`save`, then `restore`, and `cold` for the baseline) so every cost of a real restart is counted.

use std::path::{Path, PathBuf};
use std::time::Instant;

use metrics::Sampler as GpuSampler;
use model::{BackendChoice, Precision};

use crate::CliError;
use crate::bench::measure::prompt;
use crate::load::{load, load_for_snapshot};
use crate::session::{Session, Settings};
use crate::signal::Interrupt;
use crate::style::Style;
use crate::units;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Read the prompt, then save the cache.
    Save,
    /// Read the prompt after the saved start, using the snapshot.
    Restore,
    /// Read the whole prompt with no snapshot.
    Cold,
}

pub struct Options {
    pub mode: Mode,
    /// Tokens of the saved prefix; `Restore` and `Cold` add `extra` more.
    pub tokens: usize,
    pub extra: usize,
    /// Tokens generated after the prompt (the first is the one timed).
    pub reply_tokens: usize,
    pub file: PathBuf,
    pub ctx: usize,
    pub energy: bool,
    pub precision: Precision,
}

fn settings(reply_tokens: usize) -> Settings {
    Settings {
        max_tokens: reply_tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(0),
    }
}

fn ms(duration: std::time::Duration) -> String {
    format!("{:.1}", duration.as_secs_f64() * 1e3)
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    let started = Instant::now();
    let open = if options.mode == Mode::Cold {
        load
    } else {
        load_for_snapshot
    };
    let loaded = open(
        model_dir,
        Some(options.ctx),
        options.ctx,
        backend,
        options.precision,
        style,
        false,
    )?;
    let load_wall = started.elapsed();
    let length = match options.mode {
        Mode::Save => options.tokens,
        _ => options.tokens.saturating_add(options.extra),
    };
    let ids = prompt(&loaded.dir.tokenizer, length)?;
    let mut session = Session::new(loaded, settings(options.reply_tokens), *style)?;
    let mut hash_wait = None;
    if options.mode != Mode::Cold {
        session.use_snapshot(options.file.clone());
        if options.mode == Mode::Restore {
            hash_wait = Some(session.prepare_snapshot());
        }
    }
    let watch = options.energy.then(GpuSampler::start).and_then(Result::ok);
    let timed = Instant::now();
    let reply = session.complete_tokens(ids, interrupt, &mut Vec::new())?;
    let first_token = timed.elapsed().saturating_sub(reply.outcome.decode);
    let watts = watch
        .and_then(|w| w.stop().ok())
        .and_then(|s| s.power_watts)
        .map(|p| p.mean);
    println!("mode            {:?}", options.mode);
    println!(
        "kv type         {}",
        session.loaded.model.kv_format().label()
    );
    println!("prompt tokens   {}", reply.outcome.prompt_tokens);
    println!("model load ms   {}", ms(load_wall));
    if let Some(wait) = hash_wait {
        println!("hash wait ms    {}", ms(wait));
    }
    match &session.restored {
        Some(Ok(restored)) => {
            let t = restored.timing;
            println!(
                "restored        {} tokens, {}",
                restored.ids.len(),
                units::bytes(restored.bytes as u64)
            );
            println!(
                "restore ms      header {} · read {} · checksum {} · copy to gpu {}",
                ms(t.header),
                ms(t.read),
                ms(t.checksum),
                ms(t.copy)
            );
        }
        Some(Err(reject)) => println!("restored        none ({reject})"),
        None => {}
    }
    println!("kept tokens     {}", reply.outcome.reused);
    println!("first token ms  {}", ms(first_token));
    println!("reply ids       {:?}", reply.tokens);
    println!("  suffix pass   {}", ms(reply.outcome.prompt_forward));
    if let Some(watts) = watts {
        println!(
            "gpu watts       {watts:.1} (energy about {:.1} J)",
            watts * first_token.as_secs_f64()
        );
    }
    if options.mode == Mode::Save {
        let saved = session.save_snapshot();
        let next = prompt(
            &session.loaded.dir.tokenizer,
            options.tokens.saturating_add(options.extra),
        )?;
        let live = session.complete_tokens(next, interrupt, &mut Vec::new())?;
        println!("live reply ids  {:?}", live.tokens);
        match saved {
            Ok(Some(saved)) => println!(
                "saved           {} tokens, {} in {} ms",
                saved.tokens,
                units::bytes(saved.bytes as u64),
                ms(saved.took)
            ),
            Ok(None) => println!("saved           nothing (this setup cannot save)"),
            Err(reason) => return Err(CliError::BadOption(format!("save failed: {reason}"))),
        }
    }
    Ok(())
}
