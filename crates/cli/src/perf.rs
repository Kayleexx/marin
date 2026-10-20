//! `/bench` and `/verify` inside the chat: the benchmark and the reference check of `marin-run`, run on the model this
//! session has already loaded. Both work in caches of their own, so the conversation, its cache and its checkpoint are
//! untouched, and the model is not loaded again.

mod reference;

use std::path::Path;

use crate::CliError;
use crate::bench::{self, Options, measure};
use crate::memory::available_bytes;
use crate::session::Session;
use crate::signal::Interrupt;
use crate::units;
use crate::verify::{self, compares_tensors};

/// What a command has to show, and whether it found the model wanting.
pub struct Output {
    pub lines: Vec<String>,
    pub failed: bool,
}

const USAGE: &str = "usage: /bench [PROMPT [OUTPUT]] [runs=N] [profile] [quick]  (default: 128 64 runs=3, one warmup)";
const MAX_RUNS: usize = 20;

/// The workload a `/bench` line asks for.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Workload {
    pub prompt: usize,
    pub output: usize,
    pub runs: usize,
    pub warmup: usize,
    pub profile: bool,
}

pub(crate) fn parse_workload(args: &str) -> Result<Workload, CliError> {
    let mut work = Workload {
        prompt: 128,
        output: 64,
        runs: 3,
        warmup: 1,
        profile: false,
    };
    let bad = |why: String| CliError::BadOption(format!("{why}; {USAGE}"));
    let mut numbers = 0_usize;
    for word in args.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            match numbers {
                0 => work.prompt = n,
                1 => work.output = n,
                _ => return Err(bad(format!("`{word}` is one number too many"))),
            }
            numbers = numbers.saturating_add(1);
        } else if let Some(n) = word.strip_prefix("runs=") {
            work.runs = n
                .parse()
                .map_err(|_| bad(format!("`{word}` needs a number")))?;
        } else if word == "profile" {
            work.profile = true;
        } else if word == "quick" {
            (work.runs, work.warmup) = (1, 0);
        } else {
            return Err(bad(format!("`{word}` is not understood")));
        }
    }
    if work.prompt == 0 || work.output < 2 || !(1..=MAX_RUNS).contains(&work.runs) {
        return Err(bad(format!(
            "the prompt needs at least 1 token, the output at least 2 and runs 1 to {MAX_RUNS}"
        )));
    }
    Ok(work)
}

/// Times the loaded model on a fixed prompt and says what was measured, apart from what loading it cost at startup.
pub fn bench(session: &Session, interrupt: &Interrupt, args: &str) -> Result<Output, CliError> {
    let work = parse_workload(args)?;
    let needed = work.prompt.saturating_add(work.output).saturating_add(1);
    if needed > session.ctx {
        return Err(CliError::BadOption(format!(
            "that workload needs {needed} tokens and this session's window is {}: ask for less, or start marin with a larger --ctx",
            session.ctx
        )));
    }
    let loaded = &session.loaded;
    let options = Options {
        prompt_tokens: work.prompt,
        output_tokens: work.output,
        runs: work.runs,
        warmup: work.warmup,
        profile: work.profile,
        sample_gpu: false,
        sampling: measure::GREEDY,
        token_pause_ms: 0.0,
        json: None,
        precision: loaded.precision,
    };
    options.validate()?;
    interrupt.generating();
    let record = loaded
        .model
        .new_cache(needed)
        .map_err(CliError::from)
        .and_then(|mut cache| {
            let load = bench::LoadStats {
                seconds: loaded.took.as_secs_f64(),
                rss_growth: None,
                disk_read: None,
                gpu_upload: None,
                gpu_growth: None,
            };
            let stored = loaded
                .precision
                .weights
                .map_or_else(|| "none".to_string(), |q| q.label());
            bench::measure_loaded(
                loaded,
                &mut cache,
                load,
                stored,
                &options,
                interrupt.cancel_flag(),
            )
        });
    interrupt.busy();
    let record = record?;
    let mut text = bench::render::header(&record);
    text.push_str(&format!(
        "not measured by this run: loading the model took {:.2} s at startup; it was not loaded again.\nthis run kept its own cache beside the chat's: {} held for {} tokens (weights {} are shared).\n",
        record.load.seconds,
        record.load.kv_held_bytes.map_or_else(|| "an unknown amount".into(), units::bytes),
        needed,
        units::bytes(record.load.weights_bytes),
    ));
    text.push_str(&bench::render::results_text(&record));
    text.push_str("  (peak process memory is the highest since marin started, loading included)\n");
    Ok(Output {
        lines: text.lines().map(str::to_owned).collect(),
        failed: false,
    })
}

fn shorten(text: &str) -> String {
    match text.char_indices().nth(50) {
        Some((at, _)) => format!("{}…", text.get(..at).unwrap_or(text)),
        None => text.to_owned(),
    }
}

/// Checks the loaded model against a float32 reference: the file named, else one found for this model.
pub fn verify(
    session: &Session,
    interrupt: &Interrupt,
    args: &str,
    here: &Path,
) -> Result<Output, CliError> {
    let loaded = &session.loaded;
    let (path, reference) = match args.trim() {
        "" => reference::find(loaded, here, session.ctx)?,
        named => reference::named(loaded, &here.join(named), session.ctx)?,
    };
    let total = reference.cases();
    let mut lines = vec![format!(
        "verifying {} on {} against {} ({} weights, float32 math); the loaded model is used as it is",
        loaded.name,
        loaded.model.backend_name(),
        path.display(),
        reference.weights
    )];
    interrupt.generating();
    let failed = verify::check_all(loaded, &reference, interrupt.cancel_flag(), |n, report| {
        let label = format!(
            "case {}/{total}  {:?}",
            n.saturating_add(1),
            shorten(&report.prompt)
        );
        match &report.failure {
            None => lines.push(format!(
                "  pass {label}\n      {} steps, token choices match, max logit diff {:.2e}, max layer diff {:.2e}",
                report.steps, report.max_logit_diff, report.max_norm_diff
            )),
            Some(why) => lines.push(format!("  FAIL {label}\n      {why}")),
        }
        if compares_tensors(&loaded.model) {
            lines.push(format!(
                "      full tensors: max normalized {:.3e}, relative RMS {:.3e}",
                report.max_tensor_normalized, report.max_tensor_rms
            ));
        }
    });
    interrupt.busy();
    let failed = failed?;
    lines.push(if failed {
        "result: FAIL".into()
    } else {
        format!(
            "result: PASS ({total} {})",
            if total == 1 { "case" } else { "cases" }
        )
    });
    Ok(Output {
        lines: lines
            .iter()
            .flat_map(|l| l.lines().map(str::to_owned))
            .collect(),
        failed,
    })
}

/// Free memory a file of `size` bytes needs to be read and parsed, or why it should not be.
pub(crate) fn can_read(path: &Path, size: u64) -> Result<(), CliError> {
    let need = size.saturating_mul(3);
    match available_bytes() {
        Some(free) if free < need => Err(CliError::Reference {
            path: path.display().to_string(),
            reason: format!(
                "reading it needs about {} of free memory and {} is available",
                units::bytes(need),
                units::bytes(free)
            ),
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests;
