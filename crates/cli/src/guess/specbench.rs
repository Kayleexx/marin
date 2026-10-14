use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use metrics::Sampler as GpuSampler;
use model::{BackendChoice, Precision};
use runtime::{GenerateParams, Outcome, Speculation, generate};
use sampler::{Sampler, SamplerConfig};

use super::generate::read_prompts;
use super::specbench_report::{Block, FORMAT, Meta, RunRecord, Saved, record};
use super::specbench_summary::{prompts, table};
use crate::CliError;
use crate::chat::Chat;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::session::{Session, Settings};
use crate::signal::Interrupt;
use crate::style::Style;

pub struct Options {
    pub prompts: PathBuf,
    pub max_tokens: usize,
    pub ctx: usize,
    /// Rounds timed; one more is run first and thrown away.
    pub rounds: usize,
    /// Which ways of choosing tokens to time: `greedy`, `chat` (the chat defaults, fixed seed) or both.
    pub sampling: Vec<String>,
    pub energy: bool,
    pub save: Option<PathBuf>,
    pub precision: Precision,
}

#[derive(Clone)]
struct Mode {
    label: String,
    sampling: SamplerConfig,
    speculate: bool,
}

fn sampling_config(name: &str) -> Result<SamplerConfig, CliError> {
    let (temperature, top_k, top_p) = match name {
        "greedy" => (0.0, 0, 1.0),
        "chat" => (0.7, 20, 0.8),
        other => {
            return Err(CliError::BadOption(format!(
                "--sampling takes greedy or chat, not `{other}`"
            )));
        }
    };
    Ok(SamplerConfig {
        temperature,
        top_k,
        top_p,
        seed: 1,
    })
}

fn modes(options: &Options) -> Result<Vec<Mode>, CliError> {
    let mut modes = Vec::new();
    for name in &options.sampling {
        for speculate in [false, true] {
            modes.push(Mode {
                label: format!("{name}/{}", if speculate { "spec" } else { "plain" }),
                sampling: sampling_config(name)?,
                speculate,
            });
        }
    }
    Ok(modes)
}

/// Mean GPU power from the first token to the end of the reply: the energy per token is that power times the time
/// the decode took.
fn block_of(watch: Option<GpuSampler>, outcome: &Outcome) -> Block {
    let stats = watch.and_then(|w| w.stop().ok());
    let power = stats.and_then(|s| s.power_watts).map(|p| p.mean);
    let tokens = outcome.generated.saturating_sub(1);
    Block {
        power_w: power,
        clock_mhz: stats.and_then(|s| s.clock_mhz).map(|c| c.mean),
        joules_per_token: power
            .filter(|_| tokens > 0)
            .map(|w| w * outcome.decode.as_secs_f64() / tokens as f64),
    }
}

pub fn run(
    model_dir: &Path,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    let modes = modes(options)?;
    let out = options.save.as_deref().map(JsonOut::create).transpose()?;
    let prompt_files = read_prompts(&options.prompts)?;
    let loaded = load(
        model_dir,
        Some(options.ctx),
        options.ctx,
        BackendChoice::Cuda,
        options.precision,
        style,
        true,
    )?;
    let quantization = options
        .precision
        .weights
        .map_or_else(|| "none".to_string(), |q| q.label());
    let (model, backend_name) = (loaded.name.clone(), loaded.model.backend_name());
    let settings = Settings {
        max_tokens: options.max_tokens,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(1),
    };
    let mut session = Session::new(loaded, settings, *style)?;
    let mut chat = Chat::new(&session);
    let mut encoded = Vec::new();
    for file in &prompt_files {
        let ids = chat
            .prompt_tokens(&session, &file.text)
            .map_err(|e| CliError::BadOption(format!("{}/{}: {e}", file.class, file.name)))?;
        encoded.push(ids);
    }
    let usable_vocab = session.loaded.dir.tokenizer.vocab_size();
    let stop = session.loaded.stop_tokens();
    let (engine, kv) = session.engine_and_kv();
    let cancel = interrupt.cancel_flag();
    let mut records: Vec<RunRecord> = Vec::new();
    let total = options.rounds.saturating_add(1);
    for round in 0..total {
        for (index, (file, ids)) in prompt_files.iter().zip(&encoded).enumerate() {
            let turn = round
                .saturating_add(index)
                .checked_rem(modes.len())
                .unwrap_or(0);
            let mut done: Vec<(&Mode, Outcome, Vec<u32>, Block)> = Vec::new();
            for mode in modes.iter().cycle().skip(turn).take(modes.len()) {
                if cancel.load(Ordering::SeqCst) {
                    return Err(CliError::Cancelled);
                }
                let params = GenerateParams {
                    max_tokens: options.max_tokens,
                    stop_tokens: stop.clone(),
                    usable_vocab,
                    speculation: mode.speculate.then_some(Speculation::MEASURED),
                };
                let mut watch = None;
                let mut tokens = Vec::new();
                let outcome = generate(
                    engine,
                    kv,
                    &mut Sampler::new(mode.sampling)?,
                    ids,
                    &params,
                    cancel,
                    |token| {
                        if watch.is_none() && options.energy {
                            watch = GpuSampler::start().ok();
                        }
                        tokens.push(token);
                    },
                )?;
                let block = block_of(watch, &outcome);
                done.push((mode, outcome, tokens, block));
            }
            if round == 0 {
                continue;
            }
            for (mode, outcome, tokens, block) in &done {
                let baseline = mode.label.replace("spec", "plain");
                let same = done
                    .iter()
                    .find(|(m, ..)| m.label == baseline)
                    .is_some_and(|(_, _, plain, _)| plain == tokens);
                let meta = Meta {
                    class: &file.class,
                    name: &file.name,
                    round,
                    rep: 0,
                    prompt_tokens: ids.len(),
                };
                records.push(record(&meta, &mode.label, outcome, block, same));
            }
            eprint!(
                "\rround {round}/{} prompt {}/{}   ",
                options.rounds,
                index.saturating_add(1),
                prompt_files.len()
            );
        }
    }
    eprintln!();
    let mut text = format!(
        "speculation test: {model} · {quantization} · {backend_name} · up to {} tokens per reply, {} rounds\n\n",
        options.max_tokens, options.rounds
    );
    text.push_str(&prompts(&records));
    text.push_str("\n\n");
    text.push_str(&table(&records));
    text.push('\n');
    std::io::stdout().lock().write_all(text.as_bytes())?;
    if let (Some(out), Some(path)) = (out, &options.save) {
        out.finish(&Saved {
            format: FORMAT,
            model,
            quantization,
            backend: backend_name,
            max_tokens: options.max_tokens,
            runs: records,
        })?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
