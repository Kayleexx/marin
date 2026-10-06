use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use model::{BackendChoice, QuantScheme};
use runtime::StopReason;

use super::data::{FORMAT, PromptRun, Reply, Transcripts};
use crate::CliError;
use crate::chat::Chat;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::session::{Session, Settings};
use crate::signal::Interrupt;
use crate::style::Style;

pub struct Options {
    /// A folder of folders: each folder is a class of prompt, each `.txt` file in it one user message.
    pub prompts: PathBuf,
    pub max_tokens: usize,
    pub ctx: usize,
    pub out: PathBuf,
    pub quant: Option<QuantScheme>,
}

pub(super) struct Prompt {
    pub(super) class: String,
    pub(super) name: String,
    pub(super) text: String,
}

fn input_error(path: &Path, reason: impl ToString) -> CliError {
    CliError::Input {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}

fn sorted_entries(dir: &Path) -> Result<Vec<PathBuf>, CliError> {
    let mut paths = fs::read_dir(dir)
        .map_err(|e| input_error(dir, e))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| input_error(dir, e))?;
    paths.sort();
    Ok(paths)
}

fn file_name(path: &Path) -> String {
    path.file_stem()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

pub(super) fn read_prompts(root: &Path) -> Result<Vec<Prompt>, CliError> {
    let mut prompts = Vec::new();
    for class_dir in sorted_entries(root)?.into_iter().filter(|p| p.is_dir()) {
        for file in sorted_entries(&class_dir)? {
            if file.extension().is_some_and(|e| e == "txt") {
                prompts.push(Prompt {
                    class: file_name(&class_dir),
                    name: file_name(&file),
                    text: fs::read_to_string(&file).map_err(|e| input_error(&file, e))?,
                });
            }
        }
    }
    if prompts.is_empty() {
        return Err(input_error(
            root,
            "no .txt prompts found in its sub-folders",
        ));
    }
    Ok(prompts)
}

/// The two ways tokens are chosen: greedy, and what `marin` chat uses by default (fixed seed).
fn modes(max_tokens: usize) -> [(&'static str, Settings); 2] {
    let settings = |temperature, top_k, top_p| Settings {
        max_tokens,
        temperature,
        top_k,
        top_p,
        seed: Some(1),
    };
    [
        ("greedy", settings(0.0, 0, 1.0)),
        ("chat", settings(0.7, 20, 0.8)),
    ]
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    let out = JsonOut::create(&options.out)?;
    let prompts = read_prompts(&options.prompts)?;
    let loaded = load(
        model_dir,
        Some(options.ctx),
        options.ctx,
        backend,
        options.quant,
        style,
        true,
    )?;
    let quantization = options
        .quant
        .map_or_else(|| "none".to_string(), |q| q.label());
    let (model, backend_name) = (loaded.name.clone(), loaded.model.backend_name());
    let modes = modes(options.max_tokens);
    let [(_, first), _] = modes;
    let mut session = Session::new(loaded, first, *style)?;
    // Each mode answers the same prompt read from scratch, as recorded transcripts always were.
    session.reuse = false;
    let mut chat = Chat::new(&session);
    let mut runs = Vec::new();
    'prompts: for (index, prompt) in prompts.iter().enumerate() {
        let mut replies = Vec::new();
        let mut tokens = Vec::new();
        for (mode, settings) in &modes {
            chat.clear();
            session.settings = *settings;
            let reply = match chat.ask(&mut session, &prompt.text, interrupt, &mut io::sink()) {
                Err(CliError::PromptTooLong {
                    tokens: needed,
                    ctx,
                }) => {
                    eprintln!(
                        "skipped {}/{}: {needed} tokens do not fit the {ctx}-token window",
                        prompt.class, prompt.name
                    );
                    continue 'prompts;
                }
                other => other?,
            };
            if reply.outcome.stop == StopReason::Cancelled {
                return Err(CliError::Cancelled);
            }
            tokens.push(reply.tokens.len());
            replies.push(Reply {
                mode: (*mode).to_string(),
                tokens: reply.tokens,
            });
        }
        eprintln!(
            "[{}/{}] {}/{}: {} prompt tokens, replies of {:?} tokens",
            index.saturating_add(1),
            prompts.len(),
            prompt.class,
            prompt.name,
            session.last_prompt.len(),
            tokens
        );
        runs.push(PromptRun {
            class: prompt.class.clone(),
            name: prompt.name.clone(),
            prompt: session.last_prompt.clone(),
            replies,
        });
    }
    out.finish(&Transcripts {
        format: FORMAT,
        model,
        quantization,
        backend: backend_name,
        prompts: runs,
    })?;
    eprintln!("wrote {}", options.out.display());
    Ok(())
}
