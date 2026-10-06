use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use model::{BackendChoice, QuantScheme};
use runtime::CausalLm;

use crate::CliError;
use crate::json_out::JsonOut;
use crate::load::load;
use crate::quality_stats::{FORMAT, Scores, agreement, fingerprint, score, summarize};
use crate::signal::Interrupt;
use crate::style::{self, Style};

/// Tokens read as one prompt before scoring starts: enough to take the engine's prompt path, and a start of text
/// for the model to work with. Every later position is scored by a one-token pass, as when generating.
const PROMPT_TOKENS: usize = 64;

pub struct Options {
    pub text: PathBuf,
    pub tokens: usize,
    pub save: Option<PathBuf>,
    pub against: Option<PathBuf>,
    pub quant: Option<QuantScheme>,
}

fn read_reference(path: &Path) -> Result<Scores, CliError> {
    let saved = |reason: String| CliError::Saved {
        path: path.display().to_string(),
        reason,
    };
    let text = fs::read_to_string(path).map_err(|e| saved(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| saved(e.to_string()))
}

fn percent(fraction: f64) -> String {
    format!("{:.1}%", fraction * 100.0)
}

pub fn run(
    model_dir: &Path,
    backend: BackendChoice,
    options: &Options,
    style: &Style,
    interrupt: &Interrupt,
) -> Result<(), CliError> {
    interrupt.generating();
    if options.tokens < PROMPT_TOKENS.saturating_add(2) {
        return Err(CliError::BadOption(format!(
            "--tokens must be at least {}: the first {PROMPT_TOKENS} are the prompt and at least two more are scored",
            PROMPT_TOKENS.saturating_add(2)
        )));
    }
    let json = options.save.as_deref().map(JsonOut::create).transpose()?;
    let reference = options.against.as_deref().map(read_reference).transpose()?;
    let text = fs::read_to_string(&options.text).map_err(|e| CliError::Input {
        path: options.text.display().to_string(),
        reason: e.to_string(),
    })?;
    let loaded = load(
        model_dir,
        Some(options.tokens),
        options.tokens,
        backend,
        options.quant,
        style,
        true,
    )?;
    let tokenizer = &loaded.dir.tokenizer;
    let mut ids = tokenizer.encode(&text)?;
    ids.truncate(options.tokens);
    if ids.len() < PROMPT_TOKENS.saturating_add(2) {
        return Err(CliError::BadOption(format!(
            "the text is only {} tokens long; scoring needs at least {}",
            ids.len(),
            PROMPT_TOKENS.saturating_add(2)
        )));
    }
    let engine = &loaded.model;
    let mut cache = engine.new_cache(ids.len())?;
    let known = tokenizer.vocab_size();
    let cancel = interrupt.cancel_flag();
    let (prompt, rest) = ids.split_at(PROMPT_TOKENS);
    let (mut top1, mut target_logprob) = (Vec::new(), Vec::new());
    let mut logits = engine.forward(prompt, &mut cache)?;
    let progress = style::stderr_is_terminal();
    for (index, target) in rest.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(CliError::Cancelled);
        }
        let (best, logprob) = score(&logits, known, *target)?;
        top1.push(best);
        target_logprob.push(logprob);
        if progress && index.checked_rem(64) == Some(0) {
            eprint!("\rscoring {index}/{}", rest.len());
        }
        if index.saturating_add(1) < rest.len() {
            logits = engine.forward(&[*target], &mut cache)?;
        }
    }
    if progress {
        eprint!("\r{:<30}\r", "");
    }
    let scores = Scores {
        format: FORMAT,
        model: loaded.name.clone(),
        quantization: options
            .quant
            .map_or_else(|| "none".to_string(), |q| q.label()),
        prompt_tokens: PROMPT_TOKENS,
        text_tokens: ids.len(),
        text_fingerprint: fingerprint(&ids),
        top1,
        target_logprob,
    };
    let summary = summarize(&scores, rest);
    let mut out = format!(
        "quality: {} · {} · {}\n",
        loaded.name,
        scores.quantization,
        engine.backend_name()
    );
    out.push_str(&format!(
        "  text          {} tokens read; the first {PROMPT_TOKENS} are the prompt, the other {} are scored\n",
        ids.len(),
        summary.scored
    ));
    out.push_str(&format!(
        "  loss          {:.4} nats per token (perplexity {:.3}); lower is better\n",
        summary.mean_loss, summary.perplexity
    ));
    out.push_str(&format!(
        "  right guess   {} of positions had the real next token as the model's first choice\n",
        percent(summary.top1_hit_rate)
    ));
    if let Some(reference) = &reference {
        let a = agreement(reference, &scores)?;
        out.push_str(&format!(
            "vs {} · {}:\n  same choice   {} of positions pick the same first token\n  loss change   {:+.4} nats per token (perplexity {:+.2}%)\n  drift         the real token's log-probability moves by {:.4} on average\n",
            reference.model,
            reference.quantization,
            percent(a.same_top1),
            a.loss_change,
            a.perplexity_change_percent,
            a.mean_abs_logprob_diff
        ));
    }
    out.push_str(&format!(
        "{}\n",
        style.dim("reading this: loss is how surprised the model is by the real text; a quantized model that stays close to the original keeps the same loss and the same first choices.")
    ));
    std::io::stdout().lock().write_all(out.as_bytes())?;
    if let (Some(json), Some(path)) = (json, &options.save) {
        json.finish(&scores)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
