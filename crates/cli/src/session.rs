use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use model::{Engine, EngineCache};
use runtime::{
    CausalLm, GenerateParams, Kept, Outcome, Retained, SequenceCache, Speculation, StopReason,
    generate_reusing,
};
use sampler::{Sampler, SamplerConfig};

use crate::CliError;
use crate::load::Loaded;
use crate::signal::Interrupt;
use crate::snapshot::{self, Identity, Reject, Restored, Saved};
use crate::style::Style;
use crate::units;

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// None picks a new time-based seed for every reply.
    pub seed: Option<u64>,
}

impl Settings {
    pub fn sampler_config(&self, seed: u64) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed,
        }
    }
}

/// A snapshot file this session may restore from on its first prompt and save to when asked.
struct SnapshotState {
    path: PathBuf,
    weights_hash: Option<std::thread::JoinHandle<Result<u64, loader::LoaderError>>>,
    identity: Option<Identity>,
    tried: bool,
}

pub struct Session {
    pub loaded: Loaded,
    /// Private so every outside change goes through `engine_and_kv`, which forgets `retained`.
    kv: EngineCache,
    /// Dies with `loaded`: a new model, tokenizer or context size makes a new `Session`.
    retained: Retained,
    /// Off when a tool needs full replays.
    pub reuse: bool,
    pub settings: Settings,
    pub ctx: usize,
    pub last: Option<Outcome>,
    pub last_seed: u64,
    /// Tokens of the most recent prompt, for the inspect panel.
    pub last_prompt: Vec<u32>,
    /// KV positions in use right after the most recent reply.
    pub kv_used: usize,
    pub style: Style,
    /// Guess ahead by copying earlier text (GPU only, otherwise ignored). Off unless a tool turns it on.
    pub speculate: bool,
    /// Replace control characters in replies, so model output cannot drive the terminal.
    sanitize: bool,
    snapshot: Option<SnapshotState>,
    /// What happened when the first prompt tried the snapshot; `None` before that or without a snapshot.
    pub restored: Option<Result<Restored, Reject>>,
}

pub struct Reply {
    pub outcome: Outcome,
    pub seed: u64,
    /// Everything the model generated, unsanitized.
    pub text: String,
    /// The generated token ids, without the stop token.
    pub tokens: Vec<u32>,
}

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

impl Session {
    pub fn new(loaded: Loaded, settings: Settings, style: Style) -> Result<Self, CliError> {
        let ctx = loaded.ctx;
        let kv = loaded.model.new_cache(ctx)?;
        Ok(Self {
            loaded,
            kv,
            retained: Retained::default(),
            reuse: true,
            settings,
            ctx,
            last: None,
            last_seed: 0,
            last_prompt: Vec::new(),
            kv_used: 0,
            style,
            speculate: false,
            sanitize: std::io::stdout().is_terminal(),
            snapshot: None,
            restored: None,
        })
    }

    /// Makes `path` the snapshot: restored from on the first prompt if it starts the same way, saved to by
    /// `save_snapshot`. The model must have been opened with `load_for_snapshot`.
    pub fn use_snapshot(&mut self, path: PathBuf) {
        self.snapshot = Some(SnapshotState {
            path,
            weights_hash: self.loaded.weights_hash.take(),
            identity: None,
            tried: false,
        });
    }

    /// Joins the weight hash the first time it is needed.
    fn snapshot_identity(&mut self) -> Option<(PathBuf, Identity)> {
        let state = self.snapshot.as_mut()?;
        if state.identity.is_none() {
            let hash = state.weights_hash.take()?.join().ok()?.ok()?;
            state.identity = snapshot::identity(&self.loaded.model, &self.loaded.dir, hash);
        }
        Some((state.path.clone(), state.identity?))
    }

    /// Waits for the weight hash, normally long finished by the time a prompt arrives. Returns how long that took.
    pub fn prepare_snapshot(&mut self) -> std::time::Duration {
        let started = std::time::Instant::now();
        let _ = self.snapshot_identity();
        started.elapsed()
    }

    fn restore_snapshot(&mut self, prompt: &[u32]) {
        let Some(state) = self.snapshot.as_mut() else {
            return;
        };
        if std::mem::replace(&mut state.tried, true) || !self.reuse {
            return;
        }
        if self.kv.used() != 0 || !self.retained.is_empty() {
            return;
        }
        let Some((path, identity)) = self.snapshot_identity() else {
            return;
        };
        let result = snapshot::restore(&path, &self.loaded.model, &mut self.kv, prompt, &identity);
        if let Ok(restored) = &result {
            self.retained = Retained::restored(restored.ids.clone());
        }
        self.restored = Some(result);
    }

    /// Saves what the cache holds if it is exactly what the last reply left, otherwise saves nothing.
    pub fn save_snapshot(&mut self) -> Result<Option<Saved>, String> {
        let Some((path, identity)) = self.snapshot_identity() else {
            return Ok(None);
        };
        if self.retained.is_empty() {
            return Ok(None);
        }
        let ids = self.retained.tokens();
        snapshot::save(&path, &self.loaded.model, &self.kv, ids, identity).map(Some)
    }

    pub fn kv(&self) -> &EngineCache {
        &self.kv
    }

    /// Clears the record: the next reply reads its whole prompt.
    pub fn engine_and_kv(&mut self) -> (&Engine, &mut EngineCache) {
        self.retained.forget();
        (&self.loaded.model, &mut self.kv)
    }

    pub fn retained(&self) -> &Retained {
        &self.retained
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, CliError> {
        if text.trim().is_empty() {
            return Err(CliError::EmptyPrompt);
        }
        let tokens = self.loaded.dir.tokenizer.encode(text)?;
        if tokens.len() > self.ctx {
            return Err(CliError::PromptTooLong {
                tokens: tokens.len(),
                ctx: self.ctx,
            });
        }
        Ok(tokens)
    }

    /// Streams the model's continuation of `text` to `out` as it is generated.
    pub fn complete(
        &mut self,
        text: &str,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Reply, CliError> {
        let prompt = self.encode(text)?;
        self.complete_tokens(prompt, interrupt, out)
    }

    /// Like `complete`, for a prompt that is already tokens (e.g. a rendered chat).
    pub fn complete_tokens(
        &mut self,
        prompt: Vec<u32>,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Reply, CliError> {
        self.last_prompt.clone_from(&prompt);
        self.restore_snapshot(&prompt);
        let seed = self.settings.seed.unwrap_or_else(time_seed);
        let mut sampler = Sampler::new(self.settings.sampler_config(seed))?;
        let tokenizer = &self.loaded.dir.tokenizer;
        let params = GenerateParams {
            max_tokens: self.settings.max_tokens,
            stop_tokens: self.loaded.stop_tokens(),
            usable_vocab: tokenizer.vocab_size(),
            speculation: self.speculate.then_some(Speculation::MEASURED),
        };
        let sanitize = self.sanitize;
        let mut stream = tokenizer.stream();
        let mut write_error = None;
        let mut decode_error = None;
        let mut last_char = None;
        let mut text = String::new();
        let mut generated = Vec::new();
        if !self.reuse {
            self.retained.forget();
        }
        interrupt.generating();
        let result = generate_reusing(
            &self.loaded.model,
            Kept {
                cache: &mut self.kv,
                retained: &mut self.retained,
            },
            &mut sampler,
            &prompt,
            &params,
            interrupt.cancel_flag(),
            |token| {
                generated.push(token);
                if write_error.is_some() || decode_error.is_some() {
                    return;
                }
                match stream.push(token) {
                    Ok(Some(piece)) => {
                        text.push_str(&piece);
                        let piece = if sanitize { printable(&piece) } else { piece };
                        last_char = piece.chars().last().or(last_char);
                        let written = out.write_all(piece.as_bytes()).and_then(|()| out.flush());
                        if let Err(e) = written {
                            write_error = Some(e);
                            interrupt
                                .cancel_flag()
                                .store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        decode_error = Some(e);
                        interrupt
                            .cancel_flag()
                            .store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            },
        );
        interrupt.busy();
        if let Some(e) = write_error {
            return Err(e.into());
        }
        if let Some(e) = decode_error {
            return Err(e.into());
        }
        let outcome = result?;
        if last_char.is_some_and(|c| c != '\n') {
            writeln!(out)?;
        }
        self.last = Some(outcome.clone());
        self.last_seed = seed;
        self.kv_used = self.kv.used();
        Ok(Reply {
            outcome,
            seed,
            text,
            tokens: generated,
        })
    }

    /// `id  "text"` for each token.
    pub fn token_lines(&self, ids: &[u32]) -> Result<Vec<String>, CliError> {
        ids.iter()
            .map(|id| {
                Ok(format!(
                    "{id:>7}  {:?}",
                    self.loaded.dir.tokenizer.decode(&[*id])?
                ))
            })
            .collect()
    }

    /// The model's logits for the token after `prompt` (the cache is rebuilt from scratch).
    pub fn next_logits(&mut self, prompt: &[u32]) -> Result<Vec<f32>, CliError> {
        self.retained.forget();
        self.kv.clear();
        let logits = self
            .loaded
            .model
            .forward(prompt, &mut self.kv)
            .map_err(runtime::GenerateError::Model)?;
        Ok(logits)
    }

    /// The five likeliest tokens after `prompt`, as `percent  "text"  (id N)`.
    pub fn next_token_lines(&mut self, prompt: &[u32]) -> Result<Vec<String>, CliError> {
        let logits = self.next_logits(prompt)?;
        let tokenizer = &self.loaded.dir.tokenizer;
        let known = logits.get(..tokenizer.vocab_size()).unwrap_or(&logits);
        let max = known.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f64 = known.iter().map(|l| f64::from(l - max).exp()).sum();
        let mut ranked: Vec<(usize, f32)> = known.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
            .into_iter()
            .take(5)
            .map(|(id, logit)| {
                let piece = tokenizer.decode(&[u32::try_from(id).unwrap_or(0)])?;
                let percent = f64::from(logit - max).exp() / total * 100.0;
                Ok(format!("{percent:>5.1}%  {piece:?}   (id {id})"))
            })
            .collect()
    }

    /// One dim summary line: how many tokens, how fast, and why generation stopped.
    pub fn footer(&self, reply: &Reply) -> String {
        let o = &reply.outcome;
        let mut parts = vec![
            format!("{} tokens", o.generated),
            format!("first token {}", units::seconds(o.prefill)),
        ];
        let decoded = o.generated.saturating_sub(1);
        if decoded > 0 && !o.decode.is_zero() {
            parts.push(format!(
                "{:.2} tok/s",
                decoded as f64 / o.decode.as_secs_f64()
            ));
        }
        if self.settings.temperature > 0.0 {
            parts.push(format!("seed {}", reply.seed));
        }
        parts.push(format!("stopped: {}", stop_text(o.stop)));
        self.style.dim(&format!("── {} ──", parts.join(" · ")))
    }
}

/// Shows control characters (other than newline and tab) as visible symbols,
/// so a model can never emit terminal escape sequences that run on the user's screen.
pub fn printable(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' | '\t' => c,
            '\u{0}'..='\u{1f}' => {
                char::from_u32(0x2400_u32.saturating_add(u32::from(c))).unwrap_or('\u{fffd}')
            }
            '\u{7f}' => '\u{2421}',
            c if c.is_control() => '\u{fffd}',
            c => c,
        })
        .collect()
}

pub fn stop_text(stop: StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens => "max tokens (raise with --max-tokens)",
        StopReason::EndOfText => "end of text",
        StopReason::ContextFull => "context full (raise with --ctx)",
        StopReason::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::printable;

    #[test]
    fn escape_sequences_and_other_controls_become_visible_symbols() {
        assert_eq!(printable("a\u{1b}[31mred\u{7}"), "a␛[31mred␇");
        assert_eq!(printable("tab\there\nnext"), "tab\there\nnext");
        assert_eq!(printable("del\u{7f}c1\u{85}"), "del␡c1\u{fffd}");
        assert_eq!(printable("naïve café ☕"), "naïve café ☕");
    }
}
