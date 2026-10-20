use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::Instant;

use runtime::{CausalLm, Retained, SequenceCache};

use super::{Reply, Session};
use crate::CliError;
use crate::signal::Interrupt;

/// Tokens read per pass, so a cancel is heard within one pass.
const SLICE: usize = 64;

impl Session {
    /// Streams the reply to `prompt`, whose first `upto` tokens are the conversation so far (the part every later
    /// prompt shares) and the rest the reply's own start. The conversation is read a slice at a time, so Ctrl-C does
    /// not wait for a long prompt, and where the engine can hold a second copy of its state the cache returns to the
    /// end of the conversation afterwards, so the next prompt reads only what came after it instead of replaying
    /// everything. `None` if cancelled before the reply began.
    pub fn complete_after(
        &mut self,
        prompt: Vec<u32>,
        upto: usize,
        interrupt: &Interrupt,
        out: &mut impl Write,
    ) -> Result<Option<Reply>, CliError> {
        // A snapshot restores into an empty cache on the first prompt, a tool that needs full replays says so, and a
        // cache already past the conversation (no checkpoint was possible) is better kept than cut back to it.
        let plain = self.snapshot.as_ref().is_some_and(|s| !s.tried)
            || !self.reuse
            || self.held_prefix(&prompt) > Self::history(&prompt, upto);
        if plain {
            return self.complete_tokens(prompt, interrupt, out).map(Some);
        }
        let started = Instant::now();
        let kept = match self.prefill(&prompt, upto, interrupt)? {
            Some(kept) if !interrupt.cancel_flag().load(Ordering::SeqCst) => kept,
            _ => {
                self.return_to_boundary();
                return Ok(None);
            }
        };
        let read = started.elapsed();
        let reply = self.complete_tokens(prompt, interrupt, out);
        self.return_to_boundary();
        reply.map(|mut reply| {
            reply.outcome.reused = kept;
            reply.outcome.prompt_forward = reply.outcome.prompt_forward.saturating_add(read);
            reply.outcome.prefill = reply.outcome.prefill.saturating_add(read);
            Some(reply)
        })
    }

    fn history(prompt: &[u32], upto: usize) -> usize {
        upto.min(prompt.len().saturating_sub(1))
    }

    /// How many leading tokens of `prompt` the cache already holds.
    fn held_prefix(&self, prompt: &[u32]) -> usize {
        if self.retained.len() != self.kv.used() {
            return 0;
        }
        self.retained
            .tokens()
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Reads the first `upto` tokens of `prompt` (at most all but the last) into the cache and remembers them,
    /// returning how many the cache already held. `None` if cancelled; what was read stays valid.
    fn prefill(
        &mut self,
        prompt: &[u32],
        upto: usize,
        interrupt: &Interrupt,
    ) -> Result<Option<usize>, CliError> {
        let body = prompt.get(..Self::history(prompt, upto)).unwrap_or(&[]);
        let kept = self.held_prefix(body);
        if kept == 0 {
            self.kv.clear();
        } else if kept < self.kv.used() {
            self.kv
                .truncate(kept)
                .map_err(|e| CliError::BadOption(e.to_string()))?;
        }
        self.retained.forget();
        interrupt.generating();
        let mut done = kept;
        let mut finished = Ok(Some(kept));
        for part in body.get(done..).unwrap_or(&[]).chunks(SLICE) {
            if interrupt.cancel_flag().load(Ordering::SeqCst) {
                finished = Ok(None);
                break;
            }
            if let Err(e) = self.loaded.model.forward(part, &mut self.kv) {
                finished = Err(runtime::GenerateError::Model(e).into());
                break;
            }
            done = done.saturating_add(part.len());
        }
        self.retained = Retained::restored(body.get(..done).unwrap_or(&[]).to_vec());
        if matches!(finished, Ok(Some(_))) {
            // Only where the engine can hold a second copy of the state; elsewhere the next prompt truncates as before.
            self.boundary = match self.loaded.model.session_checkpoint(&self.kv) {
                Ok(Some(copy)) => Some((copy, body.to_vec())),
                _ => None,
            };
        }
        finished
    }

    /// The cache goes back to the end of the conversation `prefill` read, exactly.
    fn return_to_boundary(&mut self) {
        if let Some((copy, tokens)) = self.boundary.take() {
            self.kv = copy;
            self.retained = Retained::restored(tokens);
        }
    }
}
