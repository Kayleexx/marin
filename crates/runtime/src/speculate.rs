use std::time::{Duration, Instant};

use super::ngram::Ngrams;

/// Guessing ahead by copying: when the last `match_len` tokens of the text exactly repeat an earlier stretch, the
/// tokens that followed it (at most `max_draft`) are guessed and checked in one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    pub match_len: usize,
    pub max_draft: usize,
}

impl Speculation {
    /// The rule measured to help where text is copied (code edits) and cost nothing elsewhere: a looser match
    /// loses time on chat and fresh code at long context.
    pub const MEASURED: Self = Self {
        match_len: 8,
        max_draft: 7,
    };
}

impl Speculation {
    /// For a model whose pass over more than four tokens costs more than twice a single token's: the check is kept to
    /// four tokens (at most three guesses), which costs 1.1-1.5x one token, so a shorter match is worth trying.
    pub const SHORT_PASS: Self = Self {
        match_len: 4,
        max_draft: 3,
    };
}

impl Default for Speculation {
    fn default() -> Self {
        Self::MEASURED
    }
}

/// What guessing did during one reply.
#[derive(Debug, Clone, Default)]
pub struct SpeculationStats {
    /// Model passes after the prompt: one per step of plain decoding, one per check of guesses.
    pub passes: usize,
    /// Passes that checked guessed tokens.
    pub guessed_passes: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Time spent indexing the text and looking for matches.
    pub guesser: Duration,
    /// Time spent rolling the cache back over rejected guesses.
    pub rollback: Duration,
}

/// The guesser of one reply: the text so far, indexed.
pub(crate) struct Drafter {
    grams: Ngrams,
    max_draft: usize,
}

impl Drafter {
    /// `rows` is the most tokens the model checks in one pass, the last token included.
    pub(crate) fn new(rule: Speculation, rows: usize) -> Self {
        Self {
            grams: Ngrams::new(rule.match_len),
            max_draft: rule.max_draft.min(rows.saturating_sub(1)),
        }
    }

    pub(crate) fn push(&mut self, token: u32, stats: &mut SpeculationStats) {
        let at = Instant::now();
        self.grams.push(token);
        stats.guesser = stats.guesser.saturating_add(at.elapsed());
    }

    /// At most `room` tokens (and never more than the rule allows) to check after the last token.
    pub(crate) fn draft(&self, room: usize, stats: &mut SpeculationStats) -> Vec<u32> {
        let at = Instant::now();
        let draft = self.grams.draft(room.min(self.max_draft)).to_vec();
        stats.guesser = stats.guesser.saturating_add(at.elapsed());
        draft
    }
}
