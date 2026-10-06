use std::time::{Duration, Instant};

use super::ngram::Ngrams;

/// Longest earlier match looked for.
pub const MAX_N: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Fewest matching tokens before a guess is made.
    pub min_n: usize,
    /// Most tokens guessed at once.
    pub max_len: usize,
}

/// One pass of the model that checks a guess: it is fed the last token plus `drafted` guessed ones,
/// and `accepted` of those guesses were what the model would have chosen itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    pub drafted: usize,
    pub accepted: usize,
    /// Tokens the pass settles: the accepted guesses and the model's own next token, except at the end of the reply.
    pub produced: usize,
    /// Tokens already held when the pass ran.
    pub context: usize,
}

pub struct Replay {
    pub passes: Vec<Pass>,
    /// Time the guesser itself took, building its index over the prompt included.
    pub guesser: Duration,
    pub memory: usize,
}

/// Plays a finished reply back as if it were being generated with guessing. The model's own choices are the
/// reply, so a guess is right exactly when it equals the next reply token. The first token comes from the
/// prompt pass and is not guessed.
pub fn replay(prompt: &[u32], reply: &[u32], config: Config) -> Replay {
    let start = Instant::now();
    let mut grams = Ngrams::new(MAX_N);
    prompt.iter().for_each(|t| grams.push(*t));
    let mut passes = Vec::new();
    if let Some(first) = reply.first() {
        grams.push(*first);
    }
    let mut done = reply.len().min(1);
    while let Some(rest) = reply.get(done..).filter(|rest| !rest.is_empty()) {
        let draft = grams.draft(config.min_n, config.max_len);
        let drafted = draft.len().min(rest.len());
        let accepted = draft.iter().zip(rest).take_while(|(g, r)| g == r).count();
        let produced = accepted.saturating_add(1).min(rest.len());
        passes.push(Pass {
            drafted,
            accepted,
            produced,
            context: prompt.len().saturating_add(done),
        });
        rest.iter().take(produced).for_each(|t| grams.push(*t));
        done = done.saturating_add(produced);
    }
    Replay {
        passes,
        guesser: start.elapsed(),
        memory: grams.bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passes(prompt: &[u32], reply: &[u32], min_n: usize, max_len: usize) -> Vec<(usize, usize)> {
        replay(prompt, reply, Config { min_n, max_len })
            .passes
            .iter()
            .map(|p| (p.drafted, p.accepted))
            .collect()
    }

    #[test]
    fn copying_the_prompt_is_accepted_in_one_pass() {
        let result = passes(&[5, 6, 7, 8], &[1, 5, 6, 7, 8, 2], 1, 4);
        assert_eq!(result, [(0, 0), (4, 3)]);
    }

    #[test]
    fn a_wrong_guess_still_yields_the_models_own_token() {
        let result = passes(&[5, 6, 7, 8], &[1, 5, 9, 9], 1, 4);
        assert_eq!(result, [(0, 0), (2, 0), (0, 0)]);
    }

    #[test]
    fn every_reply_token_after_the_first_is_produced_exactly_once() {
        let reply: Vec<u32> = (0..40).map(|i| i % 7).collect();
        let replayed = replay(
            &[1, 2, 3, 4, 5, 6, 0],
            &reply,
            Config {
                min_n: 2,
                max_len: 8,
            },
        );
        let produced: usize = replayed.passes.iter().map(|p| p.produced).sum();
        assert_eq!(produced, reply.len() - 1);
    }

    #[test]
    fn guesses_are_cut_at_the_end_of_the_reply() {
        let result = passes(&[5, 6, 7, 8], &[1, 5, 6], 1, 8);
        assert_eq!(result, [(0, 0), (1, 1)]);
    }

    #[test]
    fn min_n_holds_back_weak_matches() {
        let result = passes(&[5, 6, 7, 8], &[1, 5, 6, 7, 8, 2], 2, 4);
        assert_eq!(result, [(0, 0), (0, 0), (3, 2)]);
    }

    #[test]
    fn an_empty_or_single_token_reply_needs_no_passes() {
        assert!(passes(&[1, 2], &[], 1, 4).is_empty());
        assert!(passes(&[1, 2], &[9], 1, 4).is_empty());
    }
}
