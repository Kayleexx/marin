use std::collections::HashMap;

/// Guesses the next tokens by finding the last time the text ended the same way and copying what followed.
/// Works on the prompt and the reply together, with no model.
pub struct Ngrams {
    history: Vec<u32>,
    max_n: usize,
    /// n-gram (hashed with its length) -> position just after its latest occurrence that ended before the newest token.
    last_end: HashMap<u64, u32>,
}

fn key(tokens: &[u32]) -> u64 {
    tokens.iter().fold(tokens.len() as u64, |hash, &token| {
        (hash ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl Ngrams {
    pub fn new(max_n: usize) -> Self {
        Self {
            history: Vec::new(),
            max_n,
            last_end: HashMap::new(),
        }
    }

    pub fn push(&mut self, token: u32) {
        self.history.push(token);
        // Index the n-grams that end just before the new token, so a lookup never finds the text it is asking about.
        let end = self.history.len().saturating_sub(1);
        for n in 1..=self.max_n {
            let grams = end
                .checked_sub(n)
                .and_then(|start| self.history.get(start..end));
            if let (Some(gram), Ok(end)) = (grams, u32::try_from(end)) {
                self.last_end.insert(key(gram), end);
            }
        }
    }

    /// Up to `max_len` tokens that followed the longest earlier match of the text's last `min_n..=max_n` tokens.
    /// Empty when nothing matches.
    pub fn draft(&self, min_n: usize, max_len: usize) -> &[u32] {
        self.draft_matched(min_n, max_len).0
    }

    /// `draft`, and how many tokens matched (0 when nothing did): a long match is a better sign than a short one.
    pub fn draft_matched(&self, min_n: usize, max_len: usize) -> (&[u32], usize) {
        let len = self.history.len();
        for n in (min_n.max(1)..=self.max_n.min(len)).rev() {
            let Some(suffix) = len.checked_sub(n).and_then(|s| self.history.get(s..)) else {
                continue;
            };
            let Some(&end) = self.last_end.get(&key(suffix)) else {
                continue;
            };
            let end = end as usize;
            let same = end
                .checked_sub(n)
                .and_then(|start| self.history.get(start..end))
                == Some(suffix);
            if same {
                let stop = end.saturating_add(max_len).min(len);
                return (self.history.get(end..stop).unwrap_or(&[]), n);
            }
        }
        (&[], 0)
    }

    /// Rough memory held: the history and the index.
    pub fn bytes(&self) -> usize {
        let entry = size_of::<u64>()
            .saturating_add(size_of::<u32>())
            .saturating_add(1);
        self.history
            .capacity()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.last_end.capacity().saturating_mul(entry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fed(tokens: &[u32]) -> Ngrams {
        let mut grams = Ngrams::new(4);
        tokens.iter().for_each(|t| grams.push(*t));
        grams
    }

    #[test]
    fn repeats_what_followed_the_last_match() {
        let grams = fed(&[1, 2, 3, 4, 5, 9, 1, 2, 3]);
        assert_eq!(grams.draft(2, 4), [4, 5, 9, 1]);
        assert_eq!(grams.draft(2, 2), [4, 5]);
    }

    #[test]
    fn reports_how_long_the_match_was() {
        let grams = fed(&[1, 2, 3, 4, 5, 9, 2, 3, 4, 5]);
        assert_eq!(grams.draft_matched(2, 3), (&[9, 2, 3][..], 4));
        assert_eq!(Ngrams::new(4).draft_matched(1, 3), (&[][..], 0));
    }

    #[test]
    fn needs_at_least_min_n_tokens_to_match() {
        let grams = fed(&[1, 2, 3, 7, 8, 3]);
        assert_eq!(grams.draft(1, 2), [7, 8]);
        assert!(grams.draft(2, 2).is_empty());
    }

    #[test]
    fn prefers_the_longest_match_and_the_latest_occurrence() {
        let grams = fed(&[5, 1, 2, 10, 6, 1, 2, 20, 1, 2]);
        assert_eq!(grams.draft(1, 1), [20]);
        let grams = fed(&[9, 1, 2, 10, 7, 1, 2, 20, 8, 7, 1, 2]);
        assert_eq!(grams.draft(1, 1), [20]);
        let grams = fed(&[7, 1, 2, 30, 6, 1, 2, 20, 5, 7, 1, 2]);
        assert_eq!(grams.draft(1, 1), [30]);
    }

    #[test]
    fn never_matches_the_text_against_itself() {
        let grams = fed(&[1, 2, 3, 4]);
        assert!(grams.draft(1, 4).is_empty());
        assert!(Ngrams::new(4).draft(1, 4).is_empty());
    }

    #[test]
    fn a_draft_stops_at_the_end_of_the_text() {
        let grams = fed(&[1, 2, 1, 2]);
        assert_eq!(grams.draft(2, 8), [1, 2]);
    }

    #[test]
    fn the_longest_match_wins_even_when_it_leaves_little_to_copy() {
        let grams = fed(&[4, 4, 4, 4, 4]);
        assert_eq!(grams.draft(1, 3), [4]);
    }
}
