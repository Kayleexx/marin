use std::collections::HashMap;

/// Finds the last time the text ended with the same `n` tokens and offers what followed, to guess ahead by copying.
/// Holds the prompt and the reply so far.
pub(crate) struct Ngrams {
    n: usize,
    history: Vec<u32>,
    /// Hash of an n-gram -> position just after its latest occurrence that ended before the newest token.
    last_end: HashMap<u64, u32>,
}

fn key(tokens: &[u32]) -> u64 {
    tokens.iter().fold(tokens.len() as u64, |hash, &token| {
        (hash ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl Ngrams {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            history: Vec::new(),
            last_end: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, token: u32) {
        self.history.push(token);
        // Index the n-gram that ends just before the new token, so a lookup never finds the text it asks about.
        let end = self.history.len().saturating_sub(1);
        let gram = end
            .checked_sub(self.n)
            .and_then(|start| self.history.get(start..end));
        if let (Some(gram), Ok(end)) = (gram, u32::try_from(end)) {
            self.last_end.insert(key(gram), end);
        }
    }

    /// Up to `max_len` tokens that followed the latest earlier occurrence of the text's last `n` tokens; empty when
    /// the text has not ended this way before.
    pub(crate) fn draft(&self, max_len: usize) -> &[u32] {
        let len = self.history.len();
        let Some(suffix) = len.checked_sub(self.n).and_then(|s| self.history.get(s..)) else {
            return &[];
        };
        let Some(&end) = self.last_end.get(&key(suffix)) else {
            return &[];
        };
        let end = end as usize;
        let same = end
            .checked_sub(self.n)
            .and_then(|start| self.history.get(start..end))
            == Some(suffix);
        if !same {
            return &[];
        }
        self.history
            .get(end..end.saturating_add(max_len).min(len))
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fed(n: usize, tokens: &[u32]) -> Ngrams {
        let mut grams = Ngrams::new(n);
        tokens.iter().for_each(|t| grams.push(*t));
        grams
    }

    #[test]
    fn copies_what_followed_the_last_exact_match() {
        let grams = fed(3, &[1, 2, 3, 4, 5, 9, 1, 2, 3]);
        assert_eq!(grams.draft(4), [4, 5, 9, 1]);
        assert_eq!(grams.draft(2), [4, 5]);
    }

    #[test]
    fn a_shorter_match_is_not_enough() {
        let grams = fed(3, &[7, 2, 3, 4, 5, 9, 1, 2, 3]);
        assert!(grams.draft(4).is_empty());
    }

    #[test]
    fn the_latest_occurrence_wins() {
        let grams = fed(2, &[1, 2, 10, 5, 1, 2, 20, 6, 1, 2]);
        assert_eq!(grams.draft(1), [20]);
    }

    #[test]
    fn never_matches_the_text_against_itself() {
        assert!(fed(2, &[1, 2, 3, 4]).draft(4).is_empty());
        assert!(fed(2, &[1]).draft(4).is_empty());
        assert!(Ngrams::new(2).draft(4).is_empty());
    }

    #[test]
    fn a_draft_stops_at_the_end_of_the_text() {
        assert_eq!(fed(2, &[1, 2, 1, 2]).draft(8), [1, 2]);
    }

    #[test]
    fn a_hash_collision_is_not_taken_for_a_match() {
        let mut grams = fed(2, &[1, 2, 3]);
        // Pretend `[2, 3]` hashed like `[1, 2]`, which ended at 2.
        grams.last_end.insert(key(&[2, 3]), 2);
        assert!(grams.draft(4).is_empty());
    }
}
