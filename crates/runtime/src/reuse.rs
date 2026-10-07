/// The token ids whose keys and values a cache holds, position by position, so the next prompt that starts with the
/// same ids can keep those positions instead of reading them again.
///
/// Only `generate_reusing` writes it, and only when a run returns normally (cancelled and stopped runs included). A
/// run that fails leaves it empty, so nothing it may have half written is trusted. Anything else that changes the
/// cache must call `forget`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Retained {
    ids: Vec<u32>,
}

/// A cache with the record of what it holds, handed to `generate_reusing` as one.
pub struct Kept<'a, C> {
    pub cache: &'a mut C,
    pub retained: &'a mut Retained,
}

impl Retained {
    /// The ids held, oldest first.
    pub fn tokens(&self) -> &[u32] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Stops trusting the cache: the next run reads its whole prompt.
    pub fn forget(&mut self) {
        self.ids.clear();
    }

    /// How many leading positions of a cache holding `held` positions can serve `prompt`: the ids shared exactly with
    /// the start of `prompt`, never all of `prompt` (the pass that follows needs a token to read to give the next
    /// token's logits), and none when the cache no longer holds what this records.
    pub(crate) fn reusable(&self, prompt: &[u32], held: usize) -> usize {
        if self.ids.len() != held {
            return 0;
        }
        let shared = self
            .ids
            .iter()
            .zip(prompt)
            .take_while(|(cached, wanted)| cached == wanted)
            .count();
        shared.min(prompt.len().saturating_sub(1))
    }

    pub(crate) fn take(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.ids)
    }

    pub(crate) fn set(&mut self, ids: Vec<u32>) {
        self.ids = ids;
    }
}
