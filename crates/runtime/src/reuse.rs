/// Token ids the cache holds, position by position. Set only by a run that returns normally; a failed run leaves it
/// empty. Anything else that changes the cache must call `forget`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Retained {
    ids: Vec<u32>,
}

pub struct Kept<'a, C> {
    pub cache: &'a mut C,
    pub retained: &'a mut Retained,
}

impl Retained {
    pub fn tokens(&self) -> &[u32] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn forget(&mut self) {
        self.ids.clear();
    }

    /// Never all of `prompt`: the pass after it needs a token to read. None when the cache is not what this records.
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
