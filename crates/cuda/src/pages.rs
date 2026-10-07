use thiserror::Error;

/// Must equal `1 << PAGE_SHIFT` in `kv_format.cu`.
pub const PAGE_TOKENS: usize = 64;

/// Pages allocated together, so a layer's keys stay contiguous across them. Must equal `SLAB_PAGES` in `kv_format.cu`.
pub const SLAB_PAGES: usize = 8;

pub const fn pages_for(tokens: usize) -> usize {
    tokens.div_ceil(PAGE_TOKENS)
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{needed} more kv pages are needed but only {available} are free")]
pub struct PagesExhausted {
    pub needed: usize,
    pub available: usize,
}

/// Freed pages are handed out again before any new id is issued, most recently freed first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagePool {
    limit: usize,
    issued: usize,
    freed: Vec<usize>,
}

impl PagePool {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            issued: 0,
            freed: Vec::new(),
        }
    }

    pub fn available(&self) -> usize {
        self.freed
            .len()
            .saturating_add(self.limit.saturating_sub(self.issued))
    }

    pub fn issued(&self) -> usize {
        self.issued
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    fn take(&mut self) -> Option<usize> {
        if let Some(id) = self.freed.pop() {
            return Some(id);
        }
        (self.issued < self.limit).then(|| {
            let id = self.issued;
            self.issued = self.issued.saturating_add(1);
            id
        })
    }

    fn give(&mut self, id: usize) {
        self.freed.push(id);
    }
}

/// Entry `n` holds positions `n * PAGE_TOKENS` onward.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageTable {
    ids: Vec<usize>,
}

impl PageTable {
    pub fn ids(&self) -> &[usize] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Returns where the new pages start. Changes nothing when the pool cannot supply them all.
    pub fn grow_to(&mut self, pool: &mut PagePool, pages: usize) -> Result<usize, PagesExhausted> {
        let first_new = self.ids.len();
        let needed = pages.saturating_sub(first_new);
        if needed > pool.available() {
            return Err(PagesExhausted {
                needed,
                available: pool.available(),
            });
        }
        for _ in 0..needed {
            if let Some(id) = pool.take() {
                self.ids.push(id);
            }
        }
        Ok(first_new)
    }

    /// Gives pages back lowest position first, so a refill receives them in reverse order.
    pub fn shrink_to(&mut self, pool: &mut PagePool, pages: usize) {
        for id in self.ids.drain(pages.min(self.ids.len())..) {
            pool.give(id);
        }
    }
}
