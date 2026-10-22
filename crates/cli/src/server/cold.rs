//! A session that is not running and not in the hot slot, kept as bytes in host memory: the tokens it has read, each
//! DeltaNet layer's state and each page of keys and values, exactly as the checkpoint code exports them. It holds no GPU
//! memory. Thawing builds a new cache from these bytes, so no cache is ever shared; they are never written to disk.

use model::{Engine, ModelError, PAGE_TOKENS, pages_for};
use runtime::Retained;

use super::session::Stored;

/// Most host memory the cold sessions of one server hold together.
pub(super) const COLD_BYTES: usize = 2 << 30;
/// Most cold sessions kept; the least recently used goes first.
pub(super) const COLD_SESSIONS: usize = 16;
/// Host memory that must stay available after a session is shelved.
const HOST_HEADROOM: u64 = 1 << 30;

pub(super) struct Cold {
    pub id: String,
    tokens: Vec<u32>,
    deltas: Vec<Vec<u8>>,
    pages: Vec<Vec<u8>>,
    bytes: usize,
}

fn rows_of(page: usize, tokens: usize) -> usize {
    PAGE_TOKENS.min(tokens.saturating_sub(page.saturating_mul(PAGE_TOKENS)))
}

impl Cold {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    #[cfg(test)]
    pub fn placeholder(id: &str, bytes: usize) -> Self {
        Self {
            id: id.to_owned(),
            tokens: vec![1],
            deltas: Vec::new(),
            pages: Vec::new(),
            bytes,
        }
    }
}

/// The bytes a session of `tokens` tokens takes when shelved, or `None` for an engine that cannot be shelved.
fn size_of(engine: &Engine, tokens: usize) -> Option<usize> {
    let layout = engine.state_layout()?;
    let pages = (0..pages_for(tokens)).fold(0_usize, |sum, page| {
        sum.saturating_add(layout.page_bytes(rows_of(page, tokens)))
    });
    layout
        .delta_bytes()
        .saturating_mul(layout.delta_layers)
        .saturating_add(pages)
        .checked_add(tokens.saturating_mul(4))
}

/// Copies a hot session to host memory. `None` if it cannot or should not be shelved: another engine, a cache that
/// does not hold exactly its record, more than the budget, or too little host memory left.
pub(super) fn capture(engine: &Engine, stored: &Stored) -> Result<Option<Cold>, ModelError> {
    let Ok(tokens) = engine.kept_tokens(&stored.cache) else {
        return Ok(None);
    };
    if tokens != stored.retained.tokens() {
        return Ok(None);
    }
    let Some(bytes) = size_of(engine, tokens.len()).filter(|b| *b <= COLD_BYTES) else {
        return Ok(None);
    };
    let spare = crate::memory::available_bytes().unwrap_or(0);
    if spare
        < u64::try_from(bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(HOST_HEADROOM)
    {
        return Ok(None);
    }
    let layers = engine.state_layout().map_or(0, |l| l.delta_layers);
    let deltas = (0..layers)
        .map(|index| engine.export_state_delta(&stored.cache, index))
        .collect::<Result<Vec<_>, _>>()?;
    let pages = (0..pages_for(tokens.len()))
        .map(|page| engine.export_state_page(&stored.cache, page, rows_of(page, tokens.len())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Cold {
        id: stored.id.clone(),
        tokens: tokens.to_vec(),
        deltas,
        pages,
        bytes,
    }))
}

/// A new hot session holding exactly what the cold one held. `ctx` is the server's window.
pub(super) fn thaw(engine: &Engine, ctx: usize, cold: &Cold) -> Result<Stored, ModelError> {
    let mut cache = engine.begin_state_import(ctx, cold.tokens.len())?;
    for (index, bytes) in cold.deltas.iter().enumerate() {
        engine.import_state_delta(&mut cache, index, bytes)?;
    }
    for (page, bytes) in cold.pages.iter().enumerate() {
        engine.import_state_page(&mut cache, page, rows_of(page, cold.tokens.len()), bytes)?;
    }
    engine.finish_state_import(&mut cache, &cold.tokens)?;
    Ok(Stored {
        id: cold.id.clone(),
        cache,
        retained: Retained::restored(cold.tokens.clone()),
    })
}
