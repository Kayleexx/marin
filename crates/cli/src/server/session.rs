use std::fs::File;
use std::io::{self, Read};

use model::EngineCache;
use runtime::{CausalLm, Retained, SequenceCache};

use crate::load::Loaded;

pub(super) struct Stored {
    pub id: String,
    pub cache: EngineCache,
    pub retained: Retained,
}

impl Stored {
    pub fn new(loaded: &Loaded, id: String) -> Result<Self, model::ModelError> {
        Ok(Self {
            id,
            cache: loaded.model.new_cache(loaded.ctx)?,
            retained: Retained::default(),
        })
    }

    pub fn can_reuse(&self, prompt: &[u32]) -> bool {
        let ids = self.retained.tokens();
        !ids.is_empty()
            && ids.len() == self.cache.used()
            && ids.len() < prompt.len()
            && prompt.starts_with(ids)
    }
}

pub(super) fn valid_id(id: &str) -> bool {
    id == "new" || (id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub(super) fn new_id() -> io::Result<String> {
    let mut bytes = [0_u8; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(format!("{:032x}", u128::from_be_bytes(bytes)))
}

pub(super) fn recover(
    stored: &mut Option<Stored>,
    mut active: Stored,
    loaded: &Loaded,
    previous: &[u32],
    existing: bool,
) {
    if !existing {
        return;
    }
    active.cache.clear();
    if !previous.is_empty() && loaded.model.forward(previous, &mut active.cache).is_ok() {
        active.retained = Retained::restored(previous.to_vec());
    } else {
        active.cache.clear();
        active.retained.forget();
    }
    *stored = Some(active);
}

pub(super) fn checkpoint(
    active: &mut Stored,
    loaded: &Loaded,
    history: &[u32],
) -> Result<(Option<EngineCache>, usize), model::ModelError> {
    let cached = active.retained.tokens();
    let keep = if cached.len() == active.cache.used() && history.starts_with(cached) {
        cached.len()
    } else {
        active.cache.clear();
        active.retained.forget();
        0
    };
    if keep < history.len() {
        let fresh = history.get(keep..).ok_or_else(|| {
            model::ModelError::InvalidConfig("history prefix out of bounds".into())
        })?;
        loaded.model.forward(fresh, &mut active.cache)?;
    }
    active.retained = Retained::restored(history.to_vec());
    Ok((loaded.model.session_checkpoint(&active.cache)?, keep))
}
