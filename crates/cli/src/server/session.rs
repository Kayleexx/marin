use std::fs::File;
use std::io::{self, Read};
use std::sync::{Mutex, MutexGuard, PoisonError};

use model::EngineCache;
use runtime::{CausalLm, Retained, SequenceCache};

use super::sched::{ScheduleError, Scheduled};
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

/// The one idle session kept between requests. A request takes it out for its whole run, so a second request for
/// the same ID finds nothing instead of sharing the state.
#[derive(Default)]
pub(super) struct Store(Mutex<Option<Stored>>);

impl Store {
    fn lock(&self) -> MutexGuard<'_, Option<Stored>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn take(&self, id: &str) -> Option<Stored> {
        let mut slot = self.lock();
        if slot.as_ref().is_some_and(|state| state.id == id) {
            slot.take()
        } else {
            None
        }
    }

    pub fn evict(&self) {
        let old = self.lock().take();
        drop(old);
    }

    pub fn put(&self, state: Stored) {
        let old = self.lock().replace(state);
        drop(old);
    }

    /// Puts the session back as it was before a failed or cancelled request. A new session leaves nothing behind.
    pub fn recover(
        &self,
        mut active: Stored,
        sched: Scheduled<'_>,
        previous: &[u32],
        existing: bool,
    ) {
        if !existing {
            return;
        }
        active.cache.clear();
        if !previous.is_empty() && sched.forward(previous, &mut active.cache).is_ok() {
            active.retained = Retained::restored(previous.to_vec());
        } else {
            active.cache.clear();
            active.retained.forget();
        }
        self.put(active);
    }
}

pub(super) fn checkpoint(
    active: &mut Stored,
    sched: Scheduled<'_>,
    history: &[u32],
) -> Result<(Option<EngineCache>, usize), ScheduleError> {
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
        sched.forward(fresh, &mut active.cache)?;
    }
    active.retained = Retained::restored(history.to_vec());
    Ok((sched.checkpoint(&active.cache)?, keep))
}
