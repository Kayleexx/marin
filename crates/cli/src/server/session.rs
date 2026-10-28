use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use model::EngineCache;
use model::qwen35::Mark;
use runtime::{CausalLm, Retained, SequenceCache};

use super::cold::{COLD_BYTES, COLD_SESSIONS, Cold};
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

/// Why a session id cannot be served right now.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Refused {
    /// Another request is running this session.
    Running,
    /// The server holds no session with this id (never made, dropped, or lost with a restart).
    Missing,
}

pub(super) enum Entry {
    Hot(Stored),
    Cold(Cold),
}

#[derive(Default)]
struct Inner {
    hot: Option<Stored>,
    /// Oldest first.
    cold: VecDeque<Cold>,
    cold_bytes: usize,
    /// (session id, lease number) of every session a request has claimed.
    running: Vec<(String, u64)>,
    /// Ids of hot sessions that left `hot` and are being copied to host memory.
    moving: Vec<String>,
    leases: u64,
}

/// What the server keeps of sessions between requests: at most one hot session (its cache on the GPU) and a bounded
/// number of cold ones (host memory, see `cold`). A request claims a session for its whole run, so the cache is never
/// shared and a second request for the same id is refused while the first runs.
#[derive(Default)]
pub(super) struct Store {
    inner: Mutex<Inner>,
    settled: Condvar,
}

/// A hot session on its way to host memory. A claim for its id waits until this ends, so the session is never
/// reported missing while it is only being moved.
struct Moving<'a> {
    store: &'a Store,
    id: String,
}

impl Drop for Moving<'_> {
    fn drop(&mut self) {
        self.store.lock().moving.retain(|id| *id != self.id);
        self.store.settled.notify_all();
    }
}

/// A claim on a session id; dropping it makes the id claimable again.
pub(super) struct Lease<'a> {
    store: &'a Store,
    id: String,
    number: u64,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.store
            .lock()
            .running
            .retain(|(id, number)| !(*id == self.id && *number == self.number));
    }
}

impl Store {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn claim(&self, id: &str) -> Result<(Lease<'_>, Entry), Refused> {
        let mut inner = self.lock();
        while inner.moving.iter().any(|moving| moving == id) {
            inner = self
                .settled
                .wait(inner)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if inner.running.iter().any(|(running, _)| running == id) {
            return Err(Refused::Running);
        }
        let found = if inner.hot.as_ref().is_some_and(|state| state.id == id) {
            inner.hot.take().map(Entry::Hot)
        } else {
            let at = inner.cold.iter().position(|cold| cold.id == id);
            let cold = at.and_then(|at| inner.cold.remove(at));
            if let Some(cold) = &cold {
                inner.cold_bytes = inner.cold_bytes.saturating_sub(cold.bytes());
            }
            cold.map(Entry::Cold)
        };
        let entry = found.ok_or(Refused::Missing)?;
        inner.leases = inner.leases.wrapping_add(1);
        let number = inner.leases;
        inner.running.push((id.to_owned(), number));
        Ok((
            Lease {
                store: self,
                id: id.to_owned(),
                number,
            },
            entry,
        ))
    }

    /// Makes room for a request that will build a cache: the hot session, if any, goes cold.
    pub fn make_room(&self, sched: Scheduled<'_>) {
        if let Some((old, moving)) = self.displace(None) {
            self.shelve(old, moving, sched);
        }
    }

    /// Makes `state` the hot session; the one it replaces goes cold.
    pub fn put(&self, state: Stored, sched: Scheduled<'_>) {
        if let Some((old, moving)) = self.displace(Some(state)) {
            self.shelve(old, moving, sched);
        }
    }

    /// Replaces the hot session and marks the old one as moving in the same step.
    fn displace(&self, new: Option<Stored>) -> Option<(Stored, Moving<'_>)> {
        let mut inner = self.lock();
        let old = std::mem::replace(&mut inner.hot, new)?;
        inner.moving.push(old.id.clone());
        let moving = Moving {
            store: self,
            id: old.id.clone(),
        };
        Some((old, moving))
    }

    fn shelve(&self, old: Stored, moving: Moving<'_>, sched: Scheduled<'_>) {
        if let Ok(Some(cold)) = sched.spill(&old) {
            drop(old);
            self.add_cold(cold);
        }
        drop(moving);
    }

    /// Forgets every cold session, for weights that are no longer the ones they were made with.
    pub fn clear_cold(&self) {
        let mut inner = self.lock();
        inner.cold.clear();
        inner.cold_bytes = 0;
    }

    /// Keeps `cold` as the most recent cold session; the oldest go while the count or the bytes are over the limits.
    pub fn add_cold(&self, cold: Cold) {
        let mut inner = self.lock();
        if cold.bytes() > COLD_BYTES {
            return;
        }
        if let Some(at) = inner.cold.iter().position(|old| old.id == cold.id)
            && let Some(old) = inner.cold.remove(at)
        {
            inner.cold_bytes = inner.cold_bytes.saturating_sub(old.bytes());
        }
        inner.cold_bytes = inner.cold_bytes.saturating_add(cold.bytes());
        inner.cold.push_back(cold);
        while inner.cold.len() > COLD_SESSIONS || inner.cold_bytes > COLD_BYTES {
            match inner.cold.pop_front() {
                Some(oldest) => inner.cold_bytes = inner.cold_bytes.saturating_sub(oldest.bytes()),
                None => break,
            }
        }
    }

    /// Puts the session back as it was before a failed or cancelled request. A new session leaves nothing behind.
    /// `fallback` is the host copy of a session that the request replaced: it is thawed once the request's own cache
    /// is released, and if the GPU still has no memory it stays cold, so the session is never lost to the pressure.
    /// `mark` is the end of the conversation the request had already read: the cache returns to it exactly, with no
    /// memory needed, instead of reading the previous history again.
    pub fn recover(
        &self,
        mut active: Stored,
        sched: Scheduled<'_>,
        previous: &[u32],
        existing: bool,
        fallback: Option<Cold>,
        mark: Option<(Mark, &[u32])>,
    ) {
        if !existing {
            return;
        }
        if let Some(cold) = fallback {
            drop(active);
            match sched.thaw(&cold) {
                Ok(state) => self.put(state, sched),
                Err(_) => self.add_cold(cold),
            }
            return;
        }
        if let Some((mark, history)) = mark
            && sched.rewind(&mut active.cache, mark).is_ok()
        {
            active.retained = Retained::restored(history.to_vec());
            self.put(active, sched);
            return;
        }
        active.cache.clear();
        if !previous.is_empty() && sched.forward(previous, &mut active.cache).is_ok() {
            active.retained = Retained::restored(previous.to_vec());
        } else {
            active.cache.clear();
            active.retained.forget();
        }
        self.put(active, sched);
    }
}

pub(super) fn checkpoint(
    active: &mut Stored,
    sched: Scheduled<'_>,
    history: &[u32],
) -> Result<(Option<Mark>, usize), ScheduleError> {
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
    Ok((sched.mark(&active.cache)?, keep))
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
