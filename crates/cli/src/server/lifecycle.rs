//! The one owner of the server's loaded model. With `--idle-unload` the model (and everything on the GPU that depends on
//! it: the shared prefix, the hot session) is released after a quiet period and loaded again, by the same loader call
//! that started the server, when the next completion arrives. Logical sessions survive a sleep as host bytes (see
//! `cold`). A model is never released while a request holds it, and without the option nothing here ever sleeps.

use std::fs;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use super::gate::Gate;
use super::prefix::Prefix;
use super::sched::Scheduled;
use super::session::Store;
use crate::CliError;
use crate::load::Loaded;

pub(super) type Reload<'a> = &'a (dyn Fn() -> Result<Loaded, CliError> + Sync);

/// The model with what is built on it.
pub(super) struct Awake {
    pub loaded: Loaded,
    pub prefix: Option<Prefix>,
}

enum State {
    Awake { awake: Arc<Awake>, last: Instant },
    Sleeping,
    Asleep,
    Waking,
}

/// What the server can say about the model while it is not loaded.
pub(super) struct Info {
    pub name: String,
    pub ctx: usize,
}

pub(super) struct Lifecycle<'a> {
    state: Mutex<State>,
    changed: Condvar,
    reload: Reload<'a>,
    idle: Option<Duration>,
    gate: &'a Gate,
    store: &'a Store,
    fingerprint: Mutex<Vec<FileMark>>,
    pub info: Info,
}

/// A model file as it was when the model last loaded; cold sessions are only valid for those exact weights.
type FileMark = (String, u64, Option<SystemTime>);

fn fingerprint(dir: &Path) -> Vec<FileMark> {
    let mut marks: Vec<FileMark> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok().filter(fs::Metadata::is_file)?;
            Some((
                entry.file_name().to_string_lossy().into_owned(),
                meta.len(),
                meta.modified().ok(),
            ))
        })
        .collect();
    marks.sort();
    marks
}

/// Holds the model for one request; the model stays loaded while any guard lives.
pub(super) struct Guard<'g, 'a> {
    life: &'g Lifecycle<'a>,
    pub awake: Arc<Awake>,
}

impl Drop for Guard<'_, '_> {
    fn drop(&mut self) {
        if let State::Awake { last, .. } = &mut *self.life.lock() {
            *last = Instant::now();
        }
    }
}

impl<'a> Lifecycle<'a> {
    pub fn new(
        loaded: Loaded,
        reload: Reload<'a>,
        idle: Option<Duration>,
        gate: &'a Gate,
        store: &'a Store,
    ) -> Self {
        let info = Info {
            name: loaded.name.clone(),
            ctx: loaded.ctx,
        };
        let marks = fingerprint(&loaded.dir.path);
        let prefix = Prefix::build(&loaded);
        Self {
            state: Mutex::new(State::Awake {
                awake: Arc::new(Awake { loaded, prefix }),
                last: Instant::now(),
            }),
            changed: Condvar::new(),
            reload,
            idle,
            gate,
            store,
            fingerprint: Mutex::new(marks),
            info,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The loaded model, loading it first if it is asleep. One caller loads while the others wait for it; if the load
    /// fails the model stays asleep, every session is kept, and the callers are told to try again later.
    pub fn acquire(&self) -> Result<Guard<'_, 'a>, String> {
        let mut state = self.lock();
        loop {
            match &*state {
                State::Awake { awake, .. } => {
                    let awake = Arc::clone(awake);
                    return Ok(Guard { life: self, awake });
                }
                State::Sleeping | State::Waking => {
                    state = self
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                State::Asleep => {
                    *state = State::Waking;
                    drop(state);
                    let woken = self.wake();
                    let mut state = self.lock();
                    let result = match woken {
                        Ok(awake) => {
                            let awake = Arc::new(awake);
                            *state = State::Awake {
                                awake: Arc::clone(&awake),
                                last: Instant::now(),
                            };
                            Ok(Guard { life: self, awake })
                        }
                        Err(message) => {
                            *state = State::Asleep;
                            Err(message)
                        }
                    };
                    drop(state);
                    self.changed.notify_all();
                    return result;
                }
            }
        }
    }

    fn wake(&self) -> Result<Awake, String> {
        let started = Instant::now();
        let loaded = (self.reload)().map_err(|e| format!("the model could not be loaded: {e}"))?;
        loaded.model.bind_to_thread().map_err(|e| e.to_string())?;
        let marks = fingerprint(&loaded.dir.path);
        let mut kept = self
            .fingerprint
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if *kept != marks {
            eprintln!(
                "the model files changed while it slept: sessions kept in memory are dropped"
            );
            self.store.clear_cold();
            *kept = marks;
        }
        drop(kept);
        let prefix = Prefix::build(&loaded);
        eprintln!("model awake in {:.1} s", started.elapsed().as_secs_f64());
        Ok(Awake { loaded, prefix })
    }

    pub fn idle(&self) -> Option<Duration> {
        self.idle
    }

    /// Releases the model if nobody holds it and it has been quiet for the whole idle time. The hot session goes cold
    /// first, so every session that fits the cold limits is still there for the next request.
    pub fn sleep_if_idle(&self) {
        let Some(idle) = self.idle else {
            return;
        };
        let awake = {
            let mut state = self.lock();
            let quiet = matches!(&*state, State::Awake { awake, last }
                if Arc::strong_count(awake) == 1 && last.elapsed() >= idle);
            if !quiet {
                return;
            }
            match std::mem::replace(&mut *state, State::Sleeping) {
                State::Awake { awake, .. } => awake,
                other => {
                    *state = other;
                    return;
                }
            }
        };
        if awake.loaded.model.bind_to_thread().is_ok() {
            self.store
                .make_room(Scheduled::new(&awake.loaded, self.gate, u64::MAX));
        }
        match Arc::try_unwrap(awake) {
            Ok(owned) => {
                let Awake { loaded, prefix } = owned;
                drop(prefix);
                drop(loaded);
                eprintln!("model asleep after {} s idle", idle.as_secs());
                *self.lock() = State::Asleep;
            }
            Err(shared) => {
                *self.lock() = State::Awake {
                    awake: shared,
                    last: Instant::now(),
                };
            }
        }
        self.changed.notify_all();
    }
}
