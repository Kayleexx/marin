//! Session files on disk: one private folder, plain names, whole files only. A file is written complete under a
//! temporary name and renamed into place, so a reader sees the old session or the new one, never half of either.

use std::fs::{self, DirBuilder, File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::check::valid_name;
use super::trust::{Trust, existing_ancestor};
use super::{MAX_FILE, MAX_SESSIONS, Saved, SavedError};

/// A temporary file older than this was left by a run that died while writing.
const STALE: Duration = Duration::from_secs(600);
/// How long a save waits for another run that is saving at the same moment.
const LOCK_WAIT: Duration = Duration::from_secs(5);

pub struct Store {
    pub(super) dir: PathBuf,
    trust: Trust,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Written {
    pub path: PathBuf,
    /// The number the file now carries; the next save of it must say it started from this one.
    pub revision: u64,
    pub replaced: bool,
}

pub struct Summary {
    pub name: String,
    pub title: String,
    pub saved_at: u64,
    pub model: String,
    pub exchanges: (usize, usize),
    pub code: bool,
    /// Size of the checkpoint kept for it, if there is one.
    pub state_bytes: Option<u64>,
}

pub enum Listed {
    Ok(Summary),
    Damaged { name: String, why: String },
}

fn io_error(path: &Path, e: &std::io::Error) -> SavedError {
    SavedError::Io {
        path: path.to_path_buf(),
        reason: e.to_string(),
    }
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Store {
    pub fn new(dir: PathBuf) -> Self {
        Self::trusting(dir, Trust::current())
    }

    pub fn trusting(dir: PathBuf, trust: Trust) -> Self {
        Self { dir, trust }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn path(&self, name: &str) -> Result<PathBuf, SavedError> {
        if valid_name(name) {
            Ok(self.dir.join(format!("{name}.json")))
        } else {
            Err(SavedError::BadName(name.to_owned()))
        }
    }

    /// Checks the folder is a real one only this user can enter, on a path others cannot redirect, making it when
    /// `create` is set. False if it does not exist and was not made.
    pub(super) fn ready(&self, create: bool) -> Result<bool, SavedError> {
        let unsafe_dir = |reason: String| SavedError::UnsafeDir {
            path: self.dir.clone(),
            reason,
        };
        self.trust
            .path_to(&existing_ancestor(&self.dir))
            .map_err(&unsafe_dir)?;
        match fs::symlink_metadata(&self.dir) {
            Ok(_) => self
                .trust
                .private_dir(&self.dir)
                .map_err(unsafe_dir)
                .map(|()| true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && create => {
                if let Some(parent) = self.dir.parent() {
                    fs::create_dir_all(parent).map_err(|e| io_error(parent, &e))?;
                }
                DirBuilder::new()
                    .mode(0o700)
                    .create(&self.dir)
                    .map_err(|e| io_error(&self.dir, &e))?;
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io_error(&self.dir, &e)),
        }
    }

    /// Holds the folder's lock, so that checking a session and replacing it cannot interleave with another run's
    /// save. The lock goes when the returned file is dropped, including if the process dies.
    pub(super) fn lock(&self) -> Result<File, SavedError> {
        let path = self.dir.join(".lock");
        if fs::symlink_metadata(&path).is_ok_and(|m| !m.is_file()) {
            return Err(SavedError::UnsafeDir {
                path,
                reason: "the lock must be a plain file".into(),
            });
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)
            .map_err(|e| io_error(&path, &e))?;
        let started = Instant::now();
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(file),
                Err(TryLockError::WouldBlock) if started.elapsed() < LOCK_WAIT => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(TryLockError::WouldBlock) => {
                    return Err(SavedError::Busy {
                        dir: self.dir.clone(),
                    });
                }
                Err(TryLockError::Error(e)) => return Err(io_error(&path, &e)),
            }
        }
    }

    pub fn load(&self, name: &str) -> Result<Saved, SavedError> {
        let path = self.path(name)?;
        if !self.ready(false)? {
            return Err(SavedError::NotFound(name.to_owned()));
        }
        let meta = match fs::symlink_metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(SavedError::NotFound(name.to_owned()));
            }
            Err(e) => return Err(io_error(&path, &e)),
        };
        if !meta.is_file() {
            return Err(SavedError::Corrupt {
                path,
                reason: "it is a link or a folder, not a file".into(),
            });
        }
        if meta.len() > MAX_FILE {
            return Err(SavedError::TooBig {
                path,
                size: meta.len(),
            });
        }
        let mut bytes = Vec::new();
        File::open(&path)
            .and_then(|file| {
                file.take(MAX_FILE.saturating_add(1))
                    .read_to_end(&mut bytes)
            })
            .map_err(|e| io_error(&path, &e))?;
        if bytes.len() as u64 > MAX_FILE {
            return Err(SavedError::TooBig {
                path,
                size: bytes.len() as u64,
            });
        }
        let saved = Saved::parse(&path, &bytes)?;
        if saved.name == name {
            Ok(saved)
        } else {
            Err(SavedError::Corrupt {
                path,
                reason: format!("it says it is `{}`", saved.name),
            })
        }
    }

    /// Writes `saved` under its name as the next revision. `from` is the revision this conversation was opened at or
    /// last saved as, if it came from this session: a session changed since then, or one this conversation did not
    /// come from, is not replaced. Any other file at that name is not either.
    pub fn save(&self, saved: &Saved, from: Option<u64>) -> Result<Written, SavedError> {
        let target = self.path(&saved.name)?;
        saved.check().map_err(|reason| SavedError::Corrupt {
            path: target.clone(),
            reason,
        })?;
        self.ready(true)?;
        let _lock = self.lock()?;
        let current = match fs::symlink_metadata(&target) {
            Ok(_) => Some(self.load(&saved.name).map_err(|e| SavedError::NotOurs {
                path: target.clone(),
                reason: e.to_string(),
            })?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(io_error(&target, &e)),
        };
        let revision = match (&current, from) {
            (None, _) => 1,
            (Some(kept), Some(seen)) if kept.revision == seen => seen.saturating_add(1),
            (Some(kept), seen) => {
                return Err(SavedError::Conflict {
                    name: saved.name.clone(),
                    found: kept.revision,
                    seen,
                });
            }
        };
        let mut stamped = saved.clone();
        stamped.revision = revision;
        let bytes = serde_json::to_vec(&stamped).map_err(|e| io_error(&target, &e.into()))?;
        if bytes.len() as u64 > MAX_FILE {
            return Err(SavedError::TooBig {
                path: target,
                size: bytes.len() as u64,
            });
        }
        if current.is_none() && self.names()?.len() >= MAX_SESSIONS {
            return Err(SavedError::TooMany {
                dir: self.dir.clone(),
            });
        }
        self.sweep();
        self.write_with(&target, |file| file.write_all(&bytes))?;
        let _ = fs::remove_file(self.state_path(&saved.name)?);
        Ok(Written {
            path: target,
            revision,
            replaced: current.is_some(),
        })
    }

    /// Fills a temporary file through `write`, makes it durable and renames it over `target`, so that `target` is
    /// always a whole file. A failure removes the temporary file and leaves `target` as it was.
    pub(super) fn write_with(
        &self,
        target: &Path,
        write: impl FnOnce(&mut File) -> std::io::Result<()>,
    ) -> Result<(), SavedError> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let temp = self
            .dir
            .join(format!(".{}.{unique}.tmp", std::process::id()));
        let written = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)
            .and_then(|mut file| write(&mut file).and_then(|()| file.sync_all()))
            .and_then(|()| fs::rename(&temp, target));
        if let Err(e) = written {
            let _ = fs::remove_file(&temp);
            return Err(io_error(target, &e));
        }
        if let Ok(dir) = File::open(&self.dir) {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    /// Removes what runs that died mid-write left behind.
    fn sweep(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > STALE);
            if name.starts_with('.') && name.ends_with(".tmp") && stale {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// The names of the saved sessions, sorted.
    fn names(&self) -> Result<Vec<String>, SavedError> {
        if !self.ready(false)? {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&self.dir).map_err(|e| io_error(&self.dir, &e))?;
        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|entry| {
                let file = entry.file_name().to_string_lossy().into_owned();
                file.strip_suffix(".json")
                    .filter(|name| valid_name(name))
                    .map(str::to_owned)
            })
            .collect();
        names.sort();
        Ok(names)
    }

    /// Every saved session, newest first, then the damaged ones. At most `MAX_SESSIONS` are looked at.
    pub fn list(&self) -> Result<Vec<Listed>, SavedError> {
        let mut listed: Vec<Listed> = self
            .names()?
            .into_iter()
            .take(MAX_SESSIONS)
            .map(|name| match self.load(&name) {
                Ok(saved) => Listed::Ok(Summary {
                    title: saved.title(),
                    saved_at: saved.saved_at,
                    model: saved.model.name.clone(),
                    exchanges: saved.exchanges(),
                    code: saved.code.is_some(),
                    state_bytes: fs::symlink_metadata(self.dir.join(format!("{name}.state")))
                        .ok()
                        .filter(|m| m.is_file())
                        .map(|m| m.len()),
                    name,
                }),
                Err(e) => Listed::Damaged {
                    name,
                    why: e.to_string(),
                },
            })
            .collect();
        listed.sort_by_key(|l| match l {
            Listed::Ok(s) => (0, std::cmp::Reverse(s.saved_at)),
            Listed::Damaged { .. } => (1, std::cmp::Reverse(0)),
        });
        Ok(listed)
    }
}
