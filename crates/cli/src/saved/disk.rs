//! Session files on disk: one private folder, plain names, whole files only. A file is written complete under a
//! temporary name and renamed into place, so a reader sees the old session or the new one, never half of either.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::check::valid_name;
use super::{MAX_FILE, MAX_SESSIONS, Saved, SavedError};

/// A temporary file older than this was left by a run that died while writing.
const STALE: Duration = Duration::from_secs(600);

pub struct Store {
    dir: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Written {
    Created(PathBuf),
    Replaced(PathBuf),
}

pub struct Summary {
    pub name: String,
    pub title: String,
    pub saved_at: u64,
    pub model: String,
    pub exchanges: (usize, usize),
    pub code: bool,
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
        Self { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, name: &str) -> Result<PathBuf, SavedError> {
        if valid_name(name) {
            Ok(self.dir.join(format!("{name}.json")))
        } else {
            Err(SavedError::BadName(name.to_owned()))
        }
    }

    /// Checks the folder is a real one only this user can enter, making it when `create` is set. False if it does
    /// not exist and was not made.
    fn ready(&self, create: bool) -> Result<bool, SavedError> {
        let unsafe_dir = |reason: String| SavedError::UnsafeDir {
            path: self.dir.clone(),
            reason,
        };
        match fs::symlink_metadata(&self.dir) {
            Ok(meta) if meta.is_dir() => {
                if meta.mode() & 0o077 != 0 {
                    return Err(unsafe_dir(format!(
                        "other users can enter it (mode {:o})",
                        meta.mode() & 0o777
                    )));
                }
                Ok(true)
            }
            Ok(_) => Err(unsafe_dir("it is a link or a file, not a folder".into())),
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

    /// Writes `saved` under its name. An existing session of that name is replaced; any other file there is not.
    pub fn save(&self, saved: &Saved) -> Result<Written, SavedError> {
        let target = self.path(&saved.name)?;
        saved.check().map_err(|reason| SavedError::Corrupt {
            path: target.clone(),
            reason,
        })?;
        let bytes = serde_json::to_vec(saved).map_err(|e| io_error(&target, &e.into()))?;
        if bytes.len() as u64 > MAX_FILE {
            return Err(SavedError::TooBig {
                path: target,
                size: bytes.len() as u64,
            });
        }
        self.ready(true)?;
        let existed = match fs::symlink_metadata(&target) {
            Ok(_) => {
                self.load(&saved.name).map_err(|e| SavedError::NotOurs {
                    path: target.clone(),
                    reason: e.to_string(),
                })?;
                true
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(io_error(&target, &e)),
        };
        if !existed && self.names()?.len() >= MAX_SESSIONS {
            return Err(SavedError::TooMany {
                dir: self.dir.clone(),
            });
        }
        self.sweep();
        self.write_whole(&target, &bytes)?;
        Ok(if existed {
            Written::Replaced(target)
        } else {
            Written::Created(target)
        })
    }

    fn write_whole(&self, target: &Path, bytes: &[u8]) -> Result<(), SavedError> {
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
            .and_then(|mut file| file.write_all(bytes).and_then(|()| file.sync_all()))
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
