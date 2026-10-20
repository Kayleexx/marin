//! The checkpoint file beside a session: `<name>.state`, written under the same lock, with the same care, and only for
//! the session revision it was made from. Saving the session again removes it, since it describes the old one.

use std::fs::{self, File};
use std::path::PathBuf;

use super::disk::Store;
use super::{MAX_STATE, MAX_STATES, SavedError};

fn io(path: &std::path::Path, e: &std::io::Error) -> SavedError {
    SavedError::Io {
        path: path.to_path_buf(),
        reason: e.to_string(),
    }
}

impl Store {
    pub(super) fn state_path(&self, name: &str) -> Result<PathBuf, SavedError> {
        Ok(self.path(name)?.with_extension("state"))
    }

    /// Bytes the checkpoints of other sessions take.
    fn states_elsewhere(&self, keep: &std::path::Path) -> u64 {
        fs::read_dir(&self.dir).map_or(0, |entries| {
            entries
                .flatten()
                .filter(|e| e.path() != keep && e.path().extension().is_some_and(|x| x == "state"))
                .filter_map(|e| e.metadata().ok())
                .fold(0, |sum: u64, m| sum.saturating_add(m.len()))
        })
    }

    /// Writes the checkpoint of session `name` through `write`, as long as the session on disk is still at `revision`
    /// and the checkpoints would stay within their limit. `bytes` is how much `write` will produce.
    pub fn save_state(
        &self,
        name: &str,
        revision: u64,
        bytes: u64,
        write: impl FnOnce(&mut File) -> std::io::Result<()>,
    ) -> Result<PathBuf, SavedError> {
        let target = self.state_path(name)?;
        self.ready(true)?;
        let _lock = self.lock()?;
        let current = self.load(name)?;
        if current.revision != revision {
            return Err(SavedError::Conflict {
                name: name.to_owned(),
                found: current.revision,
                seen: Some(revision),
            });
        }
        let used = self.states_elsewhere(&target).saturating_add(bytes);
        if bytes > MAX_STATE || used > MAX_STATES {
            return Err(SavedError::StateSpace {
                dir: self.dir.clone(),
                used,
            });
        }
        self.write_with(&target, write)?;
        Ok(target)
    }

    /// The checkpoint of session `name` and its size, if it has one and it is a plain file of an allowed size.
    pub fn open_state(&self, name: &str) -> Result<Option<(File, u64)>, SavedError> {
        let path = self.state_path(name)?;
        if !self.ready(false)? {
            return Ok(None);
        }
        let meta = match fs::symlink_metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(&path, &e)),
        };
        if !meta.is_file() || meta.len() > MAX_STATE {
            return Err(SavedError::Corrupt {
                path,
                reason: "it is not a plain checkpoint file of an allowed size".into(),
            });
        }
        let file = File::open(&path).map_err(|e| io(&path, &e))?;
        Ok(Some((file, meta.len())))
    }
}
