//! Whether a place on disk can be trusted to hold private files: it and everything above it must belong to this user
//! or root, and nobody else may be able to rename things in the folders on the way there.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

/// Who is running Marin, as far as ownership is concerned.
#[derive(Clone, Copy)]
pub struct Trust {
    owner: Option<u32>,
}

impl Trust {
    /// The user running this process: the owner of `/proc/self`.
    pub fn current() -> Self {
        Self {
            owner: fs::metadata("/proc/self").ok().map(|m| m.uid()),
        }
    }

    pub fn as_user(uid: u32) -> Self {
        Self { owner: Some(uid) }
    }

    fn owner(self) -> Result<u32, String> {
        self.owner
            .ok_or_else(|| "cannot tell which user is running Marin".to_owned())
    }

    /// The folder itself: a real folder of this user that no one else can enter.
    pub fn private_dir(self, dir: &Path) -> Result<(), String> {
        let meta = fs::symlink_metadata(dir).map_err(|e| e.to_string())?;
        if !meta.is_dir() {
            return Err("it is a link or a file, not a folder".into());
        }
        if meta.uid() != self.owner()? {
            return Err(format!("it belongs to another user (uid {})", meta.uid()));
        }
        if meta.mode() & 0o077 != 0 {
            return Err(format!(
                "other users can enter it (mode {:o})",
                meta.mode() & 0o777
            ));
        }
        Ok(())
    }

    /// The path down to `anchor`, an existing place on it: no link on the way belongs to someone else, and every real
    /// folder it resolves through is ours or root's and cannot be changed by others (unless sticky, like `/tmp`).
    pub fn path_to(self, anchor: &Path) -> Result<(), String> {
        let owner = self.owner()?;
        let ours = |uid: u32| uid == 0 || uid == owner;
        for step in anchor.ancestors() {
            if let Ok(meta) = fs::symlink_metadata(step)
                && meta.file_type().is_symlink()
                && !ours(meta.uid())
            {
                return Err(format!("{} is a link made by another user", step.display()));
            }
        }
        let real = fs::canonicalize(anchor).map_err(|e| format!("{}: {e}", anchor.display()))?;
        for step in real.ancestors() {
            let meta = fs::metadata(step).map_err(|e| format!("{}: {e}", step.display()))?;
            if !ours(meta.uid()) {
                return Err(format!("{} belongs to another user", step.display()));
            }
            let writable_by_others = meta.mode() & 0o022 != 0 && meta.mode() & 0o1000 == 0;
            if writable_by_others {
                return Err(format!(
                    "{} can be changed by other users (mode {:o})",
                    step.display(),
                    meta.mode() & 0o7777
                ));
            }
        }
        Ok(())
    }
}

/// The nearest place on `path` that exists, `path` itself if it does.
pub fn existing_ancestor(path: &Path) -> PathBuf {
    path.ancestors()
        .find(|p| fs::symlink_metadata(p).is_ok())
        .unwrap_or(Path::new("/"))
        .to_path_buf()
}
