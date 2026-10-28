use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::inbox::Inbox;

const MAX_FILE: u64 = 1 << 20;

/// Everything the assistant keeps lives in one private folder: the inbox and the repos it may watch.
pub struct Store {
    dir: PathBuf,
}

fn private_dir(dir: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(dir)
}

impl Store {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn read<T: DeserializeOwned + Default>(&self, name: &str) -> T {
        let Ok(file) = fs::File::open(self.dir.join(name)) else {
            return T::default();
        };
        let mut text = String::new();
        let read = file.take(MAX_FILE).read_to_string(&mut text);
        read.ok()
            .and_then(|_| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn write(&self, name: &str, value: &impl Serialize) -> io::Result<()> {
        private_dir(&self.dir)?;
        let temp = self.dir.join(format!(".{name}.tmp"));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(value).map_err(io::Error::other)?)?;
        file.sync_all()?;
        fs::rename(temp, self.dir.join(name))
    }

    /// A missing or damaged inbox is an empty one: suggestions are disposable.
    pub fn inbox(&self) -> Inbox {
        self.read("inbox.json")
    }

    pub fn save_inbox(&self, inbox: &Inbox) -> io::Result<()> {
        self.write("inbox.json", inbox)
    }

    pub fn repos(&self) -> Vec<String> {
        self.read("repos.json")
    }

    pub fn save_repos(&self, repos: &[String]) -> io::Result<()> {
        self.write("repos.json", &repos)
    }
}

/// Where the socket lives: the runtime folder when there is one, else beside the inbox. Always a private folder.
pub fn socket_path(store: &Store) -> io::Result<PathBuf> {
    let dir = match std::env::var_os("XDG_RUNTIME_DIR").filter(|v| !v.is_empty()) {
        Some(run) if std::env::var_os("MARIN_HOME").is_none() => PathBuf::from(run).join("marin"),
        _ => store.dir().to_path_buf(),
    };
    private_dir(&dir)?;
    Ok(dir.join("assist.sock"))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::assist::inbox::Event;

    #[test]
    fn files_are_private_and_a_damaged_inbox_is_empty() {
        let dir = std::env::temp_dir().join(format!("marin-assist-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::new(dir.join("a"));
        let mut inbox = Inbox::default();
        inbox.offer(
            Event {
                repo: "/r".into(),
                command: "make".into(),
                exit: 2,
                tail: "error".into(),
            },
            5,
        );
        store.save_inbox(&inbox).unwrap();
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir.join("a")), 0o700);
        assert_eq!(mode(&dir.join("a/inbox.json")), 0o600);
        assert_eq!(store.inbox().items.len(), 1);
        fs::write(dir.join("a/inbox.json"), "{not json").unwrap();
        assert!(store.inbox().items.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
