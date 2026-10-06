use std::fmt::Display;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::CliError;

/// A result file that only appears once it is complete. The destination is checked when this is created,
/// before a long measurement starts, and a failed or abandoned write leaves nothing behind.
pub struct JsonOut {
    path: PathBuf,
    temp: PathBuf,
    file: File,
    done: bool,
}

fn failed(path: &Path, reason: impl Display) -> CliError {
    CliError::Output {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}

impl JsonOut {
    pub fn create(path: &Path) -> Result<Self, CliError> {
        if path.is_dir() {
            return Err(failed(path, "it is a folder"));
        }
        let name = path
            .file_name()
            .ok_or_else(|| failed(path, "it is not a file name"))?
            .to_string_lossy();
        let temp = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
        let file = File::create(&temp).map_err(|e| failed(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            temp,
            file,
            done: false,
        })
    }

    pub fn finish(mut self, value: &impl Serialize) -> Result<(), CliError> {
        let text = serde_json::to_string_pretty(value).map_err(|e| failed(&self.path, e))?;
        self.file
            .write_all(text.as_bytes())
            .and_then(|()| self.file.write_all(b"\n"))
            .and_then(|()| self.file.sync_all())
            .map_err(|e| failed(&self.path, e))?;
        fs::rename(&self.temp, &self.path).map_err(|e| failed(&self.path, e))?;
        self.done = true;
        Ok(())
    }
}

impl Drop for JsonOut {
    fn drop(&mut self) {
        if !self.done {
            let _ = fs::remove_file(&self.temp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("marin-json-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_file_appears_complete_and_the_temporary_one_is_gone() {
        let dir = dir("ok");
        let path = dir.join("out.json");
        let out = JsonOut::create(&path).unwrap();
        assert!(!path.exists(), "nothing is visible before finish");
        out.finish(&serde_json::json!({"a": 1})).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["a"], 1);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    }

    #[test]
    fn an_abandoned_write_leaves_nothing_and_keeps_an_older_file() {
        let dir = dir("abandon");
        let path = dir.join("out.json");
        fs::write(&path, "old").unwrap();
        drop(JsonOut::create(&path).unwrap());
        assert_eq!(fs::read_to_string(&path).unwrap(), "old");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    }

    #[test]
    fn unwritable_destinations_fail_up_front() {
        let dir = dir("bad");
        assert!(JsonOut::create(&dir.join("missing").join("out.json")).is_err());
        assert!(JsonOut::create(Path::new("/")).is_err());
        assert!(JsonOut::create(&dir).is_err());
    }
}
