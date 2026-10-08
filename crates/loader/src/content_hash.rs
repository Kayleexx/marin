//! A 64-bit hash of file contents for telling model versions apart. It catches changed or damaged files, not a
//! deliberate forgery. The standard hasher is deterministic for a given Rust release; another release can give another
//! value, which only makes a saved snapshot look foreign and so unusable.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::Hasher;
use std::io::Read;
use std::path::Path;

use crate::LoaderError;

const CHUNK_BYTES: usize = 1 << 22;

/// Adds the length and the contents of the file at `path` to `hasher`.
pub(crate) fn add_file(hasher: &mut DefaultHasher, path: &Path) -> Result<(), LoaderError> {
    let io = |source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    hasher.write_u64(file.metadata().map_err(io)?.len());
    let mut chunk = vec![0_u8; CHUNK_BYTES];
    loop {
        let read = file.read(&mut chunk).map_err(io)?;
        if read == 0 {
            return Ok(());
        }
        hasher.write(chunk.get(..read).unwrap_or_default());
    }
}

pub(crate) fn hash_files(paths: &[std::path::PathBuf]) -> Result<u64, LoaderError> {
    let mut hasher = DefaultHasher::new();
    for path in paths {
        add_file(&mut hasher, path)?;
    }
    Ok(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contents_and_length_both_change_the_hash() {
        let dir = std::env::temp_dir().join(format!("marin-hash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, c) = (dir.join("a"), dir.join("b"), dir.join("c"));
        std::fs::write(&a, b"abcdef").unwrap();
        std::fs::write(&b, b"abcdeg").unwrap();
        std::fs::write(&c, b"abcdef\0").unwrap();
        let hash = |paths: &[&std::path::Path]| {
            hash_files(&paths.iter().map(|p| p.to_path_buf()).collect::<Vec<_>>()).unwrap()
        };
        assert_eq!(hash(&[&a]), hash(&[&a]));
        assert_ne!(hash(&[&a]), hash(&[&b]));
        assert_ne!(hash(&[&a]), hash(&[&c]));
        assert_ne!(hash(&[&a, &b]), hash(&[&b, &a]));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
