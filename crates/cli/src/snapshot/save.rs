use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use model::{Engine, EngineCache};
use runtime::SequenceCache;

use super::format::{Header, Identity, end_mark, ids_hash, page_hash};

pub struct Saved {
    pub tokens: usize,
    pub bytes: usize,
    pub took: Duration,
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{}.tmp", std::process::id()));
    PathBuf::from(name)
}

fn write_all(
    file: &mut File,
    engine: &Engine,
    cache: &EngineCache,
    ids: &[u32],
    identity: Identity,
) -> Result<usize, String> {
    let io = |e: std::io::Error| e.to_string();
    let tokens = u32::try_from(ids.len()).map_err(|e| e.to_string())?;
    let header = Header {
        identity,
        tokens,
        ids_hash: ids_hash(ids),
    };
    let expected = header.file_bytes().ok_or("snapshot size overflows")?;
    file.write_all(&header.encode()).map_err(io)?;
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    file.write_all(&id_bytes).map_err(io)?;
    for page in 0..ids.len().div_ceil(model::PAGE_TOKENS) {
        let payload = engine
            .export_kv_page(cache, page, header.rows_of(page))
            .map_err(|e| e.to_string())?;
        file.write_all(&payload).map_err(io)?;
        file.write_all(&page_hash(page, &payload).to_le_bytes())
            .map_err(io)?;
    }
    file.write_all(&end_mark(expected)).map_err(io)?;
    file.sync_all().map_err(io)?;
    Ok(expected)
}

/// Writes `cache`, which must hold exactly the tokens `ids`, to a temporary file next
/// to `path`, flushes it to disk and renames it over `path`, so a reader sees the old file or the whole new one.
pub fn save(
    path: &Path,
    engine: &Engine,
    cache: &EngineCache,
    ids: &[u32],
    identity: Identity,
) -> Result<Saved, String> {
    let started = Instant::now();
    if ids.is_empty() {
        return Err("nothing to save".into());
    }
    if cache.used() != ids.len() {
        return Err(format!(
            "the cache holds {} positions but {} tokens were named",
            cache.used(),
            ids.len()
        ));
    }
    let temp = temp_path(path);
    let result = File::create(&temp)
        .map_err(|e| e.to_string())
        .and_then(|mut file| write_all(&mut file, engine, cache, ids, identity))
        .and_then(|bytes| {
            fs::rename(&temp, path).map_err(|e| e.to_string())?;
            let folder = path.parent().filter(|p| !p.as_os_str().is_empty());
            if let Some(folder) = folder {
                File::open(folder)
                    .and_then(|dir| dir.sync_all())
                    .map_err(|e| e.to_string())?;
            }
            Ok(bytes)
        });
    match result {
        Ok(bytes) => Ok(Saved {
            tokens: ids.len(),
            bytes,
            took: started.elapsed(),
        }),
        Err(reason) => {
            let _ = fs::remove_file(&temp);
            Err(reason)
        }
    }
}
