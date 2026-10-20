//! Which reference file `/verify` uses: the one named, or one found beside the model or in the project's usual places
//! that was made for this model. A file for something else is never run: its numbers would say nothing about this model.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::can_read;
use crate::CliError;
use crate::load::Loaded;
use crate::verify::{Reference, read_reference, stored_weights};

/// A reference is never larger than this; bigger files are refused unread.
const LARGEST: u64 = 2 << 30;

fn refuse(path: &Path, reason: impl Into<String>) -> CliError {
    CliError::Reference {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}

/// Reads `path` if there is room for it, it can be a reference and it was made for `loaded`.
fn read_for(loaded: &Loaded, path: &Path, window: usize) -> Result<Reference, CliError> {
    let size = fs::metadata(path)
        .map_err(|e| refuse(path, e.to_string()))
        .and_then(|m| {
            if m.is_file() {
                Ok(m.len())
            } else {
                Err(refuse(path, "it is not a file"))
            }
        })?;
    if size > LARGEST {
        return Err(refuse(path, "it is larger than any reference should be"));
    }
    can_read(path, size)?;
    let reference = read_reference(path)?;
    if reference.cases() == 0 {
        return Err(refuse(path, "it holds no cases, so it checks nothing"));
    }
    reference.fits(loaded).map_err(|why| refuse(path, why))?;
    if reference.needed_context() > window {
        return Err(refuse(
            path,
            format!(
                "it needs a window of {} tokens and this session's is {window}",
                reference.needed_context()
            ),
        ));
    }
    Ok(reference)
}

pub(super) fn named(
    loaded: &Loaded,
    path: &Path,
    window: usize,
) -> Result<(PathBuf, Reference), CliError> {
    read_for(loaded, path, window).map(|reference| (path.to_path_buf(), reference))
}

/// The `"weights"` a reference file declares, read from its first bytes without parsing the file.
fn declared(path: &Path) -> Option<String> {
    let mut head = vec![0_u8; 512];
    let n = File::open(path).ok()?.read(&mut head).ok()?;
    let text = String::from_utf8_lossy(head.get(..n)?).into_owned();
    let after = text.split_once("\"weights\"")?.1;
    let value = after
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')?;
    Some(value.split('"').next()?.to_owned())
}

fn looks_like_one(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.ends_with(".json") && (name.contains("reference") || name.starts_with("hf-"))
}

fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && looks_like_one(p))
        .collect();
    found.sort();
    found
}

fn subfolders(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    found.sort();
    found
}

/// The places looked in, in order: the model's own folder, then `tests/reference`, `tests/fixtures/*` and
/// `bench-results/*` of the project in `here`.
pub(super) fn places(loaded: &Loaded, here: &Path) -> Vec<PathBuf> {
    let mut places = vec![loaded.dir.path.clone(), here.join("tests/reference")];
    places.extend(subfolders(&here.join("tests/fixtures")));
    places.extend(subfolders(&here.join("tests/reference")));
    places.extend(subfolders(&here.join("bench-results")));
    places
}

/// The first reference in the usual places that was made for `loaded`, or an account of what was found instead.
pub(super) fn find(
    loaded: &Loaded,
    here: &Path,
    window: usize,
) -> Result<(PathBuf, Reference), CliError> {
    let wanted = stored_weights(loaded);
    let mut seen = Vec::new();
    for place in places(loaded, here) {
        for path in files_in(&place) {
            if declared(&path).is_some_and(|kind| kind != wanted) {
                continue;
            }
            match read_for(loaded, &path, window) {
                Ok(reference) => return Ok((path, reference)),
                Err(CliError::Reference { reason, .. }) => {
                    seen.push(format!("{}: {reason}", path.display()));
                }
                Err(other) => return Err(other),
            }
        }
    }
    let looked = if seen.is_empty() {
        "no reference file".to_owned()
    } else {
        let more = seen.len().saturating_sub(3);
        let shown: Vec<_> = seen.into_iter().take(3).collect();
        let and = if more > 0 {
            format!("; and {more} more")
        } else {
            String::new()
        };
        format!(
            "only references for something else ({}{and})",
            shown.join("; ")
        )
    };
    Err(CliError::Reference {
        path: here.display().to_string(),
        reason: format!(
            "no reference made for {} with {wanted} weights was found ({looked}). It needs a float32 Hugging Face reference of this model's outputs (see tests/reference for the format); put it in the model's folder or tests/reference, or name it: /verify PATH",
            loaded.name
        ),
    })
}
