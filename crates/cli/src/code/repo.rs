use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use super::call::Call;

pub(super) const SCAN_BYTES: usize = 8192;
const MAX_FILE: u64 = 1 << 20;
const ELLIPSIS: &str = " …";

/// What a tool shows the model, and how the terminal summarizes it.
pub struct Output {
    pub text: String,
    pub label: String,
    pub note: String,
    pub ok: bool,
}

impl Output {
    pub(super) fn failed(label: String, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self {
            text: format!("error: {reason}"),
            label,
            note: reason,
            ok: false,
        }
    }
}

pub(super) struct Limits {
    /// Most characters one result may hold, so no single result can crowd out the conversation.
    pub chars: usize,
    pub search_time: Duration,
}

/// A directory the tools may look into, and nothing outside it.
pub struct Repo {
    root: PathBuf,
    pub(super) limits: Limits,
}

impl Repo {
    pub fn open(path: &Path, context: usize) -> std::io::Result<Self> {
        Ok(Self {
            root: path.canonicalize()?,
            limits: Limits {
                chars: (context / 2).clamp(1500, 12_000),
                search_time: Duration::from_secs(4),
            },
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The names at the top of the project, for the model to start from.
    pub fn overview(&self) -> String {
        let mut names: Vec<String> = fs::read_dir(&self.root)
            .map(|dir| {
                dir.filter_map(Result::ok)
                    .filter(|e| {
                        !matches!(
                            e.file_name().to_str(),
                            Some(".git" | "target" | "node_modules")
                        )
                    })
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if e.file_type().is_ok_and(|k| k.is_dir()) {
                            format!("{name}/")
                        } else {
                            name
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        let more = names.len().saturating_sub(40);
        names.truncate(40);
        let mut text = names.join(", ");
        if more > 0 {
            text.push_str(&format!(", and {more} more"));
        }
        text
    }

    pub fn name(&self) -> String {
        self.root.file_name().map_or_else(
            || self.root.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        )
    }

    /// The real path of `asked` if it is inside the root once every symlink is followed.
    pub(super) fn resolve(&self, asked: &str) -> Result<PathBuf, String> {
        let asked = asked.trim();
        if asked.contains('\0') {
            return Err("the path contains a NUL byte".into());
        }
        let joined = if asked.is_empty() || asked == "." {
            self.root.clone()
        } else {
            let path = Path::new(asked);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.root.join(path)
            }
        };
        let real = joined.canonicalize().map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!("`{asked}` does not exist"),
            std::io::ErrorKind::PermissionDenied => format!("`{asked}`: permission denied"),
            _ => format!("`{asked}`: {e}"),
        })?;
        if real.starts_with(&self.root) {
            Ok(real)
        } else {
            Err(format!(
                "`{asked}` is outside the project, which is all these tools can see"
            ))
        }
    }

    pub(super) fn relative(&self, path: &Path) -> String {
        match path.strip_prefix(&self.root) {
            Ok(rel) if rel.as_os_str().is_empty() => ".".into(),
            Ok(rel) => rel.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }

    pub fn run(&self, call: &Call, cancel: &AtomicBool) -> Output {
        let mut out = match call {
            Call::List { path, depth } => self.list(path, *depth),
            Call::Read { path, start, end } => self.read(path, *start, *end),
            Call::Search { pattern, path } => self.search(pattern, path, cancel),
        };
        out.text = neutralize(&out.text);
        out.text = clip(out.text, self.limits.chars);
        out
    }
}

/// File text must not be able to pass for the protocol: the tags are broken up so they never read as tool markup.
fn neutralize(text: &str) -> String {
    text.replace("<tool_", "<\u{200b}tool_")
        .replace("</tool_", "</\u{200b}tool_")
}

fn clip(mut text: String, chars: usize) -> String {
    if text.chars().count() > chars {
        let cut = text
            .char_indices()
            .nth(chars)
            .map_or(text.len(), |(at, _)| at);
        text.truncate(cut);
        text.push_str("\n[output cut here]");
    }
    text
}

pub(super) fn shorten(line: &str, chars: usize) -> String {
    if line.chars().count() <= chars {
        return line.to_owned();
    }
    let mut short: String = line.chars().take(chars).collect();
    short.push_str(ELLIPSIS);
    short
}

/// The text of a file that is small enough and not binary, or the reason it cannot be shown.
pub(super) fn read_text(path: &Path, shown: &str) -> Result<String, String> {
    let meta = fs::metadata(path).map_err(|e| format!("cannot read `{shown}`: {e}"))?;
    if meta.is_dir() {
        return Err(format!("`{shown}` is a directory; use list_files"));
    }
    if meta.len() > MAX_FILE {
        return Err(format!(
            "`{shown}` is {} bytes, too large to read; search it instead",
            meta.len()
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("cannot read `{shown}`: {e}"))?;
    if bytes.iter().take(SCAN_BYTES).any(|b| *b == 0) {
        return Err(format!(
            "`{shown}` is a binary file ({} bytes)",
            bytes.len()
        ));
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
