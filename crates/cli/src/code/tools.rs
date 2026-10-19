use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::repo::{Output, Repo, SCAN_BYTES, read_text, shorten};

const MAX_ENTRIES: usize = 100;
const READ_ENTRIES: usize = 10_000;
const MAX_DEPTH: usize = 3;
const READ_LINES: usize = 250;
const LINE_CHARS: usize = 300;
const MAX_MATCHES: usize = 40;
const MATCH_CHARS: usize = 200;
const SKIPPED: [&str; 3] = [".git", "target", "node_modules"];
const MAX_SEARCHED_FILE: u64 = 1 << 20;

fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

impl Repo {
    pub(super) fn list(&self, asked: &str, depth: Option<usize>) -> Output {
        let label = format!("list {asked}");
        let dir = match self.resolve(asked) {
            Ok(dir) => dir,
            Err(reason) => return Output::failed(label, reason),
        };
        let shown = self.relative(&dir);
        let label = format!("list {shown}");
        if !dir.is_dir() {
            return Output::failed(label, format!("`{shown}` is a file; use read_file"));
        }
        let mut lines = Vec::new();
        let mut total = 0;
        let levels = depth.unwrap_or(1).clamp(1, MAX_DEPTH);
        if let Err(e) = tree(&dir, levels, 0, &mut lines, &mut total) {
            return Output::failed(label, format!("cannot list `{shown}`: {e}"));
        }
        let mut text = format!("{shown}/ ({})\n", count(total, "entry", "entries"));
        for line in lines.iter().take(MAX_ENTRIES) {
            text.push_str(line);
            text.push('\n');
        }
        if total > MAX_ENTRIES {
            text.push_str(&format!(
                "... and {} more\n",
                total.saturating_sub(MAX_ENTRIES)
            ));
        }
        Output {
            text,
            label,
            note: count(total, "entry", "entries"),
            ok: true,
        }
    }

    pub(super) fn read(&self, asked: &str, start: Option<usize>, end: Option<usize>) -> Output {
        let label = format!("read {asked}");
        let path = match self.resolve(asked) {
            Ok(path) => path,
            Err(reason) => return Output::failed(label, reason),
        };
        let shown = self.relative(&path);
        let text = match read_text(&path, &shown) {
            Ok(text) => text,
            Err(reason) => return Output::failed(format!("read {shown}"), reason),
        };
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let first = start.unwrap_or(1).max(1);
        if first > total.max(1) {
            return Output::failed(
                format!("read {shown}"),
                format!(
                    "`{shown}` has {} (asked to start at {first})",
                    count(total, "line", "lines")
                ),
            );
        }
        let wanted = end.unwrap_or(usize::MAX).max(first);
        let mut body = String::new();
        let mut last = first.saturating_sub(1);
        for (number, line) in lines.iter().enumerate().skip(first.saturating_sub(1)) {
            let number = number.saturating_add(1);
            if number > wanted || number.saturating_sub(first) >= READ_LINES {
                break;
            }
            let row = format!("{number:>5}  {}\n", shorten(line, LINE_CHARS));
            if body.len().saturating_add(row.len()) > self.limits.chars {
                break;
            }
            body.push_str(&row);
            last = number;
        }
        let range = format!("{first}-{last}");
        let mut text = format!("{shown} (lines {range} of {total})\n{body}");
        if last < total && last < wanted {
            text.push_str(&format!(
                "[more follows: call read_file again with start={}]\n",
                last.saturating_add(1)
            ));
        }
        Output {
            text,
            label: format!("read {shown}:{range}"),
            note: count(
                last.saturating_sub(first.saturating_sub(1)),
                "line",
                "lines",
            ),
            ok: true,
        }
    }

    pub(super) fn search(&self, pattern: &str, asked: &str, cancel: &AtomicBool) -> Output {
        let label = format!("search \"{}\" in {asked}", shorten(pattern, 40));
        if pattern.trim().is_empty() {
            return Output::failed(label, "the search pattern is empty");
        }
        let from = match self.resolve(asked) {
            Ok(from) => from,
            Err(reason) => return Output::failed(label, reason),
        };
        let label = format!(
            "search \"{}\" in {}",
            shorten(pattern, 40),
            self.relative(&from)
        );
        let smart_case = pattern == pattern.to_lowercase();
        let needles: Vec<Vec<String>> = pattern
            .split('|')
            .map(|alternative| {
                alternative
                    .split(".*")
                    .filter(|part| !part.is_empty())
                    .map(|part| {
                        if smart_case {
                            part.to_lowercase()
                        } else {
                            part.to_owned()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|parts| !parts.is_empty())
            .collect();
        let started = Instant::now();
        let mut hits = Hits {
            budget: self.limits.chars.saturating_sub(200),
            ..Hits::default()
        };
        let mut stack = vec![from];
        let mut stopped = None;
        while let Some(path) = stack.pop() {
            if cancel.load(Ordering::Relaxed) {
                stopped = Some("cancelled".to_owned());
                break;
            }
            if started.elapsed() > self.limits.search_time {
                stopped = Some(format!(
                    "stopped after {} s",
                    self.limits.search_time.as_secs()
                ));
                break;
            }
            if path.is_dir() {
                stack.extend(children(&path).into_iter().rev());
            } else if hits.scan(self, &path, &needles, smart_case) {
                break;
            }
        }
        let mut text = hits.lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        if hits.found > hits.lines.len() {
            text.push_str(&format!(
                "[{} matches in total; showing the first {}. Use a more specific pattern or path]\n",
                hits.found,
                hits.lines.len()
            ));
        }
        if let Some(why) = stopped {
            text.push_str(&format!(
                "[search {why} after {} files; results may be incomplete]\n",
                hits.files
            ));
        }
        let note = match hits.found {
            0 => "no matches".to_owned(),
            n => format!(
                "{} in {}",
                count(n, "match", "matches"),
                count(hits.in_files, "file", "files")
            ),
        };
        if hits.found == 0 && looks_like_regex(pattern) {
            text.push_str("[the pattern is plain text: only `|` and `.*` are special, so regex syntax cannot match]\n");
        }
        if hits.found == 0 {
            text.insert_str(
                0,
                &format!(
                    "no matches for \"{pattern}\" in {} files searched\n",
                    hits.files
                ),
            );
        }
        Output {
            text,
            label,
            note,
            ok: true,
        }
    }
}

/// Entries of `dir` in name order, then those of its subdirectories down to `levels`; symlinks are shown, not followed.
fn tree(
    dir: &Path,
    levels: usize,
    indent: usize,
    lines: &mut Vec<String>,
    total: &mut usize,
) -> std::io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?
        .take(READ_ENTRIES)
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type();
        let is_dir = kind.as_ref().is_ok_and(|k| k.is_dir());
        let pad = "  ".repeat(indent);
        *total = total.saturating_add(1);
        lines.push(match &kind {
            Ok(k) if k.is_dir() => format!("{pad}{name}/"),
            Ok(k) if k.is_symlink() => format!("{pad}{name} (symlink)"),
            _ => format!("{pad}{name}"),
        });
        if is_dir && levels > 1 && !SKIPPED.iter().any(|skip| name == *skip) {
            tree(
                &entry.path(),
                levels.saturating_sub(1),
                indent.saturating_add(1),
                lines,
                total,
            )?;
        }
    }
    Ok(())
}

/// Whether every part occurs in `hay`, one after the other.
fn in_order(hay: &str, parts: &[String]) -> bool {
    let mut from = 0_usize;
    for part in parts {
        let Some(at) = hay.get(from..).and_then(|rest| rest.find(part.as_str())) else {
            return false;
        };
        from = from.saturating_add(at).saturating_add(part.len());
    }
    true
}

/// Regex syntax the search does not understand, which would otherwise look like a project without the text.
fn looks_like_regex(pattern: &str) -> bool {
    ["\\", "[", "]", ".+", ".?", "(?", "^", "$", "{"]
        .iter()
        .any(|token| pattern.contains(token))
}

/// Subdirectories and files below `dir`, in name order; symlinks are never followed and build output is skipped.
fn children(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            entry.file_type().is_ok_and(|kind| !kind.is_symlink())
                && !SKIPPED.iter().any(|skip| name == *skip)
        })
        .map(|entry| entry.path())
        .collect();
    found.sort();
    found
}

#[derive(Default)]
struct Hits {
    lines: Vec<String>,
    found: usize,
    in_files: usize,
    files: usize,
    /// Characters the listed matches may take, so the summary after them is never the part that gets cut.
    budget: usize,
    used: usize,
}

impl Hits {
    /// Looks through one file; true when nothing more is worth scanning.
    fn scan(
        &mut self,
        repo: &Repo,
        path: &Path,
        needles: &[Vec<String>],
        smart_case: bool,
    ) -> bool {
        if fs::metadata(path).map_or(true, |m| !m.is_file() || m.len() > MAX_SEARCHED_FILE) {
            return false;
        }
        let Ok(bytes) = fs::read(path) else {
            return false;
        };
        if bytes.iter().take(SCAN_BYTES).any(|b| *b == 0) {
            return false;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            return false;
        };
        self.files = self.files.saturating_add(1);
        let shown = repo.relative(path);
        let mut here = false;
        for (index, line) in text.lines().enumerate() {
            let lowered = if smart_case {
                line.to_lowercase()
            } else {
                String::new()
            };
            let hay = if smart_case { lowered.as_str() } else { line };
            let hit = needles.iter().any(|parts| in_order(hay, parts));
            if !hit {
                continue;
            }
            if !here {
                here = true;
                self.in_files = self.in_files.saturating_add(1);
            }
            self.found = self.found.saturating_add(1);
            let row = format!(
                "{shown}:{}: {}",
                index.saturating_add(1),
                shorten(line.trim(), MATCH_CHARS)
            );
            if self.lines.len() < MAX_MATCHES && self.used.saturating_add(row.len()) <= self.budget
            {
                self.used = self.used.saturating_add(row.len());
                self.lines.push(row);
            }
        }
        self.found >= MAX_MATCHES.saturating_mul(50)
    }
}
