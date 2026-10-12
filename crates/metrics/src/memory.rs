use std::io;
use std::path::{Component, Path, PathBuf};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))
}

fn number(text: &str) -> io::Result<u64> {
    text.trim()
        .parse()
        .map_err(|error| invalid(format!("invalid memory count: {error}")))
}

fn limit(path: &Path) -> io::Result<Option<u64>> {
    match read(path) {
        Ok(text) if text.trim() == "max" => Ok(None),
        Ok(text) => number(&text).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn mount_path(text: &str) -> PathBuf {
    PathBuf::from(
        text.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

fn location(cgroup: &str, mounts: &str) -> io::Result<Option<(PathBuf, PathBuf)>> {
    let Some(relative) = cgroup.lines().find_map(|line| line.strip_prefix("0::")) else {
        return Ok(None);
    };
    let group = Path::new(relative);
    if !group.is_absolute()
        || group
            .components()
            .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(invalid("invalid cgroup-v2 path"));
    }
    for line in mounts.lines() {
        let Some((before, after)) = line.split_once(" - ") else {
            continue;
        };
        if after.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        let mut fields = before.split_whitespace().skip(3);
        let (Some(root), Some(mount)) = (fields.next(), fields.next()) else {
            continue;
        };
        let root = mount_path(root);
        let mount = mount_path(mount);
        if !root.is_absolute()
            || !mount.is_absolute()
            || root
                .components()
                .chain(mount.components())
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(invalid("invalid cgroup-v2 mount path"));
        }
        if let Ok(suffix) = group.strip_prefix(root) {
            return Ok(Some((mount.join(suffix), mount)));
        }
    }
    Err(invalid(
        "cannot locate this process's cgroup-v2 memory limits",
    ))
}

fn clean_file_bytes(stats: &str) -> io::Result<u64> {
    let mut file = 0_u64;
    let mut excluded = 0_u64;
    for line in stats.lines() {
        let mut fields = line.split_whitespace();
        let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
            continue;
        };
        match key {
            "file" => file = number(value)?,
            "shmem" | "file_dirty" | "file_writeback" | "unevictable" => {
                excluded = excluded.saturating_add(number(value)?);
            }
            _ => {}
        }
    }
    Ok(file.saturating_sub(excluded))
}

fn constrained(mut available: u64, group: &Path, mount: &Path) -> io::Result<u64> {
    if !group.is_dir() {
        return Err(invalid("cgroup-v2 memory directory is unavailable"));
    }
    for path in group.ancestors().take_while(|path| path.starts_with(mount)) {
        let ceiling = [
            limit(&path.join("memory.max"))?,
            limit(&path.join("memory.high"))?,
        ]
        .into_iter()
        .flatten()
        .min();
        if let Some(ceiling) = ceiling {
            let current = number(&read(&path.join("memory.current"))?)?;
            // Clean file cache can be reclaimed; shared, dirty and locked pages cannot be budgeted as free.
            let clean = clean_file_bytes(&read(&path.join("memory.stat"))?)?.min(current);
            available = available.min(ceiling.saturating_sub(current.saturating_sub(clean)));
        }
    }
    Ok(available)
}

/// Host MemAvailable constrained by every visible cgroup-v2 ancestor, including memory.high.
/// None on non-Linux systems; unreadable Linux limits return an error instead of an unlimited budget.
pub fn cpu_available_bytes() -> io::Result<Option<u64>> {
    if !cfg!(target_os = "linux") {
        return Ok(None);
    }
    let info = read(Path::new("/proc/meminfo"))?;
    let kib = info
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|text| text.split_whitespace().next())
        .ok_or_else(|| invalid("/proc/meminfo has no MemAvailable"))?;
    let mut available = number(kib)?
        .checked_mul(1024)
        .ok_or_else(|| invalid("MemAvailable overflows"))?;
    let cgroup = read(Path::new("/proc/self/cgroup"))?;
    let mounts = read(Path::new("/proc/self/mountinfo"))?;
    if let Some((group, mount)) = location(&cgroup, &mounts)? {
        available = constrained(available, &group, &mount)?;
    }
    Ok(Some(available))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_roots_and_escaped_names_do_not_escape_the_hierarchy() {
        let mounts = "35 27 0:33 /tenant /cg\\040root rw - cgroup2 cgroup2 rw";
        assert_eq!(
            location("0::/tenant/child", mounts).unwrap(),
            Some((PathBuf::from("/cg root/child"), PathBuf::from("/cg root")))
        );
        assert!(location("0::/tenant/../other", mounts).is_err());
        assert!(location("0::/other/child", mounts).is_err());
        assert!(location("0::/", "").is_err());
        assert_eq!(location("1:memory:/legacy", "").unwrap(), None);
    }

    #[test]
    fn only_clean_unlocked_file_cache_is_reclaimable() {
        assert_eq!(
            clean_file_bytes("file 100\nshmem 20\nfile_dirty 10\nfile_writeback 5\nunevictable 15")
                .unwrap(),
            50
        );
        assert_eq!(clean_file_bytes("file 10\nshmem 20").unwrap(), 0);
        assert!(clean_file_bytes("file malformed").is_err());
        assert!(number("18446744073709551616").is_err());
    }

    #[test]
    fn ancestors_high_limits_and_current_usage_all_constrain_the_budget() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("marin-memory-{}-{stamp}", std::process::id()));
        let child = root.join("child");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&child).unwrap();
        for (path, ceiling, current) in [(&root, "500", "400"), (&child, "1000", "100")] {
            std::fs::write(path.join("memory.max"), ceiling).unwrap();
            std::fs::write(path.join("memory.current"), current).unwrap();
            std::fs::write(path.join("memory.stat"), "file 150\nshmem 50\n").unwrap();
        }
        assert_eq!(constrained(2000, &child, &root).unwrap(), 200);
        std::fs::write(child.join("memory.high"), "100").unwrap();
        assert_eq!(constrained(2000, &child, &root).unwrap(), 100);
        assert_eq!(constrained(50, &child, &root).unwrap(), 50);
        assert!(constrained(2000, &root.join("missing"), &root).is_err());
        std::fs::write(root.join("memory.max"), "bad").unwrap();
        assert!(constrained(2000, &child, &root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linux_reports_an_actual_budget() {
        if cfg!(target_os = "linux") {
            assert!(cpu_available_bytes().unwrap().is_some());
        }
    }
}
