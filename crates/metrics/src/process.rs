/// Resident memory of this process, from the kernel's own accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    pub rss_bytes: u64,
    pub peak_rss_bytes: u64,
}

fn kilobytes(status: &str, key: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    let value: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    value.checked_mul(1024)
}

fn parse(status: &str) -> Option<ProcessMemory> {
    Some(ProcessMemory {
        rss_bytes: kilobytes(status, "VmRSS:")?,
        peak_rss_bytes: kilobytes(status, "VmHWM:")?,
    })
}

/// `None` where `/proc/self/status` does not exist (not Linux).
pub fn process_memory() -> Option<ProcessMemory> {
    parse(&std::fs::read_to_string("/proc/self/status").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_status_lines() {
        let status = "Name:\tmarin\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\n";
        assert_eq!(
            parse(status),
            Some(ProcessMemory {
                rss_bytes: 1024 * 1024,
                peak_rss_bytes: 2048 * 1024
            })
        );
        assert_eq!(parse("Name: x\n"), None);
    }

    #[test]
    fn this_process_reports_some_memory_on_linux() {
        if cfg!(target_os = "linux") {
            let m = process_memory().unwrap();
            assert!(m.rss_bytes > 0 && m.peak_rss_bytes >= m.rss_bytes);
        }
    }
}
