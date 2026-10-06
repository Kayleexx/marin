use std::time::Duration;

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

/// `/proc` reports CPU time in ticks of 1/100 s on every Linux architecture (USER_HZ).
const TICKS_PER_SECOND: u64 = 100;

/// User plus system CPU time of this process so far, all threads together.
fn parse_cpu_time(stat: &str) -> Option<Duration> {
    // The command name sits in parentheses and may itself contain spaces and parentheses,
    // so count fields from the last `)`.
    let after = stat.get(stat.rfind(')')?.checked_add(1)?..)?;
    let mut fields = after.split_whitespace();
    let user: u64 = fields.nth(11)?.parse().ok()?;
    let system: u64 = fields.next()?.parse().ok()?;
    let millis = user
        .checked_add(system)?
        .checked_mul(1000)?
        .checked_div(TICKS_PER_SECOND)?;
    Some(Duration::from_millis(millis))
}

/// `None` where `/proc/self/stat` does not exist. Granularity is 10 ms.
pub fn cpu_time() -> Option<Duration> {
    parse_cpu_time(&std::fs::read_to_string("/proc/self/stat").ok()?)
}

fn parse_disk_reads(io: &str) -> Option<u64> {
    io.lines()
        .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
}

/// Bytes this process made the kernel fetch from storage. Reads served from the page cache count as 0.
/// `None` where `/proc/self/io` is missing or unreadable.
pub fn disk_read_bytes() -> Option<u64> {
    parse_disk_reads(&std::fs::read_to_string("/proc/self/io").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_time_survives_awkward_command_names() {
        let stat = "123 (a b) c) S 1 1 1 0 -1 4194560 100 0 0 0 250 50 0 0 20 0 1 0 99";
        assert_eq!(parse_cpu_time(stat), Some(Duration::from_secs(3)));
        let plain = "7 (marin) R 1 1 1 0 -1 0 0 0 0 0 1 2 0 0 20 0 1 0 5";
        assert_eq!(parse_cpu_time(plain), Some(Duration::from_millis(30)));
    }

    #[test]
    fn cpu_time_rejects_short_or_garbled_lines() {
        assert_eq!(parse_cpu_time(""), None);
        assert_eq!(parse_cpu_time("1 (x) S 1 2"), None);
        assert_eq!(parse_cpu_time("1 (x) S 1 1 1 0 -1 0 0 0 0 0 abc 2"), None);
        assert_eq!(parse_cpu_time("no parentheses 1 2 3"), None);
        let huge = format!("1 (x) S 1 1 1 0 -1 0 0 0 0 0 {} 1", u64::MAX);
        assert_eq!(parse_cpu_time(&huge), None);
    }

    #[test]
    fn disk_reads_come_from_the_read_bytes_line_only() {
        let io = "rchar: 900\nwchar: 1\nread_bytes: 4096\nwrite_bytes: 0\n";
        assert_eq!(parse_disk_reads(io), Some(4096));
        assert_eq!(parse_disk_reads("rchar: 900\n"), None);
        assert_eq!(parse_disk_reads("read_bytes: lots\n"), None);
        assert_eq!(parse_disk_reads(""), None);
    }

    #[test]
    fn this_process_reports_cpu_time_on_linux() {
        if cfg!(target_os = "linux") {
            assert!(cpu_time().is_some());
        }
    }

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
