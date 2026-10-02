use std::time::Duration;

pub fn bytes(count: u64) -> String {
    let value = count as f64;
    if value >= 1e9 {
        format!("{:.2} GB", value / 1e9)
    } else {
        format!("{:.0} MB", value / 1e6)
    }
}

pub fn seconds(duration: Duration) -> String {
    format!("{:.1} s", duration.as_secs_f64())
}

pub fn count(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (i, ch) in digits.chars().enumerate() {
        let remaining = digits.len().saturating_sub(i);
        if i > 0 && remaining.checked_rem(3) == Some(0) {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sizes_and_counts() {
        assert_eq!(bytes(3_440_000_000), "3.44 GB");
        assert_eq!(bytes(190_000), "0 MB");
        assert_eq!(bytes(470_000_000), "470 MB");
        assert_eq!(count(151_936), "151,936");
        assert_eq!(count(7), "7");
        assert_eq!(count(1_000), "1,000");
        assert_eq!(seconds(Duration::from_millis(1950)), "1.9 s");
    }
}
