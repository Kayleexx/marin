use std::time::Duration;

/// `amount` per second, or `None` when no time passed (so no rate can be told).
pub fn per_second(amount: u64, time: Duration) -> Option<f64> {
    (!time.is_zero()).then(|| amount as f64 / time.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_divide_by_time_and_refuse_zero() {
        assert_eq!(per_second(3_000, Duration::from_millis(500)), Some(6_000.0));
        assert_eq!(per_second(0, Duration::from_secs(1)), Some(0.0));
        assert_eq!(per_second(1, Duration::ZERO), None);
    }
}
