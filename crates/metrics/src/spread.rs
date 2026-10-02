use std::time::Duration;

/// Minimum, median, 95th percentile, maximum and mean of a set of durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    pub min: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
    pub mean: Duration,
}

/// Nearest-rank percentiles; `None` when there are no samples.
pub fn spread(samples: &[Duration]) -> Option<Spread> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = |percent: usize| {
        let n = sorted.len();
        let index = percent
            .saturating_mul(n)
            .div_ceil(100)
            .saturating_sub(1)
            .min(n.saturating_sub(1));
        sorted.get(index).copied()
    };
    let total = sorted
        .iter()
        .fold(Duration::ZERO, |sum, d| sum.saturating_add(*d));
    let count = u32::try_from(sorted.len()).ok()?;
    Some(Spread {
        min: *sorted.first()?,
        p50: rank(50)?,
        p95: rank(95)?,
        max: *sorted.last()?,
        mean: total.checked_div(count)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(values: &[u64]) -> Vec<Duration> {
        values.iter().map(|v| Duration::from_millis(*v)).collect()
    }

    #[test]
    fn nearest_rank_percentiles() {
        let s = spread(&ms(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])).unwrap();
        assert_eq!(s.min, Duration::from_millis(10));
        assert_eq!(s.p50, Duration::from_millis(50));
        assert_eq!(s.p95, Duration::from_millis(100));
        assert_eq!(s.max, Duration::from_millis(100));
        assert_eq!(s.mean, Duration::from_millis(55));
    }

    #[test]
    fn unsorted_input_and_single_samples_work_and_empty_has_no_spread() {
        let s = spread(&ms(&[30, 10, 20])).unwrap();
        assert_eq!(
            (s.min, s.p50, s.max),
            (ms(&[10])[0], ms(&[20])[0], ms(&[30])[0])
        );
        let one = spread(&ms(&[7])).unwrap();
        assert_eq!((one.p50, one.p95), (ms(&[7])[0], ms(&[7])[0]));
        assert!(spread(&[]).is_none());
    }
}
