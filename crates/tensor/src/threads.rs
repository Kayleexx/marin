//! Threads for prompt passes: how many to use, how to cut the work, and a scoped runner. A region splits its
//! output into disjoint column ranges, so every output element is computed by the same code as on one thread
//! and results do not depend on the thread count. Decode never calls this (it is limited by memory).

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Multiply-adds each helper thread must have to be worth its start-up (about 30 to 90 us per region).
const MIN_WORK: usize = if cfg!(test) { 1 } else { 1 << 24 };

static CONFIGURED: AtomicUsize = AtomicUsize::new(0);

/// Sets how many threads prompt passes may use; 0 means one per physical core.
pub fn set_threads(threads: usize) {
    CONFIGURED.store(threads, Ordering::Relaxed);
}

/// The thread count in force: the configured one, else one per physical core.
pub fn threads() -> usize {
    match CONFIGURED.load(Ordering::Relaxed) {
        0 => {
            static CORES: OnceLock<usize> = OnceLock::new();
            *CORES.get_or_init(physical_cores)
        }
        n => n,
    }
}

/// Physical cores: the distinct `thread_siblings_list` entries of the CPUs, else the logical CPU count.
pub fn physical_cores() -> usize {
    let lists = (0..)
        .map_while(|cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            std::fs::read_to_string(path).ok()
        })
        .collect::<Vec<_>>();
    match count_cores(&lists) {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        cores => cores.min(std::thread::available_parallelism().map_or(1, usize::from)),
    }
}

fn count_cores(sibling_lists: &[String]) -> usize {
    let mut distinct: Vec<&str> = sibling_lists.iter().map(|l| l.trim()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len()
}

/// Threads for a region of `work` multiply-adds that can be cut into at most `parts` pieces.
pub(crate) fn workers(work: usize, parts: usize) -> usize {
    threads()
        .min(work.checked_div(MIN_WORK).unwrap_or(1))
        .min(parts)
        .max(1)
}

/// At most `parts` consecutive ranges covering `0..total`, each a multiple of `align` long except the last,
/// with sizes differing by at most one `align` unit.
pub(crate) fn ranges(total: usize, parts: usize, align: usize) -> Vec<Range<usize>> {
    let align = align.max(1);
    let units = total.div_ceil(align);
    let parts = parts.clamp(1, units.max(1));
    let (base, extra) = (
        units.checked_div(parts).unwrap_or(0),
        units.checked_rem(parts).unwrap_or(0),
    );
    let mut start = 0_usize;
    let mut out = Vec::new();
    for part in 0..parts {
        let size = base.saturating_add(usize::from(part < extra));
        let end = start.saturating_add(size.saturating_mul(align)).min(total);
        if end > start {
            out.push(start..end);
        }
        start = end;
    }
    out
}

/// Cuts every `width`-long row of `out` at the given column lengths: part `i` holds, for each row, the slice of
/// `lens[i]` columns, so parts can be written by different threads.
pub(crate) fn split_columns<'a>(
    out: &'a mut [f32],
    width: usize,
    lens: &[usize],
) -> Vec<Vec<&'a mut [f32]>> {
    let mut parts: Vec<Vec<&mut [f32]>> = lens.iter().map(|_| Vec::new()).collect();
    for row in out.chunks_mut(width.max(1)) {
        let mut rest = row;
        for (part, len) in parts.iter_mut().zip(lens) {
            let take = (*len).min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(take);
            part.push(head);
            rest = tail;
        }
    }
    parts
}

/// Runs `work` on every job, on the calling thread and up to `threads() - 1` helpers that take jobs from a shared
/// queue. If a helper cannot be started the others carry on, so the work always completes.
pub(crate) fn run_jobs<J: Send>(jobs: Vec<J>, work: impl Fn(J) + Sync) {
    let helpers = jobs
        .len()
        .saturating_sub(1)
        .min(threads().saturating_sub(1));
    let queue = Mutex::new(jobs);
    let drain = || {
        while let Some(job) = queue.lock().ok().and_then(|mut jobs| jobs.pop()) {
            work(job);
        }
    };
    std::thread::scope(|scope| {
        for _ in 0..helpers {
            let _ = std::thread::Builder::new().spawn_scoped(scope, drain);
        }
        drain();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_everything_in_aligned_pieces() {
        for (total, parts, align) in [
            (10, 3, 4),
            (4, 8, 4),
            (0, 3, 4),
            (16, 4, 4),
            (7, 3, 1),
            (13, 2, 4),
        ] {
            let pieces = ranges(total, parts, align);
            assert!(pieces.len() <= parts.max(1));
            assert_eq!(pieces.iter().map(Range::len).sum::<usize>(), total);
            assert!(pieces.windows(2).all(|w| w[0].end == w[1].start));
            let (last, rest) = pieces
                .split_last()
                .map_or((None, &[][..]), |(l, r)| (Some(l), r));
            assert!(rest.iter().all(|r| r.len() % align == 0), "{pieces:?}");
            assert!(last.is_none_or(|l| l.start % align == 0));
        }
    }

    #[test]
    fn columns_split_every_row_at_the_same_places() {
        let mut data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let parts = split_columns(&mut data, 4, &[1, 3]);
        assert_eq!(parts[0], [[0.0_f32].as_slice(), &[4.0], &[8.0]]);
        assert_eq!(
            parts[1],
            [
                [1.0_f32, 2.0, 3.0].as_slice(),
                &[5.0, 6.0, 7.0],
                &[9.0, 10.0, 11.0]
            ]
        );
    }

    #[test]
    fn every_job_runs_once_whatever_the_thread_count() {
        let seen = Mutex::new(Vec::new());
        for count in [1, 2, 5, 16] {
            set_threads(count);
            seen.lock().unwrap().clear();
            run_jobs((0..7).collect(), |job| seen.lock().unwrap().push(job));
            let mut jobs = seen.lock().unwrap().clone();
            jobs.sort_unstable();
            assert_eq!(jobs, (0..7).collect::<Vec<_>>());
        }
        set_threads(0);
    }

    #[test]
    fn cores_are_the_distinct_sibling_lists() {
        let lists: Vec<String> = ["0,8\n", "1,9\n", "0,8\n", "1,9\n"]
            .map(String::from)
            .to_vec();
        assert_eq!(count_cores(&lists), 2);
        assert_eq!(count_cores(&[]), 0);
        assert!(physical_cores() >= 1);
        assert!(physical_cores() <= std::thread::available_parallelism().map_or(1, usize::from));
    }
}
