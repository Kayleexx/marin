use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use model::{EngineCache, ModelError};

/// Tokens of one prompt fed per GPU pass when requests share the GPU. A pass shorter than this is a decode step or
/// the end of a prompt. Smaller slices shorten the wait behind a long prefill; larger ones lose less time to the
/// per-pass staging of the cache.
pub(super) const SLICE: usize = 64;

/// A waiter that has been passed over this long runs next, so a long prefill cannot starve behind decode steps.
const AGE_LIMIT: Duration = Duration::from_secs(2);

/// After a short pass the next full slice waits this long: the request that ran it is sampling and about to ask for
/// its next decode step, which must not queue behind a whole slice.
const HOLD: Duration = Duration::from_millis(10);

/// How long the first decode step to reach the front waits for the other decoders to ask for theirs, so they can
/// share one pass. They are sampling and writing their last token, a few milliseconds at most.
const COLLECT: Duration = Duration::from_millis(4);

/// A request that has not taken a decode step for this long is no longer waited for.
const LANE_TTL: Duration = Duration::from_millis(100);

type Stepped = (EngineCache, Result<Vec<f32>, ModelError>);

struct Decode {
    ticket: u64,
    token: u32,
    cache: EngineCache,
}

struct Waiter {
    ticket: u64,
    full: bool,
    since: Instant,
}

#[derive(Default)]
struct State {
    next_ticket: u64,
    running: bool,
    waiting: Vec<Waiter>,
    hold_until: Option<Instant>,
    active: usize,
    /// Decode steps waiting for a pass, with their caches lent to whoever runs it.
    decodes: Vec<Decode>,
    /// Finished decode steps, by ticket, waiting for their request to pick them up.
    done: Vec<(u64, Stepped)>,
    /// Requests that took a decode step recently: the decoders a batch can wait for.
    lanes: Vec<(u64, Instant)>,
}

/// Lets several requests share one GPU: each forward pass runs alone. Short passes (decode steps, short prompts)
/// go first in arrival order; full slices of a long prompt run when no short pass is waiting or about to be.
pub(super) struct Gate {
    state: Mutex<State>,
    changed: Condvar,
    max_active: usize,
    max_batch: usize,
}

pub(super) struct Slot<'a>(&'a Gate);

struct Running<'a> {
    gate: &'a Gate,
    full: bool,
}

fn next_ticket(waiting: &[Waiter], now: Instant, hold_until: Option<Instant>) -> Option<u64> {
    let holding = hold_until.is_some_and(|until| until > now);
    waiting
        .iter()
        .filter(|w| now.saturating_duration_since(w.since) >= AGE_LIMIT)
        .min_by_key(|w| w.since)
        .or_else(|| {
            waiting
                .iter()
                .filter(|w| !(holding && w.full))
                .min_by_key(|w| (w.full, w.ticket))
        })
        .map(|w| w.ticket)
}

impl Gate {
    /// `batch_limit` is how many decode steps the model serves in one pass.
    pub fn new(max_active: usize, batch_limit: usize) -> Self {
        let max_active = max_active.max(1);
        Self {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            max_active,
            max_batch: batch_limit.clamp(1, max_active),
        }
    }

    pub fn batching(&self) -> bool {
        self.max_batch > 1
    }

    pub fn max_active(&self) -> usize {
        self.max_active
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Blocks until fewer than `max_active` requests are running.
    pub fn wait_for_slot(&self) {
        let mut state = self.lock();
        while state.active >= self.max_active {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Counts one running request until the slot is dropped. Only the accepting thread calls this, right after
    /// `wait_for_slot`, so the cap cannot be overshot.
    pub fn enter(&self) -> Slot<'_> {
        let mut state = self.lock();
        state.active = state.active.saturating_add(1);
        drop(state);
        Slot(self)
    }

    /// Queues a pass and blocks until it is this pass's turn, or until another request's pass finished it.
    fn queue<'a>(
        &'a self,
        mut state: MutexGuard<'a, State>,
        ticket: u64,
        full: bool,
    ) -> MutexGuard<'a, State> {
        state.waiting.push(Waiter {
            ticket,
            full,
            since: Instant::now(),
        });
        loop {
            let now = Instant::now();
            if state.done.iter().any(|(t, _)| *t == ticket) {
                return state;
            }
            if !state.running && next_ticket(&state.waiting, now, state.hold_until) == Some(ticket)
            {
                state.waiting.retain(|w| w.ticket != ticket);
                return state;
            }
            // Wakes by itself so a finished hold or an aged waiter is noticed even when nothing else changes.
            let pause = state
                .hold_until
                .map_or(AGE_LIMIT, |until| until.saturating_duration_since(now))
                .clamp(Duration::from_millis(1), AGE_LIMIT);
            state = self
                .changed
                .wait_timeout(state, pause)
                .map(|(state, _)| state)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }

    fn next_ticket(state: &mut State) -> u64 {
        let ticket = state.next_ticket;
        state.next_ticket = ticket.wrapping_add(1);
        ticket
    }

    /// Runs `work` as the only forward pass on the GPU. `cost` is the number of tokens it feeds.
    pub fn run<T>(&self, cost: usize, work: impl FnOnce() -> T) -> T {
        let full = cost >= SLICE;
        let mut state = self.lock();
        let ticket = Self::next_ticket(&mut state);
        let mut state = self.queue(state, ticket, full);
        state.running = true;
        drop(state);
        let _turn = Running { gate: self, full };
        work()
    }

    /// One decode step of the request `lane`. The step that reaches the front of the queue takes every decode step waiting with it, up to
    /// the batch limit, and `run` serves them in one pass; each request gets back its own cache and logits. `None`
    /// only if `run` returned fewer results than steps, in which case that request's cache is lost.
    pub fn decode(
        &self,
        lane: u64,
        token: u32,
        cache: EngineCache,
        run: impl FnOnce(Vec<(u32, EngineCache)>) -> Vec<Stepped>,
    ) -> Option<Stepped> {
        let mut state = self.lock();
        let ticket = Self::next_ticket(&mut state);
        state.decodes.push(Decode {
            ticket,
            token,
            cache,
        });
        self.changed.notify_all();
        let mut state = self.queue(state, ticket, false);
        if let Some(at) = state.done.iter().position(|(t, _)| *t == ticket) {
            return Some(state.done.swap_remove(at).1);
        }
        // Running from here on: while this thread waits for the other decoders, no other pass may start.
        state.running = true;
        state.lanes.retain(|(_, seen)| seen.elapsed() < LANE_TTL);
        match state.lanes.iter_mut().find(|(id, _)| *id == lane) {
            Some(entry) => entry.1 = Instant::now(),
            None => state.lanes.push((lane, Instant::now())),
        }
        let expected = state.lanes.len().min(self.max_batch);
        let deadline = Instant::now().checked_add(COLLECT);
        while state.decodes.len() < expected {
            let Some(left) = deadline.map(|d| d.saturating_duration_since(Instant::now())) else {
                break;
            };
            if left.is_zero() {
                break;
            }
            state = self
                .changed
                .wait_timeout(state, left)
                .map(|(state, _)| state)
                .unwrap_or_else(|e| e.into_inner().0);
        }
        state.decodes.sort_by_key(|d| d.ticket);
        let taken = state.decodes.len().min(self.max_batch);
        let jobs: Vec<Decode> = state.decodes.drain(..taken).collect();
        state
            .waiting
            .retain(|w| jobs.iter().all(|j| j.ticket != w.ticket));
        drop(state);
        let turn = Running {
            gate: self,
            full: false,
        };
        let tickets: Vec<u64> = jobs.iter().map(|j| j.ticket).collect();
        let results = run(jobs.into_iter().map(|j| (j.token, j.cache)).collect());
        self.lock().done.extend(tickets.into_iter().zip(results));
        drop(turn);
        let mut state = self.lock();
        let at = state.done.iter().position(|(t, _)| *t == ticket)?;
        Some(state.done.swap_remove(at).1)
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.active = state.active.saturating_sub(1);
        drop(state);
        self.0.changed.notify_all();
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.lock();
        state.running = false;
        state.hold_until = if self.full {
            None
        } else {
            Instant::now().checked_add(HOLD)
        };
        drop(state);
        self.gate.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waiter(ticket: u64, full: bool, age_ms: u64) -> Waiter {
        Waiter {
            ticket,
            full,
            since: Instant::now()
                .checked_sub(Duration::from_millis(age_ms))
                .unwrap(),
        }
    }

    #[test]
    fn short_passes_go_first_in_arrival_order() {
        let waiting = [
            waiter(0, true, 10),
            waiter(1, false, 5),
            waiter(2, false, 4),
        ];
        assert_eq!(next_ticket(&waiting, Instant::now(), None), Some(1));
    }

    #[test]
    fn a_full_slice_waits_out_the_hold_after_a_short_pass() {
        let waiting = [waiter(0, true, 10)];
        let hold = Instant::now().checked_add(HOLD);
        assert_eq!(next_ticket(&waiting, Instant::now(), hold), None);
        assert_eq!(next_ticket(&waiting, Instant::now(), None), Some(0));
    }

    #[test]
    fn a_waiter_passed_over_too_long_goes_first() {
        let waiting = [
            waiter(0, true, 2_500),
            waiter(1, false, 5),
            waiter(2, true, 3_000),
        ];
        let hold = Instant::now().checked_add(HOLD);
        assert_eq!(next_ticket(&waiting, Instant::now(), hold), Some(2));
    }

    #[test]
    fn passes_never_overlap_and_the_slot_cap_holds() {
        let gate = &Gate::new(2, 1);
        let inside = &std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for cost in [1, SLICE, 1, 8] {
                scope.spawn(move || {
                    gate.run(cost, || {
                        let now = inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        assert_eq!(now, 0);
                        std::thread::sleep(Duration::from_millis(20));
                        inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    });
                });
            }
        });
        let first = gate.enter();
        let second = gate.enter();
        assert_eq!(gate.lock().active, 2);
        drop((first, second));
        assert_eq!(gate.lock().active, 0);
    }
}
