use std::time::{Duration, Instant};

/// The parts of one forward pass that get their own time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Embedding,
    Norm,
    Qkv,
    Rope,
    KvWrite,
    Attention,
    OutProj,
    Mlp,
    Residual,
    LmHead,
    StateCopy,
    Convolution,
    Recurrence,
}

impl Stage {
    pub const ALL: [Stage; 13] = [
        Stage::Embedding,
        Stage::Norm,
        Stage::Qkv,
        Stage::Rope,
        Stage::KvWrite,
        Stage::Attention,
        Stage::OutProj,
        Stage::Mlp,
        Stage::Residual,
        Stage::LmHead,
        Stage::StateCopy,
        Stage::Convolution,
        Stage::Recurrence,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Norm => "norm",
            Self::Qkv => "qkv projections",
            Self::Rope => "rope",
            Self::KvWrite => "kv cache write",
            Self::Attention => "attention",
            Self::OutProj => "attention output proj",
            Self::Mlp => "mlp",
            Self::Residual => "residual add",
            Self::LmHead => "lm head",
            Self::StateCopy => "state transaction copy",
            Self::Convolution => "causal convolution",
            Self::Recurrence => "DeltaNet recurrence",
        }
    }
}

/// One timed stage call: which layer it ran in (`None` for embedding, final norm and LM head).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEntry {
    pub layer: Option<usize>,
    pub stage: Stage,
    pub time: Duration,
}

/// Time and call count per stage, summed over every pass that was recorded.
/// A traced one also keeps every call in order; use that for a single pass, since it grows with each call.
#[derive(Debug, Clone, Default)]
pub struct StageTimes {
    entries: Vec<(Stage, Duration, u64)>,
    trace: Option<Vec<TraceEntry>>,
    layer: Option<usize>,
}

impl StageTimes {
    pub fn traced() -> Self {
        Self {
            trace: Some(Vec::new()),
            ..Self::default()
        }
    }

    /// Tags the stages recorded from now on with `layer`.
    pub fn set_layer(&mut self, layer: Option<usize>) {
        self.layer = layer;
    }

    /// Every call in the order it was recorded; empty unless built with `traced`.
    pub fn trace(&self) -> &[TraceEntry] {
        self.trace.as_deref().unwrap_or(&[])
    }

    pub fn add(&mut self, stage: Stage, time: Duration) {
        if let Some(trace) = &mut self.trace {
            trace.push(TraceEntry {
                layer: self.layer,
                stage,
                time,
            });
        }
        match self.entries.iter_mut().find(|(s, _, _)| *s == stage) {
            Some((_, total, calls)) => {
                *total = total.saturating_add(time);
                *calls = calls.saturating_add(1);
            }
            None => self.entries.push((stage, time, 1)),
        }
    }

    pub fn total(&self, stage: Stage) -> Duration {
        self.get(stage).map_or(Duration::ZERO, |(total, _)| total)
    }

    pub fn calls(&self, stage: Stage) -> u64 {
        self.get(stage).map_or(0, |(_, calls)| calls)
    }

    /// Time of every recorded stage together.
    pub fn sum(&self) -> Duration {
        self.entries
            .iter()
            .fold(Duration::ZERO, |sum, (_, t, _)| sum.saturating_add(*t))
    }

    fn get(&self, stage: Stage) -> Option<(Duration, u64)> {
        self.entries
            .iter()
            .find(|(s, _, _)| *s == stage)
            .map(|(_, total, calls)| (*total, *calls))
    }
}

/// Switches stage timing on for one pass; when off it costs one branch per stage.
pub struct Probe<'a>(Option<&'a mut StageTimes>);

impl<'a> Probe<'a> {
    pub fn off() -> Self {
        Self(None)
    }

    pub fn on(times: &'a mut StageTimes) -> Self {
        Self(Some(times))
    }

    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }

    pub fn time<T>(&mut self, stage: Stage, work: impl FnOnce() -> T) -> T {
        match &mut self.0 {
            Some(times) => {
                let started = Instant::now();
                let out = work();
                times.add(stage, started.elapsed());
                out
            }
            None => work(),
        }
    }

    /// Tags the stages timed from now on with `layer`; no-op when off.
    pub fn layer(&mut self, layer: Option<usize>) {
        if let Some(times) = &mut self.0 {
            times.set_layer(layer);
        }
    }

    pub fn record(&mut self, stage: Stage, time: Duration) {
        if let Some(times) = &mut self.0 {
            times.add(stage, time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_add_up_and_count_calls() {
        let mut times = StageTimes::default();
        times.add(Stage::Norm, Duration::from_millis(2));
        times.add(Stage::Norm, Duration::from_millis(3));
        times.add(Stage::Mlp, Duration::from_millis(10));
        assert_eq!(times.total(Stage::Norm), Duration::from_millis(5));
        assert_eq!(times.calls(Stage::Norm), 2);
        assert_eq!(times.calls(Stage::Rope), 0);
        assert_eq!(times.sum(), Duration::from_millis(15));
    }

    #[test]
    fn an_off_probe_records_nothing_and_an_on_probe_times_the_work() {
        let mut off = Probe::off();
        assert_eq!(off.time(Stage::Rope, || 7), 7);
        assert!(!off.enabled());
        let mut times = StageTimes::default();
        let mut on = Probe::on(&mut times);
        on.time(Stage::Rope, || std::thread::sleep(Duration::from_millis(2)));
        assert!(times.total(Stage::Rope) >= Duration::from_millis(2));
        assert_eq!(times.calls(Stage::Rope), 1);
    }

    #[test]
    fn a_traced_run_keeps_each_call_with_its_layer() {
        let mut times = StageTimes::traced();
        let mut probe = Probe::on(&mut times);
        probe.record(Stage::Embedding, Duration::from_millis(1));
        probe.layer(Some(0));
        probe.record(Stage::Norm, Duration::from_millis(2));
        probe.record(Stage::Norm, Duration::from_millis(3));
        probe.layer(Some(1));
        probe.record(Stage::Mlp, Duration::from_millis(4));
        probe.layer(None);
        probe.record(Stage::LmHead, Duration::from_millis(5));
        let layers: Vec<_> = times.trace().iter().map(|e| (e.layer, e.stage)).collect();
        assert_eq!(
            layers,
            [
                (None, Stage::Embedding),
                (Some(0), Stage::Norm),
                (Some(0), Stage::Norm),
                (Some(1), Stage::Mlp),
                (None, Stage::LmHead),
            ]
        );
        assert_eq!(times.total(Stage::Norm), Duration::from_millis(5));
    }

    #[test]
    fn an_untraced_run_keeps_no_calls_and_ignores_layers() {
        let mut times = StageTimes::default();
        times.set_layer(Some(3));
        times.add(Stage::Rope, Duration::from_millis(1));
        assert!(times.trace().is_empty());
        let mut off = Probe::off();
        off.layer(Some(1));
        off.record(Stage::Rope, Duration::from_millis(1));
    }

    #[test]
    fn every_stage_has_a_name_and_is_listed_once() {
        for (i, a) in Stage::ALL.iter().enumerate() {
            assert!(!a.name().is_empty());
            assert!(Stage::ALL.iter().skip(i + 1).all(|b| a != b));
        }
    }
}
