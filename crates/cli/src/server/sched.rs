use std::sync::atomic::{AtomicBool, Ordering};

use model::{Engine, EngineCache, ModelError};
use runtime::CausalLm;
use thiserror::Error;

use super::gate::{Gate, SLICE};
use crate::load::Loaded;

/// Most requests that run at once; each holds its own KV and recurrent state on the GPU.
const MAX_ACTIVE: usize = 3;
#[derive(Debug, Error)]
pub(super) enum ScheduleError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("the request was cancelled")]
    Cancelled,
    #[error("a batched decode step returned no result")]
    Lost,
}

/// How many requests may run at once. Each holds its own state (KV and recurrent) for the whole `--ctx` window;
/// the free GPU memory after loading, minus the safety slack that covers activations and the small marks of the
/// recurrent state a pass keeps, must also hold one idle session (a full copy). A pass copies none of the keys and
/// values, and a batched decode step copies nothing. Anything but Qwen3.5 on CUDA runs one request.
pub(super) fn max_active(loaded: &Loaded) -> usize {
    let Engine::Qwen35Cuda(_) = &loaded.model else {
        return 1;
    };
    let Some((free, _)) = loaded.model.gpu_memory() else {
        return 1;
    };
    let state = model::qwen35::TextConfig::parse(
        &loaded.dir.config_json,
        loaded.dir.generation_config_json.as_deref(),
    )
    .and_then(|config| config.state_bytes(loaded.ctx))
    .map_or(u64::MAX, |bytes| u64::try_from(bytes).unwrap_or(u64::MAX));
    let copies = free
        .saturating_sub(model::GPU_SLACK_BYTES)
        .checked_div(state)
        .unwrap_or(0);
    usize::try_from(copies)
        .unwrap_or(usize::MAX)
        .saturating_sub(1)
        .clamp(1, MAX_ACTIVE)
}

/// The loaded model as one request sees it: every pass waits its turn at the gate, and long prompts go in slices.
#[derive(Clone, Copy)]
pub(super) struct Scheduled<'a> {
    loaded: &'a Loaded,
    gate: &'a Gate,
    slice: usize,
    cancel: Option<&'a AtomicBool>,
    /// Identifies the request to the gate's decode batching.
    lane: u64,
}

impl<'a> Scheduled<'a> {
    pub fn new(loaded: &'a Loaded, gate: &'a Gate, lane: u64) -> Self {
        let slice = if gate.max_active() > 1 {
            SLICE
        } else {
            usize::MAX
        };
        Self {
            loaded,
            gate,
            slice,
            cancel: None,
            lane,
        }
    }

    /// Passes stop at the next slice boundary once `cancel` is set.
    pub fn cancellable(self, cancel: &'a AtomicBool) -> Self {
        Self {
            cancel: Some(cancel),
            ..self
        }
    }

    pub fn checkpoint(&self, cache: &EngineCache) -> Result<Option<EngineCache>, ModelError> {
        self.gate
            .run(1, || self.loaded.model.session_checkpoint(cache))
    }
}

impl Scheduled<'_> {
    /// One decode step, run together with the other requests' steps that are waiting.
    fn decode(&self, tokens: &[u32], cache: &mut EngineCache) -> Result<Vec<f32>, ScheduleError> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(ScheduleError::Cancelled);
        }
        let (Some(spare), Some(token)) = (EngineCache::vacant(), tokens.first().copied()) else {
            return Err(ScheduleError::Model(ModelError::NoTokens));
        };
        let lent = std::mem::replace(cache, spare);
        let (back, logits) = self
            .gate
            .decode(self.lane, token, lent, |steps| {
                self.loaded.model.decode_batch(steps)
            })
            .ok_or(ScheduleError::Lost)?;
        *cache = back;
        Ok(logits?)
    }
}

impl CausalLm for Scheduled<'_> {
    type Error = ScheduleError;
    type Cache = EngineCache;

    fn forward(&self, tokens: &[u32], cache: &mut EngineCache) -> Result<Vec<f32>, ScheduleError> {
        if tokens.len() == 1 && self.gate.batching() {
            return self.decode(tokens, cache);
        }
        let mut logits = None;
        for part in tokens.chunks(self.slice) {
            if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(ScheduleError::Cancelled);
            }
            logits = Some(
                self.gate
                    .run(part.len(), || self.loaded.model.forward(part, cache))?,
            );
        }
        logits.ok_or(ScheduleError::Model(ModelError::NoTokens))
    }
}
