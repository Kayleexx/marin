use model::{Engine, EngineCache};
use runtime::{CausalLm, Retained};

use super::request::default_prefix;
use super::sched::Scheduled;
use super::session::Stored;
use crate::load::Loaded;

pub(super) struct Prefix {
    tokens: Vec<u32>,
    cache: EngineCache,
}

impl Prefix {
    pub fn build(loaded: &Loaded) -> Option<Self> {
        if !matches!(loaded.model, Engine::Qwen35Cuda(_)) {
            return None;
        }
        let built = default_prefix(loaded)
            .map_err(|e| e.to_string())
            .and_then(|tokens| {
                let mut cache = loaded
                    .model
                    .new_cache(loaded.ctx)
                    .map_err(|e| e.to_string())?;
                loaded
                    .model
                    .forward(&tokens, &mut cache)
                    .map_err(|e| e.to_string())?;
                Ok(Self { tokens, cache })
            });
        built
            .inspect_err(|e| eprintln!("shared prompt prefix unavailable: {e}"))
            .ok()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn bytes(&self) -> usize {
        self.cache.kv_bytes_held()
    }

    pub fn start(&self, sched: Scheduled<'_>, prompt: &[u32], id: &str) -> Option<Stored> {
        if prompt.len() <= self.tokens.len() || !prompt.starts_with(&self.tokens) {
            return None;
        }
        let cache = sched.checkpoint(&self.cache).ok()??;
        Some(Stored {
            id: id.to_owned(),
            cache,
            retained: Retained::restored(self.tokens.clone()),
        })
    }
}
