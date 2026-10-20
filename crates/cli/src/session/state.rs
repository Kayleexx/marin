use runtime::Retained;

use super::Session;
use crate::CliError;

impl Session {
    /// Drops the cache, so the memory is free before another one is filled in its place.
    pub fn release_cache(&mut self) -> Result<(), CliError> {
        self.kv = self.loaded.model.new_cache(self.ctx)?;
        self.retained.forget();
        self.boundary = None;
        self.kv_used = 0;
        Ok(())
    }

    /// Takes a cache that was filled from outside, which holds exactly `tokens`.
    pub fn install_state(&mut self, cache: model::EngineCache, tokens: Vec<u32>) {
        self.kv_used = tokens.len();
        self.kv = cache;
        self.retained = Retained::restored(tokens);
        self.boundary = None;
    }
}
