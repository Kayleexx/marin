use std::sync::atomic::AtomicBool;

use super::client::{Ask, Client, ClientError, Reply};

/// Whatever answers the tool loop: a running `marin serve`, or the model loaded in this process.
pub trait Brain {
    /// Streams one reply, passing each piece of text to `on_text`; returning false from it ends the reply early.
    fn reply(
        &mut self,
        ask: &Ask<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Reply, ClientError>;
}

impl Brain for Client {
    fn reply(
        &mut self,
        ask: &Ask<'_>,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Reply, ClientError> {
        self.chat(ask, cancel, |piece| on_text(piece))
    }
}
