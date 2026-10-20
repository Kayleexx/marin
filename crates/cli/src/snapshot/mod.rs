//! One trusted prefix of the KV cache, saved to a file and restored by the next run when its prompt starts the same
//! way. A snapshot is only ever a shortcut: anything wrong with it means the prompt is read in full.

mod format;
mod identity;
mod restore;
mod save;

pub use format::{Identity, Reject, hash_parts, ids_hash};
pub use identity::identity;
pub use restore::{Restored, Timing, restore};
pub use save::{Saved, save};

#[cfg(test)]
mod format_tests;
