#![forbid(unsafe_code)]

mod conversation;
mod qwen3;

pub use conversation::{Conversation, FitError, fit};
pub use qwen3::{ChatError, ModelKind, classify, render, render_history};

pub const DEFAULT_SYSTEM: &str = "You are Marin, a private assistant that runs on the user's own machine. Be friendly, clear and concise.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}
