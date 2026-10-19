use std::time::{SystemTime, UNIX_EPOCH};

use chat::{Message, Role};
use sampler::Sampler;
use serde::Deserialize;

use crate::load::Loaded;
use crate::session::Settings;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Completion {
    model: String,
    messages: Vec<InputMessage>,
    #[serde(default)]
    pub(super) stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    pub(super) marin_session_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputMessage {
    role: InputRole,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum InputRole {
    System,
    User,
    Assistant,
}

fn encode_chat(loaded: &Loaded, messages: &[Message]) -> Result<Vec<u32>, String> {
    let specials = loaded.dir.tokenizer.special_tokens();
    let text = chat::render(messages, &specials).map_err(|e| e.to_string())?;
    loaded
        .dir
        .tokenizer
        .encode(&text)
        .map_err(|e| e.to_string())
}

pub(super) fn default_prefix(loaded: &Loaded) -> Result<Vec<u32>, String> {
    let encode = |user: &str| {
        encode_chat(
            loaded,
            &[
                Message::new(Role::System, chat::DEFAULT_SYSTEM),
                Message::new(Role::User, user),
            ],
        )
    };
    let (first, second) = (encode("Hello")?, encode("The quick brown fox")?);
    let shared = first
        .iter()
        .zip(&second)
        .take_while(|(a, b)| a == b)
        .count();
    first
        .get(..shared)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| "the default prompts share no prefix".into())
}

pub(super) fn validated_prompt(
    loaded: &Loaded,
    input: &Completion,
    defaults: Settings,
) -> Result<(Vec<u32>, Vec<u32>, Sampler, usize), String> {
    if input.model != loaded.name {
        return Err(format!("model must be `{}` (see /v1/models)", loaded.name));
    }
    if input.messages.is_empty()
        || !matches!(
            input.messages.last().map(|m| &m.role),
            Some(InputRole::User)
        )
        || input
            .messages
            .last()
            .is_some_and(|m| m.content.trim().is_empty())
    {
        return Err("messages must end with a non-empty user message".into());
    }
    let mut messages: Vec<Message> = input
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                InputRole::System => Role::System,
                InputRole::User => Role::User,
                InputRole::Assistant => Role::Assistant,
            };
            Message::new(role, &m.content)
        })
        .collect();
    if !messages.iter().any(|m| m.role == Role::System) {
        messages.insert(0, Message::new(Role::System, chat::DEFAULT_SYSTEM));
    }
    let specials = loaded.dir.tokenizer.special_tokens();
    let prompt = encode_chat(loaded, &messages)?;
    let history = if input.marin_session_id.is_some() {
        let history = chat::render_history(&messages, &specials).map_err(|e| e.to_string())?;
        loaded
            .dir
            .tokenizer
            .encode(&history)
            .map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };
    if prompt.len() > loaded.ctx {
        return Err(format!(
            "prompt has {} tokens but context holds {}",
            prompt.len(),
            loaded.ctx
        ));
    }
    let max_tokens = input
        .max_tokens
        .unwrap_or(defaults.max_tokens.min(loaded.ctx));
    if max_tokens == 0 || max_tokens > loaded.ctx {
        return Err(format!("max_tokens must be between 1 and {}", loaded.ctx));
    }
    let settings = Settings {
        max_tokens,
        temperature: input.temperature.unwrap_or(defaults.temperature),
        top_p: input.top_p.unwrap_or(defaults.top_p),
        top_k: input.top_k.unwrap_or(defaults.top_k),
        seed: input.seed.or(defaults.seed),
    };
    let seed = settings.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let sampler = Sampler::new(settings.sampler_config(seed)).map_err(|e| e.to_string())?;
    Ok((prompt, history, sampler, max_tokens))
}
