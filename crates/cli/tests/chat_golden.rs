//! Token-level check of the chat formatter against Hugging Face's `apply_chat_template`.
//! It needs the real Qwen3 tokenizer, so it runs when MARIN_QWEN3_CHAT_DIR points at a
//! Qwen3-1.7B folder; set MARIN_REQUIRE_QWEN3_CHAT=1 to make a missing folder a failure.

use std::path::{Path, PathBuf};

use chat::{Message, Role, render};
use loader::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    meta: Meta,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Meta {
    special_tokens: std::collections::BTreeMap<String, u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Raw>,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Raw {
    role: String,
    content: String,
}

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("MARIN_QWEN3_CHAT_DIR").map(PathBuf::from)
}

#[test]
fn chat_prompts_tokenize_to_exactly_the_ids_hugging_face_produces() {
    let Some(dir) = model_dir() else {
        assert!(
            std::env::var_os("MARIN_REQUIRE_QWEN3_CHAT").is_none(),
            "MARIN_QWEN3_CHAT_DIR is not set"
        );
        eprintln!("skipping: set MARIN_QWEN3_CHAT_DIR to a Qwen3-1.7B folder to run this check");
        return;
    };
    let golden_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference/qwen3-chat-golden.json");
    let golden: Golden =
        serde_json::from_str(&std::fs::read_to_string(golden_path).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let specials = tokenizer.special_tokens();
    for (name, id) in &golden.meta.special_tokens {
        assert!(
            specials.contains(name),
            "{name} is a special token of the tokenizer"
        );
        assert_eq!(
            tokenizer.token_id(name),
            Some(*id),
            "{name} id is read from the tokenizer"
        );
    }
    for case in golden.cases {
        let messages: Vec<Message> = case
            .messages
            .iter()
            .map(|m| {
                let role = match m.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                Message::new(role, m.content.clone())
            })
            .collect();
        let ids = tokenizer
            .encode(&render(&messages, &specials).unwrap())
            .unwrap();
        assert_eq!(ids, case.ids, "{}", case.name);
    }
}
