//! The shared prompt prefix is compared by token IDs, because equal text can tokenize differently at the boundary.
use chat::{Message, Role};
use std::path::PathBuf;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Needs a Qwen3.5 model folder in `MARIN_QWEN35_Q8_DIR`.
#[test]
fn a_user_message_starting_with_a_newline_does_not_share_the_whole_prefix() -> Result {
    let Some(dir) = std::env::var_os("MARIN_QWEN35_Q8_DIR").map(PathBuf::from) else {
        eprintln!("MARIN_QWEN35_Q8_DIR is not set");
        return Ok(());
    };
    let dir = loader::ModelDir::open(&dir)?;
    let specials = dir.tokenizer.special_tokens();
    let encode = |user: &str| -> Result<Vec<u32>> {
        let messages = [
            Message::new(Role::System, chat::DEFAULT_SYSTEM),
            Message::new(Role::User, user),
        ];
        Ok(dir.tokenizer.encode(&chat::render(&messages, &specials)?)?)
    };
    let plain = [
        encode("Hello")?,
        encode("Say hello in one short sentence.")?,
        encode(" leading space")?,
    ];
    let shared = plain[0]
        .iter()
        .zip(&plain[1])
        .take_while(|(a, b)| a == b)
        .count();
    assert!(shared > 30);
    let prefix = plain[0].get(..shared).ok_or("prefix out of range")?;
    assert!(plain.iter().all(|ids| ids.starts_with(prefix)));
    let newline = encode("\nweird newline start")?;
    assert!(!newline.starts_with(prefix));
    assert!(newline.starts_with(prefix.get(..shared - 1).ok_or("short prefix")?));
    Ok(())
}
