//! Conversation continuity: every turn's prompt carries the whole history in exact roles, ending with the
//! current user text, and nothing from an earlier turn leaks in except through that history.

use chat::{DEFAULT_SYSTEM, Message, Role, render};
use cli::chat::Chat;
use cli::load::load;
use cli::session::{Session, Settings};
use cli::signal::Interrupt;
use cli::style::Style;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use model::BackendChoice;

fn session(seed: u64) -> Result<Session> {
    let model =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let loaded = load(
        &model,
        Some(512),
        512,
        BackendChoice::Cpu,
        None,
        &Style::plain(),
        false,
    )?;
    let settings = Settings {
        max_tokens: 6,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        seed: Some(seed),
    };
    Ok(Session::new(loaded, settings, Style::plain())?)
}

fn expected_prompt(session: &Session, history: &[(Role, &str)]) -> Result<Vec<u32>> {
    let mut messages = vec![Message::new(Role::System, DEFAULT_SYSTEM)];
    messages.extend(
        history
            .iter()
            .map(|(role, text)| Message::new(*role, *text)),
    );
    let specials = session.loaded.dir.tokenizer.special_tokens();
    let text = render(&messages, &specials)?;
    Ok(session.loaded.dir.tokenizer.encode(&text)?)
}

fn each_turn_sends_the_whole_history_and_the_new_user_text(interrupt: &Interrupt) -> Result<()> {
    for turns in [
        ["hi", "wsg"],
        ["hi", "how's life?"],
        ["my name is alice", "what name did i tell you?"],
        ["the secret number is 417", "what was the secret number?"],
    ] {
        let mut session = session(5)?;
        let mut chat = Chat::new(&session);
        let first = chat.ask(&mut session, turns[0], interrupt, &mut Vec::new())?;
        assert_eq!(
            session.last_prompt,
            expected_prompt(&session, &[(Role::User, turns[0])])?
        );
        chat.ask(&mut session, turns[1], interrupt, &mut Vec::new())?;
        let history = [
            (Role::User, turns[0]),
            (Role::Assistant, first.text.as_str()),
            (Role::User, turns[1]),
        ];
        assert_eq!(
            session.last_prompt,
            expected_prompt(&session, &history)?,
            "{turns:?}: turn two carries turn one and ends with its own text"
        );
    }
    Ok(())
}

fn a_new_conversation_starts_clean_and_equal_settings_repeat_exactly(
    interrupt: &Interrupt,
) -> Result<()> {
    let mut session = session(9)?;
    let mut chat = Chat::new(&session);
    let a = chat.ask(&mut session, "hi", interrupt, &mut Vec::new())?;
    chat.ask(&mut session, "wsg", interrupt, &mut Vec::new())?;
    chat.clear();
    let b = chat.ask(&mut session, "hi", interrupt, &mut Vec::new())?;
    assert_eq!(
        session.last_prompt,
        expected_prompt(&session, &[(Role::User, "hi")])?,
        "nothing from the cleared conversation remains"
    );
    assert_eq!(
        a.tokens, b.tokens,
        "same prompt and seed give the same reply"
    );
    Ok(())
}

/// The signal handler can be installed once per process, so both checks share one test.
#[test]
fn conversation_continuity() -> Result<()> {
    let interrupt = Interrupt::install()?;
    each_turn_sends_the_whole_history_and_the_new_user_text(&interrupt)?;
    a_new_conversation_starts_clean_and_equal_settings_repeat_exactly(&interrupt)
}
