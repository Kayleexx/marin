use super::menu::{Item, Menu};
use crate::CliError;
use crate::dialog::Talk;
use crate::session::Settings;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Inspect,
    Tokens,
    NextTokens,
    Temperature,
    TopK,
    TopP,
    MaxTokens,
    Seed,
    SwitchModel,
    SwitchBackend,
    Talk(Talk),
    Status,
    Bench,
    Verify,
    Settings,
    Save,
    Resume,
    Checkpoint,
    Awareness,
    Context,
    Suggestions,
    Why,
    Sessions,
    NewChat,
    Clear,
    Shortcuts,
    Quit,
}

pub const SHORTCUTS: [&str; 8] = [
    "enter send · alt+enter or ctrl+j new line · ↑/↓ history",
    "/ opens actions; type to search them, for example",
    "/mode code · /model · /status · /clear · /exit",
    "/awareness · /context · /suggestions · /why  (the optional background helper)",
    "ctrl+o inspect the last reply",
    "ctrl+c cancels a reply; press it twice at an empty prompt to exit · ctrl+d exits",
    "ctrl+a / ctrl+e start / end of line · ctrl+u / ctrl+k delete to start / end · ctrl+w delete word",
    "type after / to search the actions (also /save, /resume, /sessions), enter runs one",
];

pub fn palette(current: Talk) -> Menu<Action> {
    let item = |label: &str, detail: &str, value| Item {
        label: label.to_string(),
        detail: detail.to_string(),
        value,
    };
    let mode = |talk: Talk, label: &str, detail: &str| {
        let detail = if talk == current {
            format!("current · {detail}")
        } else {
            detail.to_string()
        };
        item(label, &detail, Action::Talk(talk))
    };
    Menu::new(vec![
        mode(Talk::Chat, "Mode: chat", "talk with the model"),
        mode(
            Talk::Code,
            "Mode: code",
            "ask about the project in this folder (read-only)",
        ),
        item(
            "Switch model…",
            "pick another model folder",
            Action::SwitchModel,
        ),
        item("Status", "model, hardware, window and mode", Action::Status),
        item(
            "Settings",
            "what generation uses right now",
            Action::Settings,
        ),
        item(
            "Inspect last reply",
            "metrics, tokens, next tokens, KV, model  (ctrl+o)",
            Action::Inspect,
        ),
        item(
            "Show tokens…",
            "how a text is split into tokens",
            Action::Tokens,
        ),
        item(
            "Next-token probabilities…",
            "the 5 likeliest tokens after a text",
            Action::NextTokens,
        ),
        item(
            "Temperature…",
            "randomness; 0 always picks the likeliest token",
            Action::Temperature,
        ),
        item(
            "Top-k…",
            "only consider the K likeliest tokens (0 = off)",
            Action::TopK,
        ),
        item(
            "Top-p…",
            "only consider tokens adding up to probability P",
            Action::TopP,
        ),
        item("Max tokens…", "longest reply", Action::MaxTokens),
        item(
            "Seed…",
            "the same seed repeats the same text; random = new each time",
            Action::Seed,
        ),
        item(
            "Switch backend…",
            "auto, cpu or cuda",
            Action::SwitchBackend,
        ),
        item(
            "Save session…",
            "keep both conversations to resume later (private to you)",
            Action::Save,
        ),
        item(
            "Resume session…",
            "continue a saved conversation",
            Action::Resume,
        ),
        item(
            "Checkpoint session…",
            "also keep the model's state of a saved session (large, private; Qwen3.5 GPU)",
            Action::Checkpoint,
        ),
        item(
            "Awareness…",
            "background helper: enable, disable, status, sources and allowlists",
            Action::Awareness,
        ),
        item(
            "Context",
            "what the helper currently retains (memory only)",
            Action::Context,
        ),
        item(
            "Suggestions…",
            "list, accept, dismiss, mute or snooze what the helper noticed",
            Action::Suggestions,
        ),
        item(
            "Why…",
            "why the latest suggestion appeared and what accepting sends",
            Action::Why,
        ),
        item("Sessions", "list the saved conversations", Action::Sessions),
        item(
            "Benchmark…",
            "time the loaded model: load, prefill, decode, first token, memory",
            Action::Bench,
        ),
        item(
            "Verify…",
            "check the loaded model against a float32 reference",
            Action::Verify,
        ),
        item(
            "Clear conversation",
            "forget the conversation of this mode",
            Action::NewChat,
        ),
        item("Clear screen", "", Action::Clear),
        item("Help", "keys and commands", Action::Shortcuts),
        item("Exit", "quit · ctrl+d", Action::Quit),
    ])
}

impl Action {
    /// The one-line question for actions that need a typed value.
    pub fn question(self) -> Option<&'static str> {
        Some(match self {
            Action::Tokens => "Text to split into tokens",
            Action::NextTokens => "Text to continue",
            Action::Temperature => "Temperature",
            Action::TopK => "Top-k",
            Action::TopP => "Top-p",
            Action::MaxTokens => "Max tokens",
            Action::Seed => "Seed (number or random)",
            Action::Bench => "Workload (blank: 128 64; for example 512 128 runs=2 profile)",
            Action::Verify => "Reference file (blank: find one made for this model)",
            Action::Save => "Save as (blank: the name it came from, or a new one)",
            Action::Resume => "Resume which (blank: the newest; Sessions lists them)",
            Action::Checkpoint => "Keep the model's state of which saved session (blank: this one)",
            Action::Awareness => {
                "Awareness: status, enable, disable, desktop on, browser on, allow-site HOST, allow-app NAME, autostart on (blank: status)"
            }
            Action::Suggestions => {
                "Suggestions: list, accept N, dismiss N, mute N, snooze N (blank: list)"
            }
            Action::Why => "Why which suggestion (blank: the newest)",
            _ => return None,
        })
    }
}

/// The value shown when a setting is edited.
pub fn current_value(settings: &Settings, action: Action) -> String {
    match action {
        Action::Temperature => settings.temperature.to_string(),
        Action::TopK => settings.top_k.to_string(),
        Action::TopP => settings.top_p.to_string(),
        Action::MaxTokens => settings.max_tokens.to_string(),
        Action::Seed => settings
            .seed
            .map_or("random".to_string(), |s| s.to_string()),
        _ => String::new(),
    }
}

/// Applies a typed value to `settings`, leaving them untouched if it is invalid.
pub fn apply_setting(
    settings: &mut Settings,
    action: Action,
    text: &str,
) -> Result<String, CliError> {
    let value = text.trim();
    let bad = |what: &str| CliError::BadOption(format!("expected {what}, got `{value}`"));
    let mut next = *settings;
    let name = match action {
        Action::Temperature => {
            next.temperature = value.parse().map_err(|_| bad("a number"))?;
            "temperature"
        }
        Action::TopK => {
            next.top_k = value.parse().map_err(|_| bad("a whole number"))?;
            "top-k"
        }
        Action::TopP => {
            next.top_p = value.parse().map_err(|_| bad("a number"))?;
            "top-p"
        }
        Action::MaxTokens => {
            next.max_tokens = value.parse().map_err(|_| bad("a whole number"))?;
            "max tokens"
        }
        Action::Seed if value == "random" => {
            next.seed = None;
            "seed"
        }
        Action::Seed => {
            next.seed = Some(
                value
                    .parse()
                    .map_err(|_| bad("a whole number or `random`"))?,
            );
            "seed"
        }
        _ => return Err(CliError::BadOption("that action has no value".into())),
    };
    next.sampler_config(0).validate()?;
    *settings = next;
    Ok(format!("{name} = {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings {
            max_tokens: 128,
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            seed: None,
        }
    }

    #[test]
    fn valid_values_are_applied_and_reported() {
        let mut s = settings();
        assert_eq!(
            apply_setting(&mut s, Action::Temperature, " 0.2 ").unwrap(),
            "temperature = 0.2"
        );
        assert_eq!(s.temperature, 0.2);
        apply_setting(&mut s, Action::MaxTokens, "16").unwrap();
        apply_setting(&mut s, Action::TopK, "40").unwrap();
        apply_setting(&mut s, Action::Seed, "7").unwrap();
        assert_eq!((s.max_tokens, s.top_k, s.seed), (16, 40, Some(7)));
        apply_setting(&mut s, Action::Seed, "random").unwrap();
        assert_eq!(s.seed, None);
    }

    #[test]
    fn invalid_values_leave_settings_unchanged() {
        let mut s = settings();
        for (action, text) in [
            (Action::Temperature, "hot"),
            (Action::Temperature, "-1"),
            (Action::TopP, "0"),
            (Action::TopP, "1.5"),
            (Action::MaxTokens, "-3"),
            (Action::TopK, "1.5"),
            (Action::Seed, "abc"),
            (Action::Quit, "x"),
        ] {
            assert!(
                apply_setting(&mut s, action, text).is_err(),
                "{action:?} {text}"
            );
        }
        assert_eq!(s.temperature, 0.7);
        assert_eq!(s.top_p, 0.9);
        assert_eq!(s.max_tokens, 128);
    }

    #[test]
    fn current_values_prefill_the_question() {
        let s = settings();
        assert_eq!(current_value(&s, Action::Temperature), "0.7");
        assert_eq!(current_value(&s, Action::Seed), "random");
        assert_eq!(Action::Tokens.question(), Some("Text to split into tokens"));
        assert_eq!(Action::Clear.question(), None);
    }

    #[test]
    fn palette_lists_every_action_once() {
        let mut menu = palette(Talk::Chat);
        let mut seen = Vec::new();
        for _ in 0..28 {
            seen.push(*menu.current().unwrap());
            menu.down();
        }
        seen.dedup();
        assert_eq!(seen.len(), 28);
        for c in "tok".chars() {
            menu.push(c);
        }
        assert_eq!(menu.current(), Some(&Action::Tokens));
    }

    fn typed(text: &str) -> Option<Action> {
        let mut menu = palette(Talk::Chat);
        text.chars().for_each(|c| menu.push(c));
        menu.current().copied()
    }

    #[test]
    fn the_commands_people_type_after_a_slash_find_their_action() {
        assert_eq!(typed("mode code"), Some(Action::Talk(Talk::Code)));
        assert_eq!(typed("mode chat"), Some(Action::Talk(Talk::Chat)));
        assert_eq!(typed("model"), Some(Action::SwitchModel));
        assert_eq!(typed("status"), Some(Action::Status));
        assert_eq!(typed("bench"), Some(Action::Bench));
        assert_eq!(typed("verify"), Some(Action::Verify));
        assert_eq!(typed("settings"), Some(Action::Settings));
        assert_eq!(typed("clear"), Some(Action::NewChat));
        assert_eq!(typed("help"), Some(Action::Shortcuts));
        assert_eq!(typed("save"), Some(Action::Save));
        assert_eq!(typed("resume"), Some(Action::Resume));
        assert_eq!(typed("sessions"), Some(Action::Sessions));
        assert_eq!(typed("awareness"), Some(Action::Awareness));
        assert_eq!(typed("context"), Some(Action::Context));
        assert_eq!(typed("suggestions"), Some(Action::Suggestions));
        assert_eq!(typed("why"), Some(Action::Why));
        assert_eq!(typed("checkpoint"), Some(Action::Checkpoint));
        assert_eq!(typed("exit"), Some(Action::Quit));
        assert_eq!(typed("quit"), Some(Action::Quit));
    }
}
