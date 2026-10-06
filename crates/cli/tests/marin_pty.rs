//! The real interactive UI on a pseudo-terminal supplied by util-linux `script`: startup layouts, the palette and
//! actions, keys typed during a reply, and a terminal that comes back usable however Marin ends or fails. After a
//! run a `stty -a` in the same pseudo-terminal must show canonical mode and echo again, and a model switch, which
//! fails on machines without CUDA, must leave a working prompt.

mod pty;

use std::path::Path;
use std::process::ExitStatus;

use pty::{Res, Sandbox, Step, have, on_terminal};

/// What a key press has to show before the next one is typed, so a slow machine only makes a test slower.
#[derive(Clone, Copy, Debug)]
enum Effect {
    /// This text is drawn.
    Shows(&'static str),
    /// One of these notices is drawn.
    AnyOf(&'static [&'static str]),
    /// A reply was drawn to its end and the prompt is back.
    ReplyDone,
    /// Nothing: the press only edits what the program already shows, like typing into a filter, and the terminal
    /// queues it in order.
    Queued,
}

const PROMPT: Effect = Effect::Shows("you >");

impl Effect {
    /// Whether `text`, everything drawn since the previous press settled, shows this effect.
    fn recognizes(self, text: &str) -> bool {
        match self {
            Self::Shows(shown) => text.contains(shown),
            Self::AnyOf(notices) => notices.iter().any(|notice| text.contains(notice)),
            Self::ReplyDone => text
                .find("stopped at")
                .and_then(|at| text.get(at..))
                .is_some_and(|rest| rest.contains("you >")),
            Self::Queued => true,
        }
    }
}

/// How a session is started.
struct Session<'a> {
    name: &'a str,
    /// Columns and rows of the terminal.
    size: (usize, usize),
    color: bool,
    max_tokens: u32,
    /// Print the terminal flags after the program ends.
    stty: bool,
}

impl Session<'_> {
    /// Types `keys` on the tiny chat model, each after the effect of the one before, then quits with Ctrl+D.
    /// Returns everything drawn and how the program ended, or None when `script` is unavailable.
    fn run(&self, keys: &[(&str, Effect)]) -> Res<Option<(String, ExitStatus)>> {
        if !have("script") {
            eprintln!("skipping: `script` is not installed");
            return Ok(None);
        }
        let sandbox = Sandbox::new(self.name)?;
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
        let command_line = format!(
            "{} --model {} --backend cpu --temp 0 --max-tokens {}{}",
            env!("CARGO_BIN_EXE_marin"),
            model.display(),
            self.max_tokens,
            if self.stty { "; stty -a" } else { "" }
        );
        let checks: Vec<_> = keys
            .iter()
            .map(|(_, effect)| {
                let effect = *effect;
                (format!("{effect:?}"), move |_: &[u8], new: &[u8]| {
                    let text = String::from_utf8_lossy(new);
                    if effect.recognizes(&text) {
                        Ok(())
                    } else {
                        Err(text.escape_debug().to_string())
                    }
                })
            })
            .collect();
        let steps: Vec<Step> = keys
            .iter()
            .zip(&checks)
            .map(|((typed, effect), (what, check))| Step {
                keys: typed,
                until: match effect {
                    Effect::Queued => None,
                    _ => Some((what.as_str(), check)),
                },
            })
            .collect();
        let (drawn, _, status) =
            on_terminal(&command_line, self.size, &sandbox, self.color, &steps)?;
        Ok(Some((String::from_utf8_lossy(&drawn).into_owned(), status)))
    }
}

/// Drives the real interactive UI: the palette, an action, a reply, the inspect panel and a clean exit.
#[test]
fn interactive_session_opens_the_palette_runs_an_action_and_exits() {
    let session = Session {
        name: "pty",
        size: (120, 40),
        color: false,
        max_tokens: 4,
        stty: false,
    };
    let keys = [
        ("", PROMPT),
        ("/tok", Effect::Shows("Show tokens")),
        ("\r", Effect::Shows("Text to split into tokens")),
        ("hello world\r", Effect::Shows("8 tokens")),
        ("hello world\r", Effect::ReplyDone),
        ("\x0f", Effect::Shows("[Reply]")),
        ("\x1b", Effect::Queued),
    ];
    let Some((screen, status)) = session.run(&keys).unwrap() else {
        return;
    };
    for expected in [
        "Status    Ready",
        "private, local-first intelligence agent",
        "built on a hardware-adaptive inference runtime",
        "Press / to open actions.",
        "you > ",
        "/ actions · ctrl+o inspect",
        "Show tokens",
        "Text to split into tokens",
        "6 tokens",
        "you > hello world",
        "[Reply]",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in {screen:?}"
        );
    }
    assert!(
        status.success(),
        "ctrl+d at an empty prompt exits cleanly: {status:?}"
    );
}

/// Keys typed while a reply streams are echoed by the terminal; they must be discarded when the prompt returns.
#[test]
fn keys_typed_during_a_reply_do_not_leak_into_the_next_prompt() {
    let session = Session {
        name: "typeahead",
        size: (100, 30),
        color: false,
        max_tokens: 3,
        stty: false,
    };
    // Both lines arrive at once, so the second is typed while the first one's reply is still being drawn.
    let keys = [("", PROMPT), ("hello world\rjunkjunk\r", Effect::ReplyDone)];
    let Some((screen, _)) = session.run(&keys).unwrap() else {
        return;
    };
    assert!(screen.contains("you > hello world"), "{screen:?}");
    assert!(
        !screen.contains("junkjunk\u{23ce}") && !screen.contains("you > junkjunk"),
        "stray keys leaked into the prompt: {screen:?}"
    );
}

/// A session at `columns` x `lines`, with an optional first message. Returns everything drawn.
fn session_output(name: &str, size: (usize, usize), color: bool, say: bool) -> Res<Option<String>> {
    let session = Session {
        name,
        size,
        color,
        max_tokens: 3,
        stty: false,
    };
    let keys: &[(&str, Effect)] = if say {
        &[("", PROMPT), ("hello world\r", Effect::ReplyDone)]
    } else {
        &[("", PROMPT)]
    };
    Ok(session.run(keys)?.map(|(screen, _)| screen))
}

const CLEAR_SCREEN: &str = "\u{1b}[2J";
const ALTERNATE_SCREEN: &str = "\u{1b}[?1049";
const PURGE_SCROLLBACK: &str = "\u{1b}[3J";

/// The welcome is ordinary terminal output: it stays above the conversation in the scrollback, so nothing
/// clears the screen, switches to the alternate screen or purges history, in any layout.
#[test]
fn the_welcome_stays_in_the_scrollback_above_the_conversation() {
    for (columns, lines, layout) in [
        (160, 52, "extra-large portrait"),
        (120, 35, "large portrait"),
        (100, 30, "face crop"),
        (80, 24, "compact"),
        (40, 20, "narrow"),
    ] {
        let Some(screen) = session_output("welcome", (columns, lines), true, true).unwrap() else {
            return;
        };
        for control in [CLEAR_SCREEN, ALTERNATE_SCREEN, PURGE_SCROLLBACK] {
            assert!(
                !screen.contains(control),
                "{layout}: {control:?} would hide or destroy terminal history"
            );
        }
        let ready = screen.find("Ready").unwrap_or(usize::MAX);
        let said = screen.find("hello world").unwrap_or(0);
        assert!(
            ready < said,
            "{layout}: the welcome is printed first and the conversation follows it"
        );
        assert!(
            screen.contains("you >") && screen.contains("\u{1b}[0m hello world"),
            "{layout}: the first prompt is echoed as a conversation line"
        );
    }
}

/// With a portrait the first prompt belongs in the right-hand column, as in the reference.
#[test]
fn with_a_portrait_the_first_prompt_is_drawn_in_the_right_hand_column() {
    let Some(screen) = session_output("column", (120, 35), true, false).unwrap() else {
        return;
    };
    assert!(
        screen.contains("\u{1b}[72G\u{1b}[J\u{1b}[38;5;147myou >\u{1b}[0m "),
        "the prompt is drawn at column 72, right of the 68-column portrait: {:?}",
        screen
            .chars()
            .rev()
            .take(200)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
}

/// NO_COLOR gets a readable monochrome startup: no color codes, but the same layout.
#[test]
fn no_color_startup_has_no_color_codes_but_the_same_content() {
    let Some(screen) = session_output("mono", (120, 35), false, false).unwrap() else {
        return;
    };
    for code in ["\u{1b}[38;", "\u{1b}[48;", "\u{1b}[0m"] {
        assert!(
            !screen.contains(code),
            "{code:?} must not appear with NO_COLOR"
        );
    }
    assert!(
        screen.contains("private, local-first intelligence agent")
            && screen.contains("Status    Ready")
    );
}

/// After the run a `stty -a` in the same terminal must show canonical mode and echo again.
fn assert_cooked(screen: &str, how: &str) {
    let flags = screen.rsplit("lflags").next().unwrap_or("");
    assert!(
        flags.contains(" icanon ") && !flags.contains("-icanon") && !flags.contains("-echo "),
        "{how}: terminal left raw: {flags:?}"
    );
}

fn restore_session(name: &str) -> Session<'_> {
    Session {
        name,
        size: (100, 30),
        color: false,
        max_tokens: 3,
        stty: true,
    }
}

#[test]
fn the_terminal_is_restored_after_ctrl_d_ctrl_c_and_the_quit_action() {
    let palette = Effect::Shows("Inspect last reply");
    for (how, keys) in [
        ("ctrl+d", vec![("", PROMPT)]),
        (
            "ctrl+c twice",
            vec![
                ("", PROMPT),
                ("\x03", Effect::Shows("press ctrl+c again to exit")),
                ("\x03", Effect::Queued),
            ],
        ),
        (
            "quit action",
            vec![
                ("", PROMPT),
                ("/", palette),
                ("quit", Effect::Queued),
                ("\r", Effect::Queued),
            ],
        ),
        (
            "after a reply",
            vec![("", PROMPT), ("hello world\r", Effect::ReplyDone)],
        ),
    ] {
        let Some((screen, _)) = restore_session("restore").run(&keys).unwrap() else {
            return;
        };
        assert_cooked(&screen, how);
    }
}

#[test]
fn a_backend_switch_attempt_leaves_a_working_prompt_and_a_restored_terminal() {
    let keys = [
        ("", PROMPT),
        ("/", Effect::Shows("Inspect last reply")),
        ("backend", Effect::Queued),
        ("\r", Effect::Shows("Backend")),
        ("cuda", Effect::Queued),
        (
            "\r",
            Effect::AnyOf(&["now using", "the switch failed", "no model is loaded"]),
        ),
        ("hello world\r", Effect::ReplyDone),
    ];
    let Some((screen, _)) = restore_session("switch").run(&keys).unwrap() else {
        return;
    };
    assert!(
        screen.contains("you > hello world") && screen.contains("marin >"),
        "the conversation continues after the switch attempt: {screen:?}"
    );
    assert!(
        !screen.contains("no model is loaded"),
        "the previous model came back: {screen:?}"
    );
    assert_cooked(&screen, "after the switch");
}
