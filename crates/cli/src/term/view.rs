use super::app::{App, Mode};
use super::editor::Editor;
use super::menu::Menu;
use super::panel::{self, KvStats, PanelData, ReplyStats};
use super::screen::{Kind, Line, display_width, height, menu_lines, width};
use crate::inspect::describe;
use crate::session::Session;
use crate::telemetry;
use runtime::StopReason;

const MAX_MENU_ROWS: usize = 6;
const MIN_MENU_ROWS: usize = 2;
/// Lines the palette keeps free for the prompt, its hint and some of the conversation above.
const MENU_CHROME: usize = 7;

/// Palette height: up to 6 rows, fewer on short terminals.
fn menu_rows_for(terminal_height: usize) -> usize {
    terminal_height
        .saturating_sub(MENU_CHROME)
        .clamp(MIN_MENU_ROWS, MAX_MENU_ROWS)
}

fn menu_rows() -> usize {
    menu_rows_for(height())
}

/// The input prompt: what the user types follows "you >"; Marin's replies are introduced by "marin >".
pub(super) const PROMPT: &str = "you > ";
pub(super) const REPLY_PREFIX: &str = "marin > ";

pub(super) fn stop_label(stop: StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens => "max tokens",
        StopReason::EndOfText => "end of text",
        StopReason::ContextFull => "context window full",
        StopReason::Cancelled => "cancelled",
    }
}

/// The prompt line and the cursor column. Long input scrolls so the cursor stays visible,
/// measured in terminal columns so wide characters never push the cursor off the line.
pub(crate) fn input_line(editor: &Editor, prefix: &str, columns: usize) -> (String, usize) {
    let chars: Vec<char> = editor
        .chars()
        .iter()
        .map(|c| if *c == '\n' { '⏎' } else { *c })
        .collect();
    let prefix_columns = display_width(prefix);
    let room = columns
        .saturating_sub(prefix_columns)
        .saturating_sub(2)
        .max(2);
    let cursor = editor.cursor().min(chars.len());
    let mut start = cursor;
    while let Some(i) = start.checked_sub(1) {
        let candidate: String = chars
            .iter()
            .skip(i)
            .take(cursor.saturating_sub(i))
            .collect();
        if display_width(&candidate) >= room {
            break;
        }
        start = i;
    }
    let before: String = chars
        .iter()
        .skip(start)
        .take(cursor.saturating_sub(start))
        .collect();
    let mut visible = String::new();
    for c in chars.iter().skip(start) {
        visible.push(*c);
        if display_width(&visible) > room {
            visible.pop();
            break;
        }
    }
    (
        format!("{prefix}{visible}"),
        prefix_columns.saturating_add(display_width(&before)),
    )
}

impl App {
    /// Columns left for blocks: the whole width, or what remains right of the startup portrait.
    fn columns(&self) -> usize {
        width().saturating_sub(self.screen.indent())
    }

    /// Runtime details live in the inspect panel; the prompt only hints at what is available.
    fn status_line(&self) -> String {
        let full = "/ actions · ctrl+o inspect";
        if display_width(full) < self.columns() {
            full.to_string()
        } else {
            "/ actions".to_string()
        }
    }

    pub(super) fn view(&self) -> (Vec<Line>, usize) {
        match &self.mode {
            Mode::Input => {
                let (text, column) = input_line(&self.editor, PROMPT, self.columns());
                let mut lines = vec![
                    Line::new(Kind::Prompt, text),
                    Line::new(Kind::Dim, self.status_line()),
                ];
                if let Some(note) = &self.notice {
                    lines.push(Line::new(Kind::Dim, note.clone()));
                }
                (lines, column)
            }
            Mode::Palette(menu) => {
                let first = format!("/{}", menu.query());
                let column = display_width(&first);
                let mut lines = vec![Line::new(Kind::Plain, first)];
                lines.extend(menu_lines(menu, menu_rows()));
                lines.push(Line::new(Kind::Dim, "↑/↓ choose · enter run · esc close"));
                (lines, column)
            }
            Mode::PickModel(menu) => Self::pick_view("Model", menu),
            Mode::PickBackend(menu) => Self::pick_view("Backend", menu),
            Mode::Ask {
                question, editor, ..
            } => {
                let (text, column) = input_line(editor, &format!("{question}: "), self.columns());
                let hint = Line::new(Kind::Dim, "enter apply · esc cancel");
                (vec![Line::new(Kind::Plain, text), hint], column)
            }
            Mode::Panel(tab) => {
                let (text, column) = input_line(&self.editor, PROMPT, self.columns());
                let mut lines = vec![
                    Line::new(Kind::Prompt, text),
                    Line::new(Kind::Strong, panel::header(*tab)),
                ];
                let body = match self.panel_data() {
                    Some(data) => panel::body(*tab, &data),
                    None => vec![
                        "no model is loaded: open / actions and choose Switch model".to_string(),
                    ],
                };
                lines.extend(
                    body.into_iter()
                        .map(|l| Line::new(Kind::Plain, format!("  {l}"))),
                );
                (lines, column)
            }
        }
    }

    pub(super) fn pick_view<T>(title: &str, menu: &Menu<T>) -> (Vec<Line>, usize) {
        let first = format!("{title}: {}", menu.query());
        let column = display_width(&first);
        let mut lines = vec![Line::new(Kind::Plain, first)];
        lines.extend(menu_lines(menu, menu_rows()));
        lines.push(Line::new(
            Kind::Dim,
            "↑/↓ choose · enter select · esc cancel",
        ));
        (lines, column)
    }

    /// What generation will actually use, so the numbers on screen are the numbers in effect.
    fn settings_lines(s: &Session) -> Vec<String> {
        let config = s.loaded.model.config();
        let tokenizer = &s.loaded.dir.tokenizer;
        let stops = s.loaded.stop_tokens();
        let stops: Vec<String> = stops
            .iter()
            .map(|id| {
                format!(
                    "{} ({id})",
                    tokenizer.token_text(*id).unwrap_or_else(|| "?".to_string())
                )
            })
            .collect();
        let source = if stops.len() > config.eos_tokens.len() {
            "config.json + tokenizer_config.json"
        } else if config.eos_from_generation_config {
            "generation_config.json"
        } else {
            "config.json"
        };
        let seed = match (s.settings.seed, s.last.as_ref()) {
            (Some(seed), _) => format!("{seed} (fixed)"),
            (None, Some(_)) => format!("random, last reply used {}", s.last_seed),
            (None, None) => "random, new every reply".to_string(),
        };
        let sampling = if s.settings.temperature == 0.0 {
            "greedy (temperature 0)"
        } else {
            "sampling"
        };
        vec![
            format!("mode          {sampling}"),
            format!("temperature   {}", s.settings.temperature),
            format!(
                "top-k         {}",
                if s.settings.top_k == 0 {
                    "off".to_string()
                } else {
                    s.settings.top_k.to_string()
                }
            ),
            format!("top-p         {}", s.settings.top_p),
            format!("seed          {seed}"),
            format!("max tokens    {}", s.settings.max_tokens),
            "thinking      off (enable_thinking=false: prompts end with an empty <think> block)"
                .to_string(),
            format!("stop tokens   {}", stops.join(", ")),
            format!("stop source   {source}"),
        ]
    }

    fn panel_data(&self) -> Option<PanelData> {
        let s = self.session.as_ref()?;
        let reply = s.last.as_ref().map(|o| ReplyStats {
            lines: telemetry::request_lines(o),
            stop: stop_label(o.stop),
            seed: (s.settings.temperature > 0.0).then_some(s.last_seed),
        });
        let model = describe(
            &s.loaded.name,
            chat::classify(s.loaded.dir.tokenizer_config_json.as_deref()),
            s.loaded.model.config(),
            s.loaded.dir.tokenizer.vocab_size(),
            s.ctx,
            &s.loaded.footprint,
            model::granite::is_granite(&s.loaded.dir.config_json),
        );
        Some(PanelData {
            reply,
            tokens: self.cache.tokens.clone(),
            next: self.cache.next.clone(),
            kv: KvStats {
                used: s.kv_used,
                capacity: s.ctx,
                bytes: s.loaded.footprint.kv_window,
                held: s.kv().kv_bytes_held(),
                format: s.loaded.model.kv_format().label(),
                backend: s.loaded.model.backend_name(),
            },
            model: model
                .lines()
                .map(|l| l.trim_start().to_string())
                .chain(telemetry::resource_lines(s))
                .collect(),
            settings: Self::settings_lines(s),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_palette_never_takes_more_than_six_rows_or_fewer_than_two() {
        assert_eq!(menu_rows_for(0), 2);
        assert_eq!(menu_rows_for(8), 2);
        assert_eq!(menu_rows_for(10), 3);
        assert_eq!(menu_rows_for(13), 6);
        assert_eq!(menu_rows_for(24), 6);
        assert_eq!(menu_rows_for(200), 6);
    }

    #[test]
    fn long_and_wide_input_scrolls_so_the_cursor_stays_inside_the_line() {
        let long: String = "x".repeat(500);
        let editor = Editor::with_text(&long);
        let (line, column) = input_line(&editor, PROMPT, width());
        assert!(
            display_width(&line) < width(),
            "{} columns on a {} wide screen",
            display_width(&line),
            width()
        );
        assert!(column <= display_width(&line));
        let wide = Editor::with_text(&"你好".repeat(100));
        let (line, column) = input_line(&wide, PROMPT, width());
        assert!(
            display_width(&line) < width() && column <= display_width(&line),
            "double-width characters are measured as two columns"
        );
    }

    const AWKWARD: [&str; 8] = [
        "❤️ love",
        "family 👨‍👩‍👧 here",
        "flag 🇯🇵 ok",
        "cafe\u{301} e\u{301}",
        "你好,世界 こんにちは",
        "⣿⣿ ██ ▀▄ ░▒▓",
        "mix 😀你a😀b",
        "tab\tand\nnewline",
    ];

    #[test]
    fn the_cursor_column_matches_the_measured_width_of_the_text_before_it() {
        for text in AWKWARD {
            for columns in [12, 20, 40, 120] {
                let editor = Editor::with_text(text);
                let (line, column) = input_line(&editor, PROMPT, columns);
                assert_eq!(
                    column,
                    display_width(&line),
                    "{text:?} at {columns}: the cursor sits at the end of what is drawn"
                );
                assert!(
                    display_width(&line) <= columns.saturating_sub(1),
                    "{text:?} at {columns}: {} columns drawn",
                    display_width(&line)
                );
            }
        }
    }
}
