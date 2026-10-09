/// The inspect panel's tabs, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Reply,
    Tokens,
    Next,
    Kv,
    Model,
    Settings,
}

impl Tab {
    pub const ALL: [Tab; 6] = [
        Tab::Reply,
        Tab::Tokens,
        Tab::Next,
        Tab::Kv,
        Tab::Model,
        Tab::Settings,
    ];

    fn title(self) -> &'static str {
        match self {
            Tab::Reply => "Reply",
            Tab::Tokens => "Tokens",
            Tab::Next => "Next tokens",
            Tab::Kv => "KV",
            Tab::Model => "Model",
            Tab::Settings => "Settings",
        }
    }

    fn position(self) -> usize {
        Self::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }

    pub fn step(self, forward: bool) -> Tab {
        let count = Self::ALL.len();
        let at = self.position();
        let next = if forward {
            at.saturating_add(1).checked_rem(count)
        } else {
            at.checked_add(count)
                .and_then(|n| n.checked_sub(1))
                .and_then(|n| n.checked_rem(count))
        };
        Self::ALL
            .get(next.unwrap_or(0))
            .copied()
            .unwrap_or(Tab::Reply)
    }
}

/// The last reply as `telemetry::request_lines` reports it, plus why it stopped.
pub struct ReplyStats {
    pub lines: Vec<String>,
    pub stop: &'static str,
    pub seed: Option<u64>,
}

pub struct KvStats {
    pub used: usize,
    pub capacity: usize,
    pub bytes: u64,
    pub format: &'static str,
    pub backend: String,
}

pub struct PanelData {
    pub reply: Option<ReplyStats>,
    /// Pre-formatted `id  "text"` lines of the last prompt.
    pub tokens: Vec<String>,
    /// Pre-formatted likeliest next tokens; None until computed.
    pub next: Option<Vec<String>>,
    pub kv: KvStats,
    pub model: Vec<String>,
    /// Effective generation settings, one `name  value` line each.
    pub settings: Vec<String>,
}

const MAX_TOKEN_ROWS: usize = 8;
const NOTHING_YET: &str = "nothing yet: send a prompt first";

/// Tab strip with the active tab in brackets, so it reads without color.
pub fn header(active: Tab) -> String {
    let tabs: Vec<String> = Tab::ALL
        .iter()
        .map(|t| {
            if *t == active {
                format!("[{}]", t.title())
            } else {
                t.title().to_string()
            }
        })
        .collect();
    format!("{}   ←/→ switch · esc close", tabs.join("  "))
}

fn bytes(count: u64) -> String {
    crate::units::bytes(count)
}

pub fn body(tab: Tab, data: &PanelData) -> Vec<String> {
    match tab {
        Tab::Reply => match &data.reply {
            None => vec![NOTHING_YET.to_string()],
            Some(r) => {
                let mut lines = r.lines.clone();
                lines.push(format!("stopped         {}", r.stop));
                if let Some(seed) = r.seed {
                    lines.push(format!("seed            {seed}"));
                }
                lines
            }
        },
        Tab::Tokens => {
            if data.tokens.is_empty() {
                return vec![NOTHING_YET.to_string()];
            }
            let mut lines: Vec<String> = data.tokens.iter().take(MAX_TOKEN_ROWS).cloned().collect();
            let hidden = data.tokens.len().saturating_sub(MAX_TOKEN_ROWS);
            if hidden > 0 {
                lines.push(format!("… and {hidden} more"));
            }
            lines
        }
        Tab::Next => match &data.next {
            Some(lines) => lines.clone(),
            None if data.tokens.is_empty() => vec![NOTHING_YET.to_string()],
            None => vec!["computing…".to_string()],
        },
        Tab::Kv => {
            let percent = data
                .kv
                .used
                .saturating_mul(100)
                .checked_div(data.kv.capacity)
                .unwrap_or(0);
            vec![
                format!(
                    "positions     {} of {} ({percent}%)",
                    data.kv.used, data.kv.capacity
                ),
                format!(
                    "reserved      {} ({} values) on {}",
                    bytes(data.kv.bytes),
                    data.kv.format,
                    data.kv.backend
                ),
                "the KV cache remembers earlier tokens so they are not recomputed".to_string(),
            ]
        }
        Tab::Model => data.model.clone(),
        Tab::Settings => data.settings.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> PanelData {
        PanelData {
            reply: Some(ReplyStats {
                lines: vec!["generated       24 tokens".to_string()],
                stop: "max tokens",
                seed: Some(7),
            }),
            tokens: (0..12).map(|i| format!("{i}  \"t\"")).collect(),
            next: None,
            kv: KvStats {
                used: 29,
                capacity: 2048,
                bytes: 470_000_000,
                format: "f16",
                backend: "cuda".into(),
            },
            model: vec!["layers 28".into()],
            settings: vec!["temperature  0.7".into()],
        }
    }

    #[test]
    fn tabs_cycle_in_both_directions() {
        assert_eq!(Tab::Reply.step(true), Tab::Tokens);
        assert_eq!(Tab::Settings.step(true), Tab::Reply);
        assert_eq!(Tab::Reply.step(false), Tab::Settings);
        assert_eq!(Tab::Model.step(true), Tab::Settings);
        assert_eq!(Tab::Next.step(false), Tab::Tokens);
    }

    #[test]
    fn header_brackets_the_active_tab() {
        let text = header(Tab::Kv);
        assert!(text.contains("[KV]") && text.contains("Reply") && !text.contains("[Reply]"));
    }

    #[test]
    fn reply_tab_shows_the_reply_report_with_stop_reason_and_seed() {
        let lines = body(Tab::Reply, &data());
        assert!(
            lines
                .iter()
                .any(|l| l.contains("generated       24 tokens"))
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("stopped         max tokens"))
        );
        assert!(lines.iter().any(|l| l.contains("seed            7")));
    }

    #[test]
    fn empty_states_say_what_to_do() {
        let mut d = data();
        d.reply = None;
        d.tokens.clear();
        assert_eq!(body(Tab::Reply, &d), [NOTHING_YET]);
        assert_eq!(body(Tab::Tokens, &d), [NOTHING_YET]);
        assert_eq!(body(Tab::Next, &d), [NOTHING_YET]);
    }

    #[test]
    fn long_token_lists_are_truncated_and_next_tokens_wait_for_compute() {
        let d = data();
        let lines = body(Tab::Tokens, &d);
        assert_eq!(lines.len(), MAX_TOKEN_ROWS + 1);
        assert_eq!(lines.last().map(String::as_str), Some("… and 4 more"));
        assert_eq!(body(Tab::Next, &d), ["computing…"]);
    }

    #[test]
    fn kv_tab_reports_usage_percentage() {
        let lines = body(Tab::Kv, &data());
        assert!(lines.iter().any(|l| l.contains("29 of 2048 (1%)")));
    }
}
