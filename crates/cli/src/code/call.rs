use serde_json::Value;

pub const OPEN: &str = "<tool_call>";
pub const CLOSE: &str = "</tool_call>";
pub const INVENTED: &str = "<tool_response>";
pub const TOOL_NAMES: &str = "list_files, read_file, search";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    List {
        path: String,
        depth: Option<usize>,
    },
    Read {
        path: String,
        start: Option<usize>,
        end: Option<usize>,
    },
    Search {
        pattern: String,
        path: String,
    },
}

pub struct Parsed {
    /// What the model said before its first call.
    pub shown: String,
    /// The reply as it goes into the history: nothing after the last complete call, nothing invented.
    pub kept: String,
    pub calls: Vec<Result<Call, String>>,
    /// The model wrote a tool result of its own; everything from there on was dropped.
    pub invented: bool,
    /// Calls that were read despite a small format slip.
    pub repaired: usize,
}

fn cut_after<'a>(text: &'a str, marker: &str) -> &'a str {
    text.rfind(marker)
        .and_then(|at| text.get(..at.saturating_add(marker.len())))
        .unwrap_or(text)
}

/// The longest malformed reply that is kept in the history, so a runaway cannot fill the window.
const KEEP_CHARS: usize = 600;

impl Call {
    /// The call written the way the model is told to write it.
    fn canonical(&self) -> String {
        let mut params: Vec<(&str, String)> = Vec::new();
        let name = match self {
            Self::List { path, depth } => {
                params.push(("path", path.clone()));
                params.extend(depth.map(|d| ("depth", d.to_string())));
                "list_files"
            }
            Self::Read { path, start, end } => {
                params.push(("path", path.clone()));
                params.extend(start.map(|n| ("start", n.to_string())));
                params.extend(end.map(|n| ("end", n.to_string())));
                "read_file"
            }
            Self::Search { pattern, path } => {
                params.push(("pattern", pattern.clone()));
                params.push(("path", path.clone()));
                "search"
            }
        };
        let body: String = params
            .iter()
            .map(|(k, v)| format!("<parameter={k}>\n{v}\n</parameter>\n"))
            .collect();
        format!("{OPEN}\n<function={name}>\n{body}</function>\n{CLOSE}")
    }
}

pub fn parse(reply: &str) -> Parsed {
    let (text, invented) = match reply.find(INVENTED) {
        Some(at) => (reply.get(..at).unwrap_or(reply), true),
        None => (reply, false),
    };
    let mut calls = Vec::new();
    let mut repaired = 0_usize;
    let mut rest = text;
    let mut complete = true;
    while let Some(start) = rest.find(OPEN) {
        let after = rest.get(start.saturating_add(OPEN.len())..).unwrap_or("");
        let Some(end) = after.find(CLOSE) else {
            // A call that never ends is still read if everything it needed was written before the rambling began.
            calls.push(match block(after) {
                Ok((call, _)) => {
                    repaired = repaired.saturating_add(1);
                    Ok(call)
                }
                Err(_) => Err("the tool call was never closed with </tool_call>".into()),
            });
            complete = false;
            break;
        };
        calls.push(block(after.get(..end).unwrap_or("")).map(|(call, slip)| {
            repaired = repaired.saturating_add(usize::from(slip));
            call
        }));
        rest = after.get(end.saturating_add(CLOSE.len())..).unwrap_or("");
    }
    let shown = text
        .find(OPEN)
        .and_then(|at| text.get(..at))
        .unwrap_or(text)
        .trim_end()
        .to_owned();
    let valid: Option<Vec<&Call>> = calls.iter().map(|c| c.as_ref().ok()).collect();
    let kept = match valid {
        _ if calls.is_empty() => text.trim_end().to_owned(),
        // What ran is what the history says, in the proper form, so a slip is not repeated from the context.
        Some(done) if repaired > 0 || !complete => {
            let rendered: Vec<String> = done.iter().map(|c| c.canonical()).collect();
            let lead = if shown.is_empty() {
                String::new()
            } else {
                format!("{shown}\n")
            };
            format!("{lead}{}", rendered.join("\n"))
        }
        Some(_) => cut_after(text, CLOSE).to_owned(),
        None => text.trim_end().chars().take(KEEP_CHARS).collect(),
    };
    Parsed {
        shown,
        kept,
        calls,
        invented,
        repaired,
    }
}

fn block(body: &str) -> Result<(Call, bool), String> {
    let body = body.trim();
    if body.starts_with('{') {
        json_call(body).map(|call| (call, false))
    } else {
        xml_call(body)
    }
}

fn json_call(body: &str) -> Result<Call, String> {
    let value: Value =
        serde_json::from_str(body).map_err(|e| format!("the JSON is invalid: {e}"))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or("the call has no \"name\"")?;
    let arguments = match value.get("arguments") {
        Some(Value::String(text)) => serde_json::from_str(text).unwrap_or(Value::Null),
        Some(other) => other.clone(),
        None => Value::Null,
    };
    let params = arguments
        .as_object()
        .map(|map| {
            map.iter()
                .map(|(k, v)| {
                    let text = v.as_str().map_or_else(|| v.to_string(), str::to_owned);
                    (k.clone(), text)
                })
                .collect()
        })
        .unwrap_or_default();
    build(name, params)
}

const PARAMETERS: [&str; 5] = ["path", "start", "end", "pattern", "depth"];

/// The model sometimes drops `parameter=` and writes `<path>value</parameter>`. That is unambiguous, so it is read,
/// but counted as a repair so the format failures stay visible.
fn bare_parameter(tag: &str) -> Option<(&'static str, &str)> {
    PARAMETERS.iter().find_map(|name| {
        tag.strip_prefix(&format!("<{name}>"))
            .map(|value| (*name, value))
    })
}

fn xml_call(body: &str) -> Result<(Call, bool), String> {
    let function = body
        .split_once("<function=")
        .ok_or("expected <function=NAME> inside the tool call")?
        .1;
    let (name, mut rest) = function
        .split_once('>')
        .ok_or("the <function=NAME> tag is not closed")?;
    let mut params = Vec::new();
    let mut repaired = false;
    while let Some(open) = rest.find('<') {
        let tag = rest.get(open..).unwrap_or("");
        let (key, value, after) = if let Some(after) = tag.strip_prefix("<parameter=") {
            let (key, after) = after
                .split_once('>')
                .ok_or("a <parameter=NAME> tag is not closed")?;
            let (value, after) = after
                .split_once("</parameter>")
                .ok_or_else(|| format!("parameter `{key}` is not closed with </parameter>"))?;
            (key.trim(), value, after)
        } else if let Some((key, after)) = bare_parameter(tag) {
            let end = [("</parameter>"), &format!("</{key}>")]
                .iter()
                .filter_map(|close| after.find(*close).map(|at| (at, close.len())))
                .min()
                .ok_or_else(|| format!("parameter `{key}` is not closed with </parameter>"))?;
            repaired = true;
            (
                key,
                after.get(..end.0).unwrap_or(""),
                after.get(end.0.saturating_add(end.1)..).unwrap_or(""),
            )
        } else {
            rest = tag.get(1..).unwrap_or("");
            continue;
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        params.push((key.to_owned(), value.to_owned()));
        rest = after;
    }
    build(name.trim(), params).map(|call| (call, repaired))
}

fn take(params: &mut Vec<(String, String)>, key: &str) -> Option<String> {
    let at = params.iter().position(|(k, _)| k == key)?;
    Some(params.swap_remove(at).1)
}

fn number(value: Option<String>, key: &str) -> Result<Option<usize>, String> {
    value
        .map(|text| {
            text.trim()
                .parse()
                .map_err(|_| format!("`{key}` must be a whole number, not {:?}", text.trim()))
        })
        .transpose()
}

fn build(name: &str, mut params: Vec<(String, String)>) -> Result<Call, String> {
    let found: Vec<String> = params.iter().map(|(k, _)| k.clone()).collect();
    let missing = |what: &str| {
        let seen = if found.is_empty() {
            "it had no <parameter=NAME> tags".to_owned()
        } else {
            format!("it had only: {}", found.join(", "))
        };
        format!(
            "{name} needs a `{what}` ({seen}); write each parameter as <parameter=NAME>, the value, then </parameter>"
        )
    };
    let call = match name {
        "list_files" => Call::List {
            path: take(&mut params, "path").unwrap_or_else(|| ".".into()),
            depth: number(take(&mut params, "depth"), "depth")?,
        },
        "read_file" => Call::Read {
            path: take(&mut params, "path").ok_or_else(|| missing("path"))?,
            start: number(take(&mut params, "start"), "start")?,
            end: number(take(&mut params, "end"), "end")?,
        },
        "search" => Call::Search {
            pattern: take(&mut params, "pattern").ok_or_else(|| missing("pattern"))?,
            path: take(&mut params, "path").unwrap_or_else(|| ".".into()),
        },
        other => {
            return Err(format!(
                "unknown tool `{other}`; the tools are {TOOL_NAMES}"
            ));
        }
    };
    match params.first() {
        Some((extra, _)) => Err(format!("`{name}` has no parameter `{extra}`")),
        None => Ok(call),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(reply: &str) -> Result<Call, String> {
        let mut parsed = parse(reply);
        assert_eq!(parsed.calls.len(), 1, "{reply}");
        parsed.calls.remove(0)
    }

    #[test]
    fn the_native_xml_form_parses_with_optional_numbers() {
        let call = one(
            "I'll read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a.rs\n</parameter>\n<parameter=start>\n3\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            call,
            Ok(Call::Read {
                path: "src/a.rs".into(),
                start: Some(3),
                end: None
            })
        );
    }

    #[test]
    fn the_json_form_parses_too() {
        let call = one(
            r#"<tool_call>{"name": "search", "arguments": {"pattern": "Gate", "path": "crates"}}</tool_call>"#,
        );
        assert_eq!(
            call,
            Ok(Call::Search {
                pattern: "Gate".into(),
                path: "crates".into()
            })
        );
    }

    #[test]
    fn defaults_apply_and_text_before_the_call_is_kept_apart() {
        let parsed =
            parse("Let me look.\n<tool_call>\n<function=list_files>\n</function>\n</tool_call>");
        assert_eq!(parsed.shown, "Let me look.");
        assert_eq!(
            parsed.calls,
            [Ok(Call::List {
                path: ".".into(),
                depth: None
            })]
        );
    }

    #[test]
    fn malformed_calls_are_errors_never_guesses() {
        for bad in [
            "<tool_call>\n<function=delete_file>\n</function>\n</tool_call>",
            "<tool_call>\n<function=read_file>\n</function>\n</tool_call>",
            "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=start>\nsoon\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=list_files>\n<parameter=colour>\nred\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>{not json}</tool_call>",
            "<tool_call>just words</tool_call>",
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.rs",
        ] {
            assert!(one(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_result_the_model_wrote_itself_is_dropped() {
        let parsed = parse(
            "<tool_call>\n<function=list_files>\n</function>\n</tool_call>\n<tool_response>\nsecret.txt\n</tool_response>\nDone.",
        );
        assert!(parsed.invented);
        assert_eq!(parsed.calls.len(), 1);
        assert!(!parsed.kept.contains("secret.txt"));
        assert!(parsed.kept.ends_with("</tool_call>"));
    }

    #[test]
    fn a_dropped_parameter_prefix_is_read_and_counted() {
        let parsed = parse(
            "<tool_call>\n<function=read_file>\n<path>src/a.rs\n</parameter>\n<parameter=start>\n2\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(parsed.repaired, 1);
        assert_eq!(
            parsed.calls,
            [Ok(Call::Read {
                path: "src/a.rs".into(),
                start: Some(2),
                end: None
            })]
        );
    }

    #[test]
    fn a_call_that_rambles_on_is_read_and_kept_in_proper_form() {
        let reply = format!(
            "<tool_call>\n<function=read_file>\n<path>\nlocked.txt\n</parameter>\n{}",
            "</parameter>\n".repeat(60)
        );
        let parsed = parse(&reply);
        assert_eq!(parsed.repaired, 1);
        assert_eq!(
            parsed.calls,
            [Ok(Call::Read {
                path: "locked.txt".into(),
                start: None,
                end: None
            })]
        );
        assert_eq!(
            parsed.kept,
            "<tool_call>\n<function=read_file>\n<parameter=path>\nlocked.txt\n</parameter>\n</function>\n</tool_call>"
        );
    }

    #[test]
    fn a_broken_call_is_kept_short_and_a_long_answer_is_kept_whole() {
        let broken = parse(&format!(
            "<tool_call>\n<function=nothing>\n{}",
            "x".repeat(5000)
        ));
        assert!(broken.calls[0].is_err() && broken.kept.chars().count() == 600);
        let long = "word ".repeat(2000);
        assert_eq!(parse(&long).kept, long.trim_end());
    }

    #[test]
    fn plain_answers_have_no_calls() {
        let parsed = parse("The answer is 4.");
        assert!(parsed.calls.is_empty());
        assert_eq!(parsed.kept, "The answer is 4.");
    }
}
