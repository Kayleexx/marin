use std::path::{Path, PathBuf};
use std::process::Command;

use super::activity::{Activity, Source, host_and_path};
use super::config::Awareness;
use super::inbox::{Kind, State};
use super::protocol::Request;
use super::store::Store;
use super::{browser, client, context, desktop, mark};
use crate::CliError;
use crate::saved::now;

/// What a control prints and, for `accept`, the draft that goes into the chat and the folder it belongs to.
#[derive(Default)]
pub struct Reply {
    pub lines: Vec<String>,
    pub draft: Option<(String, PathBuf)>,
}

fn lines(text: impl IntoIterator<Item = String>) -> Reply {
    Reply {
        lines: text.into_iter().collect(),
        draft: None,
    }
}

fn bad(text: impl Into<String>) -> CliError {
    CliError::BadOption(text.into())
}

fn restart(socket: &Path) -> Result<String, CliError> {
    if !client::running(socket) {
        return Ok("it applies when the helper starts (`/awareness enable`)".into());
    }
    let _ = client::send(socket, &Request::Stop);
    for _ in 0..40 {
        if !client::running(socket) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    super::launch(socket, false).map(|_| "the helper restarted with the change".into())
}

fn status(store: &Store, socket: &Path, config: &Awareness) -> Vec<String> {
    let mut out = config.summary();
    out.push(match client::send(socket, &Request::Status) {
        Ok(answer) => answer.message,
        Err(_) => "helper: not running, nothing is observed".into(),
    });
    out.push(match desktop::unsupported() {
        Some(why) => format!("windows: unavailable here: {why}"),
        None => "windows: available (X11)".into(),
    });
    let repos = store.repos();
    out.push(format!(
        "allowed repos: {}",
        if repos.is_empty() {
            "none".into()
        } else {
            repos.join(", ")
        }
    ));
    out
}

fn autostart(on: bool) -> Result<String, CliError> {
    let home = std::env::var_os("HOME").ok_or_else(|| bad("HOME is not set"))?;
    let dir = PathBuf::from(home).join(".config/systemd/user");
    let file = dir.join("marin-assist.service");
    let systemctl = |args: &[&str]| {
        Command::new("systemctl")
            .arg("--user")
            .args(args)
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if !on {
        let off = systemctl(&["disable", "--now", "marin-assist.service"]);
        let _ = std::fs::remove_file(&file);
        return Ok(format!(
            "login start removed (systemctl: {})",
            if off { "ok" } else { "nothing to disable" }
        ));
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(&file, super::unit_text()?)?;
    Ok(if systemctl(&["enable", "--now", "marin-assist.service"]) {
        format!(
            "starts at login now ({}); `/awareness autostart off` removes it",
            file.display()
        )
    } else {
        format!(
            "wrote {}, but `systemctl --user enable` failed; run it yourself",
            file.display()
        )
    })
}

/// `/awareness`: status, the master switch, sources, allowlists, notifications and login start.
pub fn awareness(store: &Store, socket: &Path, words: &str) -> Result<Reply, CliError> {
    let mut config = store.awareness();
    let mut parts = words.split_whitespace();
    let (what, value) = (parts.next().unwrap_or("status"), parts.next().unwrap_or(""));
    let note = match (what, value) {
        ("status" | "sources", _) => return Ok(lines(status(store, socket, &config))),
        ("autostart", "on" | "off") => return Ok(lines([autostart(value == "on")?])),
        ("enable", _) => {
            config.enabled = true;
            store.save_awareness(&config)?;
            let started = if client::running(socket) {
                restart(socket)?
            } else {
                super::launch(socket, false)?
            };
            return Ok(lines([
                started,
                "awareness is on, but nothing is watched until you approve a source:".into(),
                "  /awareness desktop on + allow-app NAME   (X11 windows)".into(),
                "  /awareness browser on + allow-site HOST  (tab titles; needs `marin assist browser install`)".into(),
                "  marin assist watch -- CMD                 (failed commands in `marin assist allow` repos)".into(),
                "no model is loaded for any of this; `/awareness disable` turns it all off".into(),
            ]));
        }
        ("disable", _) => {
            config.enabled = false;
            store.save_awareness(&config)?;
            let _ = client::send(socket, &Request::Stop);
            return Ok(lines([
                "awareness is off and the helper has stopped; nothing is observed".into(),
            ]));
        }
        _ => config.change(what, value).map_err(bad).map(|()| what)?,
    };
    store.save_awareness(&config)?;
    let applied = restart(socket)?;
    let mut out = vec![format!("{note} {value}: saved. {applied}")];
    if what == "desktop"
        && value == "on"
        && let Some(why) = desktop::unsupported()
    {
        out.push(format!("note: windows are not observable here: {why}"));
    }
    Ok(lines(out))
}

/// `/context`: what the helper currently retains. It lives in the helper's memory only.
pub fn context_view(socket: &Path) -> Reply {
    match client::send(socket, &Request::Context) {
        Ok(answer) => lines(answer.message.lines().map(String::from)),
        Err(_) => lines(["nothing retained: the helper is not running".into()]),
    }
}

fn waiting(store: &Store) -> Vec<String> {
    let inbox = store.inbox();
    let rows: Vec<String> = inbox
        .items
        .iter()
        .filter(|s| s.visible(now()))
        .map(|s| format!("#{}  {}\n    {}", s.id, s.headline(), s.first_error()))
        .collect();
    if rows.is_empty() {
        return vec!["no suggestions waiting".into()];
    }
    let mut out = rows;
    out.push("/suggestions accept N · dismiss N · mute N · snooze N · /why N".into());
    out
}

fn id_of(value: &str) -> Result<u32, CliError> {
    value
        .parse()
        .map_err(|_| bad("give the suggestion number, for example `accept 1`"))
}

/// `/suggestions`: list, accept (a draft for the chat), dismiss, mute or snooze.
pub fn suggestions(store: &Store, socket: &Path, words: &str) -> Result<Reply, CliError> {
    let mut parts = words.split_whitespace();
    let (verb, value) = (parts.next().unwrap_or("list"), parts.next().unwrap_or(""));
    match verb {
        "list" => Ok(lines(waiting(store))),
        "accept" => {
            let id = id_of(value)?;
            let inbox = store.inbox();
            let found = inbox
                .find(id)
                .ok_or_else(|| bad(format!("no suggestion #{id}")))?;
            let draft = (context::draft(found), PathBuf::from(&found.event.repo));
            mark(store, socket, id, State::Accepted)?;
            Ok(Reply {
                lines: vec![
                    "the context is in your input line; edit it and press Enter to send".into(),
                ],
                draft: Some(draft),
            })
        }
        "dismiss" => {
            mark(store, socket, id_of(value)?, State::Dismissed)?;
            Ok(lines([
                "dismissed; the same thing stays quiet for a day".into()
            ]))
        }
        "snooze" => {
            let until = now().saturating_add(3600);
            mark(store, socket, id_of(value)?, State::Snoozed { until })?;
            Ok(lines(["hidden for an hour".into()]))
        }
        "mute" => {
            let id = id_of(value)?;
            let inbox = store.inbox();
            let found = inbox
                .find(id)
                .ok_or_else(|| bad(format!("no suggestion #{id}")))?;
            let topic = match (found.kind, &found.page) {
                (Kind::Shared, Some(p)) => p.host().to_string(),
                _ => found.event.repo.clone(),
            };
            let mut config = store.awareness();
            config.change("mute", &topic).map_err(bad)?;
            store.save_awareness(&config)?;
            mark(store, socket, id, State::Dismissed)?;
            Ok(lines([format!(
                "muted {topic}; `/suggestions unmute {topic}` brings it back"
            )]))
        }
        "unmute" => {
            let mut config = store.awareness();
            config.change("unmute", value).map_err(bad)?;
            store.save_awareness(&config)?;
            Ok(lines([format!("unmuted {value}")]))
        }
        other => Err(bad(format!(
            "unknown `{other}`: list, accept, dismiss, mute, snooze or unmute"
        ))),
    }
}

/// `/why`: the rule behind a suggestion (the newest waiting one by default) and exactly what accepting would send.
pub fn why(store: &Store, words: &str) -> Result<Reply, CliError> {
    let inbox = store.inbox();
    let found = match words.split_whitespace().next() {
        Some(id) => inbox.find(id_of(id)?),
        None => inbox.items.iter().rev().find(|s| s.visible(now())),
    };
    let found = found.ok_or_else(|| bad("no such suggestion; `/suggestions` lists them"))?;
    let mut out = found.why();
    out.push("accepting would put this in your input line:".into());
    out.extend(context::draft(found).lines().map(|l| format!("  {l}")));
    Ok(lines(out))
}

/// A page or window shared by hand: the explicit, always-allowed source.
pub fn share(socket: &Path, title: &str, url: &str, text: &str) -> Result<Reply, CliError> {
    if !url.is_empty() && host_and_path(url).is_none() {
        return Err(bad("the address must start with http:// or https://"));
    }
    let activity = Activity {
        source: Source::Share,
        app: String::new(),
        title: title.into(),
        url: url.into(),
        text: text.into(),
        at: 0,
    };
    let answer = client::send(socket, &Request::Activity(activity))
        .map_err(|_| bad("the helper is not running; `marin assist start` first"))?;
    Ok(lines([answer.message]))
}

pub fn browser_action(store: &Store, what: &str) -> Result<Reply, CliError> {
    match what {
        "install" => Ok(lines(browser::install(store)?)),
        "remove" => Ok(lines(browser::remove(store)?)),
        other => Err(bad(format!("unknown `{other}`: install or remove"))),
    }
}
