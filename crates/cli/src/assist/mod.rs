//! The opt-in background assistant: a small user process that keeps a private inbox of failed commands from repos
//! the user allowed. It decides with rules only, never loads a model and does nothing while idle.

mod args;
mod client;
mod context;
mod daemon;
mod inbox;
mod protocol;
mod redact;
mod store;
mod watch;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

pub use args::AssistAction;
use inbox::{Inbox, State};
use protocol::Request;
use store::{Store, socket_path};

use crate::CliError;
use crate::config::Home;
use crate::saved::now;

/// Like `println!`, but a closed pipe (`| head`) is an error the caller handles, not a panic.
macro_rules! out {
    ($($text:tt)*) => {
        writeln!(std::io::stdout(), $($text)*)?
    };
}

/// What the caller does once the action is done.
pub enum Next {
    Done,
    Exit(ExitCode),
    Chat {
        draft: String,
        repo: PathBuf,
        code: bool,
    },
}

fn say(text: impl AsRef<str>) -> Result<Next, CliError> {
    out!("{}", text.as_ref());
    Ok(Next::Done)
}

/// The git repository around `path`, as a canonical path.
fn repo_root(path: Option<&Path>) -> Result<String, CliError> {
    let start = std::fs::canonicalize(path.unwrap_or(Path::new(".")))?;
    let root = start.ancestors().find(|p| p.join(".git").exists());
    root.map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            CliError::BadOption(format!(
                "{} is not inside a git repository",
                start.display()
            ))
        })
}

/// Changes a suggestion through the running assistant, or directly in the inbox file when none is running.
fn mark(store: &Store, socket: &Path, id: u32, state: State) -> Result<(), CliError> {
    if let Ok(answer) = client::send(
        socket,
        &Request::Mark {
            id,
            state: state.clone(),
        },
    ) {
        return if answer.ok {
            Ok(())
        } else {
            Err(CliError::BadOption(answer.message))
        };
    }
    let mut inbox = store.inbox();
    if !inbox.mark(id, state) {
        return Err(CliError::BadOption(format!("no suggestion #{id}")));
    }
    Ok(store.save_inbox(&inbox)?)
}

fn start(socket: &Path, notify: bool) -> Result<Next, CliError> {
    if client::running(socket) {
        return say("the background assistant is already running");
    }
    let exe = std::env::current_exe()?;
    let mut daemon = Command::new("setsid");
    daemon.arg("-f").arg(&exe).args(["assist", "daemon"]);
    if notify {
        daemon.arg("--notify");
    }
    daemon
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if daemon.spawn().and_then(|mut c| c.wait()).is_err() {
        let mut plain = Command::new(exe);
        plain
            .args(["assist", "daemon"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if notify {
            plain.arg("--notify");
        }
        plain.spawn()?;
    }
    let began = Instant::now();
    while began.elapsed() < Duration::from_secs(5) {
        if client::running(socket) {
            return say(
                "background assistant started (rules only; no model is loaded). Allow a repo with `marin assist allow`",
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(CliError::BadOption(
        "it did not start; run `marin assist daemon` to see why".into(),
    ))
}

fn status(store: &Store, socket: &Path) -> Result<Next, CliError> {
    match client::send(socket, &Request::Status) {
        Ok(answer) => out!("{}", answer.message),
        Err(_) => out!("off: not running, nothing is watched"),
    }
    let repos = store.repos();
    if repos.is_empty() {
        out!("allowed repos: none (`marin assist allow` in a repo)");
    }
    for repo in repos {
        out!("allowed repo: {repo}");
    }
    Ok(Next::Done)
}

fn unit() -> Result<Next, CliError> {
    let exe = std::env::current_exe()?;
    say(format!(
        "[Unit]\nDescription=Marin background assistant (rules only, no model)\n\n[Service]\nExecStart={} assist daemon\nRestart=on-failure\nNice=10\nMemoryMax=64M\nNoNewPrivileges=yes\n\n[Install]\nWantedBy=default.target",
        exe.display()
    ))
}

fn listing(inbox: &Inbox) -> Result<Next, CliError> {
    let now = now();
    let waiting: Vec<_> = inbox.items.iter().filter(|s| s.visible(now)).collect();
    if waiting.is_empty() {
        return say("inbox empty");
    }
    for s in waiting {
        out!("#{}  {}\n    {}", s.id, s.headline(), s.first_error());
    }
    out!("\nmarin assist accept ID | why ID | dismiss ID | snooze ID");
    Ok(Next::Done)
}

fn find(inbox: &Inbox, id: u32) -> Result<&inbox::Suggestion, CliError> {
    inbox
        .find(id)
        .ok_or_else(|| CliError::BadOption(format!("no suggestion #{id}")))
}

pub fn run(action: AssistAction, home: Option<&Home>) -> Result<Next, CliError> {
    let home = home.ok_or_else(|| {
        CliError::BadOption("no home folder to keep the assistant's files in".into())
    })?;
    let store = Store::new(home.assist_dir());
    let socket = socket_path(&store)?;
    match action {
        AssistAction::Daemon { notify } => daemon::run(store, socket, notify).map(|()| Next::Done),
        AssistAction::Start { notify } => start(&socket, notify),
        AssistAction::Stop => match client::send(&socket, &Request::Stop) {
            Ok(_) => say("stopped"),
            Err(_) => say("it was not running"),
        },
        AssistAction::Status => status(&store, &socket),
        AssistAction::Unit => unit(),
        AssistAction::Allow { path } => {
            let root = repo_root(path.as_deref())?;
            let mut repos = store.repos();
            if !repos.contains(&root) {
                repos.push(root.clone());
            }
            store.save_repos(&repos)?;
            say(format!("watching failures in {root}"))
        }
        AssistAction::Deny { path } => {
            let root = repo_root(path.as_deref())?;
            let mut repos = store.repos();
            repos.retain(|r| *r != root);
            store.save_repos(&repos)?;
            say(format!("no longer watching {root}"))
        }
        AssistAction::Watch { command } => watch::run(&socket, &command).map(Next::Exit),
        AssistAction::Inbox => listing(&store.inbox()),
        AssistAction::Why { id } => {
            let inbox = store.inbox();
            let s = find(&inbox, id)?;
            for line in s.why() {
                out!("{line}");
            }
            say(format!(
                "\nthis is what accepting would put in the chat:\n\n{}",
                context::draft(s)
            ))
        }
        AssistAction::Accept { id, code } => {
            let inbox = store.inbox();
            let s = find(&inbox, id)?;
            let (draft, repo) = (context::draft(s), PathBuf::from(&s.event.repo));
            mark(&store, &socket, id, State::Accepted)?;
            Ok(Next::Chat { draft, repo, code })
        }
        AssistAction::Dismiss { id } => {
            mark(&store, &socket, id, State::Dismissed)?;
            say("dismissed")
        }
        AssistAction::Snooze { id, minutes } => {
            let until = now().saturating_add(u64::from(minutes).saturating_mul(60));
            mark(&store, &socket, id, State::Snoozed { until })?;
            say(format!("hidden for {minutes} minutes"))
        }
    }
}
