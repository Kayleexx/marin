use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use super::activity::{Activity, Context, Source, related};
use super::client;
use super::desktop;
use super::inbox::{Event, Inbox, Kind, Offer, TAIL_BYTES, bound};
use super::protocol::{Request, Response};
use super::redact::redact;
use super::store::Store;
use crate::CliError;
use crate::saved::now;

const MAX_REQUEST: u64 = 128 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(2);

struct Daemon {
    store: Store,
    inbox: Inbox,
    notify: bool,
    started: u64,
    events: u32,
    quiet: u32,
    context: Context,
    watching: &'static str,
}

/// The repo root an event belongs to, if the user allowed it: the event's folder must sit inside an allowed repo.
fn allowed_root(store: &Store, folder: &str) -> Option<String> {
    let folder = fs::canonicalize(folder).ok()?;
    store
        .repos()
        .into_iter()
        .find(|repo| folder.starts_with(Path::new(repo)))
}

fn tell(text: &str) {
    if let Ok(mut child) = Command::new("notify-send")
        .args(["--app-name=marin", "marin", text])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        std::thread::spawn(move || child.wait());
    }
}

impl Daemon {
    fn reply(ok: bool, message: impl Into<String>) -> Response {
        Response {
            ok,
            message: message.into(),
        }
    }

    fn event(&mut self, mut event: Event) -> Response {
        self.events = self.events.saturating_add(1);
        let Some(root) = allowed_root(&self.store, &event.repo) else {
            self.quiet = self.quiet.saturating_add(1);
            return Self::reply(true, "ignored: not in an allowed repo");
        };
        event.repo = root;
        event.command = redact(&event.command).chars().take(200).collect();
        event.tail = bound(&redact(&event.tail), TAIL_BYTES);
        let offer = self.inbox.offer(event, now());
        self.finish(offer)
    }

    /// Counts the outcome, tells the user about a new suggestion and saves the inbox.
    fn finish(&mut self, offer: Offer) -> Response {
        let outcome = match offer {
            Offer::Surfaced(id) => {
                let headline = self.inbox.find(id).map(|s| s.headline());
                if (self.notify || self.store.awareness().notify)
                    && let Some(text) = &headline
                {
                    tell(&format!("{text}. Run `marin assist inbox`."));
                }
                format!("suggestion #{id} added")
            }
            Offer::Repeated(id) => format!("counted again on #{id}"),
            Offer::Quiet(why) => {
                self.quiet = self.quiet.saturating_add(1);
                format!("kept quiet: {why}")
            }
        };
        match self.store.save_inbox(&self.inbox) {
            Ok(()) => Self::reply(true, outcome),
            Err(e) => Self::reply(false, format!("could not save the inbox: {e}")),
        }
    }

    /// A window, tab or shared page. Passive sources need awareness on, their source approved and the app or site
    /// allowed; a share is an explicit act. Activity only becomes a suggestion through the rules below.
    fn activity(&mut self, activity: Activity) -> Response {
        let (now, config) = (now(), self.store.awareness());
        let Some(activity) = activity.cleaned(now) else {
            return Self::reply(true, "ignored: nothing usable");
        };
        let allowed = match activity.source {
            Source::Window => config.app_allowed(&activity.app),
            Source::Tab => config.site_allowed(activity.host()),
            Source::Share => true,
        };
        if !allowed {
            return Self::reply(true, "ignored: not approved");
        }
        self.events = self.events.saturating_add(1);
        self.context.push(activity.clone());
        if activity.source == Source::Share {
            if config.is_muted(activity.host()) {
                return Self::reply(true, "kept quiet: you muted this site");
            }
            let share = Event {
                repo: String::new(),
                command: "share".into(),
                exit: 0,
                tail: String::new(),
            };
            let offer = self.inbox.offer_page(Kind::Shared, share, activity, now);
            return self.finish(offer);
        }
        let failure = self.store.repos().into_iter().find_map(|repo| {
            let failure = self.inbox.recent_failure(&repo, now)?;
            (!config.is_muted(&repo) && related(&failure.first_error(), &activity))
                .then(|| failure.event.clone())
        });
        let Some(failure) = failure else {
            return Self::reply(true, "noted; nothing to suggest");
        };
        let offer = self.inbox.offer_page(Kind::Docs, failure, activity, now);
        self.finish(offer)
    }

    fn context(&mut self) -> Response {
        self.context.expire(now());
        let lines: Vec<String> = self.context.items().iter().map(|a| a.line(now())).collect();
        let head = format!(
            "{} item(s) kept in memory only (at most {}, {} min); nothing is written to disk",
            lines.len(),
            super::activity::MAX_CONTEXT,
            super::activity::CONTEXT_TTL / 60
        );
        Self::reply(
            true,
            std::iter::once(head)
                .chain(lines)
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn status(&self) -> Response {
        let waiting = self.inbox.items.iter().filter(|i| i.visible(now())).count();
        Self::reply(
            true,
            format!(
                "running (pid {}) for {} s · {} events, {} kept quiet · {waiting} waiting · notifications {} · windows: {} · no model loaded",
                std::process::id(),
                now().saturating_sub(self.started),
                self.events,
                self.quiet,
                if self.notify { "on" } else { "off" },
                self.watching
            ),
        )
    }

    fn handle(&mut self, stream: &mut UnixStream) -> io::Result<bool> {
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;
        let mut line = String::new();
        BufReader::new(Read::take(&*stream, MAX_REQUEST)).read_line(&mut line)?;
        let mut stop = false;
        let response = match serde_json::from_str(&line) {
            Err(e) => Self::reply(false, format!("not understood: {e}")),
            Ok(Request::Event(event)) => self.event(event),
            Ok(Request::Activity(activity)) => self.activity(activity),
            Ok(Request::Context) => self.context(),
            Ok(Request::Status) => self.status(),
            Ok(Request::Stop) => {
                stop = true;
                Self::reply(true, "stopping")
            }
            Ok(Request::Mark { id, state }) => {
                let found = self.inbox.mark(id, state);
                match found.then(|| self.store.save_inbox(&self.inbox)) {
                    Some(Ok(())) => Self::reply(true, "done"),
                    Some(Err(e)) => Self::reply(false, e.to_string()),
                    None => Self::reply(false, format!("no suggestion #{id}")),
                }
            }
        };
        let mut text = serde_json::to_vec(&response).map_err(io::Error::other)?;
        text.push(b'\n');
        stream.write_all(&text)?;
        Ok(stop)
    }
}

/// Serves requests one at a time until told to stop. It blocks in `accept`, so it uses no CPU while idle, and it
/// never loads a model.
pub fn run(store: Store, socket: PathBuf, notify: bool) -> Result<(), CliError> {
    let already = || CliError::BadOption("the background assistant is already running".into());
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(socket.with_file_name("assist.lock"))?;
    lock.try_lock().map_err(|_| already())?;
    if client::running(&socket) {
        return Err(already());
    }
    let _ = fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let config = store.awareness();
    let (observer, watching) = match (config.enabled && config.desktop, desktop::unsupported()) {
        (false, _) => (None, "off"),
        (true, Some(_)) => (None, "unavailable here"),
        (true, None) => match desktop::watch(socket.clone(), store.dir().to_path_buf()) {
            Some(child) => (Some(child), "on"),
            None => (None, "unavailable here"),
        },
    };
    let cleanup = (socket.clone(), observer.clone());
    ctrlc::set_handler(move || {
        let _ = fs::remove_file(&cleanup.0);
        if let Some(child) = &cleanup.1 {
            desktop::stop(child);
        }
        std::process::exit(0);
    })
    .map_err(|e| CliError::BadOption(format!("could not install the shutdown handler: {e}")))?;
    let mut daemon = Daemon {
        inbox: store.inbox(),
        store,
        notify,
        started: now(),
        events: 0,
        quiet: 0,
        context: Context::default(),
        watching,
    };
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if daemon.handle(&mut stream).unwrap_or(false) {
                    break;
                }
            }
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    if let Some(child) = &observer {
        desktop::stop(child);
    }
    let _ = fs::remove_file(&socket);
    drop(lock);
    Ok(())
}
