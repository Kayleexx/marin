use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use super::client;
use super::inbox::{Event, Inbox, Offer, TAIL_BYTES, bound};
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
        let headline;
        let outcome = match self.inbox.offer(event, now()) {
            Offer::Surfaced(id) => {
                headline = self.inbox.find(id).map(|s| s.headline());
                if self.notify
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

    fn status(&self) -> Response {
        let waiting = self.inbox.items.iter().filter(|i| i.visible(now())).count();
        Self::reply(
            true,
            format!(
                "running (pid {}) for {} s · {} events, {} kept quiet · {waiting} waiting · notifications {} · no model loaded",
                std::process::id(),
                now().saturating_sub(self.started),
                self.events,
                self.quiet,
                if self.notify { "on" } else { "off" }
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
    if client::running(&socket) {
        return Err(CliError::BadOption(
            "the background assistant is already running".into(),
        ));
    }
    let _ = fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let mut daemon = Daemon {
        inbox: store.inbox(),
        store,
        notify,
        started: now(),
        events: 0,
        quiet: 0,
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
    let _ = fs::remove_file(&socket);
    Ok(())
}
