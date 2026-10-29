use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use super::activity::{Activity, Source};
use super::client;
use super::protocol::Request;
use super::store::Store;

pub type Observer = Arc<Mutex<Child>>;

/// Why windows cannot be observed here, or `None` when they can. Only X11 (and XWayland-free X sessions) expose the
/// focused window to ordinary programs; Wayland compositors deliberately do not, and Marin does not work around that.
pub fn unsupported() -> Option<&'static str> {
    if std::env::var("XDG_SESSION_TYPE").as_deref() != Ok("x11")
        || std::env::var_os("DISPLAY").is_none()
    {
        return Some(
            "this is not an X11 session (Wayland does not let programs watch other windows); use `marin assist share`",
        );
    }
    let found = Command::new("xprop")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok();
    (!found).then_some("`xprop` is not installed (package x11-utils)")
}

fn xprop(args: &[&str]) -> String {
    Command::new("xprop")
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// `window id # 0x4a00007` becomes `0x4a00007`; "0x0" (no focus) is nothing.
fn window_id(line: &str) -> Option<&str> {
    let id = line.rsplit('#').next()?.trim();
    (id.starts_with("0x") && id != "0x0").then_some(id)
}

/// The text inside the last quotes of `NAME(TYPE) = "a", "b"`.
fn last_quoted(output: &str, name: &str) -> String {
    output
        .lines()
        .find(|l| l.starts_with(name))
        .and_then(|l| l.split_once(" = "))
        .map(|(_, value)| value.split("\", \"").last().unwrap_or(value))
        .map(|v| v.trim().trim_matches('"').to_string())
        .unwrap_or_default()
}

fn describe(id: &str, store: &Store) -> Option<Activity> {
    let class = last_quoted(&xprop(&["-id", id, "WM_CLASS"]), "WM_CLASS").to_ascii_lowercase();
    if !store.awareness().app_allowed(&class) {
        return None;
    }
    let title = last_quoted(&xprop(&["-id", id, "_NET_WM_NAME"]), "_NET_WM_NAME");
    Some(Activity {
        source: Source::Window,
        app: class,
        title,
        url: String::new(),
        text: String::new(),
        at: 0,
    })
}

/// Experimental, unverified on a real X11 session. Follows the focused window through the X server's own property-change notifications: the child process blocks
/// until the focus changes, so nothing is polled. The title is read only for applications the user allowed.
pub fn watch(socket: PathBuf, dir: PathBuf) -> Option<Observer> {
    let mut child = Command::new("sh")
        .args([
            "-c",
            "xprop -spy -root _NET_ACTIVE_WINDOW & watcher=$!; read _; kill $watcher",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let lines = BufReader::new(child.stdout.take()?).lines();
    let store = Store::new(dir);
    std::thread::spawn(move || {
        for line in lines.map_while(Result::ok) {
            let activity = window_id(&line).and_then(|id| describe(id, &store));
            if let Some(activity) = activity {
                let _ = client::send(&socket, &Request::Activity(activity));
            }
        }
    });
    Some(Arc::new(Mutex::new(child)))
}

/// Closing the shell's input ends it and its `xprop`. The same happens if the daemon dies any way at all, because
/// the pipe closes with it, so no watcher outlives the daemon.
pub fn stop(observer: &Observer) {
    if let Ok(mut child) = observer.lock() {
        drop(child.stdin.take());
        let _ = child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xprop_output_is_read_as_the_class_and_the_title() {
        let class = "WM_CLASS(STRING) = \"navigator\", \"firefox\"\n";
        assert_eq!(last_quoted(class, "WM_CLASS"), "firefox");
        let title = "_NET_WM_NAME(UTF8_STRING) = \"cargo test - Terminal\"\n";
        assert_eq!(last_quoted(title, "_NET_WM_NAME"), "cargo test - Terminal");
        assert_eq!(last_quoted("garbage", "WM_CLASS"), "");
        assert_eq!(
            window_id("_NET_ACTIVE_WINDOW(WINDOW): window id # 0x4a00007"),
            Some("0x4a00007")
        );
        assert_eq!(
            window_id("_NET_ACTIVE_WINDOW(WINDOW): window id # 0x0"),
            None
        );
    }
}
