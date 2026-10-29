//! Awareness through the real binary: nothing is observed until approved, tabs and shares become suggestions only
//! by rule, and muting, dismissing and disabling keep it quiet.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn marin(home: &Path, cwd: &Path, args: &[&str], stdin: &[u8]) -> Output {
    let spawned = Command::new(env!("CARGO_BIN_EXE_marin"))
        .env("MARIN_HOME", home)
        .env("NO_COLOR", "1")
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = spawned else {
        return Output {
            status: Default::default(),
            stdout: Vec::new(),
            stderr: b"could not start marin".to_vec(),
        };
    };
    if let Some(mut pipe) = child.stdin.take() {
        let _ = pipe.write_all(stdin);
    }
    child.wait_with_output().unwrap_or(Output {
        status: Default::default(),
        stdout: Vec::new(),
        stderr: Vec::new(),
    })
}

fn run(home: &Path, repo: &Path, args: &[&str]) -> String {
    let out = marin(home, repo, args, b"");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// What Firefox sends: a length (native byte order) and JSON. Returns the host's JSON answer.
fn tab(home: &Path, repo: &Path, url: &str, title: &str) -> String {
    let body = serde_json::json!({ "url": url, "title": title }).to_string();
    let mut frame = u32::try_from(body.len())
        .unwrap_or(0)
        .to_ne_bytes()
        .to_vec();
    frame.extend(body.as_bytes());
    let out = marin(home, repo, &["assist", "browser", "host"], &frame);
    String::from_utf8_lossy(out.stdout.get(4..).unwrap_or_default()).into_owned()
}

fn scratch(name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("marin-aware-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(dir.join("repo/.git"));
    (dir.join("home"), dir.join("repo"), dir)
}

const FAIL: &[&str] = &[
    "assist",
    "watch",
    "--",
    "sh",
    "-c",
    "echo 'error[E0382]: borrow of moved value: config' >&2; exit 101",
];
const DOCS: &str = "https://doc.rust-lang.org/error_codes/E0382.html?secret=1#x";

fn approve(home: &Path, repo: &Path) {
    run(home, repo, &["assist", "allow"]);
    run(home, repo, &["assist", "awareness", "enable"]);
    run(home, repo, &["assist", "awareness", "browser", "on"]);
    run(
        home,
        repo,
        &["assist", "awareness", "allow-site", "doc.rust-lang.org"],
    );
}

#[test]
fn nothing_is_observed_until_it_is_approved_and_disabling_stops_it_all() {
    let (home, repo, dir) = scratch("off");
    assert!(run(&home, &repo, &["assist", "context"]).contains("not running"));
    assert!(run(&home, &repo, &["assist", "awareness"]).contains("awareness: off"));
    run(&home, &repo, &["assist", "allow"]);
    run(&home, &repo, &["assist", "start"]);
    run(&home, &repo, FAIL);
    let ignored = tab(&home, &repo, DOCS, "E0382");
    assert!(ignored.contains("site not allowed"), "{ignored}");
    run(&home, &repo, &["assist", "awareness", "enable"]);
    assert!(
        tab(&home, &repo, DOCS, "E0382").contains("site not allowed"),
        "browser still off"
    );
    run(&home, &repo, &["assist", "awareness", "browser", "on"]);
    assert!(
        tab(&home, &repo, DOCS, "E0382").contains("site not allowed"),
        "site not allowed"
    );
    run(
        &home,
        &repo,
        &["assist", "awareness", "allow-site", "doc.rust-lang.org"],
    );
    assert!(tab(&home, &repo, DOCS, "E0382").contains("suggestion #2"));

    let off = run(&home, &repo, &["assist", "awareness", "disable"]);
    assert!(off.contains("nothing is observed"), "{off}");
    assert!(
        tab(&home, &repo, DOCS, "E0382").contains("ignored"),
        "disabled: dropped before the helper"
    );
    assert!(run(&home, &repo, &["assist", "context"]).contains("not running"));
    assert!(!home.join("assist/assist.sock").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn docs_opened_after_a_failure_become_one_explained_suggestion_that_accepts_into_a_draft() {
    let (home, repo, dir) = scratch("docs");
    approve(&home, &repo);
    run(&home, &repo, &["assist", "start"]);
    assert_eq!(marin(&home, &repo, FAIL, b"").status.code(), Some(101));

    let unrelated = tab(
        &home,
        &repo,
        "https://doc.rust-lang.org/book/ch04-01.html",
        "Ownership",
    );
    assert!(unrelated.contains("nothing to suggest"), "{unrelated}");
    assert!(tab(&home, &repo, "https://evil.example/E0382", "E0382").contains("site not allowed"));
    assert!(tab(&home, &repo, DOCS, "Error code E0382").contains("suggestion #2"));
    assert!(tab(&home, &repo, DOCS, "Error code E0382").contains("counted again"));

    let context = run(&home, &repo, &["assist", "context"]);
    assert!(
        context.contains("doc.rust-lang.org/error_codes/E0382.html"),
        "{context}"
    );
    assert!(
        !context.contains("secret=1") && !context.contains("evil.example"),
        "{context}"
    );
    let why = run(&home, &repo, &["assist", "why", "2"]);
    assert!(
        why.contains("within 15 minutes") && why.contains("data left the device: no"),
        "{why}"
    );

    let accepted = run(&home, &repo, &["assist", "suggestions", "accept", "2"]);
    assert!(
        accepted.contains("Afterwards I opened this page") && accepted.contains("E0382"),
        "{accepted}"
    );
    let stored = fs::read_to_string(home.join("assist/inbox.json")).unwrap_or_default();
    assert!(!stored.contains("secret=1"), "queries are never kept");
    run(&home, &repo, &["assist", "stop"]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_share_is_filtered_and_muting_it_keeps_the_site_quiet() {
    let (home, repo, dir) = scratch("share");
    run(&home, &repo, &["assist", "start"]);
    let out = marin(
        &home,
        &repo,
        &[
            "assist",
            "share",
            "Deploy notes",
            "--url",
            "https://wiki.example/deploy?x=1",
        ],
        b"token=abcdef123456\nrollback steps",
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("suggestion #1"));
    let draft = run(&home, &repo, &["assist", "why", "1"]);
    assert!(
        draft.contains("rollback steps") && !draft.contains("abcdef123456"),
        "{draft}"
    );
    assert!(
        run(&home, &repo, &["assist", "suggestions", "mute", "1"]).contains("muted wiki.example")
    );
    let again = marin(
        &home,
        &repo,
        &[
            "assist",
            "share",
            "Deploy notes 2",
            "--url",
            "https://wiki.example/other",
        ],
        b"more",
    );
    assert!(String::from_utf8_lossy(&again.stdout).contains("you muted this site"));
    assert!(run(&home, &repo, &["assist", "suggestions"]).contains("no suggestions waiting"));
    run(&home, &repo, &["assist", "stop"]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_oversized_browser_message_is_refused() {
    let (home, repo, dir) = scratch("big");
    let out = marin(
        &home,
        &repo,
        &["assist", "browser", "host"],
        &u32::MAX.to_ne_bytes(),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("too large"));
    let _ = fs::remove_dir_all(&dir);
}

fn daemon_pid(home: &Path, repo: &Path) -> Option<String> {
    let status = run(home, repo, &["assist", "status"]);
    let rest = status.split("(pid ").nth(1)?;
    rest.split(')').next().map(String::from)
}

#[test]
fn a_second_daemon_is_refused_and_a_killed_one_leaves_nothing_that_blocks_a_restart() {
    let (home, repo, dir) = scratch("kill");
    run(&home, &repo, &["assist", "start"]);
    let pid = daemon_pid(&home, &repo).unwrap_or_default();
    assert!(!pid.is_empty());
    let second = run(&home, &repo, &["assist", "daemon"]);
    assert!(second.contains("already running"), "{second}");
    assert_eq!(
        daemon_pid(&home, &repo).as_deref(),
        Some(pid.as_str()),
        "the first one is untouched"
    );

    assert!(
        Command::new("kill")
            .args(["-KILL", &pid])
            .status()
            .is_ok_and(|s| s.success())
    );
    for _ in 0..50 {
        if daemon_pid(&home, &repo).is_none() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(
        daemon_pid(&home, &repo).is_none(),
        "a killed daemon is reported as not running"
    );
    assert!(run(&home, &repo, &["assist", "start"]).contains("started"));
    assert_ne!(daemon_pid(&home, &repo).unwrap_or_default(), pid);
    run(&home, &repo, &["assist", "stop"]);
    assert!(!home.join("assist/assist.sock").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_browser_host_only_serves_the_marin_extension() {
    let (home, repo, dir) = scratch("sender");
    let denied = run(
        &home,
        &repo,
        &["assist", "browser", "host", "/m.json", "other@evil"],
    );
    assert!(
        denied.contains("not started by the Marin extension"),
        "{denied}"
    );
    let _ = fs::remove_dir_all(&dir);
}
