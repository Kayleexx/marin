//! The background assistant through the real binary: off by default, rules only, private, and quiet about repeats.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

fn marin(home: &Path, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_marin"))
        .env("MARIN_HOME", home)
        .env("NO_COLOR", "1")
        .current_dir(cwd)
        .args(args)
        .output()
        .unwrap_or_else(|e| Output {
            status: ExitStatus::from_raw(127 << 8),
            stdout: Vec::new(),
            stderr: e.to_string().into_bytes(),
        })
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("marin-assist-e2e-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(dir.join("repo/.git"));
    dir
}

const FAIL: &[&str] = &[
    "assist",
    "watch",
    "--",
    "sh",
    "-c",
    "echo 'error: broken at line 7' >&2; echo 'token=abcdef123456' >&2; exit 3",
];

#[test]
fn it_is_off_until_started_and_then_collects_one_suggestion_per_failure() {
    let dir = scratch("flow");
    let (home, repo) = (dir.join("home"), dir.join("repo"));
    let off = marin(&home, &repo, FAIL);
    assert_eq!(off.status.code(), Some(3), "the command's own exit code");
    assert!(
        text(&off).contains("broken at line 7"),
        "output passes through"
    );
    assert!(!home.join("assist/assist.sock").exists());
    assert!(text(&marin(&home, &repo, &["assist", "status"])).contains("off"));

    assert!(text(&marin(&home, &repo, &["assist", "allow"])).contains("watching"));
    assert!(text(&marin(&home, &repo, &["assist", "start"])).contains("started"));
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&home.join("assist")), 0o700);
    assert_eq!(mode(&home.join("assist/assist.sock")), 0o600);

    for _ in 0..2 {
        assert_eq!(marin(&home, &repo, FAIL).status.code(), Some(3));
    }
    let inbox = text(&marin(&home, &repo, &["assist", "inbox"]));
    assert!(
        inbox.contains("#1") && inbox.contains("seen 2 times"),
        "{inbox}"
    );
    assert!(!inbox.contains("#2"));
    let stored = fs::read_to_string(home.join("assist/inbox.json")).unwrap();
    assert!(stored.contains("broken at line") && !stored.contains("abcdef123456"));

    let elsewhere = dir.join("home");
    marin(&home, &elsewhere, FAIL);
    let status = text(&marin(&home, &repo, &["assist", "status"]));
    assert!(
        status.contains("1 kept quiet") && status.contains("no model loaded"),
        "{status}"
    );

    let why = text(&marin(&home, &repo, &["assist", "why", "1"]));
    assert!(
        why.contains("data left the device: no") && why.contains("<<<"),
        "{why}"
    );
    assert!(text(&marin(&home, &repo, &["assist", "dismiss", "1"])).contains("dismissed"));
    assert!(text(&marin(&home, &repo, &["assist", "inbox"])).contains("inbox empty"));

    assert!(text(&marin(&home, &repo, &["assist", "stop"])).contains("stopped"));
    assert!(!home.join("assist/assist.sock").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_folder_that_is_not_a_repo_cannot_be_allowed() {
    let dir = scratch("norepo");
    let loose = dir.join("loose");
    fs::create_dir_all(&loose).unwrap();
    let out = marin(&dir.join("home"), &loose, &["assist", "allow"]);
    assert!(!out.status.success());
    assert!(text(&out).contains("not inside a git repository"));
    let _ = fs::remove_dir_all(&dir);
}
