use std::process::Command;

fn git(args: &[&str]) -> Option<std::process::Output> {
    Command::new("git").args(args).output().ok()
}

/// Short commit hash, with `-dirty` when tracked files differ from it, or `unknown` outside a git checkout.
fn revision() -> String {
    let hash = git(&["rev-parse", "--short=12", "HEAD"])
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty());
    let Some(hash) = hash else {
        return "unknown".into();
    };
    let clean =
        git(&["diff-index", "--quiet", "HEAD", "--"]).is_some_and(|out| out.status.success());
    if clean { hash } else { format!("{hash}-dirty") }
}

fn main() {
    println!("cargo:rustc-env=MARIN_REVISION={}", revision());
    for path in [
        ".git/HEAD",
        ".git/index",
        ".git/refs/heads",
        "crates",
        "Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed=../../{path}");
    }
}
