use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// A scratch folder that is removed when it goes out of scope.
struct Sandbox(PathBuf);

impl Sandbox {
    fn new(name: &str) -> io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("marin-show-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn show(args: &[&str], stdin: &[u8], columns: &str) -> io::Result<Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin"))
        .arg("show")
        .args(args)
        .env("COLUMNS", columns)
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("no stdin"))?;
    let bytes = stdin.to_vec();
    let feeder = std::thread::spawn(move || {
        let _ = input.write_all(&bytes);
    });
    let out = child.wait_with_output()?;
    feeder
        .join()
        .map_err(|_| io::Error::other("feeder panicked"))?;
    Ok(out)
}

fn show_file(name: &str, content: &[u8]) -> io::Result<Output> {
    let sandbox = Sandbox::new(&name.replace('.', "-"))?;
    let path = sandbox.0.join(name);
    std::fs::write(&path, content)?;
    show(&[&path.display().to_string()], b"", "60")
}

#[test]
fn files_are_shown_according_to_their_kind() {
    let md = show_file("notes.md", b"# Title\n\nSome **bold** text\n").unwrap();
    assert_eq!(text(&md.stdout), "Title\n=====\n\nSome bold text\n");

    let rs = show_file("main.rs", b"fn main() {\n    let x = 1;\n}\n").unwrap();
    assert_eq!(text(&rs.stdout), "| fn main() {\n|     let x = 1;\n| }\n");

    let json = show_file("data.json", b"{\"a\": [1, true]}\n").unwrap();
    assert_eq!(text(&json.stdout), "| {\"a\": [1, true]}\n");

    let csv = show_file("scores.csv", b"name,score\nann,10\nbob,7\n").unwrap();
    assert_eq!(
        text(&csv.stdout),
        "+------+-------+\n| name | score |\n+------+-------+\n| ann  | 10    |\n| bob  | 7     |\n+------+-------+\n"
    );

    let log = show_file(
        "run.log",
        b"2024-01-01 INFO started\n2024-01-01 ERROR boom\n",
    )
    .unwrap();
    assert_eq!(
        text(&log.stdout),
        "2024-01-01 INFO started\n2024-01-01 ERROR boom\n"
    );

    let plain = show_file("notes.txt", b"  indented  text\n# not a heading\n").unwrap();
    assert_eq!(
        text(&plain.stdout),
        "  indented  text\n# not a heading\n",
        "plain text is shown as written"
    );
    for out in [md, rs, json, csv, log, plain] {
        assert!(out.status.success() && out.stderr.is_empty());
    }
}

#[test]
fn unknown_and_unrecognized_files_fall_back_to_plain_text() {
    let unknown = show_file("data.weird", b"just some words\nmore words\n").unwrap();
    assert_eq!(text(&unknown.stdout), "just some words\nmore words\n");
    let empty = show_file("empty.md", b"").unwrap();
    assert!(empty.status.success() && empty.stdout.is_empty());
    let odd = show_file("x.rb", b"puts 1\n").unwrap();
    assert_eq!(
        text(&odd.stdout),
        "| puts 1\n",
        "known code in a language without colors still gets the code layout"
    );
}

#[test]
fn binary_and_invalid_text_is_reported_not_printed() {
    for (name, bytes) in [
        ("image.png", &b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"[..]),
        ("code.rs", b"fn main\0() {}"),
        ("bad.txt", &[b'a', 0xff, 0xfe, b'b'][..]),
    ] {
        let out = show_file(name, bytes).unwrap();
        assert!(out.status.success(), "{name}");
        assert_eq!(
            text(&out.stdout),
            format!("{name}: binary content, not shown\n"),
            "{name}"
        );
    }
}

#[test]
fn problems_with_the_input_are_explained() {
    let sandbox = Sandbox::new("show-errors").unwrap();
    let missing = sandbox.0.join("nope.md");
    let out = show(&[missing.to_str().unwrap()], b"", "60").unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("error: cannot read ") && err.contains("nope.md"),
        "{err}"
    );
    assert!(err.contains("hint: check that the path exists"), "{err}");
    assert!(out.stdout.is_empty());
    let dir = show(&[sandbox.0.to_str().unwrap()], b"", "60").unwrap();
    assert_eq!(dir.status.code(), Some(1));
    assert!(
        text(&dir.stderr).contains("error: cannot read "),
        "{}",
        text(&dir.stderr)
    );
}

#[test]
fn standard_input_is_read_and_its_kind_detected_or_chosen() {
    let diff = show(
        &["-"],
        b"diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n",
        "60",
    )
    .unwrap();
    assert_eq!(
        text(&diff.stdout),
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
    );
    let md = show(&["--as", "markdown", "-"], b"## Part\n- a\n- b\n", "60").unwrap();
    assert_eq!(text(&md.stdout), "Part\n----\n\n- a\n- b\n");
    let table = show(&["--as", "csv", "-"], b"a;b\n1;2\n", "60").unwrap();
    assert_eq!(
        text(&table.stdout),
        "+-----+\n| a;b |\n+-----+\n| 1;2 |\n+-----+\n",
        "the chosen kind wins over what the text looks like"
    );
    let weird = show(&["--as", "klingon", "-"], b"qapla\n", "60").unwrap();
    assert_eq!(
        text(&weird.stdout),
        "| qapla\n",
        "an unknown kind is plain code, never an error"
    );
    let nothing = show(&["-"], b"", "60").unwrap();
    assert!(nothing.status.success() && nothing.stdout.is_empty());
}

#[test]
fn output_wraps_to_the_terminal_width_and_survives_huge_input() {
    let long = "word ".repeat(40);
    let out = show(&["--as", "markdown", "-"], long.as_bytes(), "30").unwrap();
    for line in text(&out.stdout).lines() {
        assert!(line.chars().count() <= 29, "{line:?}");
    }
    let huge = "A paragraph with **bold** and `code`.\n\n".repeat(5000);
    let out = show(&["--as", "markdown", "-"], huge.as_bytes(), "80").unwrap();
    assert!(out.status.success());
    assert_eq!(
        text(&out.stdout)
            .matches("A paragraph with bold and `code`.")
            .count(),
        5000
    );
}

#[test]
fn nothing_to_show_on_a_terminal_is_an_explained_error() {
    if Command::new("script").arg("--version").output().is_err() {
        return;
    }
    let out = Command::new("script")
        .args([
            "-qec",
            &format!("{} show", env!("CARGO_BIN_EXE_marin")),
            "/dev/null",
        ])
        .env("NO_COLOR", "1")
        .env("TERM", "xterm")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let screen = text(&out.stdout);
    assert!(screen.contains("error: nothing to show"), "{screen}");
    assert!(
        screen.contains("hint: pass a file, or pipe text into `marin show`"),
        "{screen}"
    );
}
