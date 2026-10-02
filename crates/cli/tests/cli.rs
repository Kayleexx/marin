use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3")
}

fn marin(args: &[&str], stdin: &str) -> io::Result<Output> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(args)
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(stdin.as_bytes())?;
    }
    child.wait_with_output()
}

fn model() -> String {
    fixture().display().to_string()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn run_ok(args: &[&str]) -> io::Result<Output> {
    let out = marin(args, "")?;
    assert!(out.status.success(), "{}", text(&out.stderr));
    Ok(out)
}

fn run_err(args: &[&str]) -> io::Result<(Option<i32>, String)> {
    let out = marin(args, "")?;
    Ok((out.status.code(), text(&out.stderr)))
}

fn temp_dir(name: &str) -> io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("marin-cli-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn copy_fixture_files(dir: &std::path::Path, names: &[&str]) -> io::Result<()> {
    for name in names {
        fs::copy(fixture().join(name), dir.join(name))?;
    }
    Ok(())
}

#[test]
fn prompt_mode_is_deterministic_and_keeps_stderr_quiet_when_piped() {
    let args = [
        "-p",
        "hello world",
        "--temp",
        "0",
        "--max-tokens",
        "6",
        &model(),
    ];
    let first = run_ok(&args).unwrap();
    let second = run_ok(&args).unwrap();
    assert!(!first.stdout.is_empty());
    assert_eq!(first.stdout, second.stdout);
    assert!(first.stderr.is_empty(), "{}", text(&first.stderr));
}

#[test]
fn the_same_seed_repeats_and_another_seed_differs() {
    let run = |seed: &str| {
        run_ok(&[
            "-p",
            "the cat",
            "--temp",
            "1.5",
            "--max-tokens",
            "24",
            "--seed",
            seed,
            &model(),
        ])
        .unwrap()
        .stdout
    };
    assert_eq!(run("7"), run("7"));
    assert_ne!(run("7"), run("8"));
}

#[test]
fn missing_folder_is_a_clear_error() {
    let (code, err) = run_err(&["-p", "x", "/definitely/not/here"]).unwrap();
    assert_eq!(code, Some(1));
    assert!(
        err.contains("error:") && err.contains("not a directory"),
        "{err}"
    );
    assert!(err.contains("hint:"), "{err}");
}

#[test]
fn empty_folder_lists_everything_that_is_missing() {
    let dir = temp_dir("empty").unwrap();
    let (code, err) = run_err(&["-p", "x", &dir.display().to_string()]).unwrap();
    assert_eq!(code, Some(1));
    for needed in ["config.json", "tokenizer.json", "model.safetensors"] {
        assert!(err.contains(needed), "{err}");
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn truncated_weights_are_reported_as_a_partial_download() {
    let dir = temp_dir("truncated").unwrap();
    copy_fixture_files(
        &dir,
        &["config.json", "generation_config.json", "tokenizer.json"],
    )
    .unwrap();
    let weights = fs::read(fixture().join("model.safetensors")).unwrap();
    let half = weights.get(..weights.len() / 2).unwrap();
    fs::write(dir.join("model.safetensors"), half).unwrap();
    let (code, err) = run_err(&["-p", "x", &dir.display().to_string()]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("partial download"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_config_for_another_architecture_is_refused() {
    let dir = temp_dir("llama").unwrap();
    copy_fixture_files(
        &dir,
        &[
            "generation_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    )
    .unwrap();
    let config = fs::read_to_string(fixture().join("config.json")).unwrap();
    fs::write(
        dir.join("config.json"),
        config.replace("\"qwen3\"", "\"llama\""),
    )
    .unwrap();
    let (code, err) = run_err(&["-p", "x", &dir.display().to_string()]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("llama") && err.contains("qwen3"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_config_that_disagrees_with_the_weights_lists_the_problems() {
    let dir = temp_dir("mismatch").unwrap();
    copy_fixture_files(
        &dir,
        &[
            "generation_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    )
    .unwrap();
    let config = fs::read_to_string(fixture().join("config.json")).unwrap();
    let wrong = config.replace("\"num_hidden_layers\": 2", "\"num_hidden_layers\": 3");
    fs::write(dir.join("config.json"), wrong).unwrap();
    let (code, err) = run_err(&["-p", "x", &dir.display().to_string()]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("missing tensor model.layers.2"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn empty_and_oversized_prompts_and_contexts_are_rejected() {
    let (code, err) = run_err(&["-p", "   ", &model()]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("prompt is empty"), "{err}");
    let (_, err) = run_err(&["-p", "the cat sat on the mat", "--ctx", "4", &model()]).unwrap();
    assert!(
        err.contains("tokens long") && err.contains("--ctx"),
        "{err}"
    );
    let (_, err) = run_err(&["-p", "x", "--ctx", "999999", &model()]).unwrap();
    assert!(err.contains("128"), "{err}");
    let (_, err) = run_err(&["-p", "x", "--ctx", "0", &model()]).unwrap();
    assert!(err.contains("above 0"), "{err}");
}

#[test]
fn invalid_sampling_values_fail_before_the_model_loads() {
    let (code, err) = run_err(&["-p", "x", "--temp=-1", &model()]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("temperature"), "{err}");
    let (_, err) = run_err(&["-p", "x", "--top-p", "0", &model()]).unwrap();
    assert!(err.contains("top_p"), "{err}");
}

#[test]
fn usage_mistakes_exit_with_code_two() {
    let (code, err) = run_err(&[]).unwrap();
    assert_eq!(code, Some(2));
    assert!(err.contains("Usage") && err.contains("Examples"), "{err}");
    assert_eq!(run_err(&["--no-such-flag", &model()]).unwrap().0, Some(2));
}

#[test]
fn inspect_explains_the_model() {
    let out = run_ok(&["inspect", &model()]).unwrap();
    let text = text(&out.stdout);
    for expected in [
        "decoder-only transformer",
        "layers",
        "grouped-query",
        "kv cache",
        "memory",
    ] {
        assert!(text.contains(expected), "{text}");
    }
}

#[test]
fn verify_passes_and_catches_a_corrupted_reference() {
    let reference = fixture().join("reference.json").display().to_string();
    let ok = run_ok(&["verify", &model(), "--reference", &reference]).unwrap();
    assert!(text(&ok.stdout).contains("result: PASS"));

    let dir = temp_dir("verify").unwrap();
    let original = fs::read_to_string(&reference).unwrap();
    let broken = original.replacen("\"token\": ", "\"token\": 1", 1);
    assert_ne!(broken, original);
    let broken_path = dir.join("broken.json");
    fs::write(&broken_path, broken).unwrap();
    let broken_arg = broken_path.display().to_string();
    let out = marin(&["verify", &model(), "--reference", &broken_arg], "").unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stdout).contains("result: FAIL"));

    let (code, err) = run_err(&["verify", &model(), "--reference", "/no/such/file.json"]).unwrap();
    assert_eq!(code, Some(1));
    assert!(err.contains("cannot use reference file"), "{err}");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn interactive_session_handles_commands_prompts_and_mistakes() {
    let script = "/help\n/tokens hello world\n/top hello world\n/set temp 0\n/set temp abc\n\
                  /set bogus 1\nhello world\n\n/nonsense\n/model\n/kv\n/memory\n/metrics\n\
                  /backend\n/quit\n";
    let out = marin(&["--max-tokens", "5", &model()], script).unwrap();
    assert!(out.status.success());
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    let expected = [
        "/quit",
        "tokens:",
        "likeliest next tokens",
        "temp = 0",
        "unknown command /nonsense",
        "prompt       ",
        "bf16 weights, f32 math",
    ];
    for item in expected {
        assert!(stdout.contains(item), "missing {item:?} in {stdout}");
    }
    assert!(stderr.contains("temp needs a number"), "{stderr}");
    assert!(stderr.contains("unknown setting `bogus`"), "{stderr}");
    assert!(
        stderr.contains("5 tokens") && stderr.contains("stopped: max tokens"),
        "{stderr}"
    );
}

#[test]
fn session_survives_errors_and_ends_cleanly_on_end_of_input() {
    let out = marin(
        &["--temp", "0", "--max-tokens", "2", &model()],
        "   \nhello world\n",
    )
    .unwrap();
    assert!(out.status.success());
    assert!(text(&out.stderr).contains("2 tokens"));
}

#[test]
fn a_full_context_window_is_reported_as_the_stop_reason() {
    let args = ["--ctx", "8", "--max-tokens", "50", "--temp", "0", &model()];
    let out = marin(&args, "hello world\n").unwrap();
    assert!(
        text(&out.stderr).contains("context full"),
        "{}",
        text(&out.stderr)
    );
}

#[test]
fn zero_max_tokens_generates_nothing() {
    let out = run_ok(&["-p", "hello", "--max-tokens", "0", &model()]).unwrap();
    assert!(out.stdout.is_empty());
}

#[test]
fn the_cpu_backend_can_be_forced_and_is_reported() {
    let out = marin(
        &["--backend", "cpu", "--max-tokens", "1", &model()],
        "/backend\n",
    )
    .unwrap();
    assert!(out.status.success());
    assert!(
        text(&out.stdout).contains("cpu (bf16 weights, f32 math)"),
        "{}",
        text(&out.stdout)
    );
    let (code, _) = run_err(&["--backend", "tpu", &model()]).unwrap();
    assert_eq!(code, Some(2));
}
