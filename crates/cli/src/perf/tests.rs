use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use model::{BackendChoice, Precision};
use runtime::{CausalLm, SequenceCache};

use super::*;
use crate::load::load;
use crate::session::Settings;
use crate::style::Style;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

struct Temp(PathBuf);

impl Temp {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("marin-perf-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The tiny Qwen3 model without the reference that sits beside it in the fixtures.
fn bare_model(into: &Path) -> PathBuf {
    let to = into.join("model");
    fs::create_dir_all(&to).unwrap();
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(fixtures().join("tiny-qwen3").join(name), to.join(name)).unwrap();
    }
    to
}

fn session(path: &Path, ctx: usize) -> Session {
    let loaded = load(
        path,
        Some(ctx),
        ctx,
        BackendChoice::Cpu,
        Precision::default(),
        &Style::plain(),
        false,
    )
    .unwrap();
    let settings = Settings {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(1),
    };
    Session::new(loaded, settings, Style::plain()).unwrap()
}

/// Puts a conversation's worth of tokens in the session's cache.
fn fill(session: &mut Session, ids: &[u32]) {
    let (engine, kv) = session.engine_and_kv();
    engine.forward(ids, kv).unwrap();
}

fn reference() -> PathBuf {
    fixtures().join("tiny-qwen3/reference.json")
}

fn shown(out: &Output) -> String {
    out.lines.join("\n")
}

fn refusal(result: Result<Output, CliError>) -> String {
    match result {
        Err(e) => e.to_string(),
        Ok(out) => panic!("expected a refusal, got:\n{}", shown(&out)),
    }
}

#[test]
fn a_workload_is_read_from_short_arguments() {
    let w = |args: &str| parse_workload(args).unwrap();
    assert_eq!(
        w(""),
        Workload {
            prompt: 128,
            output: 64,
            runs: 3,
            warmup: 1,
            profile: false
        }
    );
    assert_eq!(
        w("512 128 runs=2 profile"),
        Workload {
            prompt: 512,
            output: 128,
            runs: 2,
            warmup: 1,
            profile: true
        }
    );
    assert_eq!(
        w("1024"),
        Workload {
            prompt: 1024,
            output: 64,
            runs: 3,
            warmup: 1,
            profile: false
        }
    );
    assert_eq!(w("quick").runs, 1);
    assert_eq!(w("quick").warmup, 0);
    assert_eq!(w("  runs=5   ").runs, 5);
    for bad in [
        "1 2 3", "banana", "runs=x", "runs=0", "runs=99", "0", "8 1", "-3", "runs",
    ] {
        let why = parse_workload(bad).unwrap_err().to_string();
        assert!(why.contains("usage: /bench"), "{bad}: {why}");
    }
}

#[test]
fn a_benchmark_measures_the_loaded_model_in_a_cache_of_its_own() {
    let dir = Temp::new("bench");
    let mut session = session(&bare_model(&dir.0), 64);
    let ids = session.encode("the cat sat on the mat").unwrap();
    fill(&mut session, &ids);
    let held = session.kv().used();
    assert!(held > 0);
    let out = bench(&session, &Interrupt::unhooked(), "8 4 runs=2").unwrap();
    let text = shown(&out);
    for needle in [
        "not measured by this run: loading the model took",
        "it was not loaded again",
        "own cache",
        "results (median of runs):",
        "first token (TTFT)",
        "prefill",
        "decode",
        "peak process memory",
    ] {
        assert!(text.contains(needle), "{needle}:\n{text}");
    }
    assert!(!out.failed);
    assert_eq!(
        session.kv().used(),
        held,
        "the conversation's cache was not used"
    );
}

#[test]
fn a_workload_beyond_the_window_is_refused_before_anything_runs() {
    let dir = Temp::new("window");
    let session = session(&bare_model(&dir.0), 64);
    let why = refusal(bench(&session, &Interrupt::unhooked(), "100 10"));
    assert!(
        why.contains("this session's window is 64") && why.contains("--ctx"),
        "{why}"
    );
    let why = refusal(bench(&session, &Interrupt::unhooked(), "8 4 runs=0"));
    assert!(why.contains("usage"), "{why}");
}

#[test]
fn verify_finds_the_reference_beside_the_model_and_passes_it() {
    let session = session(&fixtures().join("tiny-qwen3"), 64);
    let here = Temp::new("nowhere");
    let out = verify(&session, &Interrupt::unhooked(), "", &here.0).unwrap();
    let text = shown(&out);
    assert!(
        text.contains("reference.json") && text.contains("result: PASS (3 cases)"),
        "{text}"
    );
    assert!(!out.failed);
    let named = verify(
        &session,
        &Interrupt::unhooked(),
        reference().to_str().unwrap(),
        &here.0,
    )
    .unwrap();
    assert!(!named.failed && shown(&named).contains("result: PASS"));
}

#[test]
fn an_altered_reference_fails_and_is_never_a_pass() {
    let dir = Temp::new("altered");
    let session = session(&bare_model(&dir.0), 64);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(reference()).unwrap()).unwrap();
    let logit = value.pointer_mut("/cases/0/steps/1/top/0/1").unwrap();
    *logit = serde_json::json!(logit.as_f64().unwrap() + 1.0);
    let path = dir.0.join("altered-reference.json");
    fs::write(&path, value.to_string()).unwrap();
    let out = verify(
        &session,
        &Interrupt::unhooked(),
        path.to_str().unwrap(),
        &dir.0,
    )
    .unwrap();
    let text = shown(&out);
    assert!(
        out.failed && text.contains("FAIL") && text.contains("result: FAIL"),
        "{text}"
    );
    assert!(!text.contains("result: PASS"), "{text}");
}

#[test]
fn a_reference_for_something_else_is_explained_not_judged() {
    let dir = Temp::new("else");
    let session = session(&bare_model(&dir.0), 64);
    let go = |name: &str| verify(&session, &Interrupt::unhooked(), name, &dir.0);
    let hybrid = fixtures().join("tiny-qwen35/reference.json");
    let why = refusal(go(hybrid.to_str().unwrap()));
    assert!(
        why.contains("layers") && why.contains("this model has 2"),
        "{why}"
    );
    let packed = fixtures().join("tiny-qwen35-int8/quant-reference.json");
    let why = refusal(go(packed.to_str().unwrap()));
    assert!(
        why.contains("int8-dequantized") && why.contains("bfloat16"),
        "{why}"
    );
    let why = refusal(go("missing.json"));
    assert!(why.contains("missing.json"), "{why}");
    let why = refusal(go("."));
    assert!(why.contains("not a file"), "{why}");
    fs::write(
        dir.0.join("empty.json"),
        r#"{"format": 1, "weights": "bfloat16", "cases": []}"#,
    )
    .unwrap();
    let why = refusal(go("empty.json"));
    assert!(
        why.contains("checks nothing"),
        "an empty reference cannot pass: {why}"
    );
    fs::write(dir.0.join("garbage.json"), "{not json").unwrap();
    assert!(refusal(go("garbage.json")).contains("garbage.json"));
    fs::write(dir.0.join("future.json"), r#"{"format": 9, "cases": []}"#).unwrap();
    assert!(refusal(go("future.json")).contains("unknown format"));
}

#[test]
fn a_session_window_smaller_than_the_reference_needs_is_said_not_run() {
    let dir = Temp::new("small");
    let session = session(&bare_model(&dir.0), 8);
    let why = refusal(verify(
        &session,
        &Interrupt::unhooked(),
        reference().to_str().unwrap(),
        &dir.0,
    ));
    assert!(
        why.contains("needs a window of") && why.contains("session's is 8"),
        "{why}"
    );
}

#[test]
fn with_no_usable_reference_the_message_says_what_is_missing() {
    let dir = Temp::new("missing");
    let model = bare_model(&dir.0);
    let session = session(&model, 64);
    let project = dir.0.join("project");
    fs::create_dir_all(project.join("tests/reference")).unwrap();
    fs::create_dir_all(project.join("bench-results/q")).unwrap();
    let wrong = fixtures().join("tiny-qwen35/reference.json");
    fs::copy(
        &wrong,
        project.join("tests/reference/qwen35-reference.json"),
    )
    .unwrap();
    let packed = fixtures().join("tiny-qwen35-int8/quant-reference.json");
    fs::copy(&packed, project.join("bench-results/q/hf-int8.json")).unwrap();
    fs::write(
        project.join("tests/reference/notes.json"),
        "not a reference",
    )
    .unwrap();
    let why = refusal(verify(&session, &Interrupt::unhooked(), "", &project));
    assert!(
        why.contains("no reference made for model with bfloat16 weights"),
        "{why}"
    );
    assert!(
        why.contains("qwen35-reference.json") && why.contains("layers"),
        "{why}"
    );
    assert!(
        !why.contains("hf-int8.json"),
        "a file for other weights is passed over: {why}"
    );
    assert!(!why.contains("notes.json"), "{why}");
    assert!(why.contains("/verify PATH"), "{why}");
    let bare = refusal(verify(
        &session,
        &Interrupt::unhooked(),
        "",
        &dir.0.join("empty"),
    ));
    assert!(bare.contains("no reference file"), "{bare}");
}

#[test]
fn the_places_looked_in_are_the_model_then_the_project() {
    let dir = Temp::new("places");
    let session = session(&bare_model(&dir.0), 64);
    let project = dir.0.join("project");
    for place in ["tests/reference", "tests/fixtures/a", "bench-results/b"] {
        fs::create_dir_all(project.join(place)).unwrap();
    }
    let places = reference::places(&session.loaded, &project);
    assert_eq!(places.first(), Some(&session.loaded.dir.path));
    let tail: Vec<_> = places
        .iter()
        .skip(1)
        .map(|p| p.strip_prefix(&project).unwrap().to_owned())
        .collect();
    assert_eq!(
        tail,
        [
            PathBuf::from("tests/reference"),
            PathBuf::from("tests/fixtures/a"),
            PathBuf::from("bench-results/b"),
        ]
    );
}

#[test]
fn ctrl_c_stops_a_benchmark_and_a_check_and_leaves_the_session_alone() {
    let dir = Temp::new("cancel");
    let mut session = session(&bare_model(&dir.0), 128);
    let ids = session.encode("the cat sat on the mat").unwrap();
    fill(&mut session, &ids);
    let held = session.kv().used();
    for (what, run) in [
        (
            "bench",
            Box::new(|s: &Session, i: &Interrupt| bench(s, i, "100 20 runs=20").map(drop))
                as Box<dyn Fn(&Session, &Interrupt) -> Result<(), CliError> + Sync>,
        ),
        (
            "verify",
            Box::new(|s: &Session, i: &Interrupt| {
                verify(s, i, reference().to_str().unwrap(), Path::new(".")).map(drop)
            }),
        ),
    ] {
        let interrupt = Interrupt::unhooked();
        let done = std::sync::atomic::AtomicBool::new(false);
        let result = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::SeqCst) {
                    interrupt.cancel_flag().store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_micros(50));
                }
            });
            let result = run(&session, &interrupt);
            done.store(true, Ordering::SeqCst);
            result
        });
        assert!(
            matches!(result, Err(CliError::Cancelled)),
            "{what}: {result:?}"
        );
        assert_eq!(session.kv().used(), held, "{what} left the cache as it was");
    }
    let again = bench(&session, &Interrupt::unhooked(), "8 4 runs=1 quick").unwrap();
    assert!(
        shown(&again).contains("results (median of runs):"),
        "the model works after a cancel"
    );
}
