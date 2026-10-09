//! The KV and Model tabs of the inspect panel name the cache type and the weight format, so the real UI on a
//! pseudo-terminal shows whether `--kv-type` and `--quantize` (or their defaults) reached the loaded model, at the
//! start and after a model switch. Without a GPU the GPU
//! test skips unless `MARIN_REQUIRE_CUDA` is set.

mod pty;

use std::path::{Path, PathBuf};
use std::process::Command;

use pty::{Check, Res, Sandbox, Step, have, on_terminal};

fn step<'a>(keys: &'a str, until: Option<(&'a str, Check<'a>)>) -> Step<'a> {
    Step { keys, until }
}

/// Opens the inspect panel on the KV tab, then the Model tab, and closes it.
fn look<'a>(steps: &mut Vec<Step<'a>>, [reply, kv, model]: [Check<'a>; 3]) {
    steps.extend([
        step("\x0f", Some(("reply tab", reply))),
        step("4", Some(("kv tab", kv))),
        step("5", Some(("model tab", model))),
        step("\x1b", None),
    ]);
}

fn tiny_chat() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat")
}

fn have_gpu() -> bool {
    let ok = Command::new(env!("CARGO_BIN_EXE_marin"))
        .env("MARIN_HOME", std::env::temp_dir())
        .args(["--backend", "cuda", "--max-tokens", "1", "--model"])
        .args([tiny_chat().as_os_str(), "run".as_ref(), "hi".as_ref()])
        .output()
        .is_ok_and(|out| out.status.success());
    assert!(
        ok || std::env::var_os("MARIN_REQUIRE_CUDA").is_none(),
        "MARIN_REQUIRE_CUDA is set but the GPU run failed"
    );
    ok
}

/// What the KV and Model tabs report, in order: at the start and, if `switch`, after loading the model again.
#[derive(Debug, PartialEq)]
struct Panels {
    kv: Vec<String>,
    weights: Vec<String>,
}

fn panels(args: &str, switch: bool) -> Res<Option<Panels>> {
    if !have("script") {
        eprintln!("skipping: `script` is not installed");
        return Ok(None);
    }
    let shows = |what: &'static str| {
        move |_: &[u8], new: &[u8]| {
            if String::from_utf8_lossy(new).contains(what) {
                Ok(())
            } else {
                Err(String::from_utf8_lossy(new).escape_debug().to_string())
            }
        }
    };
    let (prompt, reply, kv, model, menu, backend, now) = (
        shows("you >"),
        shows("[Reply]"),
        shows("values) on"),
        shows(" · "),
        shows("Inspect last reply"),
        shows("Backend"),
        shows("now using"),
    );
    let tabs = [&reply as Check, &kv, &model];
    let mut steps = vec![step("", Some(("prompt", &prompt)))];
    look(&mut steps, tabs);
    if switch {
        steps.extend([
            step("/", Some(("menu", &menu))),
            step("backend", None),
            step("\r", Some(("backend menu", &backend))),
            step("cuda", None),
            step("\r", Some(("switched", &now))),
        ]);
        look(&mut steps, tabs);
    }
    let sandbox = Sandbox::new("panels")?;
    let line = format!(
        "{} --model {} {args} --temp 0 --max-tokens 3",
        env!("CARGO_BIN_EXE_marin"),
        tiny_chat().display()
    );
    let (drawn, _, _) = on_terminal(&line, (110, 30), &sandbox, false, &steps)?;
    let screen = String::from_utf8_lossy(&drawn).into_owned();
    let after = |marker: &str, until: &str| -> Vec<String> {
        screen
            .split(marker)
            .skip(1)
            .filter_map(|part| {
                let before = part.split_once(until)?.0;
                Some(before.rsplit_once(['(', '·'])?.1.trim().to_string())
            })
            .collect()
    };
    Ok(Some(Panels {
        kv: after("reserved", " values)"),
        weights: weights_lines(&screen),
    }))
}

/// The text after the bullet on each `weights ... · ...` line of the Model tab.
fn weights_lines(screen: &str) -> Vec<String> {
    screen
        .split("weights         ")
        .skip(1)
        .filter_map(|part| {
            let line = part.split(['\r', '\n', '\u{1b}']).next()?;
            Some(line.split_once(" · ")?.1.trim().to_string())
        })
        .collect()
}

#[test]
fn the_cache_type_and_weight_format_reach_the_gpu_and_survive_a_model_switch() {
    if !have_gpu() {
        return;
    }
    for (args, kv, weights) in [
        ("--backend cuda", "f16", "bf16 weights"),
        ("--backend cuda --kv-type f32", "f32", "bf16 weights"),
        ("--backend cuda --quantize none", "f16", "bf16 weights"),
        (
            "--backend cuda --quantize int8 --quant-group 32",
            "f16",
            "int8 weights, groups of 32",
        ),
        (
            "--backend cuda --quantize int8 --quant-group 32 --kv-type f32",
            "f32",
            "int8 weights, groups of 32",
        ),
    ] {
        let Some(found) = panels(args, true).unwrap() else {
            return;
        };
        let expected = Panels {
            kv: vec![kv.into(); 2],
            weights: vec![weights.into(); 2],
        };
        assert_eq!(found, expected, "{args}");
    }
}

#[test]
fn without_flags_the_cpu_keeps_f32_and_bf16() {
    if let Some(found) = panels("--backend cpu", false).unwrap() {
        assert_eq!(
            found,
            Panels {
                kv: vec!["f32".into()],
                weights: vec!["bf16 weights".into()],
            }
        );
    }
}
