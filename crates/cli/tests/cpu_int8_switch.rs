//! The real terminal keeps INT8 across backend switches and discards an old model's chat.

mod pty;

use std::fs;
use std::path::Path;

use pty::{Check, Res, Sandbox, Step, have, on_terminal};

#[test]
fn int8_survives_backend_and_model_switches() -> Res<()> {
    switches(false)
}

#[test]
fn int8_survives_gpu_backend_and_model_switches() -> Res<()> {
    if let Err(error) = model::Gpu::new() {
        if std::env::var_os("MARIN_REQUIRE_CUDA").is_some() {
            return Err(error.into());
        }
        eprintln!("skipping: {error}");
        return Ok(());
    }
    switches(true)
}

fn switches(gpu: bool) -> Res<()> {
    if !have("script") {
        eprintln!("skipping: `script` is not installed");
        return Ok(());
    }
    let sandbox = Sandbox::new(if gpu {
        "gpu-int8-switch"
    } else {
        "cpu-int8-switch"
    })?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let other = sandbox.0.join("OtherChat");
    fs::create_dir(&other)?;
    for entry in fs::read_dir(&fixture)? {
        let entry = entry?;
        fs::copy(entry.path(), other.join(entry.file_name()))?;
    }
    let config = other.join("config.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&config)?)?;
    *value
        .get_mut("max_position_embeddings")
        .ok_or("missing context")? = 512.into();
    fs::write(config, serde_json::to_vec(&value)?)?;
    let shows = |text: &'static str| {
        move |_: &[u8], new: &[u8]| {
            let seen = String::from_utf8_lossy(new);
            if seen.contains(text) {
                Ok(())
            } else {
                Err(seen.escape_debug().to_string())
            }
        }
    };
    let prompt = shows("you >");
    let done = shows("stopped at");
    let palette = shows("Inspect last reply");
    let backend = shows("Backend");
    let failed = shows("the switch failed");
    let switched = shows("now using");
    let model = shows("OtherChat");
    let reply = shows("[Reply]");
    let empty = shows("nothing yet: send a prompt first");
    let weights = shows("int8 weights, groups of 32");
    let mut steps = vec![
        Step {
            keys: "",
            until: Some(("prompt", &prompt)),
        },
        Step {
            keys: "hello world\r",
            until: Some(("reply", &done)),
        },
    ];
    for (target, notice) in [
        ("cuda", if gpu { &switched as Check } else { &failed }),
        ("auto", &switched),
        ("cpu", &switched),
        (if gpu { "cuda" } else { "cpu" }, &switched),
    ] {
        steps.extend([
            Step {
                keys: "/",
                until: Some(("palette", &palette)),
            },
            Step {
                keys: "backend",
                until: None,
            },
            Step {
                keys: "\r",
                until: Some(("backends", &backend)),
            },
            Step {
                keys: target,
                until: None,
            },
            Step {
                keys: "\r",
                until: Some(("switch", notice)),
            },
            Step {
                keys: "\x0f",
                until: Some(("inspect", &reply)),
            },
            Step {
                keys: "5",
                until: Some(("INT8 retained", &weights)),
            },
            Step {
                keys: "\x1b",
                until: None,
            },
            Step {
                keys: "hello again\r",
                until: Some(("reply after switch", &done)),
            },
        ]);
    }
    steps.extend([
        Step {
            keys: "/",
            until: Some(("palette", &palette)),
        },
        Step {
            keys: "switch model",
            until: None,
        },
        Step {
            keys: "\r",
            until: Some(("models", &model)),
        },
        Step {
            keys: "OtherChat",
            until: None,
        },
        Step {
            keys: "\r",
            until: Some(("model changed", &switched)),
        },
        Step {
            keys: "\x0f",
            until: Some(("old model reply discarded", &empty)),
        },
        Step {
            keys: "5",
            until: Some(("new model INT8", &weights)),
        },
        Step {
            keys: "\x1b",
            until: None,
        },
        Step {
            keys: "hello new model\r",
            until: Some(("new model reply", &done)),
        },
    ]);
    let line = format!(
        "{}MARIN_MODELS='{}' {} --model '{}' --backend cpu --quantize int8 --quant-group 32 --ctx 256 --threads 8 --temp 0 --max-tokens 3; stty -a",
        if gpu { "" } else { "CUDA_VISIBLE_DEVICES=-1 " },
        sandbox.0.display(),
        env!("CARGO_BIN_EXE_marin"),
        fixture.display()
    );
    let (drawn, _, status) = on_terminal(&line, (110, 30), &sandbox, false, &steps)?;
    assert!(status.success());
    let screen = String::from_utf8_lossy(&drawn);
    assert!(!screen.contains("no model is loaded"));
    assert!(screen.contains("now using OtherChat"));
    let flags = screen.rsplit("lflags").next().unwrap_or("");
    assert!(flags.contains(" icanon ") && !flags.contains("-icanon") && !flags.contains("-echo "));
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(sandbox.0.join("config.json"))?)?;
    assert_eq!(
        saved.get("model").ok_or("missing model")?,
        other.to_string_lossy().as_ref()
    );
    Ok(())
}
