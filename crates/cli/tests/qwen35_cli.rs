use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35")
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let p = std::env::temp_dir().join(format!("marin-qwen35-{}-{stamp}", std::process::id()));
        fs::create_dir(&p)?;
        Ok(Self(p))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn both_existing_clis_generate_and_reject_unsupported_formats() -> Result {
    let temp = Temp::new()?;
    let path = fixture();
    let chat = temp.0.join("hybrid-chat");
    fs::create_dir(&chat)?;
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(path.join(name), chat.join(name))?;
    }
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-chat/tokenizer_config.json"),
        chat.join("tokenizer_config.json"),
    )?;

    for (bin, args) in [
        (
            env!("CARGO_BIN_EXE_marin-run"),
            vec![
                path.to_str().ok_or("path")?,
                "-p",
                "The capital of France is",
            ],
        ),
        (
            env!("CARGO_BIN_EXE_marin"),
            vec![
                "--model",
                chat.to_str().ok_or("path")?,
                "run",
                "The capital of France is",
            ],
        ),
    ] {
        let output = Command::new(bin)
            .env("MARIN_HOME", temp.0.join("home"))
            .args(&args)
            .args(["--backend", "cpu", "--max-tokens", "4", "--temp", "0"])
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stdout.is_empty());
        for flags in [
            ["--quantize", "int8"],
            ["--kv-type", "f16"],
            ["--backend", "cuda"],
        ] {
            let output = Command::new(bin)
                .env("MARIN_HOME", temp.0.join("home"))
                .args(&args)
                .args(flags)
                .output()?;
            assert!(!output.status.success());
        }
    }
    Ok(())
}
#[test]
fn full_hf_verification_requires_layer_and_logit_vectors() -> Result {
    let temp = Temp::new()?;
    let path = fixture();
    let verify = |refpath: &Path| {
        Command::new(env!("CARGO_BIN_EXE_marin-run"))
            .arg("verify")
            .arg(&path)
            .args(["--backend", "cpu", "--reference"])
            .arg(refpath)
            .output()
    };
    let good = verify(&path.join("reference.json"))?;
    assert!(
        good.status.success(),
        "{}",
        String::from_utf8_lossy(&good.stdout)
    );
    let raw: serde_json::Value = serde_json::from_slice(&fs::read(path.join("reference.json"))?)?;
    for mode in 0..3 {
        let mut value = raw.clone();
        let case = value
            .get_mut("cases")
            .and_then(|x| x.as_array_mut())
            .and_then(|x| x.first_mut())
            .ok_or("cases")?;
        if mode == 0 {
            case.as_object_mut().ok_or("case")?.remove("layer_values");
        } else {
            let step = case
                .get_mut("steps")
                .and_then(|x| x.as_array_mut())
                .and_then(|x| x.first_mut())
                .ok_or("step")?;
            if mode == 1 {
                step.as_object_mut().ok_or("step")?.remove("logits");
            } else {
                *step
                    .get_mut("logits")
                    .and_then(|x| x.as_array_mut())
                    .and_then(|x| x.first_mut())
                    .ok_or("logit")? = 10.0.into();
            }
        }
        let file = temp.0.join(format!("bad-{mode}.json"));
        fs::write(&file, serde_json::to_vec(&value)?)?;
        assert!(!verify(&file)?.status.success());
    }
    Ok(())
}
