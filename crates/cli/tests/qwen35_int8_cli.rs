use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn both_clis_select_opt_in_packed_hybrid_int8_and_preserve_bf16_default() -> Result {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let root =
        Temp(std::env::temp_dir().join(format!("marin-q35-q8-cli-{}-{stamp}", std::process::id())));
    let source = root.0.join("chat");
    fs::create_dir_all(&source)?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8");
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(fixture.join(name), source.join(name))?;
    }
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-chat/tokenizer_config.json"),
        source.join("tokenizer_config.json"),
    )?;
    let packed = root.0.join("packed");
    let dir = loader::ModelDir::open(&source)?;
    model::export_int8(&dir, &packed, 32)?;
    for (bin, prefix) in [
        (
            env!("CARGO_BIN_EXE_marin-run"),
            vec![packed.to_str().ok_or("path")?],
        ),
        (
            env!("CARGO_BIN_EXE_marin"),
            vec!["--model", packed.to_str().ok_or("path")?, "run"],
        ),
    ] {
        let run = |extra: &[&str]| {
            Command::new(bin)
                .env("MARIN_HOME", root.0.join("home"))
                .args(&prefix)
                .args(["--backend", "cpu", "--max-tokens", "4", "--temp", "0"])
                .args(extra)
                .args(if bin == env!("CARGO_BIN_EXE_marin-run") {
                    vec!["-p"]
                } else {
                    vec![]
                })
                .arg("The capital of France is")
                .output()
        };
        let default = run(&[])?;
        assert!(!default.status.success());
        let q = run(&["--quantize", "int8", "--quant-group", "32"])?;
        assert!(q.status.success(), "{}", String::from_utf8_lossy(&q.stderr));
        assert!(!q.stdout.is_empty());
        if bin == env!("CARGO_BIN_EXE_marin-run") {
            let verified = Command::new(bin)
                .arg("verify")
                .arg(&packed)
                .args([
                    "--backend",
                    "cpu",
                    "--quantize",
                    "int8",
                    "--quant-group",
                    "32",
                    "--reference",
                ])
                .arg(fixture.join("quant-reference.json"))
                .output()?;
            assert!(
                verified.status.success(),
                "{} {}",
                String::from_utf8_lossy(&verified.stdout),
                String::from_utf8_lossy(&verified.stderr)
            );
        }
        assert!(
            !run(&["--quantize", "int8", "--quant-group", "64"])?
                .status
                .success()
        );
    }
    Ok(())
}
