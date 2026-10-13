use std::{fs, path::Path, process::Command};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn both_clis_require_explicit_int8_and_use_the_exported_bundle() -> Result {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("marin-packed-cli-{}-{stamp}", std::process::id()));
    fs::create_dir(&root)?;
    let target = root.join("Qwen3-packed");
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat");
    let export = Command::new(env!("CARGO_BIN_EXE_marin-run"))
        .args(["export-int8"])
        .arg(&source)
        .arg(&target)
        .args(["--quant-group", "32"])
        .output()?;
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    for (bin, args) in [
        (
            env!("CARGO_BIN_EXE_marin-run"),
            vec![target.to_str().ok_or("path")?, "-p", "hello"],
        ),
        (
            env!("CARGO_BIN_EXE_marin"),
            vec!["--model", target.to_str().ok_or("path")?, "run", "hello"],
        ),
    ] {
        let mut plain = Command::new(bin);
        plain
            .env("MARIN_HOME", root.join("home"))
            .args(&args)
            .args(["--backend", "cpu", "--max-tokens", "1", "--temp", "0"]);
        assert!(
            !plain.output()?.status.success(),
            "BF16 default cannot silently change precision"
        );
        let output = plain
            .args(["--quantize", "int8", "--quant-group", "32"])
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
