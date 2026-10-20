//! The tiny Qwen3.5 model as a packed 8-bit folder a chat session can load on the GPU.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub type Res<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// True if a GPU is there; without one the tests that need it skip, unless `MARIN_REQUIRE_CUDA` is set.
pub fn have_gpu() -> Res<bool> {
    match model::Gpu::new() {
        Ok(_) => Ok(true),
        Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping: no usable GPU ({e})");
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

/// Builds `root/packed` from the fixtures and returns it.
pub fn packed_tiny(root: &Path) -> Res<PathBuf> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = root.join("chat");
    fs::create_dir_all(&source)?;
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        fs::copy(
            fixtures.join("tiny-qwen35-int8").join(name),
            source.join(name),
        )?;
    }
    fs::copy(
        fixtures.join("tiny-chat/tokenizer_config.json"),
        source.join("tokenizer_config.json"),
    )?;
    let packed = root.join("packed");
    model::export_int8(&loader::ModelDir::open(&source)?, &packed, 32).map_err(io::Error::other)?;
    Ok(packed)
}

/// A scratch folder that goes when it does.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(name: &str) -> io::Result<Self> {
        let dir = std::env::temp_dir().join(format!("marin-q35-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
