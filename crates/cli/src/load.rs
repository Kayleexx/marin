use std::path::Path;
use std::time::Duration;

use loader::ModelDir;
use model::{Backend, BackendChoice, Engine, Qwen3Config};

use crate::CliError;
use crate::memory::{Footprint, footprint, resolve_ctx};
use crate::progress::Progress;
use crate::style::Style;
use crate::units;

pub struct Loaded {
    pub dir: ModelDir,
    pub model: Engine,
    pub name: String,
    pub footprint: Footprint,
    pub ctx: usize,
    pub took: Duration,
    /// Free GPU memory right before the weights were uploaded, when running on the GPU.
    pub gpu_free_before: Option<u64>,
}

pub fn model_name(path: &Path) -> String {
    path.canonicalize()
        .ok()
        .as_deref()
        .and_then(Path::file_name)
        .or_else(|| path.file_name())
        .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned())
}

/// Opens the folder, checks the memory it needs, then reads the weights.
/// `chrome` controls the progress display on stderr.
pub fn load(
    path: &Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    style: &Style,
    chrome: bool,
) -> Result<Loaded, CliError> {
    let dir = ModelDir::open(path)?;
    let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
    let ctx = resolve_ctx(&config, ctx, default_ctx)?;
    let footprint = footprint(&dir, &config, ctx)?;
    let selected = Backend::select(backend, footprint.weights.saturating_add(footprint.kv))?;
    if matches!(selected.backend, Backend::Cpu) {
        footprint.ensure_fits()?;
    }
    if let (Some(note), true) = (&selected.note, chrome) {
        eprintln!("{} {note}", style.yellow("note:"));
    }
    if chrome && !crate::style::stderr_is_terminal() {
        eprintln!(
            "loading {} ({})",
            model_name(path),
            units::bytes(footprint.weights)
        );
    }
    let gpu_free_before = match &selected.backend {
        Backend::Cuda(gpu) => gpu
            .info()
            .ok()
            .and_then(|info| u64::try_from(info.free_bytes).ok()),
        Backend::Cpu => None,
    };
    let mut progress = Progress::new(*style, chrome);
    let model = Engine::load(&dir, selected.backend, |done, total| {
        progress.update(done, total)
    })?;
    progress.finish();
    Ok(Loaded {
        name: model_name(path),
        took: progress.elapsed(),
        dir,
        model,
        footprint,
        ctx,
        gpu_free_before,
    })
}
