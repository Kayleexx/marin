use std::path::Path;
use std::time::Duration;

use loader::ModelDir;
use model::{Backend, BackendChoice, Engine, KvFormat, Precision, Qwen3Config};

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
    /// Running while the weights load, when a snapshot was asked for.
    pub weights_hash: Option<std::thread::JoinHandle<Result<u64, loader::LoaderError>>>,
}

pub fn model_name(path: &Path) -> String {
    path.canonicalize()
        .ok()
        .as_deref()
        .and_then(Path::file_name)
        .or_else(|| path.file_name())
        .map_or_else(|| "model".to_string(), |n| n.to_string_lossy().into_owned())
}

struct Request<'a> {
    path: &'a Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    precision: Precision,
    style: &'a Style,
    chrome: bool,
    hash_weights: bool,
}

/// Opens the folder, checks the memory it needs, then reads the weights.
/// `chrome` controls the progress display on stderr.
pub fn load(
    path: &Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    precision: Precision,
    style: &Style,
    chrome: bool,
) -> Result<Loaded, CliError> {
    open(Request {
        path,
        ctx,
        default_ctx,
        backend,
        precision,
        style,
        chrome,
        hash_weights: false,
    })
}

/// Like `load`, and also hashes the weight files on another thread while they load, which a snapshot needs to prove it
/// belongs to these weights.
pub fn load_for_snapshot(
    path: &Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    precision: Precision,
    style: &Style,
    chrome: bool,
) -> Result<Loaded, CliError> {
    open(Request {
        path,
        ctx,
        default_ctx,
        backend,
        precision,
        style,
        chrome,
        hash_weights: true,
    })
}

fn open(request: Request<'_>) -> Result<Loaded, CliError> {
    let Request {
        path,
        ctx,
        default_ctx,
        backend,
        precision,
        style,
        chrome,
        hash_weights,
    } = request;
    if backend == BackendChoice::Cpu {
        if precision.weights.is_some() {
            return Err(CliError::BadOption(
                "--quantize needs the GPU, but --backend cpu was given".into(),
            ));
        }
        if precision.kv != KvFormat::default() {
            return Err(CliError::BadOption(
                "--kv-type needs the GPU, but --backend cpu was given".into(),
            ));
        }
    }
    let dir = ModelDir::open(path)?;
    let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
    let ctx = resolve_ctx(&config, ctx, default_ctx)?;
    let footprint = footprint(&dir, &config, ctx, precision)?;
    let selected = Backend::select(
        backend,
        footprint.weights.saturating_add(footprint.kv_startup),
        footprint
            .weights
            .saturating_add(footprint.kv_window.max(footprint.kv_startup)),
    )?;
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
    let weights_hash = hash_weights.then(|| dir.weights.spawn_content_hash());
    let mut progress = Progress::new(*style, chrome);
    let model = Engine::load(&dir, selected.backend, precision.weights, |done, total| {
        progress.update(done, total)
    })?
    .with_kv_format(precision.kv)?;
    progress.finish();
    Ok(Loaded {
        name: model_name(path),
        took: progress.elapsed(),
        dir,
        model,
        footprint,
        ctx,
        gpu_free_before,
        weights_hash,
    })
}
