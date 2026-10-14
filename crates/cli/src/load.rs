use std::path::Path;
use std::time::Duration;

use loader::ModelDir;
use model::{Backend, BackendChoice, Engine, KvFormat, Precision, QuantScheme};

use crate::CliError;
use crate::memory::Footprint;
use crate::progress::Progress;
use crate::style::Style;
use crate::units;

pub struct Loaded {
    pub dir: ModelDir,
    pub model: Engine,
    pub name: String,
    pub footprint: Footprint,
    /// How the weights and the KV cache are stored, after the backend was chosen.
    pub precision: Precision,
    pub ctx: usize,
    pub took: Duration,
    /// Free GPU memory right before the weights were uploaded, when running on the GPU.
    pub gpu_free_before: Option<u64>,
    /// Running while the weights load, when a snapshot was asked for.
    pub weights_hash: Option<std::thread::JoinHandle<Result<u64, loader::LoaderError>>>,
}

impl Loaded {
    pub fn stop_tokens(&self) -> Vec<u32> {
        let mut stops = self.model.config().eos_tokens.clone();
        if chat::classify(self.dir.tokenizer_config_json.as_deref()) == chat::ModelKind::Chat
            && let Some(id) = self.dir.tokenizer.token_id("<|im_end|>")
            && !stops.contains(&id)
        {
            stops.push(id);
        }
        stops
    }
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
    /// The KV type was not asked for: keep `precision.kv` on the GPU and use f32 on the CPU.
    kv_follows_backend: bool,
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
        kv_follows_backend: false,
    })
}

/// Like `load`, and hashes the weight files while they load when the selected GPU backend supports snapshots.
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
        kv_follows_backend: false,
    })
}

/// What the chat user asked for: `kv` is `None` for f16 on the GPU and f32 on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct ChatPrecision {
    pub weights: Option<QuantScheme>,
    pub kv: Option<KvFormat>,
}

pub fn load_chat(
    path: &Path,
    (ctx, default_ctx): (Option<usize>, usize),
    backend: BackendChoice,
    choice: ChatPrecision,

    style: &Style,
    (chrome, hash_weights): (bool, bool),
) -> Result<Loaded, CliError> {
    open(Request {
        path,
        ctx,
        default_ctx,
        backend,
        precision: Precision {
            weights: choice.weights,
            kv: choice.kv.unwrap_or(KvFormat::F16),
        },
        style,
        chrome,
        hash_weights,
        kv_follows_backend: choice.kv.is_none(),
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
        kv_follows_backend,
    } = request;
    let planned = crate::plan::build(
        path,
        ctx,
        default_ctx,
        backend,
        precision,
        kv_follows_backend,
    )?;
    let crate::plan::Plan {
        dir,
        selected,
        footprint,
        precision,
        ctx,
        notes,
        ..
    } = planned;
    for note in &notes {
        if chrome || note.starts_with("default context reduced") {
            eprintln!("{} {note}", style.yellow("note:"));
        }
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
    let weights_hash = (hash_weights && matches!(&selected.backend, Backend::Cuda(_)))
        .then(|| dir.weights.spawn_content_hash());
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
        precision,
        ctx,
        gpu_free_before,
        weights_hash,
    })
}
