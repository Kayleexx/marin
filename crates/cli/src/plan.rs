use std::path::Path;

use loader::ModelDir;
use model::{Backend, BackendChoice, Engine, KvFormat, Precision, Qwen3Config, Selected};

use crate::CliError;
use crate::load::ChatPrecision;
use crate::memory::{Footprint, footprint, resolve_ctx};
use crate::units;

pub(crate) struct Plan {
    pub dir: ModelDir,
    pub config: Qwen3Config,
    pub selected: Selected,
    pub footprint: Footprint,
    pub precision: Precision,
    pub ctx: usize,
    pub notes: Vec<String>,
    pub gpu_startup: u64,
    pub gpu_window: u64,
    pub host_staging: u64,
}

fn staging_bytes(dir: &ModelDir, precision: Precision) -> u64 {
    let hybrid = model::qwen35::is_qwen35(&dir.config_json);
    let largest = dir
        .weights
        .names()
        .filter(|name| !hybrid || name.starts_with("model.language_model."))
        .filter_map(|name| dir.weights.info(name))
        .map(|info| info.byte_len())
        .max()
        .unwrap_or(0);
    let copies = if precision.weights.is_some() && dir.weights.packed_group().is_none() {
        2
    } else {
        1
    };
    largest
        .saturating_mul(copies)
        .saturating_add(512 * 1024 * 1024)
}

fn fit_hybrid_cpu(
    dir: &ModelDir,
    config: &Qwen3Config,
    requested: Option<usize>,
    ctx: usize,
    precision: Precision,
    usage: Footprint,
) -> Result<(usize, Footprint), CliError> {
    let Some(available) = usage.available else {
        return Ok((ctx, usage));
    };
    if usage.needed <= available || requested.is_some() {
        usage.ensure_fits()?;
        return Ok((ctx, usage));
    }
    let mut low = 1;
    let mut high = ctx;
    let mut best = None;
    while low <= high {
        let middle = low.saturating_add(high.saturating_sub(low) / 2);
        let mut candidate = footprint(dir, config, middle, precision)?;
        candidate.available = Some(available);
        if candidate.needed <= available {
            best = Some((middle, candidate));
            low = middle.saturating_add(1);
        } else {
            high = middle.saturating_sub(1);
        }
    }
    match best {
        Some(fit) => Ok(fit),
        None => Err(CliError::NotEnoughMemory {
            needed: footprint(dir, config, 1, precision)?.needed,
            available,
        }),
    }
}

fn cpu_failure(choice: BackendChoice, notes: &[String], error: CliError) -> CliError {
    if choice == BackendChoice::Auto && !notes.is_empty() {
        CliError::NoFit {
            gpu: notes.join("; "),
            cpu: error.to_string(),
        }
    } else {
        error
    }
}

pub(crate) fn build(
    path: &Path,
    requested_ctx: Option<usize>,
    default_ctx: usize,
    choice: BackendChoice,
    mut precision: Precision,
    kv_follows_backend: bool,
) -> Result<Plan, CliError> {
    if choice == BackendChoice::Cpu {
        if precision
            .weights
            .is_some_and(|q| q.bits != model::QuantBits::Int8)
        {
            return Err(CliError::BadOption(
                "CPU weights support int8 only; int4 needs the GPU".into(),
            ));
        }
        if precision.kv != KvFormat::default() && !kv_follows_backend {
            return Err(CliError::BadOption(
                "--kv-type needs the GPU, but --backend cpu was given".into(),
            ));
        }
    }
    let dir = ModelDir::open(path)?;
    let config = Engine::parse_config(&dir)?;
    let hybrid = model::qwen35::is_qwen35(&dir.config_json);
    let granite = model::granite::is_granite(&dir.config_json);
    if granite {
        if choice == BackendChoice::Cuda && precision.weights.is_some() {
            return Err(CliError::BadOption(
                "Granite CUDA currently supports BF16 weights only".into(),
            ));
        }
        if precision.kv != KvFormat::F32 && !kv_follows_backend {
            return Err(CliError::BadOption("Granite MoE requires FP32 KV".into()));
        }
    }
    if hybrid && !kv_follows_backend && precision.kv != KvFormat::F32 {
        return Err(CliError::BadOption("Qwen3.5 requires FP32 state".into()));
    }
    let mut ctx = resolve_ctx(&config, requested_ctx, default_ctx)?;
    let mut usage = footprint(&dir, &config, ctx, precision)?;
    let (startup, window) = if hybrid {
        let text = model::qwen35::TextConfig::parse(
            &dir.config_json,
            dir.generation_config_json.as_deref(),
        )?;
        model::qwen35::gpu_budget_bytes(&text, ctx)?
    } else if granite {
        let c = model::granite::Config::parse(&dir.config_json)?;
        let workspace = model::granite::gpu_workspace_bytes(&c, ctx)?;
        let (startup, window) = model::granite::gpu_kv_budget(&c, ctx)?;
        (
            startup.saturating_add(workspace),
            window.saturating_add(workspace),
        )
    } else {
        (usage.kv_startup, usage.kv_window.max(usage.kv_startup))
    };
    let gpu_startup = usage.weights.saturating_add(startup);
    let gpu_window = usage.weights.saturating_add(window);
    let mut selected = if granite && choice == BackendChoice::Auto {
        Selected {
            backend: Backend::Cpu,
            note: Some(if precision.weights.is_some() {
                "Granite CUDA BF16 does not support INT8; using the CPU".into()
            } else {
                "Granite CUDA BF16 is experimental; use --backend cuda to select it".into()
            }),
        }
    } else {
        Backend::select(choice, gpu_startup, gpu_window)?
    };
    let host_staging = staging_bytes(&dir, precision);
    let mut notes = Vec::new();
    if let Some(note) = selected.note.take() {
        notes.push(note);
    }
    if matches!(selected.backend, Backend::Cuda(_))
        && let Some(available) = metrics::cpu_available_bytes()?
        && available < host_staging
    {
        let reason = format!(
            "CUDA upload needs {} host staging but only {} is available",
            units::bytes(host_staging),
            units::bytes(available)
        );
        if choice != BackendChoice::Auto {
            return Err(CliError::NotEnoughMemory {
                needed: host_staging,
                available,
            });
        }
        notes.push(format!("{reason}; using the CPU"));
        selected.backend = Backend::Cpu;
    }
    if matches!(selected.backend, Backend::Cuda(_)) {
        if hybrid || granite {
            usage.kv_startup = startup;
            usage.kv_window = window;
        }
    } else {
        if precision
            .weights
            .is_some_and(|q| q.bits != model::QuantBits::Int8)
        {
            return Err(cpu_failure(
                choice,
                &notes,
                CliError::BadOption("CPU weights support int8 only; int4 needs the GPU".into()),
            ));
        }
        if precision.kv != KvFormat::default() && !kv_follows_backend {
            return Err(cpu_failure(
                choice,
                &notes,
                CliError::BadOption("--kv-type needs the GPU, but no CUDA backend fits".into()),
            ));
        }
        let cpu_fit = if hybrid || granite {
            fit_hybrid_cpu(&dir, &config, requested_ctx, ctx, precision, usage)
        } else {
            usage
                .fit_cpu(&config, ctx, requested_ctx.is_none())
                .map(|chosen| (chosen, usage))
        };
        (ctx, usage) = cpu_fit.map_err(|error| cpu_failure(choice, &notes, error))?;
        if requested_ctx.is_none() && ctx < default_ctx.min(config.max_positions) {
            notes.push(format!(
                "default context reduced to {ctx} tokens to fit available CPU memory"
            ));
        }
    }
    if kv_follows_backend {
        precision.kv = if hybrid || granite || matches!(selected.backend, Backend::Cpu) {
            KvFormat::F32
        } else {
            precision.kv
        };
    }
    Ok(Plan {
        dir,
        config,
        selected,
        footprint: usage,
        precision,
        ctx,
        notes,
        gpu_startup,
        gpu_window,
        host_staging,
    })
}

pub fn run(
    path: &Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    precision: Precision,
) -> Result<(), CliError> {
    display(build(path, ctx, default_ctx, backend, precision, false)?)
}

pub fn run_chat(
    path: &Path,
    ctx: Option<usize>,
    default_ctx: usize,
    backend: BackendChoice,
    choice: ChatPrecision,
) -> Result<(), CliError> {
    display(build(
        path,
        ctx,
        default_ctx,
        backend,
        Precision {
            weights: choice.weights,
            kv: choice.kv.unwrap_or(KvFormat::F16),
        },
        choice.kv.is_none(),
    )?)
}

fn display(plan: Plan) -> Result<(), CliError> {
    let format = match (plan.dir.weights.packed_group(), plan.precision.weights) {
        (Some(group), _) => format!("packed INT8, group {group}"),
        (None, Some(q)) => format!("BF16 source, {} at load", q.label()),
        (None, None) => "BF16 source".to_string(),
    };
    let workspace = if model::qwen35::is_qwen35(&plan.dir.config_json) {
        let text = model::qwen35::TextConfig::parse(
            &plan.dir.config_json,
            plan.dir.generation_config_json.as_deref(),
        )?;
        text.workspace_bytes(plan.ctx, plan.ctx)? as u64
    } else {
        model::cpu_workspace_bytes(&plan.config, plan.ctx)?
    };
    let backend_name = match &plan.selected.backend {
        Backend::Cpu => "cpu",
        Backend::Cuda(_) => "cuda",
    };
    println!("model       {}", plan.dir.path.display());
    println!("artifact    {format}");
    println!("backend     {backend_name}");
    if backend_name == "cuda" {
        println!(
            "threads     GPU execution; {} CPU fallback prefill",
            tensor::threads()
        );
    } else {
        println!("threads     {} CPU prefill", tensor::threads());
    }
    println!("context     {} tokens", plan.ctx);
    println!("weights     {}", units::bytes(plan.footprint.weights));
    if plan.footprint.recurrent > 0 {
        println!(
            "recurrent   {} fixed state",
            units::bytes(plan.footprint.recurrent)
        );
        println!(
            "KV window   {}",
            units::bytes(plan.footprint.kv.saturating_sub(plan.footprint.recurrent))
        );
        println!(
            "rollback    {} state transaction reserve",
            units::bytes(plan.footprint.kv)
        );
    } else {
        println!("KV window   {}", units::bytes(plan.footprint.kv));
    }
    if backend_name == "cuda" {
        println!("CPU work    {} fallback prefill", units::bytes(workspace));
    } else {
        println!("workspace   {} full prefill", units::bytes(workspace));
    }
    println!("safety      {}", units::bytes(model::CPU_SLACK_BYTES));
    println!(
        "CPU-only need {} estimated",
        units::bytes(plan.footprint.needed)
    );
    match plan.footprint.available {
        Some(bytes) => println!("RAM free    {}", units::bytes(bytes)),
        None => println!("RAM free    unknown; fit is unverified"),
    }
    if let Backend::Cuda(gpu) = &plan.selected.backend {
        let free = u64::try_from(gpu.info().map_err(model::ModelError::from)?.free_bytes)
            .unwrap_or(u64::MAX);
        println!(
            "GPU budget  {} startup / {} full window + 512 MiB safety",
            units::bytes(plan.gpu_startup),
            units::bytes(plan.gpu_window)
        );
        println!("VRAM free   {}", units::bytes(free));
        println!("host stage  {}", units::bytes(plan.host_staging));
        if free < plan.gpu_window.saturating_add(512 * 1024 * 1024) {
            println!("window      startup fits; full context is not guaranteed");
        }
    }
    for note in &plan.notes {
        println!("note        {note}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_default_context_fits_but_an_explicit_context_is_not_reduced() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35");
        let dir = ModelDir::open(&path).unwrap();
        let config = Engine::parse_config(&dir).unwrap();
        let precision = Precision::default();
        let minimum = footprint(&dir, &config, 1, precision).unwrap();
        let mut full = footprint(&dir, &config, 128, precision).unwrap();
        assert!(minimum.needed < full.needed);
        full.available = Some(minimum.needed);
        let (ctx, chosen) = fit_hybrid_cpu(&dir, &config, None, 128, precision, full).unwrap();
        assert!((1..128).contains(&ctx));
        assert!(chosen.needed <= minimum.needed);
        assert!(fit_hybrid_cpu(&dir, &config, Some(128), 128, precision, full).is_err());
    }
}
