use std::path::Path;

use loader::ModelDir;
use model::{Precision, Qwen3Config};

use crate::CliError;
use crate::load::model_name;
use crate::memory::{DEFAULT_CTX, Footprint, footprint, resolve_ctx};
use crate::style::Style;
use crate::units;

pub fn is_base_model(name: &str) -> bool {
    name.to_lowercase().contains("base")
}

/// The same facts `inspect` prints, shared with the `/model` command.
pub fn describe(
    name: &str,
    cfg: &Qwen3Config,
    vocab_known: usize,
    ctx: usize,
    fp: &Footprint,
) -> String {
    let params = cfg.parameter_count().map_or_else(
        || "unknown".to_string(),
        |p| format!("{:.2} billion", p as f64 / 1e9),
    );
    let share = cfg.heads.checked_div(cfg.kv_heads).unwrap_or(1);
    let kind = if is_base_model(name) {
        "base model: it continues text, it does not follow instructions"
    } else {
        "plain-text completion (chat templates are not supported yet)"
    };
    let rows = [
        (
            "parameters",
            format!("{params}   (the numbers the model learned)"),
        ),
        (
            "weights",
            format!(
                "{} stored as bf16 (16-bit numbers)",
                units::bytes(fp.weights)
            ),
        ),
        (
            "layers",
            format!("{}   (each layer = attention + MLP)", cfg.layers),
        ),
        (
            "hidden size",
            format!(
                "{}   (numbers that represent one token)",
                units::count(cfg.hidden_size as u64)
            ),
        ),
        (
            "attention",
            format!(
                "{} query heads, {} key/value heads, head size {}\n{:16}each key/value head serves {share} query heads (grouped-query attention)",
                cfg.heads, cfg.kv_heads, cfg.head_dim, ""
            ),
        ),
        ("mlp size", units::count(cfg.intermediate_size as u64)),
        (
            "vocabulary",
            format!(
                "{} ids, tokenizer knows {}",
                units::count(cfg.vocab_size as u64),
                units::count(vocab_known as u64)
            ),
        ),
        (
            "context",
            format!(
                "up to {} tokens, {} in this session",
                units::count(cfg.max_positions as u64),
                units::count(ctx as u64)
            ),
        ),
        (
            "kv cache",
            format!(
                "up to {} for this context, {} before the first token on the GPU   (remembers earlier tokens so they are not recomputed)",
                units::bytes(fp.kv),
                units::bytes(fp.kv_startup)
            ),
        ),
        (
            "output layer",
            if cfg.tie_embeddings {
                "shares the embedding table".to_string()
            } else {
                "its own matrix".to_string()
            },
        ),
        ("kind", kind.to_string()),
    ];
    rows.iter()
        .map(|(label, value)| format!("  {label:<13} {value}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn run(
    path: &Path,
    ctx: Option<usize>,
    precision: Precision,
    style: &Style,
) -> Result<(), CliError> {
    let dir = ModelDir::open(path)?;
    let cfg = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
    let ctx = resolve_ctx(&cfg, ctx, DEFAULT_CTX)?;
    let fp = footprint(&dir, &cfg, ctx, precision)?;
    let name = model_name(path);
    println!("{}", style.bold(&name));
    println!("  A decoder-only transformer: it reads tokens and predicts the next one.\n");
    println!(
        "{}\n",
        describe(&name, &cfg, dir.tokenizer.vocab_size(), ctx, &fp)
    );
    if let Some(scheme) = precision.weights {
        println!(
            "  quantized     {}: the weights take {} in memory\n",
            scheme.label(),
            units::bytes(fp.weights)
        );
    }
    let needed = format!("about {}", units::bytes(fp.needed));
    match fp.available {
        Some(available) if available < fp.needed => println!(
            "  memory        needs {needed}, only {} free  {}",
            units::bytes(available),
            style.red("does not fit")
        ),
        Some(available) => println!(
            "  memory        needs {needed}, {} free  {}",
            units::bytes(available),
            style.green("fits")
        ),
        None => println!("  memory        needs {needed} (free memory unknown on this system)"),
    }
    Ok(())
}
