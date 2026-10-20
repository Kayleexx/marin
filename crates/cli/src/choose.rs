//! Which model `marin` talks to: the one named, the one saved, the only one found, or the one the user picks from
//! what the planner says fits this machine. A terminal with nothing to pick from is asked for a folder.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use model::{Backend, Precision};

use crate::CliError;
use crate::config::{self, Choice};
use crate::marin_args::Options;
use crate::setup::{
    CHAT_CTX, Explicit, asked_precision, backend_choice, packed_group, precision_for_pick,
};
use crate::style::Style;
use crate::term::menu::Item;
use crate::term::picker;
use crate::{plan, units};

const ASKS: usize = 3;

/// A packed 8-bit model chosen by a name, not from a list: on a terminal the user is asked once, and the answer is
/// the opt-in. True to load it as 8-bit.
pub fn confirm_packed(model: &Path, group: usize, style: Style) -> Result<bool, CliError> {
    print!(
        "  {} is stored as packed 8-bit weights (groups of {group}) and can only be loaded that way. Load it? {} ",
        style.bold(&config::display_name(model)),
        style.dim("[Y/n]")
    );
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_lowercase().as_str(),
        "" | "y" | "yes"
    ))
}

/// The model a launch would load without asking anyone: the one named, the saved one, or the only one found.
pub fn unambiguous(options: &Options, saved: Option<&Path>) -> Option<PathBuf> {
    let remembered = saved.filter(|path| config::is_model_dir(path));
    match config::choose_chat_model(options.model.as_deref(), remembered, &config::model_roots()) {
        Ok(Choice::Use(path)) => Some(path),
        Ok(Choice::Pick(found)) if found.len() == 1 => found.into_iter().next(),
        _ => None,
    }
}

/// Whether a running server can stand in for loading a model here, or why not. It must run the model this launch
/// would load, in a window code mode can work in, and nothing may have been asked of the model that a server cannot
/// honor (backend, window, weights, KV format, snapshot).
pub fn server_fits(
    options: &Options,
    explicit: &Explicit,
    wanted: Option<&Path>,
    (model, context): (&str, usize),
) -> Result<(), String> {
    if explicit.backend
        || explicit.quantize
        || options.ctx.is_some()
        || options.kv_type.is_some()
        || options.snapshot.is_some()
    {
        return Err(
            "you asked for a backend, window, weights or KV format a running server cannot change"
                .into(),
        );
    }
    let Some(wanted) = wanted else {
        return Err("it is not clear which model you want; pass --model".into());
    };
    let name = crate::load::model_name(wanted);
    if name != model {
        return Err(format!("it runs {model}, not {name}"));
    }
    if context < crate::code::MIN_CONTEXT {
        return Err(format!(
            "its window is {context} tokens, code mode needs {}",
            crate::code::MIN_CONTEXT
        ));
    }
    Ok(())
}

/// A model chosen from a list, and whether it was a packed 8-bit one (so choosing it was the opt-in).
pub struct Chosen {
    pub path: PathBuf,
    pub picked_packed: bool,
}

/// Explicit `--model`, then the saved model, then the chat models found on disk (the only one, or the user's pick).
/// Without a model anywhere, a terminal asks for a folder and a pipe gets an error that says what to do.
pub fn choose_model(
    options: &Options,
    saved: Option<&Path>,
    interactive: bool,
    style: Style,
) -> Result<Chosen, CliError> {
    let remembered = saved.filter(|path| config::is_model_dir(path));
    if let Some(gone) = saved.filter(|path| !config::is_model_dir(path))
        && options.model.is_none()
    {
        eprintln!(
            "{} the saved model {} is no longer there",
            style.yellow("note:"),
            gone.display()
        );
    }
    let roots = config::model_roots();
    let found = match config::choose_chat_model(options.model.as_deref(), remembered, &roots) {
        Ok(Choice::Use(path)) => {
            return Ok(Chosen {
                path,
                picked_packed: false,
            });
        }
        Ok(Choice::Pick(found)) => found,
        Err(CliError::NoChatModel { .. }) if interactive => {
            return ask_for_model(&roots, style).map(listed);
        }
        Err(e) => return Err(e),
    };
    match found.as_slice() {
        [only] => Ok(Chosen {
            path: only.clone(),
            picked_packed: false,
        }),
        _ if !interactive => {
            let names: Vec<String> = found.iter().map(|p| config::display_name(p)).collect();
            Err(CliError::ChooseModel(names.join(", ")))
        }
        _ => {
            eprintln!("{}", style.dim("Checking which models fit this machine…"));
            let items = ranked(found, options);
            picker::pick(style, "Choose a model:", items)?
                .map(listed)
                .ok_or(CliError::Cancelled)
        }
    }
}

fn listed(path: PathBuf) -> Chosen {
    let picked_packed = packed_group(&path).is_some();
    Chosen {
        path,
        picked_packed,
    }
}

/// The models with what the planner says about each, the ones that fit the GPU first and larger before smaller.
fn ranked(found: Vec<PathBuf>, options: &Options) -> Vec<Item<PathBuf>> {
    let mut rows: Vec<(u8, u64, Item<PathBuf>)> = found
        .into_iter()
        .map(|path| {
            let verdict = fit(&path, options);
            let tag = if packed_group(&path).is_some() {
                "packed 8-bit · "
            } else {
                ""
            };
            let (rank, weights, detail) = match verdict {
                Ok((true, weights)) => (
                    0,
                    weights,
                    format!("{tag}fits the GPU · {}", units::bytes(weights)),
                ),
                Ok((false, weights)) => (
                    1,
                    weights,
                    format!("{tag}runs on the CPU · {}", units::bytes(weights)),
                ),
                Err(why) => (2, 0, format!("will not fit: {why}")),
            };
            let item = Item {
                label: config::display_name(&path),
                detail,
                value: path,
            };
            (rank, weights, item)
        })
        .collect();
    rows.sort_by_key(|row| (row.0, std::cmp::Reverse(row.1)));
    rows.into_iter().map(|(_, _, item)| item).collect()
}

/// `(on the GPU, weight bytes)` from the same planner a load would use, or the first words of why it cannot run.
fn fit(path: &Path, options: &Options) -> Result<(bool, u64), String> {
    let precision = asked_precision(options).map_err(|e| e.to_string())?;
    let precision = precision_for_pick(precision, path);
    let precision = Precision {
        weights: precision.weights,
        kv: precision.kv.unwrap_or(model::KvFormat::F16),
    };
    let planned = plan::build(
        path,
        options.ctx,
        CHAT_CTX,
        backend_choice(options.backend),
        precision,
        options.kv_type.is_none(),
    )
    .map_err(|e| match e {
        CliError::NoFit { .. } | CliError::NotEnoughMemory { .. } => {
            "too big for this machine's memory (try --quantize int8)".to_string()
        }
        other => other.to_string().chars().take(70).collect(),
    })?;
    Ok((
        matches!(planned.selected.backend, Backend::Cuda(_)),
        planned.footprint.weights,
    ))
}

/// First run without any model on disk: asks for a folder a few times.
fn ask_for_model(roots: &[PathBuf], style: Style) -> Result<PathBuf, CliError> {
    let places: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
    println!(
        "\n  {}\n  {}\n",
        style.bold("No chat model found."),
        style.dim(&format!(
            "Looked in {}. Download one (for example Qwen/Qwen3-1.7B) or give the folder of a model you have.",
            if places.is_empty() { "nowhere".to_string() } else { places.join(", ") }
        ))
    );
    let stdin = io::stdin();
    for _ in 0..ASKS {
        print!("  model folder (Enter to quit): ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Err(CliError::Cancelled);
        }
        let path = expand_home(line.trim());
        match check_chat_model(&path) {
            Ok(()) => return Ok(path),
            Err(e) => println!("  {} {e}", style.red("error:")),
        }
    }
    Err(CliError::NoModel)
}

fn expand_home(text: &str) -> PathBuf {
    match (text.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(text),
    }
}

pub fn check_chat_model(path: &Path) -> Result<(), CliError> {
    if !config::is_model_dir(path) {
        return Err(CliError::NotAModel(path.display().to_string()));
    }
    if config::model_kind(path) != chat::ModelKind::Chat {
        return Err(CliError::NotAChatModel(config::display_name(path)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marin_args::MarinCli;
    use clap::{CommandFactory, FromArgMatches};

    fn fits(
        args: &[&str],
        wanted: Option<&str>,
        model: &str,
        context: usize,
    ) -> Result<(), String> {
        let matches = MarinCli::command().get_matches_from(args);
        let options = MarinCli::from_arg_matches(&matches).unwrap().options;
        let explicit = Explicit::from_matches(&matches);
        server_fits(&options, &explicit, wanted.map(Path::new), (model, context))
    }

    #[test]
    fn a_server_running_the_wanted_model_is_used() {
        assert!(
            fits(
                &["marin", "code"],
                Some("/m/Qwen3.5-4B-Q8"),
                "Qwen3.5-4B-Q8",
                8192
            )
            .is_ok()
        );
        assert!(
            fits(
                &["marin", "--temp", "0.2", "code"],
                Some("/m/Qwen3.5-4B-Q8"),
                "Qwen3.5-4B-Q8",
                8192
            )
            .is_ok()
        );
    }

    #[test]
    fn a_different_model_is_never_picked_up_silently() {
        let why = fits(
            &["marin", "code"],
            Some("/m/Qwen3-1.7B"),
            "Qwen3.5-4B-Q8",
            8192,
        )
        .unwrap_err();
        assert!(why.contains("runs Qwen3.5-4B-Q8, not Qwen3-1.7B"), "{why}");
        assert!(
            fits(&["marin", "code"], None, "Qwen3.5-4B-Q8", 8192)
                .unwrap_err()
                .contains("not clear")
        );
    }

    #[test]
    fn what_a_server_cannot_honor_keeps_the_model_local() {
        for flags in [
            vec!["marin", "--backend", "cpu", "code"],
            vec!["marin", "code", "--ctx", "4096"],
            vec!["marin", "code", "--quantize", "int8"],
            vec!["marin", "code", "--kv-type", "f16"],
        ] {
            let why = fits(&flags, Some("/m/Qwen3.5-4B-Q8"), "Qwen3.5-4B-Q8", 8192).unwrap_err();
            assert!(why.contains("cannot change"), "{flags:?}: {why}");
        }
        let small = fits(
            &["marin", "code"],
            Some("/m/Qwen3.5-4B-Q8"),
            "Qwen3.5-4B-Q8",
            1024,
        )
        .unwrap_err();
        assert!(small.contains("1024"), "{small}");
    }
}
