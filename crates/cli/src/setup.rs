//! What `marin` settles before the first prompt: which model, with which weights and settings. Every choice comes
//! from a flag, else the saved config, else the folder or the hardware; nothing a flag said is changed.

use std::path::{Path, PathBuf};

use clap::ArgMatches;
use clap::ValueEnum;
use clap::parser::ValueSource;
use loader::ModelDir;
use model::{BackendChoice, QuantBits, QuantScheme};

use crate::CliError;
use crate::arg_types::KvTypeArg;
use crate::args::BackendArg;
use crate::config::{Config, Home};
use crate::load::ChatPrecision;
use crate::marin_args::Options;
use crate::session::Settings;

/// Chat keeps more history than a one-off completion, so it gets a larger window by default.
pub const CHAT_CTX: usize = 4096;
/// Code mode keeps tool results in the conversation, so it starts with a larger window (reduced if memory is short).
pub const CODE_CTX: usize = 8192;

pub fn backend_choice(arg: BackendArg) -> BackendChoice {
    match arg {
        BackendArg::Auto => BackendChoice::Auto,
        BackendArg::Cpu => BackendChoice::Cpu,
        BackendArg::Cuda => BackendChoice::Cuda,
    }
}

pub fn settings(options: &Options) -> Result<Settings, CliError> {
    let settings = Settings {
        max_tokens: options.max_tokens,
        temperature: options.temp,
        top_k: options.top_k,
        top_p: options.top_p,
        seed: options.seed,
    };
    settings.sampler_config(0).validate()?;
    Ok(settings)
}

pub fn absolute(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Which of the settings that the config can remember were given on the command line.
#[derive(Debug, Default, Clone, Copy)]
pub struct Explicit {
    pub backend: bool,
    pub temp: bool,
    pub max_tokens: bool,
    pub quantize: bool,
}

impl Explicit {
    pub fn from_matches(matches: &ArgMatches) -> Self {
        // A global flag typed after the subcommand is recorded with that subcommand.
        let given = |id: &str| {
            let mut level = Some(matches);
            while let Some(m) = level {
                if m.try_contains_id(id).unwrap_or(false)
                    && m.value_source(id) == Some(ValueSource::CommandLine)
                {
                    return true;
                }
                level = m.subcommand().map(|(_, sub)| sub);
            }
            false
        };
        Self {
            backend: given("backend"),
            temp: given("temp"),
            max_tokens: given("max_tokens"),
            quantize: given("quantize") || given("quant_group"),
        }
    }
}

/// Fills in what the flags left open from the saved config.
pub fn apply_config(options: &mut Options, explicit: &Explicit, config: &Config) {
    if !explicit.backend
        && let Some(backend) = config
            .backend
            .as_deref()
            .and_then(|name| BackendArg::from_str(name, true).ok())
    {
        options.backend = backend;
    }
    if !explicit.temp
        && let Some(temp) = config.temp
    {
        options.temp = temp;
    }
    if !explicit.max_tokens
        && let Some(max_tokens) = config.max_tokens
    {
        options.max_tokens = max_tokens;
    }
}

/// The weights and KV format the flags ask for.
pub fn asked_precision(options: &Options) -> Result<ChatPrecision, CliError> {
    Ok(ChatPrecision {
        weights: options.quantize.scheme(options.quant_group)?,
        kv: options.kv_type.map(KvTypeArg::format),
    })
}

/// The group size of a packed 8-bit model, which can only be read as 8-bit.
pub fn packed_group(model: &Path) -> Option<usize> {
    ModelDir::open(model).ok()?.weights.packed_group()
}

fn int8(group: usize) -> Option<QuantScheme> {
    QuantScheme::new(QuantBits::Int8, group).ok()
}

/// What to load `model` with: as the flags ask. A packed model also loads the way the user opted into when they
/// picked it, which the config remembers for that model; nothing else changes the precision.
pub fn precision_for(
    asked: ChatPrecision,
    explicit: &Explicit,
    model: &Path,
    saved: &Config,
) -> ChatPrecision {
    let remembered = saved
        .model
        .as_deref()
        .is_some_and(|m| absolute(m) == absolute(model))
        && saved.quantize.as_deref() == Some("int8");
    match saved.quant_group {
        Some(group) if remembered && !explicit.quantize => ChatPrecision {
            weights: int8(group),
            ..asked
        },
        _ => asked,
    }
}

/// The model came from a list that said it is packed 8-bit: choosing it is the opt-in.
pub fn precision_for_pick(asked: ChatPrecision, model: &Path) -> ChatPrecision {
    match packed_group(model) {
        Some(group) => ChatPrecision {
            weights: int8(group),
            ..asked
        },
        None => asked,
    }
}

pub fn chat_precision(
    options: &Options,
    explicit: &Explicit,
    model: &Path,
    saved: &Config,
) -> Result<ChatPrecision, CliError> {
    Ok(precision_for(
        asked_precision(options)?,
        explicit,
        model,
        saved,
    ))
}

/// Remembers the model for next time, with the 8-bit opt-in it needed, if any. A failure to write is not worth
/// stopping for.
pub fn remember(home: Option<&Home>, path: &Path, weights: Option<QuantScheme>) {
    let Some(home) = home else { return };
    let packed = packed_group(path).and(weights).map(|w| w.group);
    let _ = home.update_config(|c| {
        c.model = Some(path.to_path_buf());
        c.quantize = packed.map(|_| "int8".to_string());
        c.quant_group = packed;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marin_args::MarinCli;
    use clap::{CommandFactory, FromArgMatches};

    fn parse(args: &[&str]) -> (Options, Explicit) {
        let matches = MarinCli::command().get_matches_from(args);
        let cli = MarinCli::from_arg_matches(&matches).unwrap();
        (cli.options, Explicit::from_matches(&matches))
    }

    fn saved() -> Config {
        Config {
            model: Some(PathBuf::from("/models/packed")),
            backend: Some("cpu".into()),
            temp: Some(0.2),
            max_tokens: Some(3),
            quantize: Some("int8".into()),
            quant_group: Some(64),
        }
    }

    #[test]
    fn a_flag_is_explicit_before_or_after_the_subcommand_and_defaults_are_not() {
        let (_, none) = parse(&["marin"]);
        assert!(!none.backend && !none.temp && !none.max_tokens && !none.quantize);
        let (_, before) = parse(&["marin", "--backend", "cpu", "run", "x"]);
        assert!(before.backend && !before.temp);
        let (_, after) = parse(&["marin", "run", "--temp", "0.9", "--max-tokens", "5", "x"]);
        assert!(after.temp && after.max_tokens && !after.backend);
        let (_, quant) = parse(&["marin", "code", "--quantize", "int8"]);
        assert!(quant.quantize);
    }

    #[test]
    fn saved_settings_fill_gaps_and_flags_always_win() {
        let (mut options, explicit) = parse(&["marin"]);
        apply_config(&mut options, &explicit, &saved());
        assert_eq!((options.temp, options.max_tokens), (0.2, 3));
        assert_eq!(options.backend, BackendArg::Cpu);
        let (mut options, explicit) = parse(&["marin", "--temp", "0.9", "--backend", "cuda"]);
        apply_config(&mut options, &explicit, &saved());
        assert_eq!((options.temp, options.max_tokens), (0.9, 3));
        assert_eq!(options.backend, BackendArg::Cuda);
        let (mut options, explicit) = parse(&["marin"]);
        apply_config(
            &mut options,
            &explicit,
            &Config {
                backend: Some("nonsense".into()),
                ..Config::default()
            },
        );
        assert_eq!(
            options.backend,
            BackendArg::Auto,
            "an unreadable saved value is ignored"
        );
    }

    #[test]
    fn a_remembered_packed_opt_in_follows_its_model_and_never_overrides_a_flag() {
        let (options, explicit) = parse(&["marin"]);
        let asked = asked_precision(&options).unwrap();
        let packed = Path::new("/models/packed");
        let weights = precision_for(asked, &explicit, packed, &saved()).weights;
        assert_eq!(
            weights.map(|w| (w.bits, w.group)),
            Some((QuantBits::Int8, 64))
        );
        let other = precision_for(asked, &explicit, Path::new("/models/other"), &saved());
        assert!(
            other.weights.is_none(),
            "the opt-in belongs to the model it was given for"
        );
        let (options, explicit) = parse(&["marin", "--quantize", "int8", "--quant-group", "32"]);
        let flagged = precision_for(
            asked_precision(&options).unwrap(),
            &explicit,
            packed,
            &saved(),
        );
        assert_eq!(flagged.weights.map(|w| w.group), Some(32));
        let nothing = precision_for(asked, &explicit, packed, &Config::default());
        assert!(
            nothing.weights.is_none(),
            "without an opt-in nothing changes the precision"
        );
    }

    #[test]
    fn a_folder_that_is_not_a_model_has_no_packed_format() {
        assert_eq!(packed_group(Path::new("/definitely/not/here")), None);
    }
}
