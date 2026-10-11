#![forbid(unsafe_code)]

use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use model::BackendChoice;
use runtime::StopReason;

use cli::arg_types::KvTypeArg;
use cli::args::BackendArg;
use cli::chat::Chat;
use cli::config::{self, Choice, Config, Home};
use cli::error::CliError;
use cli::load::ChatPrecision;
use cli::marin_args::{MarinCli, MarinCommand, Options};
use cli::render::{MarkdownStream, MarkdownWriter, Theme};
use cli::session::{Session, Settings};
use cli::signal::{INTERRUPTED_EXIT_CODE, Interrupt};
use cli::style::{Style, stderr_is_terminal, stdin_is_terminal};
use cli::term::app::{self, Launch};
use cli::term::caps::Caps;
use cli::term::menu::Item;
use cli::term::picker;
use cli::term::screen;
use cli::term::splash;
use cli::{load, report, show};

/// Chat keeps more history than a one-off completion, so it gets a larger window by default.
const CHAT_CTX: usize = 4096;

fn main() -> ExitCode {
    let cli = MarinCli::parse();
    tensor::set_threads(cli.options.threads.unwrap_or(0));
    let style = Style::detect(cli.options.no_color);
    match run(cli, style) {
        Ok(code) => code,
        Err(e) if e.is_broken_pipe() => ExitCode::SUCCESS,
        Err(e) => {
            report(&e, &style);
            ExitCode::FAILURE
        }
    }
}

fn run(cli: MarinCli, style: Style) -> Result<ExitCode, CliError> {
    let home = Home::detect();
    match cli.command {
        Some(MarinCommand::Model { path }) => model_command(path.as_deref(), home.as_ref()),
        Some(MarinCommand::Show { path, kind }) => {
            show::run(path.as_deref(), kind.as_deref(), cli.options.no_color)
                .map(|()| ExitCode::SUCCESS)
        }
        Some(MarinCommand::Run { prompt }) => {
            one_shot(&prompt.join(" "), &cli.options, home, style)
        }
        None => session(&cli.options, home, style),
    }
}

fn backend_choice(arg: BackendArg) -> BackendChoice {
    match arg {
        BackendArg::Auto => BackendChoice::Auto,
        BackendArg::Cpu => BackendChoice::Cpu,
        BackendArg::Cuda => BackendChoice::Cuda,
    }
}

fn settings(options: &Options) -> Result<Settings, CliError> {
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

fn absolute(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn model_command(path: Option<&Path>, home: Option<&Home>) -> Result<ExitCode, CliError> {
    let Some(path) = path else {
        let config = home.map(Home::load_config).unwrap_or_default();
        match config.model {
            Some(model) => println!(
                "default: {} ({})",
                config::display_name(&model),
                model.display()
            ),
            None => println!("default: none yet"),
        }
        let (chat, base): (Vec<PathBuf>, Vec<PathBuf>) = config::discover(&config::model_roots())
            .into_iter()
            .partition(|p| config::model_kind(p) == chat::ModelKind::Chat);
        for found in chat {
            println!(
                "chat:    {} ({})",
                config::display_name(&found),
                found.display()
            );
        }
        for found in base {
            println!("base:    {} (use marin-run)", config::display_name(&found));
        }
        return Ok(ExitCode::SUCCESS);
    };
    if !config::is_model_dir(path) {
        return Err(CliError::NotAModel(path.display().to_string()));
    }
    if config::model_kind(path) != chat::ModelKind::Chat {
        return Err(CliError::NotAChatModel(config::display_name(path)));
    }
    let path = absolute(path);
    if let Some(home) = home {
        home.save_config(&Config {
            model: Some(path.clone()),
        })?;
    }
    println!("default model: {}", config::display_name(&path));
    Ok(ExitCode::SUCCESS)
}

/// Explicit `--model`, then the remembered default, then the chat models found on disk.
fn choose_model(
    options: &Options,
    home: Option<&Home>,
    interactive: bool,
    style: Style,
) -> Result<PathBuf, CliError> {
    let remembered = home.map(Home::load_config).unwrap_or_default().model;
    let found = match config::choose_chat_model(
        options.model.as_deref(),
        remembered.as_deref(),
        &config::model_roots(),
    )? {
        Choice::Use(path) => return Ok(path),
        Choice::Pick(found) => found,
    };
    match found.as_slice() {
        [only] if !interactive => Ok(only.clone()),
        _ if !interactive => {
            let names: Vec<String> = found.iter().map(|p| config::display_name(p)).collect();
            Err(CliError::ChooseModel(names.join(", ")))
        }
        _ => {
            let items = found
                .into_iter()
                .map(|path| Item {
                    label: config::display_name(&path),
                    detail: path.display().to_string(),
                    value: path,
                })
                .collect();
            picker::pick(style, "Choose a model:", items)?.ok_or(CliError::Cancelled)
        }
    }
}

fn remember(home: Option<&Home>, path: &Path) {
    let Some(home) = home else { return };
    if home.load_config().model.as_deref() != Some(path) {
        let _ = home.save_config(&Config {
            model: Some(path.to_path_buf()),
        });
    }
}

fn chat_precision(options: &Options) -> Result<ChatPrecision, CliError> {
    Ok(ChatPrecision {
        weights: options.quantize.scheme(options.quant_group)?,
        kv: options.kv_type.map(KvTypeArg::format),
    })
}

fn open(
    options: &Options,
    home: Option<&Home>,
    interactive: bool,
    style: Style,
) -> Result<Session, CliError> {
    let settings = settings(options)?;
    let path = absolute(&choose_model(options, home, interactive, style)?);
    let backend = backend_choice(options.backend);
    let loaded = load::load_chat(
        &path,
        (options.ctx, CHAT_CTX),
        backend,
        chat_precision(options)?,
        &style,
        (stderr_is_terminal(), options.snapshot.is_some()),
    )?;
    remember(home, &path);
    let mut session = Session::new(loaded, settings, style)?;
    if let Some(file) = &options.snapshot {
        session.use_snapshot(file.clone());
    }
    Ok(session)
}

fn keep_snapshot(session: &mut Session, style: &Style) {
    if let Err(reason) = session.save_snapshot() {
        eprintln!(
            "{} could not save the snapshot: {reason}",
            style.yellow("note:")
        );
    }
}

fn one_shot(
    prompt: &str,
    options: &Options,
    home: Option<Home>,
    style: Style,
) -> Result<ExitCode, CliError> {
    let mut text = prompt.to_string();
    if text.trim().is_empty() {
        if stdin_is_terminal() {
            return Err(CliError::NoPrompt);
        }
        text.clear();
        io::stdin().read_to_string(&mut text)?;
    }
    settings(options)?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let mut session = open(options, home.as_ref(), false, style)?;
    let mut chat = Chat::new(&session);
    let reply = if io::IsTerminal::is_terminal(&io::stdout()) {
        let theme = Theme::new(&Caps::detect(options.no_color));
        let stream = MarkdownStream::new(theme, screen::width().saturating_sub(1), 0, false);
        let mut out = MarkdownWriter::new(io::stdout().lock(), stream);
        let reply = chat.ask(&mut session, text.trim_end(), &interrupt, &mut out);
        out.finish()?;
        reply?
    } else {
        chat.ask(
            &mut session,
            text.trim_end(),
            &interrupt,
            &mut io::stdout().lock(),
        )?
    };
    keep_snapshot(&mut session, &style);
    Ok(if reply.outcome.stop == StopReason::Cancelled {
        ExitCode::from(u8::try_from(INTERRUPTED_EXIT_CODE).unwrap_or(130))
    } else {
        ExitCode::SUCCESS
    })
}

fn session(options: &Options, home: Option<Home>, style: Style) -> Result<ExitCode, CliError> {
    settings(options)?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let interactive = stdin_is_terminal() && std::io::IsTerminal::is_terminal(&io::stdout());
    let mut session = open(options, home.as_ref(), interactive, style)?;
    let mut chat = Chat::new(&session);
    if interactive {
        let caps = Caps::detect(options.no_color);
        let splash = splash::show(&caps)?;
        let code = app::run(Launch {
            session,
            chat,
            interrupt,
            home,
            backend: backend_choice(options.backend),
            precision: chat_precision(options)?,
            style,
            caps,
            splash: Some(splash),
        })?;
        return Ok(ExitCode::from(u8::try_from(code).unwrap_or(1)));
    }
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut out = io::stdout().lock();
        if let Err(e) = chat.ask(&mut session, line.trim_end(), &interrupt, &mut out) {
            if e.is_broken_pipe() {
                return Err(e);
            }
            report(&e, &style);
        }
    }
    keep_snapshot(&mut session, &style);
    Ok(ExitCode::SUCCESS)
}
