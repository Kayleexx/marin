#![forbid(unsafe_code)]

use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{ArgMatches, CommandFactory, FromArgMatches};
use runtime::StopReason;

use cli::choose::{self, Chosen};
use cli::config::{self, Home};
use cli::dialog::{self, Dialog, Talk};
use cli::error::CliError;
use cli::marin_args::{MarinCli, MarinCommand, Options};
use cli::render::{MarkdownStream, MarkdownWriter, Theme};
use cli::session::Session;
use cli::setup::{self, CHAT_CTX, CODE_CTX, Explicit, absolute, backend_choice};
use cli::signal::{INTERRUPTED_EXIT_CODE, Interrupt};
use cli::style::{Style, stderr_is_terminal, stdin_is_terminal};
use cli::term::app::{self, Launch};
use cli::term::caps::Caps;
use cli::term::screen;
use cli::term::splash;
use cli::{code, load, plan, report, show};

/// The port `marin serve` listens on unless told otherwise; `marin code` looks for a server there.
const SERVER_PORT: u16 = 8000;

fn main() -> ExitCode {
    let matches = MarinCli::command().get_matches();
    let cli = match MarinCli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };
    tensor::set_threads(cli.options.threads.unwrap_or(0));
    let style = Style::detect(cli.options.no_color);
    match run(cli, &matches, style) {
        Ok(code) => code,
        Err(e) if e.is_broken_pipe() => ExitCode::SUCCESS,
        Err(e) => {
            report(&e, &style);
            ExitCode::FAILURE
        }
    }
}

/// What a session starts with: the saved preferences under the flags.
struct Launching {
    options: Options,
    explicit: Explicit,
    home: Option<Home>,
    /// The config as it was at launch, read once.
    saved: config::Config,
}

/// `saved` lets the remembered backend and sampling settings fill in what the flags left open.
fn launching(mut options: Options, matches: &ArgMatches, style: Style, saved: bool) -> Launching {
    let explicit = Explicit::from_matches(matches);
    let home = Home::detect();
    let (config, damaged) = home.as_ref().map(Home::read_config).unwrap_or_default();
    if let Some(problem) = damaged {
        eprintln!(
            "{} ignoring the saved settings: {problem}",
            style.yellow("note:")
        );
    }
    if saved {
        setup::apply_config(&mut options, &explicit, &config);
    }
    Launching {
        options,
        explicit,
        home,
        saved: config,
    }
}

fn run(cli: MarinCli, matches: &ArgMatches, style: Style) -> Result<ExitCode, CliError> {
    match cli.command {
        Some(MarinCommand::Model { path }) => {
            model_command(path.as_deref(), Home::detect().as_ref())
        }
        Some(MarinCommand::Plan) => {
            let l = launching(cli.options, matches, style, false);
            let path = pick_model(&l, false, style)?.path;
            plan::run_chat(
                &path,
                l.options.ctx,
                CHAT_CTX,
                backend_choice(l.options.backend),
                setup::chat_precision(&l.options, &l.explicit, &path, &l.saved)?,
            )
            .map(|()| ExitCode::SUCCESS)
        }
        Some(MarinCommand::Show { path, kind }) => {
            show::run(path.as_deref(), kind.as_deref(), cli.options.no_color)
                .map(|()| ExitCode::SUCCESS)
        }
        Some(MarinCommand::Run { prompt }) => {
            let l = launching(cli.options, matches, style, true);
            let text = prompt.join(" ");
            if text.trim().is_empty() && stdin_is_terminal() {
                return session(&l, style, Talk::Chat, PathBuf::from("."), None);
            }
            one_shot(&text, &l, style)
        }
        Some(MarinCommand::Code { path, port, log }) => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let l = launching(cli.options, matches, style, true);
            let remote = match port {
                Some(port) => Some(port),
                None => running_server(&l, style),
            };
            match remote {
                Some(port) => code::run(code::Options {
                    path: Some(root),
                    port,
                    log,
                    no_color: l.options.no_color,
                }),
                None => session(&l, style, Talk::Code, root, log),
            }
        }
        Some(MarinCommand::Serve { port, idle_unload }) => {
            if cli.options.snapshot.is_some() {
                return Err(CliError::BadOption(
                    "--snapshot is not supported by marin serve".into(),
                ));
            }
            // Fail before the model takes seconds to load.
            if let Err(e) = std::net::TcpListener::bind(("127.0.0.1", port))
                && e.kind() == io::ErrorKind::AddrInUse
            {
                return Err(CliError::PortInUse(port));
            }
            let l = launching(cli.options, matches, style, false);
            let defaults = setup::settings(&l.options)?;
            let path = absolute(&pick_model(&l, false, style)?.path);
            let precision = setup::chat_precision(&l.options, &l.explicit, &path, &l.saved)?;
            let backend = backend_choice(l.options.backend);
            let reload = || {
                load::load_chat(
                    &path,
                    (l.options.ctx, CHAT_CTX),
                    backend,
                    precision,
                    &style,
                    (false, false),
                )
            };
            let idle = idle_unload.map(std::time::Duration::from_secs);
            cli::server::run(reload()?, &reload, defaults, port, idle)?;
            Ok(ExitCode::SUCCESS)
        }
        None => {
            let l = launching(cli.options, matches, style, true);
            session(&l, style, Talk::Chat, PathBuf::from("."), None)
        }
    }
}

/// A Marin server already on the usual port holds a model on this machine: asking about the project through it
/// avoids loading a second copy, but only if it is the model this launch would load, as far as can be checked.
fn running_server(l: &Launching, style: Style) -> Option<u16> {
    let (model, context) = code::server_at(SERVER_PORT)?;
    let wanted = choose::unambiguous(&l.options, l.saved.model.as_deref());
    match choose::server_fits(
        &l.options,
        &l.explicit,
        wanted.as_deref(),
        (&model, context),
    ) {
        Ok(()) => {
            eprintln!(
                "{}",
                style.dim(&format!(
                    "using the Marin server on port {SERVER_PORT} ({model})"
                ))
            );
            Some(SERVER_PORT)
        }
        Err(why) => {
            eprintln!(
                "{}",
                style.dim(&format!(
                    "not using the Marin server on port {SERVER_PORT}: {why}; loading a model here"
                ))
            );
            None
        }
    }
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
    choose::check_chat_model(path)?;
    let path = absolute(path);
    if let Some(home) = home {
        home.update_config(|c| c.model = Some(path.clone()))?;
    }
    println!("default model: {}", config::display_name(&path));
    Ok(ExitCode::SUCCESS)
}

fn pick_model(l: &Launching, interactive: bool, style: Style) -> Result<Chosen, CliError> {
    choose::choose_model(&l.options, l.saved.model.as_deref(), interactive, style)
}

fn open(
    l: &Launching,
    (interactive, default_ctx): (bool, usize),
    style: Style,
) -> Result<Session, CliError> {
    let settings = setup::settings(&l.options)?;
    let chosen = pick_model(l, interactive, style)?;
    let path = absolute(&chosen.path);
    let mut precision = if chosen.picked_packed {
        setup::precision_for_pick(setup::asked_precision(&l.options)?, &path)
    } else {
        setup::chat_precision(&l.options, &l.explicit, &path, &l.saved)?
    };
    if interactive
        && precision.weights.is_none()
        && !l.explicit.quantize
        && let Some(group) = setup::packed_group(&path)
    {
        if !choose::confirm_packed(&path, group, style)? {
            return Err(CliError::Cancelled);
        }
        precision = setup::precision_for_pick(precision, &path);
    }
    let loaded = load::load_chat(
        &path,
        (l.options.ctx, default_ctx),
        backend_choice(l.options.backend),
        precision,
        &style,
        (stderr_is_terminal(), l.options.snapshot.is_some()),
    )?;
    setup::remember(l.home.as_ref(), &path, loaded.precision.weights);
    let mut session = Session::new(loaded, settings, style)?;
    if let Some(file) = &l.options.snapshot {
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

fn one_shot(prompt: &str, l: &Launching, style: Style) -> Result<ExitCode, CliError> {
    let mut text = prompt.to_string();
    if text.trim().is_empty() {
        if stdin_is_terminal() {
            return Err(CliError::NoPrompt);
        }
        text.clear();
        io::stdin().read_to_string(&mut text)?;
    }
    setup::settings(&l.options)?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let mut session = open(l, (false, CHAT_CTX), style)?;
    let mut chat = cli::chat::Chat::new(&session);
    let asked = if io::IsTerminal::is_terminal(&io::stdout()) {
        let theme = Theme::new(&Caps::detect(l.options.no_color));
        let stream = MarkdownStream::new(theme, screen::width().saturating_sub(1), 0, false);
        let mut out = MarkdownWriter::new(io::stdout().lock(), stream);
        let reply = chat.ask(&mut session, text.trim_end(), &interrupt, &mut out);
        out.finish()?;
        reply
    } else {
        chat.ask(
            &mut session,
            text.trim_end(),
            &interrupt,
            &mut io::stdout().lock(),
        )
    };
    let cancelled = matches!(asked, Err(CliError::Cancelled));
    let reply = match asked {
        Ok(reply) => Some(reply),
        Err(CliError::Cancelled) => None,
        Err(e) => return Err(e),
    };
    keep_snapshot(&mut session, &style);
    Ok(
        if cancelled || reply.is_some_and(|r| r.outcome.stop == StopReason::Cancelled) {
            ExitCode::from(u8::try_from(INTERRUPTED_EXIT_CODE).unwrap_or(130))
        } else {
            ExitCode::SUCCESS
        },
    )
}

/// The conversation: a terminal gets the interactive UI, anything else reads one question or command per line.
fn session(
    l: &Launching,
    style: Style,
    talk: Talk,
    root: PathBuf,
    log: Option<PathBuf>,
) -> Result<ExitCode, CliError> {
    setup::settings(&l.options)?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let interactive = stdin_is_terminal() && io::IsTerminal::is_terminal(&io::stdout());
    let default_ctx = if talk == Talk::Code {
        CODE_CTX
    } else {
        CHAT_CTX
    };
    let mut session = open(l, (interactive, default_ctx), style)?;
    let mut dialog = Dialog::new(&session, talk, absolute(&root), log)?;
    if interactive {
        let caps = Caps::detect(l.options.no_color);
        let splash = splash::show(&caps)?;
        let code = app::run(Launch {
            session,
            dialog,
            interrupt,
            home: l.home.clone(),
            backend: backend_choice(l.options.backend),
            precision: setup::asked_precision(&l.options)?,
            style,
            caps,
            splash: Some(splash),
        })?;
        return Ok(ExitCode::from(u8::try_from(code).unwrap_or(1)));
    }
    dialog::run_piped(
        &mut session,
        &mut dialog,
        &interrupt,
        (style, l.home.as_ref()),
        io::stdin().lock().lines(),
    )?;
    keep_snapshot(&mut session, &style);
    Ok(ExitCode::SUCCESS)
}
