#![forbid(unsafe_code)]

use std::io;
use std::process::ExitCode;

use clap::Parser;
use model::BackendChoice;
use runtime::StopReason;

use cli::args::{self, Cli, Command, RunArgs};
use cli::error::CliError;
use cli::memory::DEFAULT_CTX;
use cli::session::{Session, Settings};
use cli::signal::{self, Interrupt};
use cli::style::{self, Style};
use cli::{bench, inspect, load, repl, report, telemetry, verify};

fn main() -> ExitCode {
    let cli = Cli::parse();
    let style = Style::detect(cli.no_color);
    match run(cli, &style) {
        Ok(code) => code,
        Err(e) if e.is_broken_pipe() => ExitCode::SUCCESS,
        Err(e) => {
            report(&e, &style);
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli, style: &Style) -> Result<ExitCode, CliError> {
    match cli.command {
        Some(Command::Inspect { model_dir, ctx }) => {
            inspect::run(&model_dir, ctx.ctx, style).map(|()| ExitCode::SUCCESS)
        }
        Some(Command::Bench {
            model_dir,
            backend,
            prompt_tokens,
            tokens,
            runs,
            warmup,
            profile,
            json,
            ctx,
        }) => bench::run(
            &model_dir,
            backend_choice(backend),
            ctx.ctx,
            &bench::Options {
                prompt_tokens,
                output_tokens: tokens,
                runs,
                warmup,
                profile,
                json,
            },
            style,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::Verify {
            model_dir,
            reference,
            backend,
        }) => verify::run(&model_dir, &reference, backend_choice(backend), style)
            .map(|()| ExitCode::SUCCESS),
        None => run_model(cli.run, *style),
    }
}

fn backend_choice(arg: args::BackendArg) -> BackendChoice {
    match arg {
        args::BackendArg::Auto => BackendChoice::Auto,
        args::BackendArg::Cpu => BackendChoice::Cpu,
        args::BackendArg::Cuda => BackendChoice::Cuda,
    }
}

fn run_model(args: RunArgs, style: Style) -> Result<ExitCode, CliError> {
    let Some(model_dir) = args.model_dir else {
        return Err(CliError::BadOption("a model folder is required".into()));
    };
    let settings = Settings {
        max_tokens: args.max_tokens,
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    settings.sampler_config(0).validate()?;
    let interrupt = Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))?;
    let chrome = style::stderr_is_terminal();
    let loaded = load::load(
        &model_dir,
        args.ctx.ctx,
        DEFAULT_CTX,
        backend_choice(args.backend),
        &style,
        chrome,
    )?;
    let mut session = Session::new(loaded, settings, style)?;
    match args.prompt {
        Some(prompt) => {
            let reply = session.complete(&prompt, &interrupt, &mut io::stdout().lock())?;
            if chrome {
                eprintln!("{}", session.footer(&reply));
            }
            if args.metrics {
                for line in telemetry::request_lines(&reply.outcome)
                    .into_iter()
                    .chain(telemetry::resource_lines(&session))
                {
                    eprintln!("  {line}");
                }
            }
            Ok(if reply.outcome.stop == StopReason::Cancelled {
                ExitCode::from(signal::INTERRUPTED_EXIT_CODE as u8)
            } else {
                ExitCode::SUCCESS
            })
        }
        None => {
            eprintln!("{}", repl::banner(&session));
            repl::run(&mut session, &interrupt).map(|()| ExitCode::SUCCESS)
        }
    }
}
