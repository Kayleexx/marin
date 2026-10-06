#![forbid(unsafe_code)]

use std::io;
use std::process::ExitCode;

use clap::Parser;
use model::BackendChoice;
use runtime::{CausalLm, StopReason};
use sampler::SamplerConfig;

use cli::args::{self, Cli, Command, RunArgs};
use cli::error::CliError;
use cli::memory::DEFAULT_CTX;
use cli::session::{Session, Settings};
use cli::signal::{self, Interrupt};
use cli::style::{self, Style};
use cli::{bench, guess, inspect, load, probe, quality, repl, report, telemetry, trace, verify};

fn main() -> ExitCode {
    let cli = Cli::parse();
    let style = Style::detect(cli.no_color);
    match run(cli, &style) {
        Ok(code) => code,
        Err(e) if e.is_broken_pipe() => ExitCode::SUCCESS,
        Err(CliError::Cancelled) => {
            eprintln!("cancelled");
            ExitCode::from(signal::INTERRUPTED_EXIT_CODE as u8)
        }
        Err(e) => {
            report(&e, &style);
            ExitCode::FAILURE
        }
    }
}

fn interrupt() -> Result<Interrupt, CliError> {
    Interrupt::install().map_err(|e| CliError::Io(io::Error::other(e)))
}

fn run(cli: Cli, style: &Style) -> Result<ExitCode, CliError> {
    match cli.command {
        Some(Command::Inspect {
            model_dir,
            quant,
            ctx,
        }) => inspect::run(&model_dir, ctx.ctx, quant.scheme()?, style).map(|()| ExitCode::SUCCESS),
        Some(Command::Bench {
            model_dir,
            backend,
            prompt_tokens,
            tokens,
            runs,
            warmup,
            profile,
            sample_gpu,
            temp,
            top_k,
            top_p,
            token_pause_ms,
            seed,
            json,
            quant,
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
                sample_gpu,
                sampling: SamplerConfig {
                    temperature: temp,
                    top_k,
                    top_p,
                    seed,
                },
                token_pause_ms,
                json,
                quant: quant.scheme()?,
            },
            style,
            &interrupt()?,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::Probe { seconds, json }) => {
            probe::run(&probe::Options { seconds, json }, &interrupt()?).map(|()| ExitCode::SUCCESS)
        }
        Some(Command::Compare { old, new }) => {
            bench::compare(&old, &new).map(|()| ExitCode::SUCCESS)
        }
        Some(Command::Trace {
            model_dir,
            backend,
            prompt_tokens,
            pass,
            json,
            ctx,
        }) => trace::run(
            &model_dir,
            backend_choice(backend),
            ctx.ctx,
            &trace::Options {
                prompt_tokens,
                pass,
                json,
            },
            style,
            &interrupt()?,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::Quality {
            model_dir,
            text,
            tokens,
            backend,
            save,
            against,
            quant,
        }) => quality::run(
            &model_dir,
            backend_choice(backend),
            &quality::Options {
                text,
                tokens,
                save,
                against,
                quant: quant.scheme()?,
            },
            style,
            &interrupt()?,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::Guess {
            model_dir,
            prompts,
            max_tokens,
            ctx,
            save,
            backend,
            quant,
        }) => guess::generate::run(
            &model_dir,
            backend_choice(backend),
            &guess::generate::Options {
                prompts,
                max_tokens,
                ctx,
                out: save,
                quant: quant.scheme()?,
            },
            style,
            &interrupt()?,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::GuessReport {
            transcripts,
            costs,
            min_match,
            draft,
        }) => guess::report::run(&guess::report::Options {
            transcripts,
            costs,
            detail: guess::replay::Config {
                min_n: min_match,
                max_len: draft,
            },
        })
        .map(|()| ExitCode::SUCCESS),
        Some(Command::PassCost {
            model_dir,
            text,
            contexts,
            sizes,
            rounds,
            warmup,
            reps,
            save,
            stages,
            backend,
            quant,
        }) => guess::passcost::run(
            &model_dir,
            backend_choice(backend),
            &guess::passcost::Options {
                text,
                contexts,
                sizes,
                rounds,
                warmup,
                reps,
                save,
                stages,
                quant: quant.scheme()?,
            },
            style,
            &interrupt()?,
        )
        .map(|()| ExitCode::SUCCESS),
        Some(Command::SpecBench {
            model_dir,
            prompts,
            max_tokens,
            ctx,
            rounds,
            sampling,
            no_energy,
            save,
            quant,
        }) => {
            if rounds == 0 {
                return Err(CliError::BadOption("--rounds must be at least 1".into()));
            }
            guess::specbench::run(
                &model_dir,
                &guess::specbench::Options {
                    prompts,
                    max_tokens,
                    ctx,
                    rounds,
                    sampling,
                    energy: !no_energy,
                    save,
                    quant: quant.scheme()?,
                },
                style,
                &interrupt()?,
            )
            .map(|()| ExitCode::SUCCESS)
        }
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
    let interrupt = interrupt()?;
    let chrome = style::stderr_is_terminal();
    let loaded = load::load(
        &model_dir,
        args.ctx.ctx,
        DEFAULT_CTX,
        backend_choice(args.backend),
        args.quant.scheme()?,
        &style,
        chrome,
    )?;
    let mut session = Session::new(loaded, settings, style)?;
    session.speculate = args.speculate;
    if args.speculate && chrome && session.loaded.model.max_rows() == 0 {
        eprintln!("note: guessing ahead needs the GPU; decoding normally");
    }
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
