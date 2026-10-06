use runtime::SequenceCache;

use crate::CliError;
use crate::bench;
use crate::inspect::describe;
use crate::memory::available_bytes;
use crate::session::Session;
use crate::telemetry;
use crate::units;

const HELP: &str = "\
Type text and the model continues it. Each prompt starts fresh.

  /help                 this list
  /model                what model is loaded
  /memory               how much memory it uses
  /kv                   the context window (the model's short-term memory)
  /backend              where the math runs
  /metrics              speed of the last reply
  /profile              where the time went for the last prompt, stage by stage
  /tokens TEXT          show how TEXT is split into tokens
  /top TEXT             show the 5 likeliest next tokens after TEXT
  /set NAME VALUE       change temp, top_k, top_p, max_tokens or seed (random)
  /quit                 leave (Ctrl-D works too)";

pub enum Next {
    Continue,
    Quit,
}

pub fn run(session: &mut Session, line: &str) -> Result<Next, CliError> {
    let (name, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let rest = rest.trim();
    match name {
        "/help" | "/?" => println!("{HELP}"),
        "/quit" | "/exit" => return Ok(Next::Quit),
        "/model" => model(session),
        "/memory" => memory(session),
        "/kv" => kv(session),
        "/backend" => println!(
            "{} ({})",
            session.loaded.model.backend_name(),
            model::PRECISION
        ),
        "/metrics" => metrics(session),
        "/profile" => profile(session)?,
        "/tokens" => tokens(session, rest)?,
        "/top" => top(session, rest)?,
        "/set" => set(session, rest)?,
        other => println!("unknown command {other}. Type /help for the list."),
    }
    Ok(Next::Continue)
}

fn model(s: &Session) {
    let cfg = s.loaded.model.config();
    let vocab = s.loaded.dir.tokenizer.vocab_size();
    println!("{}", s.style.bold(&s.loaded.name));
    println!(
        "{}",
        describe(&s.loaded.name, cfg, vocab, s.ctx, &s.loaded.footprint)
    );
}

fn memory(s: &Session) {
    for line in telemetry::resource_lines(s) {
        println!("  {line}");
    }
    if let Some(free) = available_bytes() {
        println!("  free now        {}", units::bytes(free));
    }
}

fn kv(s: &Session) {
    println!(
        "  window       {} tokens",
        units::count(SequenceCache::capacity(&s.kv) as u64)
    );
    println!(
        "  last reply   used {} positions",
        units::count(s.kv.used() as u64)
    );
}

fn metrics(s: &Session) {
    let Some(o) = &s.last else {
        println!("nothing generated yet");
        return;
    };
    for line in telemetry::request_lines(o) {
        println!("  {line}");
    }
}

fn profile(s: &mut Session) -> Result<(), CliError> {
    if s.last_prompt.is_empty() {
        println!("nothing generated yet");
        return Ok(());
    }
    let room = s.last_prompt.len() < SequenceCache::capacity(&s.kv);
    let vocab = s.loaded.dir.tokenizer.vocab_size();
    let report = bench::profile_report(
        &s.loaded.model,
        &mut s.kv,
        &s.last_prompt,
        if room { 2 } else { 1 },
        vocab,
    )?;
    for line in report.lines() {
        println!("  {line}");
    }
    Ok(())
}

fn tokens(s: &Session, text: &str) -> Result<(), CliError> {
    let ids = s.encode(text)?;
    println!("{} tokens:", ids.len());
    for line in s.token_lines(&ids)? {
        println!("{line}");
    }
    Ok(())
}

fn top(s: &mut Session, text: &str) -> Result<(), CliError> {
    let prompt = s.encode(text)?;
    println!("likeliest next tokens:");
    for line in s.next_token_lines(&prompt)? {
        println!("  {line}");
    }
    Ok(())
}

fn set(s: &mut Session, args: &str) -> Result<(), CliError> {
    let (name, value) = args
        .split_once(char::is_whitespace)
        .map(|(n, v)| (n, v.trim()))
        .ok_or_else(|| CliError::BadOption("usage: /set NAME VALUE (try /set temp 0.5)".into()))?;
    let mut next = s.settings;
    let bad = |what: &str| CliError::BadOption(format!("{name} needs {what}, got `{value}`"));
    match name {
        "temp" => next.temperature = value.parse().map_err(|_| bad("a number"))?,
        "top_k" => next.top_k = value.parse().map_err(|_| bad("a whole number"))?,
        "top_p" => next.top_p = value.parse().map_err(|_| bad("a number"))?,
        "max_tokens" => next.max_tokens = value.parse().map_err(|_| bad("a whole number"))?,
        "seed" if value == "random" => next.seed = None,
        "seed" => {
            next.seed = Some(
                value
                    .parse()
                    .map_err(|_| bad("a whole number or `random`"))?,
            )
        }
        _ => {
            return Err(CliError::BadOption(format!(
                "unknown setting `{name}`; see /help"
            )));
        }
    }
    next.sampler_config(0).validate()?;
    s.settings = next;
    println!("{name} = {value}");
    Ok(())
}
