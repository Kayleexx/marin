//! Shared set-up for the snapshot tests: a tiny model on the GPU, restarted by loading it again.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cli::load::load_for_snapshot;
use cli::session::{Reply, Session, Settings};
use cli::signal::Interrupt;
use cli::style::Style;
use model::{BackendChoice, KvFormat, Precision, QuantBits, QuantScheme};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn interrupt() -> Result<&'static Interrupt> {
    static INTERRUPT: OnceLock<Option<Interrupt>> = OnceLock::new();
    INTERRUPT
        .get_or_init(|| Interrupt::install().ok())
        .as_ref()
        .ok_or_else(|| "could not install the interrupt handler".into())
}

pub fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat")
}

pub fn work_dir(name: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("marin-snapshot-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn precision(kv: KvFormat, int8: bool) -> Precision {
    Precision {
        weights: int8.then_some(QuantScheme {
            bits: QuantBits::Int8,
            group: 32,
        }),
        kv,
    }
}

pub fn tokens(count: usize, seed: usize) -> Vec<u32> {
    (0..count)
        .map(|i| {
            let spread = i.saturating_mul(37).saturating_add(seed.saturating_mul(11));
            u32::try_from(spread.checked_rem(250).unwrap_or(0).saturating_add(5)).unwrap_or(5)
        })
        .collect()
}

pub struct Setup<'a> {
    pub model: PathBuf,
    pub precision: Precision,
    pub snapshot: Option<&'a Path>,
    pub temperature: f32,
    pub speculate: bool,
    pub max_tokens: usize,
}

impl<'a> Setup<'a> {
    pub fn new(precision: Precision) -> Self {
        Self {
            model: fixture(),
            precision,
            snapshot: None,
            temperature: 0.0,
            speculate: false,
            max_tokens: 10,
        }
    }

    pub fn at(mut self, path: &'a Path) -> Self {
        self.snapshot = Some(path);
        self
    }

    pub fn open(&self) -> Result<Option<Session>> {
        let loaded = match load_for_snapshot(
            &self.model,
            Some(1024),
            1024,
            BackendChoice::Cuda,
            self.precision,
            &Style::plain(),
            false,
        ) {
            Ok(loaded) => loaded,
            Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
                eprintln!("skipping: {e}");
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        let settings = Settings {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: 20,
            top_p: 0.9,
            seed: Some(3),
        };
        let mut session = Session::new(loaded, settings, Style::plain())?;
        session.speculate = self.speculate;
        if let Some(path) = self.snapshot {
            session.use_snapshot(path.to_path_buf());
        }
        Ok(Some(session))
    }
}

pub fn run(session: &mut Session, prompt: &[u32]) -> Result<Reply> {
    Ok(session.complete_tokens(prompt.to_vec(), interrupt()?, &mut Vec::new())?)
}

pub fn then(session: &Session, extra: usize) -> Vec<u32> {
    let mut prompt = session.retained().tokens().to_vec();
    prompt.extend(tokens(extra, 9));
    prompt
}

/// A session that has read `prompt` and saved its cache to `file`, and the tokens it then holds.
pub fn saved(setup: &Setup, file: &Path, prompt: &[u32]) -> Result<Option<(Session, Vec<u32>)>> {
    let Some(mut first) = setup.open()? else {
        return Ok(None);
    };
    run(&mut first, prompt)?;
    let saved = first.save_snapshot().map_err(|e| e.to_string())?;
    assert!(saved.is_some() && file.is_file());
    let held = first.retained().tokens().to_vec();
    Ok(Some((first, held)))
}
