//! A checkpoint: the model's processed state of a saved conversation (Qwen3.5 on the GPU), so that resuming it does
//! not read the history again. It is only ever a shortcut. The file says what it was made from, every block carries a
//! hash, and anything that does not match the session, the model, the weights or itself means the history is read
//! again instead. It holds a transformation of the private conversation: treat it as private as the conversation.
//!
//! ```text
//! magic "MARINSTA", version u32, header length u32, header (JSON), hash of the header
//! ids:               tokens * u32, hash
//! per DeltaNet layer: its block, hash of the layer number and the block
//! per page:          its block, hash of the page number and the block
//! end:               "MARINEND", total length
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};

use model::PAGE_TOKENS;
use model::pages_for;
use model::qwen35::StateLayout;
use serde::{Deserialize, Serialize};

use crate::CliError;
use crate::saved::MAX_STATE;
use crate::session::Session;
use crate::snapshot::{Reject, hash_parts, ids_hash};

/// Below this many tokens, reading the history again takes about as long as checking and loading a file.
pub const MIN_TOKENS: usize = 256;
const MAGIC: &[u8; 8] = b"MARINSTA";
const END: &[u8; 8] = b"MARINEND";
const VERSION: u32 = 1;
const MAX_HEADER: usize = 64 << 10;
/// Magic, version and header length; the two hashes after the header and the ids; the end mark and total length.
const FIXED: usize = (8 + 4 + 4) + (8 + 8) + (8 + 8);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub config: String,
    pub weights: String,
    pub weight_bits: u32,
    pub weight_group: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Layout {
    delta_layers: usize,
    conv_values: usize,
    recurrent_values: usize,
    full_layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    page_tokens: usize,
}

impl Layout {
    fn of(layout: &StateLayout) -> Self {
        Self {
            delta_layers: layout.delta_layers,
            conv_values: layout.conv_values,
            recurrent_values: layout.recurrent_values,
            full_layers: layout.full_layers,
            heads: layout.heads,
            kv_heads: layout.kv_heads,
            head_dim: layout.head_dim,
            page_tokens: PAGE_TOKENS,
        }
    }
}

/// What the file must say it came from, for it to be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub session: String,
    pub revision: u64,
    pub mode: String,
    pub model: Model,
}

#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    source: Source,
    layout: Layout,
    tokens: usize,
    ids_hash: String,
}

fn rows_of(page: usize, tokens: usize) -> usize {
    PAGE_TOKENS.min(tokens.saturating_sub(page.saturating_mul(PAGE_TOKENS)))
}

fn id_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

fn header_bytes(source: &Source, layout: &StateLayout, ids: &[u32]) -> Result<Vec<u8>, CliError> {
    serde_json::to_vec(&Header {
        version: VERSION,
        source: source.clone(),
        layout: Layout::of(layout),
        tokens: ids.len(),
        ids_hash: format!("{:016x}", ids_hash(ids)),
    })
    .map_err(|e| CliError::BadOption(e.to_string()))
}

/// The size of the file `write` makes for `tokens` tokens, with a header of `header` bytes.
pub fn file_bytes(layout: &StateLayout, tokens: usize, header: usize) -> u64 {
    let blocks = layout
        .delta_layers
        .saturating_mul(layout.delta_bytes().saturating_add(8));
    let pages = (0..pages_for(tokens)).fold(0_usize, |sum, page| {
        sum.saturating_add(layout.page_bytes(rows_of(page, tokens)).saturating_add(8))
    });
    [FIXED, header, tokens.saturating_mul(4), blocks, pages]
        .into_iter()
        .fold(0_u64, |sum, part| sum.saturating_add(part as u64))
}

/// The size of the file `write` will make for a cache holding `ids`.
pub fn planned_bytes(source: &Source, layout: &StateLayout, ids: &[u32]) -> Result<u64, CliError> {
    let header = header_bytes(source, layout, ids)?.len();
    Ok(file_bytes(layout, ids.len(), header))
}

struct Out<'a> {
    file: BufWriter<&'a mut File>,
    written: u64,
}

impl Out<'_> {
    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.written = self.written.saturating_add(bytes.len() as u64);
        self.file.write_all(bytes)
    }

    fn block(&mut self, number: u64, bytes: &[u8]) -> std::io::Result<()> {
        self.put(bytes)?;
        self.put(&hash_parts(&[&number.to_le_bytes(), bytes]).to_le_bytes())
    }
}

fn failed(e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

/// Writes the checkpoint of the session's cache into `file`: `planned_bytes` bytes.
pub fn write(session: &Session, source: &Source, file: &mut File) -> std::io::Result<()> {
    let engine = &session.loaded.model;
    let ids = engine.kept_tokens(session.kv()).map_err(failed)?;
    let layout = engine
        .state_layout()
        .ok_or_else(|| failed("no state layout"))?;
    let header = header_bytes(source, &layout, ids).map_err(failed)?;
    let mut out = Out {
        file: BufWriter::with_capacity(1 << 20, file),
        written: 0,
    };
    out.put(MAGIC)?;
    out.put(&VERSION.to_le_bytes())?;
    out.put(&u32::try_from(header.len()).map_err(failed)?.to_le_bytes())?;
    out.put(&header)?;
    out.put(&hash_parts(&[&header]).to_le_bytes())?;
    let raw = id_bytes(ids);
    out.put(&raw)?;
    out.put(&hash_parts(&[&raw]).to_le_bytes())?;
    for index in 0..layout.delta_layers {
        let bytes = engine
            .export_state_delta(session.kv(), index)
            .map_err(failed)?;
        out.block(index as u64, &bytes)?;
    }
    for page in 0..pages_for(ids.len()) {
        let rows = rows_of(page, ids.len());
        let bytes = engine
            .export_state_page(session.kv(), page, rows)
            .map_err(failed)?;
        out.block(page as u64, &bytes)?;
    }
    let total = out.written.saturating_add(16);
    out.put(END)?;
    out.put(&total.to_le_bytes())?;
    out.file.flush()
}

struct In {
    file: BufReader<File>,
}

impl In {
    fn take(&mut self, len: usize) -> Result<Vec<u8>, Reject> {
        let mut bytes = vec![0_u8; len];
        self.file
            .read_exact(&mut bytes)
            .map_err(|e| match e.kind() {
                ErrorKind::UnexpectedEof => Reject::Damaged("the file is shorter than it says"),
                _ => Reject::Failed(e.to_string()),
            })?;
        Ok(bytes)
    }

    fn word(&mut self) -> Result<u64, Reject> {
        let bytes = self.take(8)?;
        let array: [u8; 8] = bytes.try_into().map_err(|_| Reject::Damaged("a number"))?;
        Ok(u64::from_le_bytes(array))
    }

    fn block(&mut self, number: u64, len: usize) -> Result<Vec<u8>, Reject> {
        let bytes = self.take(len)?;
        if self.word()? != hash_parts(&[&number.to_le_bytes(), &bytes]) {
            return Err(Reject::Damaged("a block does not match its hash"));
        }
        Ok(bytes)
    }
}

/// Why a checkpoint was not used, in words about checkpoints.
pub fn why(reject: &Reject) -> String {
    match reject {
        Reject::Damaged(what) => format!("it is damaged: {what}"),
        Reject::Foreign(what) => format!("it was made for something else: {what}"),
        Reject::Failed(what) => what.clone(),
        Reject::Missing | Reject::NoMatch => "it does not apply".into(),
    }
}

fn foreign(what: &'static str) -> Reject {
    Reject::Foreign(what)
}

fn failure(e: impl std::fmt::Display) -> Reject {
    Reject::Failed(e.to_string())
}

/// Reads the checkpoint in `file` (`len` bytes) into the session's cache if, and only if, it is whole, was made for
/// `expect` and this model's state layout, and fits the context. Returns the tokens the cache now holds.
pub fn restore(
    session: &mut Session,
    file: File,
    len: u64,
    expect: &Source,
) -> Result<Vec<u32>, Reject> {
    let layout = session
        .loaded
        .model
        .state_layout()
        .ok_or(foreign("this model cannot keep its state"))?;
    let mut input = In {
        file: BufReader::with_capacity(1 << 20, file),
    };
    let start = input.take(16)?;
    if start.get(..8) != Some(MAGIC.as_slice()) {
        return Err(Reject::Damaged("it is not a checkpoint"));
    }
    let word = |index: usize| -> Option<u32> {
        let chunk = start.chunks_exact(4).nth(index)?;
        Some(u32::from_le_bytes(chunk.try_into().ok()?))
    };
    if word(2) != Some(VERSION) {
        return Err(foreign("another checkpoint version"));
    }
    let header_len = usize::try_from(word(3).ok_or(Reject::Damaged("header"))?)
        .ok()
        .filter(|n| *n <= MAX_HEADER)
        .ok_or(Reject::Damaged("header size"))?;
    let header_raw = input.take(header_len)?;
    if input.word()? != hash_parts(&[&header_raw]) {
        return Err(Reject::Damaged("the header does not match its hash"));
    }
    let header: Header =
        serde_json::from_slice(&header_raw).map_err(|_| Reject::Damaged("the header"))?;
    if header.source.session != expect.session || header.source.revision != expect.revision {
        return Err(foreign("it was made for another version of the session"));
    }
    if header.source.mode != expect.mode {
        return Err(foreign("it was made in the other mode"));
    }
    if header.source.model != expect.model {
        return Err(foreign("another model, its files or its precision"));
    }
    if header.layout != Layout::of(&layout) {
        return Err(foreign("another state layout"));
    }
    let tokens = header.tokens;
    if tokens == 0 || tokens >= session.ctx {
        return Err(foreign("it does not fit the context window"));
    }
    if file_bytes(&layout, tokens, header_len) != len || len > MAX_STATE {
        return Err(Reject::Damaged("the file is not the size it should be"));
    }
    let ids_raw = input.take(tokens.saturating_mul(4))?;
    if input.word()? != hash_parts(&[&ids_raw]) {
        return Err(Reject::Damaged("the token list does not match its hash"));
    }
    let ids: Vec<u32> = ids_raw
        .chunks_exact(4)
        .filter_map(|c| c.try_into().ok().map(u32::from_le_bytes))
        .collect();
    if format!("{:016x}", ids_hash(&ids)) != header.ids_hash {
        return Err(Reject::Damaged("the token list"));
    }
    session.release_cache().map_err(failure)?;
    let engine = &session.loaded.model;
    let mut cache = engine
        .begin_state_import(session.ctx, tokens)
        .map_err(failure)?;
    for index in 0..layout.delta_layers {
        let bytes = input.block(index as u64, layout.delta_bytes())?;
        engine
            .import_state_delta(&mut cache, index, &bytes)
            .map_err(failure)?;
    }
    for page in 0..pages_for(tokens) {
        let rows = rows_of(page, tokens);
        let bytes = input.block(page as u64, layout.page_bytes(rows))?;
        engine
            .import_state_page(&mut cache, page, rows, &bytes)
            .map_err(failure)?;
    }
    if input.take(END.len())? != END.as_slice() || input.word()? != len {
        return Err(Reject::Damaged("the end of the file"));
    }
    engine
        .finish_state_import(&mut cache, &ids)
        .map_err(failure)?;
    session.install_state(cache, ids.clone());
    Ok(ids)
}
