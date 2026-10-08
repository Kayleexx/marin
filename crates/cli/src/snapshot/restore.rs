use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant};

use model::PAGE_TOKENS;
use model::{Engine, EngineCache};
use runtime::SequenceCache;

use super::format::{
    END_BYTES, HEADER_BYTES, Header, Identity, PAGE_HASH_BYTES, Reject, end_is_valid, ids_hash,
    page_hash,
};

pub struct Restored {
    pub ids: Vec<u32>,
    pub bytes: usize,
    pub timing: Timing,
}

#[derive(Default, Clone, Copy)]
pub struct Timing {
    pub header: Duration,
    pub read: Duration,
    pub checksum: Duration,
    pub copy: Duration,
}

fn read_exact(file: &mut File, len: usize) -> Result<Vec<u8>, Reject> {
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes).map_err(|e| match e.kind() {
        ErrorKind::UnexpectedEof => Reject::Damaged("file is shorter than it says"),
        _ => Reject::Failed(e.to_string()),
    })?;
    Ok(bytes)
}

fn shared_start(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whole pages only, unless the prompt covers the entire snapshot: a page that is cut short would have to be copied
/// partly.
fn restorable(shared: usize, saved: usize) -> usize {
    if shared >= saved {
        saved
    } else {
        shared
            .checked_div(PAGE_TOKENS)
            .unwrap_or(0)
            .saturating_mul(PAGE_TOKENS)
    }
}

struct Opened {
    file: File,
    header: Header,
    ids: Vec<u32>,
}

fn open(path: &Path, now: &Identity) -> Result<Opened, Reject> {
    let mut file = File::open(path).map_err(|e| match e.kind() {
        ErrorKind::NotFound => Reject::Missing,
        _ => Reject::Failed(e.to_string()),
    })?;
    let header = Header::decode(&read_exact(&mut file, HEADER_BYTES)?)?;
    header.check_against(now)?;
    let expected = header
        .file_bytes()
        .ok_or(Reject::Damaged("sizes overflow"))?;
    let actual = file
        .metadata()
        .map_err(|e| Reject::Failed(e.to_string()))?
        .len();
    if header.tokens == 0 || u64::try_from(expected).ok() != Some(actual) {
        return Err(Reject::Damaged("length"));
    }
    let bytes = read_exact(&mut file, (header.tokens as usize).saturating_mul(4))?;
    let ids: Vec<u32> = bytes
        .chunks_exact(4)
        .filter_map(|c| c.try_into().ok().map(u32::from_le_bytes))
        .collect();
    if ids_hash(&ids) != header.ids_hash {
        return Err(Reject::Damaged("token ids"));
    }
    Ok(Opened { file, header, ids })
}

fn load_pages(
    opened: &mut Opened,
    engine: &Engine,
    cache: &mut EngineCache,
    tokens: usize,
    timing: &mut Timing,
) -> Result<usize, Reject> {
    let failed = |e: model::ModelError| Reject::Failed(e.to_string());
    engine.begin_kv_import(cache, tokens).map_err(failed)?;
    let mut total = 0_usize;
    for page in 0..tokens.div_ceil(PAGE_TOKENS) {
        let rows = opened.header.rows_of(page);
        let payload_len = opened
            .header
            .page_bytes(rows)
            .ok_or(Reject::Damaged("sizes overflow"))?;
        let started = Instant::now();
        let payload = read_exact(&mut opened.file, payload_len)?;
        let stored = read_exact(&mut opened.file, PAGE_HASH_BYTES)?;
        timing.read = timing.read.saturating_add(started.elapsed());
        let started = Instant::now();
        let good = stored.as_slice() == page_hash(page, &payload).to_le_bytes();
        timing.checksum = timing.checksum.saturating_add(started.elapsed());
        if !good {
            return Err(Reject::Damaged("page data"));
        }
        let started = Instant::now();
        engine
            .import_kv_page(cache, page, rows, &payload)
            .map_err(failed)?;
        timing.copy = timing.copy.saturating_add(started.elapsed());
        total = total.saturating_add(payload_len);
    }
    let started = Instant::now();
    let expected = opened
        .header
        .file_bytes()
        .ok_or(Reject::Damaged("sizes overflow"))?;
    let end_at = expected.saturating_sub(END_BYTES) as u64;
    opened
        .file
        .seek(SeekFrom::Start(end_at))
        .map_err(|e| Reject::Failed(e.to_string()))?;
    let end = read_exact(&mut opened.file, END_BYTES)?;
    timing.read = timing.read.saturating_add(started.elapsed());
    if !end_is_valid(&end, expected) {
        return Err(Reject::Damaged("end mark"));
    }
    engine.finish_kv_import(cache, tokens).map_err(failed)?;
    Ok(total)
}

/// Loads the longest start of the snapshot at `path` that `prompt` shares into an empty `cache`. The cache holds
/// either exactly the returned tokens or nothing: its length only advances after every page was copied and checked, so
/// a failure leaves a cache that reads as empty (pages already mapped for it are overwritten by the next write).
pub fn restore(
    path: &Path,
    engine: &Engine,
    cache: &mut EngineCache,
    prompt: &[u32],
    now: &Identity,
) -> Result<Restored, Reject> {
    let started = Instant::now();
    if cache.used() != 0 {
        return Err(Reject::Failed("the cache is not empty".into()));
    }
    let mut opened = open(path, now)?;
    let mut timing = Timing {
        header: started.elapsed(),
        ..Timing::default()
    };
    let tokens = restorable(
        shared_start(&opened.ids, prompt),
        opened.header.tokens as usize,
    );
    if tokens == 0 {
        return Err(Reject::NoMatch);
    }
    load_pages(&mut opened, engine, cache, tokens, &mut timing).map(|bytes| {
        opened.ids.truncate(tokens);
        Restored {
            ids: opened.ids,
            bytes,
            timing,
        }
    })
}
