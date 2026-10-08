//! The snapshot file: a header that says what the cache was made of, the token ids, one block per 64-position page, and
//! an end mark. Everything is little-endian. Every part carries a hash, so a damaged or half-written file is refused.
//!
//! ```text
//! header (80 bytes): magic, version, format, layers, kv heads, head size, page positions, positions,
//!                    weight bits, weight group, weights hash, config hash, ids hash, hash of the header so far
//! ids:               positions * u32
//! per page:          [layer][keys, values][row] bytes, then a hash of the page number and those bytes
//! end:               "MARINEND", total length
//! ```

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::Hasher;

use model::{KvFormat, PAGE_TOKENS, pages_for};

pub const HEADER_BYTES: usize = 80;
pub const END_BYTES: usize = 16;
pub const PAGE_HASH_BYTES: usize = 8;
const MAGIC: &[u8; 8] = b"MARINKV\0";
const END_MAGIC: &[u8; 8] = b"MARINEND";
const VERSION: u32 = 1;

pub fn hash_parts(parts: &[&[u8]]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for part in parts {
        hasher.write(part);
    }
    hasher.finish()
}

pub fn ids_hash(ids: &[u32]) -> u64 {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    hash_parts(&[&bytes])
}

pub fn page_hash(page: usize, payload: &[u8]) -> u64 {
    hash_parts(&[&(page as u64).to_le_bytes(), payload])
}

/// Why a snapshot was not used. None of these is an error for the user: the prompt is simply read in full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    Missing,
    Damaged(&'static str),
    Foreign(&'static str),
    NoMatch,
    Failed(String),
}

impl fmt::Display for Reject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(f, "no snapshot"),
            Self::Damaged(what) => write!(f, "snapshot is damaged: {what}"),
            Self::Foreign(what) => write!(f, "snapshot is from another setup: {what}"),
            Self::NoMatch => write!(f, "snapshot does not start like this prompt"),
            Self::Failed(why) => write!(f, "snapshot could not be used: {why}"),
        }
    }
}

/// Everything that decides what a stored key or value means. Two caches may be exchanged only if these are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    pub format: KvFormat,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub weight_bits: u32,
    pub weight_group: u32,
    pub weights: u64,
    pub config: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub identity: Identity,
    pub tokens: u32,
    pub ids_hash: u64,
}

fn format_code(format: KvFormat) -> u32 {
    match format {
        KvFormat::F32 => 0,
        KvFormat::F16 => 1,
    }
}

fn format_from(code: u32) -> Option<KvFormat> {
    KvFormat::ALL.into_iter().find(|f| format_code(*f) == code)
}

fn word(bytes: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(4)?;
    Some(u32::from_le_bytes(
        bytes.get(start..start.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn long(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let id = &self.identity;
        let mut out = Vec::with_capacity(HEADER_BYTES);
        out.extend_from_slice(MAGIC);
        for value in [
            VERSION,
            format_code(id.format),
            id.layers,
            id.kv_heads,
            id.head_dim,
            PAGE_TOKENS as u32,
            self.tokens,
            id.weight_bits,
            id.weight_group,
            0,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in [id.weights, id.config, self.ids_hash] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let hash = hash_parts(&[&out]);
        out.extend_from_slice(&hash.to_le_bytes());
        let mut fixed = [0_u8; HEADER_BYTES];
        fixed.copy_from_slice(&out);
        fixed
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, Reject> {
        let damaged = Reject::Damaged("header");
        if bytes.len() != HEADER_BYTES || bytes.get(..8) != Some(MAGIC.as_slice()) {
            return Err(Reject::Damaged("not a snapshot"));
        }
        let (body, stored) = bytes.split_at(HEADER_BYTES - 8);
        if long(stored, 0) != Some(hash_parts(&[body])) {
            return Err(damaged);
        }
        let field = |index: usize| word(body.get(8..).unwrap_or_default(), index);
        if field(0) != Some(VERSION) {
            return Err(Reject::Foreign("file format version"));
        }
        if field(5) != Some(PAGE_TOKENS as u32) {
            return Err(Reject::Foreign("page size"));
        }
        let format = field(1)
            .and_then(format_from)
            .ok_or(Reject::Foreign("KV type"))?;
        let get = |index: usize| field(index).ok_or(Reject::Damaged("header"));
        Ok(Self {
            identity: Identity {
                format,
                layers: get(2)?,
                kv_heads: get(3)?,
                head_dim: get(4)?,
                weight_bits: get(7)?,
                weight_group: get(8)?,
                weights: long(body, 48).ok_or(damaged.clone())?,
                config: long(body, 56).ok_or(damaged.clone())?,
            },
            tokens: get(6)?,
            ids_hash: long(body, 64).ok_or(damaged)?,
        })
    }

    /// Names the first thing that differs from what this run would store.
    pub fn check_against(&self, now: &Identity) -> Result<(), Reject> {
        let then = &self.identity;
        let checks = [
            (then.format == now.format, "KV type"),
            (then.layers == now.layers, "layer count"),
            (then.kv_heads == now.kv_heads, "kv heads"),
            (then.head_dim == now.head_dim, "head size"),
            (then.weight_bits == now.weight_bits, "weight quantization"),
            (
                then.weight_group == now.weight_group,
                "weight quantization group",
            ),
            (
                then.config == now.config,
                "model configuration or tokenizer",
            ),
            (then.weights == now.weights, "model weights"),
        ];
        match checks.into_iter().find(|(same, _)| !same) {
            Some((_, what)) => Err(Reject::Foreign(what)),
            None => Ok(()),
        }
    }

    pub fn row_bytes(&self) -> Option<usize> {
        self.identity.format.row_bytes(
            self.identity.kv_heads as usize,
            self.identity.head_dim as usize,
        )
    }

    /// Bytes of the block for a page of `rows` positions, without its hash.
    pub fn page_bytes(&self, rows: usize) -> Option<usize> {
        self.row_bytes()?
            .checked_mul(rows)?
            .checked_mul(self.identity.layers as usize)?
            .checked_mul(2)
    }

    /// Positions in page `page` of the whole snapshot.
    pub fn rows_of(&self, page: usize) -> usize {
        (self.tokens as usize)
            .saturating_sub(page.saturating_mul(PAGE_TOKENS))
            .min(PAGE_TOKENS)
    }

    pub fn pages_offset(&self) -> usize {
        HEADER_BYTES.saturating_add((self.tokens as usize).saturating_mul(4))
    }

    /// Exact size of a complete file.
    pub fn file_bytes(&self) -> Option<usize> {
        let mut total = self.pages_offset().checked_add(END_BYTES)?;
        for page in 0..pages_for(self.tokens as usize) {
            total = total
                .checked_add(self.page_bytes(self.rows_of(page))?)?
                .checked_add(PAGE_HASH_BYTES)?;
        }
        Some(total)
    }
}

pub fn end_mark(total: usize) -> Vec<u8> {
    let mut out = END_MAGIC.to_vec();
    out.extend_from_slice(&(total as u64).to_le_bytes());
    out
}

pub fn end_is_valid(bytes: &[u8], total: usize) -> bool {
    bytes == end_mark(total)
}
