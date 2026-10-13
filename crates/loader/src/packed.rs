use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{LoaderError, Weights};

pub const MANIFEST: &str = "marin-int8.json";
const FORMAT: &str = "marin-q8-v1";
const QUANTIZER: &str = "symmetric-maxabs-round-v1";

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    format: String,
    quantizer: String,
    architecture: String,
    pub group: usize,
    #[serde(deserialize_with = "crate::packed_read::unique")]
    files: BTreeMap<String, [u8; 32]>,
    #[serde(deserialize_with = "crate::packed_read::unique")]
    pub tensors: BTreeMap<String, [u8; 32]>,
    #[serde(deserialize_with = "crate::packed_read::unique")]
    source_tensors: BTreeMap<String, [u8; 32]>,
}

pub(crate) fn bad(reason: impl Into<String>) -> LoaderError {
    LoaderError::Packed(reason.into())
}

pub(crate) fn io(path: &Path, source: std::io::Error) -> LoaderError {
    LoaderError::Io {
        path: path.into(),
        source,
    }
}

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn file_digest(path: &Path) -> Result<[u8; 32], LoaderError> {
    let mut file = File::open(path).map_err(|e| io(path, e))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(|e| io(path, e))?;
        if n == 0 {
            break;
        }
        hash.update(buffer.get(..n).ok_or_else(|| bad("digest read length"))?);
    }
    Ok(hash.finalize().into())
}

impl Manifest {
    pub(crate) fn open(dir: &Path) -> Result<Self, LoaderError> {
        let path = dir.join(MANIFEST);
        if !fs::symlink_metadata(&path)
            .map_err(|e| io(&path, e))?
            .file_type()
            .is_file()
            || !fs::symlink_metadata(dir.join("model.safetensors"))
                .map_err(|e| io(dir, e))?
                .file_type()
                .is_file()
        {
            return Err(bad("artifact files must be regular files"));
        }
        let file = File::open(&path).map_err(|e| io(&path, e))?;
        if file.metadata().map_err(|e| io(&path, e))?.len() > 4 << 20 {
            return Err(bad("manifest exceeds 4 MiB"));
        }
        let result: Self = serde_json::from_reader(file).map_err(|e| bad(e.to_string()))?;
        if result.format != FORMAT
            || result.quantizer != QUANTIZER
            || result.architecture != "qwen3"
            || result.group == 0
            || result.group.checked_rem(32) != Some(0)
        {
            return Err(bad("incompatible format, architecture, quantizer or group"));
        }
        for required in ["config.json", "tokenizer.json"] {
            if !result.files.contains_key(required) {
                return Err(bad(format!("missing {required} digest")));
            }
        }
        for name in [
            "config.json",
            "tokenizer.json",
            "generation_config.json",
            "tokenizer_config.json",
        ] {
            if dir.join(name).try_exists().map_err(|e| io(dir, e))?
                && !fs::symlink_metadata(dir.join(name))
                    .map_err(|e| io(dir, e))?
                    .file_type()
                    .is_file()
            {
                return Err(bad("artifact metadata must be regular files"));
            }
            match (
                result.files.get(name),
                dir.join(name).try_exists().map_err(|e| io(dir, e))?,
            ) {
                (Some(want), true) if *want == file_digest(&dir.join(name))? => {}
                (None, false) => {}
                _ => return Err(bad(format!("metadata checksum/presence mismatch: {name}"))),
            }
        }
        if result.files.len()
            != 2_usize
                .saturating_add(usize::from(
                    result.files.contains_key("generation_config.json"),
                ))
                .saturating_add(usize::from(
                    result.files.contains_key("tokenizer_config.json"),
                ))
        {
            return Err(bad("unknown metadata file"));
        }
        Ok(result)
    }
}

#[derive(Serialize)]
struct Descriptor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Streaming writer for one immutable bundle; dropping an unfinished writer removes only its own temporary directory.
pub struct PackedWriter {
    temp: PathBuf,
    destination: PathBuf,
    file: BufWriter<File>,
    manifest: Manifest,
    remaining: BTreeMap<String, u64>,
    order: Vec<String>,
    next: usize,
    failed: bool,
}

impl PackedWriter {
    pub fn new(
        source: &Path,
        destination: &Path,
        group: usize,
        tensors: Vec<(String, Vec<usize>, bool)>,
    ) -> Result<Self, LoaderError> {
        if group == 0 || group.checked_rem(32) != Some(0) {
            return Err(bad("group must be a positive multiple of 32"));
        }
        if destination.exists() {
            return Err(bad("destination exists; refusing to overwrite it"));
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| bad(e.to_string()))?
            .as_nanos();
        let temp = parent.join(format!(".marin-q8-{}-{stamp}", std::process::id()));
        fs::create_dir(&temp).map_err(|e| io(&temp, e))?;
        // Construct the guard before any fallible writes so failed exports clean up too.
        let path = temp.join("model.safetensors");
        let file = match File::options().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(e) => {
                let _ = fs::remove_dir(&temp);
                return Err(io(&path, e));
            }
        };
        let mut writer = Self {
            temp,
            destination: destination.into(),
            file: BufWriter::with_capacity(1 << 20, file),
            manifest: Manifest {
                format: FORMAT.into(),
                quantizer: QUANTIZER.into(),
                architecture: "qwen3".into(),
                group,
                files: BTreeMap::new(),
                tensors: BTreeMap::new(),
                source_tensors: BTreeMap::new(),
            },
            remaining: BTreeMap::new(),
            order: Vec::new(),
            next: 0,
            failed: false,
        };
        for name in [
            "config.json",
            "tokenizer.json",
            "generation_config.json",
            "tokenizer_config.json",
        ] {
            let input = source.join(name);
            if input.is_file() {
                let output = writer.temp.join(name);
                fs::copy(&input, &output).map_err(|e| io(&input, e))?;
                File::open(&output)
                    .and_then(|f| f.sync_all())
                    .map_err(|e| io(&output, e))?;
                writer
                    .manifest
                    .files
                    .insert(name.into(), file_digest(&output)?);
            }
        }
        let mut header = BTreeMap::new();
        let mut offset = 0_u64;
        for (name, shape, codes) in tensors {
            let size = shape
                .iter()
                .try_fold(1_u64, |n, d| n.checked_mul(u64::try_from(*d).ok()?))
                .and_then(|n| n.checked_mul(if codes { 1 } else { 4 }))
                .ok_or_else(|| bad("tensor size overflow"))?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| bad("offset overflow"))?;
            if header
                .insert(
                    name.clone(),
                    Descriptor {
                        dtype: if codes { "I8" } else { "F32" }.into(),
                        shape,
                        data_offsets: [offset, end],
                    },
                )
                .is_some()
            {
                return Err(bad("duplicate tensor"));
            }
            writer.remaining.insert(name.clone(), size);
            writer.order.push(name);
            offset = end;
        }
        let mut bytes = serde_json::to_vec(&header).map_err(|e| bad(e.to_string()))?;
        while bytes.len().checked_rem(8) != Some(0) {
            bytes.push(b' ');
        }
        writer
            .file
            .write_all(
                &u64::try_from(bytes.len())
                    .map_err(|e| bad(e.to_string()))?
                    .to_le_bytes(),
            )
            .and_then(|()| writer.file.write_all(&bytes))
            .map_err(|e| io(&path, e))?;
        Ok(writer)
    }

    pub fn write(
        &mut self,
        name: &str,
        chunks: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Result<(), LoaderError> {
        if self.failed {
            return Err(bad("a previous write failed; discard this export"));
        }
        if self.order.get(self.next).map(String::as_str) != Some(name) {
            return Err(bad("tensor write order mismatch"));
        }
        let want = self
            .remaining
            .get(name)
            .copied()
            .ok_or_else(|| bad("unknown/duplicate tensor write"))?;
        // A caller may catch an I/O error; partial writes cannot be retried or published.
        self.failed = true;
        let mut count = 0_u64;
        let mut hash = Sha256::new();
        for chunk in chunks {
            let bytes = chunk.as_ref();
            count = count
                .checked_add(u64::try_from(bytes.len()).map_err(|e| bad(e.to_string()))?)
                .filter(|n| *n <= want)
                .ok_or_else(|| bad("tensor write size mismatch"))?;
            hash.update(bytes);
            self.file.write_all(bytes).map_err(|e| io(&self.temp, e))?;
        }
        if count != want {
            return Err(bad("incomplete tensor write"));
        }
        self.manifest
            .tensors
            .insert(name.into(), hash.finalize().into());
        self.remaining.remove(name);
        self.failed = false;
        self.next = self.next.saturating_add(1);
        Ok(())
    }

    pub fn source_digest(
        &mut self,
        name: &str,
        chunks: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) {
        let mut hash = Sha256::new();
        for bytes in chunks {
            hash.update(bytes.as_ref());
        }
        self.manifest
            .source_tensors
            .insert(name.into(), hash.finalize().into());
    }

    pub fn finish(mut self) -> Result<(), LoaderError> {
        if self.failed || !self.remaining.is_empty() {
            return Err(bad("incomplete export"));
        }
        self.file.flush().map_err(|e| io(&self.temp, e))?;
        self.file
            .get_ref()
            .sync_all()
            .map_err(|e| io(&self.temp, e))?;
        let path = self.temp.join(MANIFEST);
        let bytes = serde_json::to_vec(&self.manifest).map_err(|e| bad(e.to_string()))?;
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| io(&path, e))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|e| io(&path, e))?;
        File::open(&self.temp)
            .and_then(|f| f.sync_all())
            .map_err(|e| io(&self.temp, e))?;
        if let Err(error) = fs::rename(&self.temp, &self.destination) {
            if !self.destination.is_dir() {
                return Err(io(&self.destination, error));
            }
            if Manifest::open(&self.destination)? != self.manifest {
                return Err(bad("concurrent export differs"));
            }
            let existing = Weights::open(&self.destination)?;
            for name in existing.names() {
                existing.verify_packed(name)?;
            }
        }
        let parent = self
            .destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| io(parent, e))?;
        Ok(())
    }
}

impl Drop for PackedWriter {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.temp);
    }
}
