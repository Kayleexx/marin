use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::LoaderError;

const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 1 << 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    Other,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    dtype_name: String,
    pub shape: Vec<usize>,
    begin: u64,
    end: u64,
}

impl TensorInfo {
    pub fn byte_len(&self) -> u64 {
        self.end.saturating_sub(self.begin)
    }

    pub fn dtype_name(&self) -> &str {
        &self.dtype_name
    }
}

#[derive(Debug)]
pub struct SafetensorsFile {
    path: PathBuf,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl SafetensorsFile {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        let bad = |reason: String| LoaderError::Safetensors {
            path: path.to_path_buf(),
            reason,
        };
        let io = |source| LoaderError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|_| bad("file is shorter than 8 bytes".into()))?;
        let header_len = u64::from_le_bytes(prefix);
        let data_start = header_len
            .checked_add(8)
            .filter(|_| header_len <= MAX_HEADER_BYTES)
            .filter(|start| *start <= file_len)
            .ok_or_else(|| bad(format!("header length {header_len} is invalid")))?;
        let mut header = vec![0_u8; usize::try_from(header_len).map_err(|e| bad(e.to_string()))?];
        file.read_exact(&mut header).map_err(io)?;
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|e| bad(format!("bad header JSON: {e}")))?;
        let data_len = file_len.saturating_sub(data_start);
        let mut tensors = BTreeMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let tensor: RawTensor =
                serde_json::from_value(value).map_err(|e| bad(format!("tensor {name}: {e}")))?;
            let info = validate(&name, tensor, data_len).map_err(bad)?;
            tensors.insert(name, info);
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn read_bf16(&self, name: &str) -> Result<Vec<u16>, LoaderError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LoaderError::UnknownTensor(name.to_string()))?;
        if info.dtype != Dtype::Bf16 {
            return Err(LoaderError::UnsupportedDtype {
                name: name.to_string(),
                found: info.dtype_name.clone(),
            });
        }
        let io = |source| LoaderError::Io {
            path: self.path.clone(),
            source,
        };
        let mut file = File::open(&self.path).map_err(io)?;
        let offset = self.data_start.saturating_add(info.begin);
        file.seek(SeekFrom::Start(offset)).map_err(io)?;
        let total = usize::try_from(info.byte_len()).map_err(|e| LoaderError::Safetensors {
            path: self.path.clone(),
            reason: e.to_string(),
        })?;
        let mut out = Vec::with_capacity(total.checked_div(2).unwrap_or(0));
        let mut buffer = vec![0_u8; READ_CHUNK_BYTES.min(total)];
        let mut remaining = total;
        while remaining > 0 {
            let want = remaining.min(buffer.len());
            let chunk = buffer.get_mut(..want).unwrap_or(&mut []);
            file.read_exact(chunk).map_err(io)?;
            out.extend(
                chunk
                    .chunks_exact(2)
                    .filter_map(|b| <[u8; 2]>::try_from(b).ok())
                    .map(u16::from_le_bytes),
            );
            remaining = remaining.saturating_sub(want);
        }
        Ok(out)
    }
}

fn validate(name: &str, tensor: RawTensor, data_len: u64) -> Result<TensorInfo, String> {
    let [begin, end] = tensor.data_offsets;
    if begin > end {
        return Err(format!("tensor {name}: start offset is after end offset"));
    }
    if end > data_len {
        return Err(format!(
            "tensor {name} ends at byte {end} but the file only has {data_len} bytes of data (partial download?)"
        ));
    }
    let stored = end.saturating_sub(begin);
    let (dtype, width) = match tensor.dtype.as_str() {
        "BF16" => (Dtype::Bf16, 2_u64),
        "F16" => (Dtype::Other, 2),
        "F32" => (Dtype::Other, 4),
        _ => (Dtype::Other, 0),
    };
    if width > 0 {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1_u64, |acc, d| acc.checked_mul(u64::try_from(*d).ok()?))
            .and_then(|n| n.checked_mul(width));
        if elements != Some(stored) {
            return Err(format!(
                "tensor {name}: shape {:?} does not match its {stored} stored bytes",
                tensor.shape
            ));
        }
    }
    Ok(TensorInfo {
        dtype,
        dtype_name: tensor.dtype,
        shape: tensor.shape,
        begin,
        end,
    })
}
