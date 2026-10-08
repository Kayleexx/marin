use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::LoaderError;
use crate::safetensors::{SafetensorsFile, TensorInfo};

pub const SINGLE_FILE: &str = "model.safetensors";
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// All tensors of a model, from one file or from the shards named by the index.
#[derive(Debug)]
pub struct Weights {
    files: Vec<SafetensorsFile>,
    owner: BTreeMap<String, usize>,
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

impl Weights {
    pub fn exists_in(dir: &Path) -> bool {
        dir.join(SINGLE_FILE).is_file() || dir.join(INDEX_FILE).is_file()
    }

    pub fn open(dir: &Path) -> Result<Self, LoaderError> {
        let single = dir.join(SINGLE_FILE);
        let paths = if single.is_file() {
            vec![single]
        } else {
            shard_paths(dir)?
        };
        let mut files = Vec::new();
        let mut owner = BTreeMap::new();
        for (index, path) in paths.iter().enumerate() {
            let file = SafetensorsFile::open(path)?;
            for name in file.tensors().keys() {
                if owner.insert(name.clone(), index).is_some() {
                    return Err(LoaderError::DuplicateTensor(name.clone()));
                }
            }
            files.push(file);
        }
        Ok(Self { files, owner })
    }

    /// Hashes every weight file on another thread, so it can overlap loading the weights. Join it before trusting
    /// anything that was computed with these weights.
    pub fn spawn_content_hash(&self) -> std::thread::JoinHandle<Result<u64, LoaderError>> {
        let paths: Vec<_> = self.files.iter().map(|f| f.path().to_path_buf()).collect();
        std::thread::spawn(move || crate::content_hash::hash_files(&paths))
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let file = self.files.get(*self.owner.get(name)?)?;
        file.tensors().get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.owner.keys().map(String::as_str)
    }

    pub fn total_bytes(&self) -> u64 {
        self.files
            .iter()
            .flat_map(|f| f.tensors().values())
            .map(TensorInfo::byte_len)
            .sum()
    }

    pub fn read_bf16(&self, name: &str) -> Result<Vec<u16>, LoaderError> {
        let file = self
            .owner
            .get(name)
            .and_then(|i| self.files.get(*i))
            .ok_or_else(|| LoaderError::UnknownTensor(name.to_string()))?;
        file.read_bf16(name)
    }
}

fn shard_paths(dir: &Path) -> Result<Vec<std::path::PathBuf>, LoaderError> {
    let index_path = dir.join(INDEX_FILE);
    let text = fs::read_to_string(&index_path).map_err(|source| LoaderError::Io {
        path: index_path.clone(),
        source,
    })?;
    let index: ShardIndex = serde_json::from_str(&text).map_err(|source| LoaderError::Json {
        path: index_path,
        source,
    })?;
    let shards: BTreeSet<&String> = index.weight_map.values().collect();
    let missing: Vec<String> = shards
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .map(|name| name.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(LoaderError::MissingFiles {
            dir: dir.to_path_buf(),
            files: missing,
        });
    }
    Ok(shards.into_iter().map(|name| dir.join(name)).collect())
}
