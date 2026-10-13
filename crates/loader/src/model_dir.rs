use std::fs;
use std::path::{Path, PathBuf};

use crate::weights::{INDEX_FILE, SINGLE_FILE};
use crate::{LoaderError, Tokenizer, Weights};

const CONFIG: &str = "config.json";
const GENERATION_CONFIG: &str = "generation_config.json";
const TOKENIZER: &str = "tokenizer.json";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";

/// Everything a Hugging Face model folder provides, opened and checked for presence.
pub struct ModelDir {
    pub path: PathBuf,
    pub config_json: String,
    pub generation_config_json: Option<String>,
    pub tokenizer_config_json: Option<String>,
    pub tokenizer: Tokenizer,
    pub weights: Weights,
}

impl ModelDir {
    pub fn open(path: &Path) -> Result<Self, LoaderError> {
        if !path.is_dir() {
            return Err(LoaderError::NotADirectory(path.to_path_buf()));
        }
        let mut missing = Vec::new();
        for required in [CONFIG, TOKENIZER] {
            if !path.join(required).is_file() {
                missing.push(required.to_string());
            }
        }
        if !Weights::exists_in(path) {
            missing.push(format!("{SINGLE_FILE} (or {INDEX_FILE})"));
        }
        if !missing.is_empty() {
            return Err(LoaderError::MissingFiles {
                dir: path.to_path_buf(),
                files: missing,
            });
        }
        let weights = Weights::open(path)?;
        let generation_path = path.join(GENERATION_CONFIG);
        let generation_config_json = generation_path
            .is_file()
            .then(|| read_text(&generation_path))
            .transpose()?;
        let tokenizer_config_path = path.join(TOKENIZER_CONFIG);
        let tokenizer_config_json = tokenizer_config_path
            .is_file()
            .then(|| read_text(&tokenizer_config_path))
            .transpose()?;
        Ok(Self {
            config_json: read_text(&path.join(CONFIG))?,
            generation_config_json,
            tokenizer_config_json,
            tokenizer: Tokenizer::from_file(&path.join(TOKENIZER))?,
            weights,
            path: path.to_path_buf(),
        })
    }
}

impl ModelDir {
    /// Hash of the files that decide what a cached key or value means besides the weights: the model configuration
    /// and the tokenizer.
    pub fn config_hash(&self) -> Result<u64, LoaderError> {
        crate::content_hash::hash_files(&[self.path.join(CONFIG), self.path.join(TOKENIZER)])
    }
}

fn read_text(path: &Path) -> Result<String, LoaderError> {
    fs::read_to_string(path).map_err(|source| LoaderError::Io {
        path: path.to_path_buf(),
        source,
    })
}
