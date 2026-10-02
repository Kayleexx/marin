use std::fs;
use std::path::PathBuf;

use loader::{LoaderError, ModelDir, Weights};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> std::io::Result<Self> {
        let path = std::env::temp_dir().join(format!("marin-loader-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Builds a safetensors file: (name, dtype, shape, bytes).
fn safetensors(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> serde_json::Result<Vec<u8>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let begin = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    Ok(file)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> std::io::Result<()> {
    fs::write(dir.0.join(name), bytes)
}

#[test]
fn reads_bf16_tensor_and_reports_shape() {
    let dir = TempDir::new("roundtrip").unwrap();
    let file = safetensors(&[("w", "BF16", &[2, 2], bf16_bytes(&[1.0, 2.0, 3.0, 4.0]))]).unwrap();
    write(&dir, "model.safetensors", &file).unwrap();
    let weights = Weights::open(&dir.0).unwrap();
    assert_eq!(weights.info("w").unwrap().shape, [2, 2]);
    assert_eq!(weights.total_bytes(), 8);
    let bits = weights.read_bf16("w").unwrap();
    assert_eq!(bits, [0x3f80, 0x4000, 0x4040, 0x4080]);
}

#[test]
fn truncated_file_is_reported_as_partial_download() {
    let dir = TempDir::new("truncated").unwrap();
    let mut file = safetensors(&[("w", "BF16", &[2, 2], bf16_bytes(&[1.0; 4]))]).unwrap();
    file.truncate(file.len() - 3);
    write(&dir, "model.safetensors", &file).unwrap();
    let err = Weights::open(&dir.0).unwrap_err().to_string();
    assert!(err.contains("partial download"), "{err}");
}

#[test]
fn absurd_header_length_is_rejected() {
    let dir = TempDir::new("header").unwrap();
    write(&dir, "model.safetensors", &u64::MAX.to_le_bytes()).unwrap();
    assert!(matches!(
        Weights::open(&dir.0),
        Err(LoaderError::Safetensors { .. })
    ));
}

#[test]
fn file_shorter_than_prefix_is_rejected() {
    let dir = TempDir::new("short").unwrap();
    write(&dir, "model.safetensors", &[1, 2, 3]).unwrap();
    assert!(Weights::open(&dir.0).is_err());
}

#[test]
fn shape_that_disagrees_with_byte_length_is_rejected() {
    let dir = TempDir::new("shape").unwrap();
    let file = safetensors(&[("w", "BF16", &[3, 3], bf16_bytes(&[1.0; 4]))]).unwrap();
    write(&dir, "model.safetensors", &file).unwrap();
    let err = Weights::open(&dir.0).unwrap_err().to_string();
    assert!(err.contains("does not match"), "{err}");
}

#[test]
fn non_bf16_tensor_opens_but_cannot_be_read() {
    let dir = TempDir::new("f32").unwrap();
    let file = safetensors(&[("w", "F32", &[1], vec![0; 4])]).unwrap();
    write(&dir, "model.safetensors", &file).unwrap();
    let weights = Weights::open(&dir.0).unwrap();
    assert!(matches!(
        weights.read_bf16("w"),
        Err(LoaderError::UnsupportedDtype { .. })
    ));
    assert!(matches!(
        weights.read_bf16("missing"),
        Err(LoaderError::UnknownTensor(_))
    ));
}

#[test]
fn sharded_weights_are_merged_through_the_index() {
    let dir = TempDir::new("shards").unwrap();
    write(
        &dir,
        "a.safetensors",
        &safetensors(&[("x", "BF16", &[1], bf16_bytes(&[1.0]))]).unwrap(),
    )
    .unwrap();
    write(
        &dir,
        "b.safetensors",
        &safetensors(&[("y", "BF16", &[1], bf16_bytes(&[2.0]))]).unwrap(),
    )
    .unwrap();
    let index = r#"{"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}"#;
    write(&dir, "model.safetensors.index.json", index.as_bytes()).unwrap();
    let weights = Weights::open(&dir.0).unwrap();
    assert_eq!(weights.names().collect::<Vec<_>>(), ["x", "y"]);
    assert_eq!(weights.read_bf16("y").unwrap(), [0x4000]);
}

#[test]
fn missing_shard_is_named() {
    let dir = TempDir::new("missing-shard").unwrap();
    write(
        &dir,
        "a.safetensors",
        &safetensors(&[("x", "BF16", &[1], bf16_bytes(&[1.0]))]).unwrap(),
    )
    .unwrap();
    let index = r#"{"weight_map": {"x": "a.safetensors", "y": "gone.safetensors"}}"#;
    write(&dir, "model.safetensors.index.json", index.as_bytes()).unwrap();
    let err = Weights::open(&dir.0).unwrap_err().to_string();
    assert!(err.contains("gone.safetensors"), "{err}");
}

#[test]
fn tensor_in_two_shards_is_rejected() {
    let dir = TempDir::new("duplicate").unwrap();
    let shard = safetensors(&[("x", "BF16", &[1], bf16_bytes(&[1.0]))]).unwrap();
    write(&dir, "a.safetensors", &shard).unwrap();
    write(&dir, "b.safetensors", &shard).unwrap();
    let index = r#"{"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}"#;
    write(&dir, "model.safetensors.index.json", index.as_bytes()).unwrap();
    assert!(matches!(
        Weights::open(&dir.0),
        Err(LoaderError::DuplicateTensor(_))
    ));
}

#[test]
fn model_dir_lists_every_missing_file_at_once() {
    let dir = TempDir::new("empty-model").unwrap();
    let Err(err) = ModelDir::open(&dir.0) else {
        panic!("empty directory must not open");
    };
    let text = err.to_string();
    for needed in ["config.json", "tokenizer.json", "model.safetensors"] {
        assert!(text.contains(needed), "{text}");
    }
}

#[test]
fn model_dir_rejects_a_file_path() {
    let dir = TempDir::new("not-a-dir").unwrap();
    write(&dir, "file", b"x").unwrap();
    assert!(matches!(
        ModelDir::open(&dir.0.join("file")),
        Err(LoaderError::NotADirectory(_))
    ));
}
