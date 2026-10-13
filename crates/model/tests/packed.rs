use std::fs;
use std::path::{Path, PathBuf};

use loader::{Dtype, ModelDir};
use model::{QuantBits, QuantScheme, Qwen3, export_int8};
use runtime::{CausalLm, SequenceCache};
use tensor::{bf16_to_f32, quantize};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("marin-packed-{}-{stamp}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-chat")
}
fn scheme() -> Result<QuantScheme> {
    Ok(QuantScheme::new(QuantBits::Int8, 32)?)
}

#[test]
fn packed_data_logits_reuse_and_rollback_equal_current_int8() -> Result {
    let temp = Temp::new()?;
    let source = ModelDir::open(&fixture())?;
    let target = temp.0.join("packed");
    export_int8(&source, &target, 32)?;
    let packed = ModelDir::open(&target)?;
    for name in source.weights.names() {
        let info = source.weights.info(name).ok_or("missing tensor")?;
        if name == "lm_head.weight" {
            continue;
        }
        let bits = source.weights.read_bf16(name)?;
        if let [rows, cols] = info.shape.as_slice() {
            let expected = quantize(&bits, *rows, *cols, scheme()?)?;
            assert_eq!(packed.weights.read_i8(name)?, expected.codes);
            assert_eq!(
                packed.weights.read_f32(&format!("{name}.scales"))?,
                expected.scales
            );
        } else {
            assert_eq!(
                packed.weights.read_f32(name)?,
                bits.into_iter().map(bf16_to_f32).collect::<Vec<_>>()
            );
        }
    }
    assert!(!packed.weights.names().any(|n| n == "lm_head.weight"));
    assert!(Qwen3::load(&packed, |_, _| {}).is_err());
    assert!(
        Qwen3::load_quantized(
            &packed,
            Some(QuantScheme::new(QuantBits::Int8, 64)?),
            |_, _| {}
        )
        .is_err()
    );
    let old = Qwen3::load_quantized(&source, Some(scheme()?), |_, _| {})?;
    let new = Qwen3::load_quantized(&packed, Some(scheme()?), |_, _| {})?;
    for threads in [1, 8] {
        tensor::set_threads(threads);
        let mut a = old.new_cache(1024)?;
        let mut b = new.new_cache(1024)?;
        let prompt: Vec<_> = (0..513).map(|i| (i % 319) as u32).collect();
        assert_eq!(old.forward(&prompt, &mut a)?, new.forward(&prompt, &mut b)?);
        for token in [10, 15, 30] {
            assert_eq!(
                old.forward(&[token], &mut a)?,
                new.forward(&[token], &mut b)?
            );
        }
        a.truncate(127)?;
        b.truncate(127)?;
        assert_eq!(
            old.forward(&[71, 90], &mut a)?,
            new.forward(&[71, 90], &mut b)?
        );
        assert_eq!(a.used(), b.used());
    }
    tensor::set_threads(0);
    assert!(
        export_int8(&source, &target, 32).is_err(),
        "never overwrite a published bundle"
    );
    Ok(())
}

fn mutate_manifest(path: &Path, change: impl FnOnce(&mut serde_json::Value)) -> Result {
    let file = path.join("marin-int8.json");
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&file)?)?;
    change(&mut value);
    fs::write(file, serde_json::to_vec(&value)?)?;
    Ok(())
}

#[test]
fn corrupted_incompatible_and_nonfinite_artifacts_fail() -> Result {
    let temp = Temp::new()?;
    let source = ModelDir::open(&fixture())?;
    for case in [
        "checksum",
        "truncated",
        "version",
        "group",
        "metadata",
        "missing-meta",
        "nan",
        "negative",
        "code",
        "shape",
        "overlap",
        "duplicate",
    ] {
        let path = temp.0.join(case);
        export_int8(&source, &path, 32)?;
        let weights = path.join("model.safetensors");
        let mut bytes = fs::read(&weights)?;
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])?;
        let data = 8 + header_len;
        match case {
            "version" => mutate_manifest(&path, |v| v["format"] = "foreign-v9".into())?,
            "group" => mutate_manifest(&path, |v| v["group"] = 0.into())?,
            "metadata" => {
                fs::write(path.join("config.json"), b"{}")?;
            }
            "missing-meta" => {
                fs::remove_file(path.join("generation_config.json"))?;
            }
            "checksum" => {
                bytes[data] ^= 1;
                fs::write(weights, bytes)?;
            }
            "truncated" => {
                bytes.truncate(bytes.len() - 7);
                fs::write(weights, bytes)?;
            }
            "nan" | "negative" | "code" => {
                let name = if case == "code" {
                    "model.embed_tokens.weight"
                } else {
                    "model.embed_tokens.weight.scales"
                };
                let begin = header[name]["data_offsets"][0].as_u64().ok_or("offset")? as usize;
                let end = header[name]["data_offsets"][1].as_u64().ok_or("offset")? as usize;
                if case == "code" {
                    bytes[data + begin] = 128;
                } else {
                    bytes[data + begin..data + begin + 4].copy_from_slice(
                        &if case == "nan" { f32::NAN } else { -1.0_f32 }.to_le_bytes(),
                    );
                }
                let hash = loader::packed::digest(&bytes[data + begin..data + end]);
                mutate_manifest(&path, |v| v["tensors"][name] = serde_json::json!(hash))?;
                fs::write(weights, bytes)?;
            }
            "shape" => {
                header["model.embed_tokens.weight"]["shape"] = serde_json::json!([160, 128]);
                rewrite_header(&weights, &header, &bytes[data..])?;
            }
            "overlap" => {
                header["model.norm.weight"]["data_offsets"] = serde_json::json!([0, 256]);
                rewrite_header(&weights, &header, &bytes[data..])?;
            }
            "duplicate" => {
                let text = serde_json::to_string(&header)?;
                let text = text.replacen('{', "{\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[64],\"data_offsets\":[0,256]},",1);
                let mut out = (text.len() as u64).to_le_bytes().to_vec();
                out.extend(text.as_bytes());
                out.extend(&bytes[data..]);
                fs::write(weights, out)?;
            }
            _ => return Err("unknown case".into()),
        }
        let result = ModelDir::open(&path)
            .map_err(model::ModelError::from)
            .and_then(|dir| {
                Qwen3::load_quantized(
                    &dir,
                    Some(scheme().map_err(|e| model::ModelError::Unsupported(e.to_string()))?),
                    |_, _| {},
                )
            });
        assert!(result.is_err(), "accepted {case}");
    }
    Ok(())
}
fn rewrite_header(path: &Path, header: &serde_json::Value, data: &[u8]) -> Result {
    let header = serde_json::to_vec(header)?;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn interrupted_and_concurrent_publication_exposes_only_complete_bundles() -> Result {
    let temp = Temp::new()?;
    let target = temp.0.join("published");
    let descriptors = vec![("codes".to_owned(), vec![32], true)];
    let first = loader::packed::PackedWriter::new(&fixture(), &target, 32, descriptors.clone())?;
    drop(first);
    assert!(!target.exists());
    assert_eq!(fs::read_dir(&temp.0)?.count(), 0);
    let failed_target = temp.0.join("failed");
    let mut failed =
        loader::packed::PackedWriter::new(&fixture(), &failed_target, 32, descriptors.clone())?;
    assert!(failed.write("codes", [vec![0_u8; 16]]).is_err());
    assert!(
        failed.finish().is_err(),
        "a caught write error must not publish partial data"
    );
    assert!(!failed_target.exists());
    let mut a = loader::packed::PackedWriter::new(&fixture(), &target, 32, descriptors.clone())?;
    let mut b = loader::packed::PackedWriter::new(&fixture(), &target, 32, descriptors)?;
    a.write("codes", [vec![0_u8; 32]])?;
    b.write("codes", [vec![0_u8; 32]])?;
    a.finish()?;
    b.finish()?;
    let loaded = ModelDir::open(&target)?;
    assert_eq!(
        loaded.weights.info("codes").map(|i| i.dtype),
        Some(Dtype::I8)
    );
    assert_eq!(loaded.weights.read_i8("codes")?, vec![0; 32]);
    assert_eq!(fs::read_dir(&temp.0)?.count(), 1);
    Ok(())
}

#[test]
fn cuda_upload_preserves_packed_logits_and_format_identity() -> Result {
    let gpu = match model::Gpu::new() {
        Ok(gpu) => gpu,
        Err(error) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("CUDA unavailable: {error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let temp = Temp::new()?;
    let source = ModelDir::open(&fixture())?;
    let target = temp.0.join("packed");
    export_int8(&source, &target, 32)?;
    let packed = ModelDir::open(&target)?;
    let old = model::Qwen3Gpu::load(&source, gpu, Some(scheme()?), |_, _| {})?;
    let new = model::Qwen3Gpu::load(&packed, model::Gpu::new()?, Some(scheme()?), |_, _| {})?;
    assert_eq!(old.quantization(), new.quantization());
    let mut a = old.new_cache(64)?;
    let mut b = new.new_cache(64)?;
    for tokens in [&[1, 2, 3][..], &[4], &[5, 6]] {
        assert_eq!(old.forward(tokens, &mut a)?, new.forward(tokens, &mut b)?);
    }
    assert!(model::Qwen3Gpu::load(&packed, model::Gpu::new()?, None, |_, _| {}).is_err());
    Ok(())
}

#[test]
fn untied_heads_are_exported_and_loaded_separately() -> Result {
    let temp = Temp::new()?;
    let source = temp.0.join("source");
    fs::create_dir(&source)?;
    for name in [
        "config.json",
        "tokenizer.json",
        "generation_config.json",
        "tokenizer_config.json",
    ] {
        fs::copy(fixture().join(name), source.join(name))?;
    }
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(source.join("config.json"))?)?;
    config["tie_word_embeddings"] = false.into();
    fs::write(source.join("config.json"), serde_json::to_vec(&config)?)?;
    let bytes = fs::read(fixture().join("model.safetensors"))?;
    let len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len])?;
    let mut data = bytes[8 + len..].to_vec();
    let mut head = header["model.embed_tokens.weight"].clone();
    let begin = head["data_offsets"][0].as_u64().ok_or("offset")? as usize;
    let end = head["data_offsets"][1].as_u64().ok_or("offset")? as usize;
    let copied = data[begin..end].to_vec();
    head["data_offsets"] = serde_json::json!([data.len(), data.len() + copied.len()]);
    header["lm_head.weight"] = head;
    data.extend(copied);
    rewrite_header(&source.join("model.safetensors"), &header, &data)?;
    let dir = ModelDir::open(&source)?;
    let target = temp.0.join("packed");
    export_int8(&dir, &target, 32)?;
    let packed = ModelDir::open(&target)?;
    assert!(packed.weights.info("lm_head.weight.scales").is_some());
    let old = Qwen3::load_quantized(&dir, Some(scheme()?), |_, _| {})?;
    let new = Qwen3::load_quantized(&packed, Some(scheme()?), |_, _| {})?;
    assert_eq!(
        old.forward(&[1, 2, 3], &mut old.new_cache(64)?)?,
        new.forward(&[1, 2, 3], &mut new.new_cache(64)?)?
    );
    Ok(())
}
