use std::path::Path;

use loader::{ModelDir, packed::PackedWriter};
use tensor::{QuantBits, QuantScheme, bf16_to_f32, quantize};

use super::{
    config::Qwen3Config,
    weights::{expected_tensors, validate},
};
use crate::ModelError;

pub fn export_int8(dir: &ModelDir, destination: &Path, group: usize) -> Result<(), ModelError> {
    if dir.weights.packed_group().is_some() {
        return Err(ModelError::Unsupported(
            "export requires original BF16 weights".into(),
        ));
    }
    let scheme = QuantScheme::new(QuantBits::Int8, group)?;
    let expected = if crate::qwen35::is_qwen35(&dir.config_json) {
        let config = crate::qwen35::TextConfig::parse(
            &dir.config_json,
            dir.generation_config_json.as_deref(),
        )?;
        crate::qwen35::layout::validate(&dir.weights, &config)?;
        crate::qwen35::layout::tensors(&config)?
    } else {
        let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
        let expected = expected_tensors(&config)?;
        validate(&dir.weights, &expected)?;
        expected
            .into_iter()
            .map(|w| (w.name, w.shape.clone(), w.shape.len() == 2))
            .collect()
    };
    let mut descriptors = Vec::new();
    for (name, shape, codes) in &expected {
        descriptors.push((name.clone(), shape.clone(), *codes));
        if *codes {
            let [rows, cols] = shape.as_slice() else {
                return Err(ModelError::Unsupported("quantized matrix shape".into()));
            };
            if cols.checked_rem(group) != Some(0) {
                return Err(ModelError::Unsupported(
                    "matrix columns are not divisible by quantization group".into(),
                ));
            }
            descriptors.push((
                format!("{name}.scales"),
                vec![*rows, cols.checked_div(group).unwrap_or(0)],
                false,
            ));
        }
    }
    let mut writer = PackedWriter::new(&dir.path, destination, group, descriptors)?;
    for (name, shape, codes) in expected {
        let bytes = dir.weights.info(&name).map_or(0, |i| i.byte_len());
        super::cpu_memory::ensure_available(bytes.saturating_mul(2).saturating_add(64 << 20))?;
        let f32_source = dir
            .weights
            .info(&name)
            .is_some_and(|i| i.dtype == loader::Dtype::F32);
        if f32_source {
            if codes {
                return Err(ModelError::Unsupported(
                    "quantization requires BF16 matrix source".into(),
                ));
            }
            let values = dir.weights.read_f32(&name)?;
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.source_digest(&name, [&bytes]);
            writer.write(&name, [&bytes])?;
            continue;
        }
        let bits = dir.weights.read_bf16(&name)?;
        writer.source_digest(
            &name,
            bits.chunks(1 << 18).map(|chunk| {
                chunk
                    .iter()
                    .flat_map(|b| b.to_le_bytes())
                    .collect::<Vec<_>>()
            }),
        );
        if codes {
            let [rows, cols] = shape.as_slice() else {
                return Err(ModelError::Unsupported("matrix shape".into()));
            };
            let q = quantize(&bits, *rows, *cols, scheme)?;
            writer.write(&name, [&q.codes])?;
            writer.write(
                &format!("{}.scales", name),
                q.scales.chunks(1 << 16).map(|chunk| {
                    chunk
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                }),
            )?;
        } else {
            if bits.iter().any(|v| !bf16_to_f32(*v).is_finite()) {
                return Err(ModelError::Unsupported("nonfinite norm tensor".into()));
            }
            writer.write(
                &name,
                bits.chunks(1 << 16).map(|chunk| {
                    chunk
                        .iter()
                        .flat_map(|b| bf16_to_f32(*b).to_le_bytes())
                        .collect::<Vec<_>>()
                }),
            )?;
        }
    }
    writer.finish()?;
    Ok(())
}
