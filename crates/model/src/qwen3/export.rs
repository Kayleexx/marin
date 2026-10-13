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
    let config = Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())?;
    let scheme = QuantScheme::new(QuantBits::Int8, group)?;
    let expected = expected_tensors(&config)?;
    validate(&dir.weights, &expected)?;
    let mut descriptors = Vec::new();
    for want in &expected {
        descriptors.push((want.name.clone(), want.shape.clone(), want.shape.len() == 2));
        if let [rows, cols] = want.shape.as_slice() {
            if cols.checked_rem(group) != Some(0) {
                return Err(ModelError::Unsupported(
                    "matrix columns are not divisible by quantization group".into(),
                ));
            }
            descriptors.push((
                format!("{}.scales", want.name),
                vec![*rows, cols.checked_div(group).unwrap_or(0)],
                false,
            ));
        }
    }
    let mut writer = PackedWriter::new(&dir.path, destination, group, descriptors)?;
    for want in expected {
        let bytes = dir.weights.info(&want.name).map_or(0, |i| i.byte_len());
        super::cpu_memory::ensure_available(bytes.saturating_mul(2).saturating_add(64 << 20))?;
        let bits = dir.weights.read_bf16(&want.name)?;
        writer.source_digest(
            &want.name,
            bits.chunks(1 << 18).map(|chunk| {
                chunk
                    .iter()
                    .flat_map(|b| b.to_le_bytes())
                    .collect::<Vec<_>>()
            }),
        );
        if let [rows, cols] = want.shape.as_slice() {
            let q = quantize(&bits, *rows, *cols, scheme)?;
            writer.write(&want.name, [&q.codes])?;
            writer.write(
                &format!("{}.scales", want.name),
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
                &want.name,
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
