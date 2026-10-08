use loader::ModelDir;
use model::{Engine, QuantBits};

use super::Identity;

/// What this run's cache is made of, or `None` when it cannot be saved (the CPU cache) or the model files cannot be
/// hashed. `weights` is the hash of the weight files.
pub fn identity(engine: &Engine, dir: &ModelDir, weights: u64) -> Option<Identity> {
    let layout = engine.kv_layout()?;
    let (weight_bits, weight_group) = engine.quantization().map_or((0, 0), |scheme| {
        let bits = match scheme.bits {
            QuantBits::Int8 => 8,
            QuantBits::Int4 => 4,
        };
        (bits, u32::try_from(scheme.group).unwrap_or(u32::MAX))
    });
    Some(Identity {
        format: layout.format,
        layers: u32::try_from(layout.layers).ok()?,
        kv_heads: u32::try_from(layout.kv_heads).ok()?,
        head_dim: u32::try_from(layout.head_dim).ok()?,
        weight_bits,
        weight_group,
        weights,
        config: dir.config_hash().ok()?,
    })
}
