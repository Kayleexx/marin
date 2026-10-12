#![forbid(unsafe_code)]

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvError {
    #[error("context is full: need {needed} positions but capacity is {capacity}")]
    ContextFull { needed: usize, capacity: usize },
    #[error("{what}: expected a multiple of {stride} values, got {actual}")]
    Shape {
        what: &'static str,
        stride: usize,
        actual: usize,
    },
    #[error("layer {layer} is out of range (cache has {layers})")]
    Layer { layer: usize, layers: usize },
    #[error("cannot roll back to {len} positions: only {held} are held")]
    RollbackPastEnd { len: usize, held: usize },
    #[error("cache size arithmetic overflowed")]
    Overflow,
    #[error("cannot allocate KV storage: {0}")]
    Allocation(String),
}

#[derive(Debug)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// Contiguous per-layer cache laid out as `[position][kv_head][head_dim]`.
/// All layers write the same positions during one forward pass, then `advance` commits them.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKv>,
    stride: usize,
    capacity: usize,
    len: usize,
}

impl KvCache {
    pub fn new(
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, KvError> {
        let stride = kv_heads.checked_mul(head_dim).ok_or(KvError::Overflow)?;
        Self::bytes_for(layers, kv_heads, head_dim, capacity).ok_or(KvError::Overflow)?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(layers).map_err(allocation)?;
        storage.resize_with(layers, || LayerKv {
            keys: Vec::new(),
            values: Vec::new(),
        });
        Ok(Self {
            layers: storage,
            stride,
            capacity,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Actual vector capacity, including storage retained after rollback or clear.
    pub fn bytes_held(&self) -> usize {
        self.layers.iter().fold(0_usize, |bytes, layer| {
            bytes.saturating_add(
                layer
                    .keys
                    .capacity()
                    .saturating_add(layer.values.capacity())
                    .saturating_mul(size_of::<f32>()),
            )
        })
    }

    pub fn bytes_initialized(&self) -> usize {
        self.layers.iter().fold(0_usize, |bytes, layer| {
            bytes.saturating_add(
                layer
                    .keys
                    .len()
                    .saturating_add(layer.values.len())
                    .saturating_mul(size_of::<f32>()),
            )
        })
    }

    /// Largest possible temporary copy among vectors that actually need to grow.
    pub fn growth_copy_bytes(&self, positions: usize) -> Result<usize, KvError> {
        let (_, floats) = self.reservation_size(positions)?;
        Ok(self
            .layers
            .iter()
            .flat_map(|layer| [&layer.keys, &layer.values])
            .filter(|data| data.capacity() < floats)
            .map(|data| data.capacity())
            .max()
            .unwrap_or(0)
            .saturating_mul(size_of::<f32>()))
    }

    fn reservation_size(&self, positions: usize) -> Result<(usize, usize), KvError> {
        if positions > self.capacity {
            return Err(KvError::ContextFull {
                needed: positions,
                capacity: self.capacity,
            });
        }
        let used = positions
            .checked_mul(self.stride)
            .ok_or(KvError::Overflow)?;
        let positions = positions
            .checked_next_multiple_of(256)
            .ok_or(KvError::Overflow)?
            .min(self.capacity);
        let floats = positions
            .checked_mul(self.stride)
            .ok_or(KvError::Overflow)?;
        Ok((used, floats))
    }

    /// Reserve every layer before a pass can write any of them. Length commits separately.
    pub fn reserve(&mut self, positions: usize) -> Result<(), KvError> {
        let (used, floats) = self.reservation_size(positions)?;
        if self
            .layers
            .first()
            .is_some_and(|layer| layer.keys.len() >= used)
        {
            return Ok(());
        }
        for layer in &mut self.layers {
            for data in [&mut layer.keys, &mut layer.values] {
                data.try_reserve_exact(floats.saturating_sub(data.len()))
                    .map_err(allocation)?;
            }
        }
        for layer in &mut self.layers {
            layer.keys.resize(used.max(layer.keys.len()), 0.0);
            layer.values.resize(used.max(layer.values.len()), 0.0);
        }
        Ok(())
    }

    pub fn bytes_for(
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Option<usize> {
        [2, layers, kv_heads, head_dim, capacity, size_of::<f32>()]
            .iter()
            .try_fold(1_usize, |acc, n| acc.checked_mul(*n))
    }

    /// Stores keys and values for the chunk starting at the committed length.
    pub fn write(&mut self, layer: usize, keys: &[f32], values: &[f32]) -> Result<(), KvError> {
        let tokens = self.tokens_in(keys, "keys")?;
        if self.tokens_in(values, "values")? != tokens {
            return Err(KvError::Shape {
                what: "values",
                stride: self.stride,
                actual: values.len(),
            });
        }
        let end_pos = self.len.checked_add(tokens).ok_or(KvError::Overflow)?;
        if end_pos > self.capacity {
            return Err(KvError::ContextFull {
                needed: end_pos,
                capacity: self.capacity,
            });
        }
        let start = self.len.checked_mul(self.stride).ok_or(KvError::Overflow)?;
        let end = start.checked_add(keys.len()).ok_or(KvError::Overflow)?;
        let layers = self.layers.len();
        if layer >= layers {
            return Err(KvError::Layer { layer, layers });
        }
        self.reserve(end_pos)?;
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or(KvError::Layer { layer, layers })?;
        copy_range(&mut slot.keys, start, end, keys)?;
        copy_range(&mut slot.values, start, end, values)
    }

    /// Keys and values for the first `positions` positions of `layer`.
    pub fn read(&self, layer: usize, positions: usize) -> Result<(&[f32], &[f32]), KvError> {
        let end = positions
            .checked_mul(self.stride)
            .ok_or(KvError::Overflow)?;
        let layers = self.layers.len();
        let slot = self
            .layers
            .get(layer)
            .ok_or(KvError::Layer { layer, layers })?;
        match (slot.keys.get(..end), slot.values.get(..end)) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => Err(KvError::ContextFull {
                needed: positions,
                capacity: self.capacity,
            }),
        }
    }

    pub fn advance(&mut self, tokens: usize) -> Result<(), KvError> {
        let new_len = self.len.checked_add(tokens).ok_or(KvError::Overflow)?;
        if new_len > self.capacity {
            return Err(KvError::ContextFull {
                needed: new_len,
                capacity: self.capacity,
            });
        }
        self.reserve(new_len)?;
        self.len = new_len;
        Ok(())
    }

    /// Forgets every position from `len` on, so the next write lands there.
    pub fn truncate(&mut self, len: usize) -> Result<(), KvError> {
        if len > self.len {
            return Err(KvError::RollbackPastEnd {
                len,
                held: self.len,
            });
        }
        self.len = len;
        Ok(())
    }

    fn tokens_in(&self, data: &[f32], what: &'static str) -> Result<usize, KvError> {
        let shape = KvError::Shape {
            what,
            stride: self.stride,
            actual: data.len(),
        };
        match data.len().checked_rem(self.stride) {
            Some(0) => data.len().checked_div(self.stride).ok_or(shape),
            _ => Err(shape),
        }
    }
}

fn allocation(error: std::collections::TryReserveError) -> KvError {
    KvError::Allocation(error.to_string())
}

fn copy_range(dst: &mut [f32], start: usize, end: usize, src: &[f32]) -> Result<(), KvError> {
    dst.get_mut(start..end)
        .ok_or(KvError::Overflow)?
        .copy_from_slice(src);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> KvCache {
        KvCache::new(2, 1, 2, 3).unwrap()
    }

    #[test]
    fn written_chunks_are_readable_in_order() {
        let mut kv = cache();
        kv.write(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        kv.write(1, &[0.0; 4], &[0.0; 4]).unwrap();
        kv.advance(2).unwrap();
        kv.write(0, &[9.0, 9.0], &[8.0, 8.0]).unwrap();
        let (k, v) = kv.read(0, 3).unwrap();
        assert_eq!(k, [1.0, 2.0, 3.0, 4.0, 9.0, 9.0]);
        assert_eq!(v, [5.0, 6.0, 7.0, 8.0, 8.0, 8.0]);
    }

    #[test]
    fn overflowing_capacity_is_an_error() {
        let mut kv = cache();
        let err = kv.write(0, &[0.0; 8], &[0.0; 8]).unwrap_err();
        assert_eq!(
            err,
            KvError::ContextFull {
                needed: 4,
                capacity: 3
            }
        );
    }

    #[test]
    fn advance_cannot_pass_capacity() {
        let mut kv = cache();
        kv.advance(3).unwrap();
        assert!(kv.advance(1).is_err());
    }

    #[test]
    fn truncate_forgets_the_tail_and_the_next_write_replaces_it() {
        let mut kv = cache();
        kv.write(0, &[1.0, 2.0, 3.0, 4.0], &[0.0; 4]).unwrap();
        kv.advance(2).unwrap();
        kv.truncate(1).unwrap();
        assert_eq!(kv.len(), 1);
        kv.write(0, &[7.0, 8.0], &[0.0; 2]).unwrap();
        kv.advance(1).unwrap();
        assert_eq!(kv.read(0, 2).unwrap().0, [1.0, 2.0, 7.0, 8.0]);
        assert_eq!(
            kv.truncate(3),
            Err(KvError::RollbackPastEnd { len: 3, held: 2 })
        );
    }

    #[test]
    fn clear_resets_length() {
        let mut kv = cache();
        kv.advance(2).unwrap();
        kv.clear();
        assert!(kv.is_empty());
    }

    #[test]
    fn misaligned_and_mismatched_writes_are_rejected() {
        let mut kv = cache();
        assert!(kv.write(0, &[0.0; 3], &[0.0; 3]).is_err());
        assert!(kv.write(0, &[0.0; 4], &[0.0; 2]).is_err());
        assert!(kv.write(9, &[0.0; 2], &[0.0; 2]).is_err());
    }

    #[test]
    fn size_estimate_matches_allocation_and_detects_overflow() {
        assert_eq!(KvCache::bytes_for(2, 1, 2, 3), Some(2 * 2 * 2 * 3 * 4));
        assert_eq!(KvCache::bytes_for(usize::MAX, 2, 1, 1), None);
        assert!(KvCache::new(1, usize::MAX, 2, 1).is_err());
    }

    #[test]
    fn growth_preserves_all_layers_and_reuses_the_high_water_mark() {
        let mut kv = KvCache::new(2, 1, 2, 8192).unwrap();
        assert_eq!(kv.bytes_held(), 0);
        assert_eq!(kv.bytes_initialized(), 0);
        assert_eq!(kv.growth_copy_bytes(127).unwrap(), 0);
        for layer in 0..2 {
            kv.write(layer, &[layer as f32; 254], &[3.0; 254]).unwrap();
        }
        kv.advance(127).unwrap();
        assert_eq!(kv.capacity(), 8192);
        assert_eq!(kv.bytes_held(), KvCache::bytes_for(2, 1, 2, 256).unwrap());
        assert_eq!(
            kv.bytes_initialized(),
            KvCache::bytes_for(2, 1, 2, 127).unwrap()
        );
        assert_eq!(kv.growth_copy_bytes(127).unwrap(), 0);
        assert_eq!(kv.growth_copy_bytes(257).unwrap(), 256 * 2 * 4);
        kv.reserve(257).unwrap();
        let held = kv.bytes_held();
        for layer in 0..2 {
            assert_eq!(kv.read(layer, 127).unwrap().0, &[layer as f32; 254]);
        }
        kv.truncate(126).unwrap();
        kv.write(0, &[9.0, 10.0], &[11.0, 12.0]).unwrap();
        kv.advance(1).unwrap();
        assert_eq!(
            kv.read(0, 127).unwrap().0.get(252..),
            Some([9.0, 10.0].as_slice())
        );
        kv.clear();
        kv.reserve(1).unwrap();
        assert_eq!(kv.bytes_held(), held);
        assert_eq!(kv.growth_copy_bytes(257).unwrap(), 0);
        assert_eq!(
            kv.bytes_initialized(),
            KvCache::bytes_for(2, 1, 2, 257).unwrap()
        );
    }

    #[test]
    fn refused_growth_keeps_committed_data_and_length() {
        let mut kv = KvCache::new(2, 1, 1, usize::MAX / 16).unwrap();
        for layer in 0..2 {
            kv.write(layer, &[7.0], &[8.0]).unwrap();
        }
        kv.advance(1).unwrap();
        let held = kv.bytes_held();
        assert!(matches!(
            kv.reserve(usize::MAX / 16),
            Err(KvError::Allocation(_))
        ));
        assert_eq!(kv.len(), 1);
        assert_eq!(kv.bytes_held(), held);
        for layer in 0..2 {
            assert_eq!(
                kv.read(layer, 1).unwrap(),
                ([7.0].as_slice(), [8.0].as_slice())
            );
        }
        assert!(kv.write(2, &[0.0; 257], &[0.0; 257]).is_err());
        assert_eq!(kv.bytes_held(), held);
    }
}
