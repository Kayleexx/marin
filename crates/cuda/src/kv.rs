use cudarc::driver::CudaSlice;

use crate::{Buffer, CudaError, Gpu};

/// Per-layer key/value cache in GPU memory, laid out as `[position][kv_head][head_dim]`.
pub struct DeviceKv {
    pub(crate) keys: Vec<CudaSlice<f32>>,
    pub(crate) values: Vec<CudaSlice<f32>>,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
}

impl DeviceKv {
    pub fn new(
        gpu: &Gpu,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self, CudaError> {
        let overflow = CudaError::Overflow { what: "kv cache" };
        let floats = kv_heads
            .checked_mul(head_dim)
            .and_then(|stride| stride.checked_mul(capacity))
            .ok_or(overflow)?;
        let mut keys = Vec::with_capacity(layers);
        let mut values = Vec::with_capacity(layers);
        for _ in 0..layers {
            keys.push(gpu.stream.alloc_zeros::<f32>(floats)?);
            values.push(gpu.stream.alloc_zeros::<f32>(floats)?);
        }
        Ok(Self {
            keys,
            values,
            kv_heads,
            head_dim,
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

    pub fn advance(&mut self, tokens: usize) -> Result<(), CudaError> {
        let new_len = self
            .len
            .checked_add(tokens)
            .filter(|n| *n <= self.capacity)
            .ok_or(CudaError::Shape {
                what: "kv positions",
                expected: self.capacity,
                actual: self.len.saturating_add(tokens),
            })?;
        self.len = new_len;
        Ok(())
    }

    /// Stores keys and values for the chunk starting at the committed length.
    pub fn write(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        keys: &Buffer,
        values: &Buffer,
    ) -> Result<(), CudaError> {
        let stride = self.kv_heads.saturating_mul(self.head_dim);
        let tokens = keys.len.checked_div(stride).unwrap_or(0);
        if keys.len != tokens.saturating_mul(stride) || values.len != keys.len {
            return Err(CudaError::Shape {
                what: "kv chunk",
                expected: tokens.saturating_mul(stride),
                actual: keys.len,
            });
        }
        let end_pos = self.len.saturating_add(tokens);
        if end_pos > self.capacity {
            return Err(CudaError::Shape {
                what: "kv positions",
                expected: self.capacity,
                actual: end_pos,
            });
        }
        let start = self.len.saturating_mul(stride);
        let end = start.saturating_add(keys.len);
        let layers = self.keys.len();
        let (Some(k_dst), Some(v_dst)) = (self.keys.get_mut(layer), self.values.get_mut(layer))
        else {
            return Err(CudaError::Shape {
                what: "kv layer",
                expected: layers,
                actual: layer,
            });
        };
        gpu.stream
            .memcpy_dtod(&keys.data, &mut k_dst.slice_mut(start..end))?;
        gpu.stream
            .memcpy_dtod(&values.data, &mut v_dst.slice_mut(start..end))?;
        Ok(())
    }
}
