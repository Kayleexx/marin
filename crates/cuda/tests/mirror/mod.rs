use std::error::Error;

use crate::attention_reference::reference_attention;
use crate::common::{assert_close, values};
use crate::kv_round::round_trip;
use cuda::{DeviceKv, Gpu, KvFormat};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const LAYERS: usize = 2;
pub const SMALL: (usize, usize, usize) = (4, 2, 16);

#[derive(Default, Clone)]
pub struct Layer {
    pub keys: Vec<f32>,
    pub vals: Vec<f32>,
}

pub struct Mirror {
    pub kv: DeviceKv,
    pub dims: (usize, usize, usize),
    pub format: KvFormat,
    pub layers: Vec<Layer>,
    pub seed: u64,
}

impl Mirror {
    /// `layers` holds what the cache should return, so the values after rounding to `format`.
    pub fn new(
        gpu: &Gpu,
        dims: (usize, usize, usize),
        capacity: usize,
        format: KvFormat,
    ) -> Result<Self> {
        Ok(Self {
            kv: DeviceKv::new(gpu, LAYERS, dims, capacity, format)?,
            dims,
            format,
            layers: vec![Layer::default(); LAYERS],
            seed: 1,
        })
    }

    fn stride(&self) -> usize {
        self.dims.1.saturating_mul(self.dims.2)
    }

    fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(1);
        self.seed
    }

    fn fresh(&mut self, tokens: usize) -> Vec<Layer> {
        let count = tokens.saturating_mul(self.stride());
        (0..LAYERS)
            .map(|_| Layer {
                keys: values(count, self.next_seed()),
                vals: values(count, self.next_seed()),
            })
            .collect()
    }

    pub fn append(&mut self, gpu: &Gpu, tokens: usize) -> Result<()> {
        let new = self.fresh(tokens);
        for (layer, (held, add)) in self.layers.iter_mut().zip(new).enumerate() {
            let (k, v) = (gpu.upload(&add.keys)?, gpu.upload(&add.vals)?);
            self.kv.write(gpu, layer, &k, &v)?;
            held.keys.extend(round_trip(self.format, &add.keys));
            held.vals.extend(round_trip(self.format, &add.vals));
        }
        self.kv.advance(tokens)?;
        Ok(())
    }

    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.kv.truncate(len)?;
        let floats = len.saturating_mul(self.stride());
        for layer in &mut self.layers {
            layer.keys.truncate(floats);
            layer.vals.truncate(floats);
        }
        Ok(())
    }

    /// The attention output of every layer for `tokens` new tokens, next to the f64 reference for them. Frees the pages
    /// it opened.
    pub fn outputs(&mut self, gpu: &Gpu, tokens: usize) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let (heads, kv_heads, head_dim) = self.dims;
        let len = self.kv.len();
        let mut pairs = Vec::new();
        let seed = self.next_seed();
        let q = values(tokens.saturating_mul(heads).saturating_mul(head_dim), seed);
        let new = self.fresh(tokens);
        for (layer, (held, add)) in self.layers.iter().zip(new).enumerate() {
            self.kv
                .write(gpu, layer, &gpu.upload(&add.keys)?, &gpu.upload(&add.vals)?)?;
            let mut out = gpu.zeros(q.len())?;
            gpu.attention(
                &gpu.upload(&q)?,
                &mut self.kv,
                layer,
                heads,
                tokens,
                &mut out,
            )?;
            let keys = [held.keys.as_slice(), &round_trip(self.format, &add.keys)].concat();
            let vals = [held.vals.as_slice(), &round_trip(self.format, &add.vals)].concat();
            let expected =
                reference_attention(&q, &keys, &vals, (heads, kv_heads, head_dim, len, tokens));
            pairs.push((gpu.download(&out)?, expected));
        }
        // Frees a page the check may have opened.
        self.kv.truncate(len)?;
        Ok(pairs)
    }

    /// Checks the scalar kernels against the reference to 1e-4.
    pub fn check(&mut self, gpu: &Gpu, tokens: usize) -> Result<()> {
        gpu.set_fast_attention(false);
        for (got, expected) in self.outputs(gpu, tokens)? {
            assert_close(&got, &expected, 1e-4);
        }
        Ok(())
    }

    pub fn check_every_kernel(&mut self, gpu: &Gpu) -> Result<()> {
        for tokens in [1, 2, 5, 8, 12] {
            self.check(gpu, tokens)?;
        }
        Ok(())
    }
}
