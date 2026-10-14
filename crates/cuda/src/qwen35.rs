use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::ops::{blocks, expect_len, to_u32};
use crate::{Buffer, CudaError, Gpu};

#[derive(Clone, Copy)]
pub struct DeltaDims {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
}

pub struct DeltaInputs<'a> {
    pub mixed: &'a Buffer,
    pub z: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub a_log: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub norm: &'a Buffer,
}

impl DeltaDims {
    fn totals(self) -> Result<(usize, usize, usize), CudaError> {
        if self.key_heads == 0
            || self.value_heads == 0
            || self.value_heads.checked_rem(self.key_heads) != Some(0)
            || self.key_dim == 0
            || self.key_dim > 256
            || self.value_dim == 0
            || self.value_dim > 256
            || self.kernel == 0
        {
            return Err(CudaError::Unsupported {
                what: "Qwen3.5 Delta dimensions",
                max: 256,
                actual: self.key_dim.max(self.value_dim),
            });
        }
        let key = self.key_heads.checked_mul(self.key_dim);
        let value = self.value_heads.checked_mul(self.value_dim);
        let channels = key
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(value?));
        match (key, value, channels) {
            (Some(k), Some(v), Some(c)) => Ok((k, v, c)),
            _ => Err(CudaError::Overflow {
                what: "Delta dimensions",
            }),
        }
    }
}

impl Gpu {
    pub fn copy_buffer(&self, src: &Buffer) -> Result<Buffer, CudaError> {
        let mut out = self.zeros(src.len)?;
        self.stream.memcpy_dtod(&src.data, &mut out.data)?;
        Ok(out)
    }

    pub fn q35_convolution(
        &self,
        qkv: &Buffer,
        weight: &Buffer,
        history: &mut Buffer,
        mixed: &mut Buffer,
        dims: DeltaDims,
        tokens: usize,
    ) -> Result<(), CudaError> {
        let (_, _, channels) = dims.totals()?;
        expect_len("Delta input", channels.checked_mul(tokens), qkv)?;
        expect_len("Delta output", Some(qkv.len), mixed)?;
        expect_len(
            "Delta convolution weights",
            channels.checked_mul(dims.kernel),
            weight,
        )?;
        expect_len("Delta convolution state", Some(weight.len), history)?;
        let config = LaunchConfig {
            grid_dim: (blocks(channels, 256)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (channels, kernel, tokens) = (
            to_u32(channels, "Delta channels")?,
            to_u32(dims.kernel, "convolution width")?,
            to_u32(tokens, "token count")?,
        );
        self.launch(&self.kernels.q35_convolution, config, |args| {
            args.arg(&qkv.data)
                .arg(&weight.data)
                .arg(&mut history.data)
                .arg(&mut mixed.data)
                .arg(&channels)
                .arg(&kernel)
                .arg(&tokens);
        })
    }

    pub fn q35_recurrent(
        &self,
        input: DeltaInputs<'_>,
        state: &mut Buffer,
        core: &mut Buffer,
        dims: DeltaDims,
        tokens: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let DeltaInputs {
            mixed,
            z,
            a,
            b,
            a_log,
            dt_bias,
            norm,
        } = input;
        let (_, value, channels) = dims.totals()?;
        for (name, expected, buffer) in [
            (
                "Delta convolution output",
                channels.checked_mul(tokens),
                mixed,
            ),
            ("Delta gate", value.checked_mul(tokens), z),
            ("Delta decay", dims.value_heads.checked_mul(tokens), a),
            ("Delta beta", dims.value_heads.checked_mul(tokens), b),
            ("Delta A_log", Some(dims.value_heads), a_log),
            ("Delta dt_bias", Some(dims.value_heads), dt_bias),
            ("Delta norm", Some(dims.value_dim), norm),
            ("Delta state", dims.key_dim.checked_mul(value), &*state),
            ("Delta output", value.checked_mul(tokens), &*core),
        ] {
            expect_len(name, expected, buffer)?;
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(CudaError::Shape {
                what: "Delta epsilon",
                expected: 1,
                actual: 0,
            });
        }
        let block = dims.value_dim.next_power_of_two().max(32);
        let config = LaunchConfig {
            grid_dim: (to_u32(dims.value_heads, "Delta heads")?, 1, 1),
            block_dim: (to_u32(block, "Delta block")?, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kh, vh, kd, vd, channels, tokens) = (
            to_u32(dims.key_heads, "key heads")?,
            to_u32(dims.value_heads, "value heads")?,
            to_u32(dims.key_dim, "key dimension")?,
            to_u32(dims.value_dim, "value dimension")?,
            to_u32(channels, "channels")?,
            to_u32(tokens, "tokens")?,
        );
        self.launch(&self.kernels.q35_recurrent, config, |args| {
            args.arg(&mixed.data)
                .arg(&z.data)
                .arg(&a.data)
                .arg(&b.data)
                .arg(&a_log.data)
                .arg(&dt_bias.data)
                .arg(&norm.data)
                .arg(&mut state.data)
                .arg(&mut core.data)
                .arg(&kh)
                .arg(&vh)
                .arg(&kd)
                .arg(&vd)
                .arg(&channels)
                .arg(&tokens)
                .arg(&eps);
        })
    }

    pub fn q35_split_query(
        &self,
        projected: &Buffer,
        query: &mut Buffer,
        gate: &mut Buffer,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), CudaError> {
        let n = query.len;
        expect_len("gated query projection", n.checked_mul(2), projected)?;
        expect_len("gated query gate", Some(n), gate)?;
        let per_token = heads.checked_mul(head_dim).ok_or(CudaError::Overflow {
            what: "gated query heads",
        })?;
        if per_token == 0 || n.checked_rem(per_token) != Some(0) {
            return Err(CudaError::Shape {
                what: "gated query heads",
                expected: per_token,
                actual: n,
            });
        }
        let config = LaunchConfig {
            grid_dim: (blocks(n, 256)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (heads, head_dim, n) = (
            to_u32(heads, "heads")?,
            to_u32(head_dim, "head dimension")?,
            to_u32(n, "query length")?,
        );
        self.launch(&self.kernels.q35_split_query, config, |args| {
            args.arg(&projected.data)
                .arg(&mut query.data)
                .arg(&mut gate.data)
                .arg(&heads)
                .arg(&head_dim)
                .arg(&n);
        })
    }

    pub fn q35_partial_rope(
        &self,
        x: &mut Buffer,
        heads: usize,
        head_dim: usize,
        rotary: usize,
        start: usize,
        theta: f32,
    ) -> Result<(), CudaError> {
        if heads == 0
            || head_dim == 0
            || rotary == 0
            || rotary > head_dim
            || !rotary.is_multiple_of(2)
            || x.len.checked_rem(heads.checked_mul(head_dim).unwrap_or(0)) != Some(0)
        {
            return Err(CudaError::Shape {
                what: "partial RoPE",
                expected: head_dim,
                actual: rotary,
            });
        }
        let config = LaunchConfig {
            grid_dim: (
                to_u32(
                    x.len
                        .checked_div(head_dim)
                        .ok_or(CudaError::Overflow { what: "RoPE heads" })?,
                    "RoPE heads",
                )?,
                1,
                1,
            ),
            block_dim: (to_u32((rotary / 2).max(1), "RoPE block")?, 1, 1),
            shared_mem_bytes: 0,
        };
        let (heads, head_dim, rotary, start) = (
            to_u32(heads, "heads")?,
            to_u32(head_dim, "head dimension")?,
            to_u32(rotary, "rotary dimension")?,
            to_u32(start, "position")?,
        );
        self.launch(&self.kernels.q35_partial_rope, config, |args| {
            args.arg(&mut x.data)
                .arg(&heads)
                .arg(&head_dim)
                .arg(&rotary)
                .arg(&start)
                .arg(&theta);
        })
    }

    pub fn q35_gate_attention(&self, context: &mut Buffer, gate: &Buffer) -> Result<(), CudaError> {
        expect_len("attention gate", Some(context.len), gate)?;
        let config = LaunchConfig {
            grid_dim: (blocks(context.len, 256)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = to_u32(context.len, "attention width")?;
        self.launch(&self.kernels.q35_gate_attention, config, |args| {
            args.arg(&mut context.data).arg(&gate.data).arg(&n);
        })
    }
}
