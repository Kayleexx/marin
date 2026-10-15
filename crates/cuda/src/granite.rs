use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::ops::{blocks, expect_len, to_u32};
use crate::{Buffer, CudaError, DeviceMatrix, Gpu, Tokens};

fn grid(count: usize, block: usize) -> Result<LaunchConfig, CudaError> {
    Ok(LaunchConfig {
        grid_dim: (blocks(count, block)?, 1, 1),
        block_dim: (to_u32(block, "Granite block")?, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn row_grid(rows: usize) -> Result<LaunchConfig, CudaError> {
    Ok(LaunchConfig {
        grid_dim: (blocks(rows, 4)?, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    })
}

impl Gpu {
    pub fn granite_route_ids(
        &self,
        routes: &Tokens,
        start: usize,
        len: usize,
    ) -> Result<Vec<u32>, CudaError> {
        let end = start
            .checked_add(len)
            .filter(|end| *end <= routes.capacity)
            .ok_or(CudaError::Shape {
                what: "Granite route download",
                expected: routes.capacity,
                actual: start.saturating_add(len),
            })?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&routes.data.slice(start..end))?)
    }

    pub fn granite_scale(&self, x: &mut Buffer, scale: f32) -> Result<(), CudaError> {
        if !scale.is_finite() {
            return Err(CudaError::Shape {
                what: "Granite scale",
                expected: 1,
                actual: 0,
            });
        }
        if x.is_empty() || scale == 1.0 {
            return Ok(());
        }
        let n = to_u32(x.len, "Granite scale length")?;
        self.launch(&self.kernels.granite_scale, grid(x.len, 256)?, |a| {
            a.arg(&mut x.data).arg(&n).arg(&scale);
        })
    }

    pub fn granite_scaled_add(
        &self,
        x: &mut Buffer,
        update: &Buffer,
        scale: f32,
    ) -> Result<(), CudaError> {
        expect_len("Granite residual", Some(x.len), update)?;
        if !scale.is_finite() {
            return Err(CudaError::Shape {
                what: "Granite residual scale",
                expected: 1,
                actual: 0,
            });
        }
        if x.is_empty() {
            return Ok(());
        }
        let n = to_u32(x.len, "Granite residual length")?;
        self.launch(&self.kernels.granite_scaled_add, grid(x.len, 256)?, |a| {
            a.arg(&mut x.data).arg(&update.data).arg(&n).arg(&scale);
        })
    }

    pub fn granite_moe(
        &self,
        input: &Buffer,
        router: &Buffer,
        weights: (&DeviceMatrix, &DeviceMatrix),
        dims: (usize, usize, usize, usize, usize),
        scratch: (
            &mut Tokens,
            &mut Buffer,
            &mut Buffer,
            &mut Buffer,
            &mut Buffer,
        ),
    ) -> Result<(), CudaError> {
        let (input_weights, output_weights) = weights;
        let (tokens, experts, top_k, hidden, intermediate) = dims;
        let (ids, gates, projected, expert_output, mixture) = scratch;
        let assignments = tokens.checked_mul(top_k).ok_or(CudaError::Overflow {
            what: "Granite assignments",
        })?;
        let input_rows = experts
            .checked_mul(intermediate)
            .and_then(|n| n.checked_mul(2));
        let output_rows = experts.checked_mul(hidden);
        if experts == 0
            || experts > 32
            || top_k == 0
            || top_k > experts
            || ids.capacity < assignments
            || input_weights.rows != input_rows.unwrap_or(0)
            || input_weights.cols != hidden
            || output_weights.rows != output_rows.unwrap_or(0)
            || output_weights.cols != intermediate
        {
            return Err(CudaError::Shape {
                what: "Granite expert layout",
                expected: assignments,
                actual: ids.capacity,
            });
        }
        expect_len("Granite input", tokens.checked_mul(hidden), input)?;
        expect_len("Granite router", tokens.checked_mul(experts), router)?;
        expect_len("Granite gates", Some(assignments), gates)?;
        expect_len(
            "Granite projection",
            assignments
                .checked_mul(intermediate)
                .and_then(|n| n.checked_mul(2)),
            projected,
        )?;
        expect_len(
            "Granite expert output",
            assignments.checked_mul(hidden),
            expert_output,
        )?;
        expect_len("Granite mixture", tokens.checked_mul(hidden), mixture)?;
        if tokens == 0 {
            return Ok(());
        }
        let (t, e, k, h, i, a) = (
            to_u32(tokens, "Granite tokens")?,
            to_u32(experts, "Granite experts")?,
            to_u32(top_k, "Granite top-k")?,
            to_u32(hidden, "Granite hidden")?,
            to_u32(intermediate, "Granite intermediate")?,
            to_u32(assignments, "Granite assignments")?,
        );
        self.launch(&self.kernels.granite_route, grid(tokens, 128)?, |args| {
            args.arg(&router.data)
                .arg(&mut ids.data)
                .arg(&mut gates.data);
            args.arg(&t).arg(&e).arg(&k);
        })?;
        self.launch(
            &self.kernels.granite_expert_input,
            row_grid(assignments.saturating_mul(intermediate).saturating_mul(2))?,
            |args| {
                args.arg(&input_weights.data)
                    .arg(&input.data)
                    .arg(&ids.data)
                    .arg(&mut projected.data);
                args.arg(&t).arg(&k).arg(&h).arg(&i);
            },
        )?;
        self.launch(
            &self.kernels.granite_silu,
            grid(assignments.saturating_mul(intermediate), 256)?,
            |args| {
                args.arg(&mut projected.data).arg(&a).arg(&i);
            },
        )?;
        self.launch(
            &self.kernels.granite_expert_output,
            row_grid(assignments.saturating_mul(hidden))?,
            |args| {
                args.arg(&output_weights.data)
                    .arg(&projected.data)
                    .arg(&ids.data)
                    .arg(&mut expert_output.data);
                args.arg(&a).arg(&h).arg(&i);
            },
        )?;
        self.launch(
            &self.kernels.granite_mix,
            grid(tokens.saturating_mul(hidden), 256)?,
            |args| {
                args.arg(&expert_output.data)
                    .arg(&gates.data)
                    .arg(&mut mixture.data);
                args.arg(&t).arg(&k).arg(&h);
            },
        )
    }
}
