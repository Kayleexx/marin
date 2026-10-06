//! Measurements of what the GPU itself can do, so the engine's numbers have something to be judged against:
//! how fast memory can be read, how fast it does floating-point math, and what copying to the host costs.
//! Each probe runs in short timed windows, because a laptop GPU that is held to a power limit gives a
//! different answer in its first moments than after a few seconds.

use std::time::{Duration, Instant};

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::{CudaError, Gpu};

const SOURCE: &str = include_str!("probe.cu");
const BLOCK: u32 = 256;
/// Blocks per multiprocessor for each probe: enough to keep it busy without exceeding what it can hold.
const READ_BLOCKS_PER_SM: u32 = 8;
const FMA_BLOCKS_PER_SM: u32 = 6;
const MMA_BLOCKS_PER_SM: u32 = 4;
/// Smaller buffers could sit in the GPU's cache and say nothing about memory.
const MIN_READ_BYTES: usize = 4_000_000;
const FMA_CHAINS: f64 = 8.0;
/// Tensor-core multiply-accumulates per iteration, and the operations in one: 16 x 8 x 16 multiply-adds.
const MMA_PER_ITER: f64 = 4.0;
const MMA_FLOPS: f64 = 4096.0;
/// Longest a single launch should run, so a window can end close to its deadline.
const LAUNCH_TARGET: Duration = Duration::from_millis(30);
const FMA_A: f32 = 0.999_9;
const FMA_B: f32 = 0.000_1;

/// One timed stretch of a probe: `work` (bytes or floating-point operations) done in `time`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub work: f64,
    pub time: Duration,
}

impl Window {
    pub fn per_second(&self) -> f64 {
        self.work / self.time.as_secs_f64().max(f64::MIN_POSITIVE)
    }
}

/// How long each of a probe's copies took, on the same path the engine uses to bring logits to the host.
#[derive(Debug, Clone)]
pub struct CopyTimes {
    pub bytes: usize,
    pub times: Vec<Duration>,
}

fn grid(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn wrong(what: &'static str) -> CudaError {
    CudaError::SelfCheck { what }
}

impl Gpu {
    fn probe_function(&self, name: &str) -> Result<CudaFunction, CudaError> {
        let options = CompileOptions {
            arch: Some("compute_80"),
            ..CompileOptions::default()
        };
        let module = self
            .ctx
            .load_module(compile_ptx_with_opts(SOURCE, options)?)?;
        Ok(module.load_function(name)?)
    }

    pub fn sm_count(&self) -> Result<usize, CudaError> {
        Ok(self.sm_count)
    }

    fn blocks(&self, per_sm: u32) -> Result<u32, CudaError> {
        let sms = u32::try_from(self.sm_count()?).map_err(|_| CudaError::Overflow {
            what: "multiprocessor count",
        })?;
        sms.checked_mul(per_sm)
            .ok_or(CudaError::Overflow { what: "grid size" })
    }

    /// Time of `launch(iters)` on the GPU's clock, with the host waiting for it to finish.
    fn timed(
        &self,
        iters: u32,
        launch: &mut impl FnMut(u32) -> Result<(), CudaError>,
    ) -> Result<Duration, CudaError> {
        let start = self.mark()?;
        launch(iters)?;
        let end = self.mark()?;
        self.synchronize()?;
        end.since(&start)
    }

    /// An iteration count that makes one launch last about `LAUNCH_TARGET`.
    fn calibrate(
        &self,
        launch: &mut impl FnMut(u32) -> Result<(), CudaError>,
    ) -> Result<u32, CudaError> {
        let mut iters = 64_u32;
        for _ in 0..24 {
            let took = self.timed(iters, launch)?;
            if took >= Duration::from_millis(2) {
                let scale = LAUNCH_TARGET.as_secs_f64() / took.as_secs_f64();
                let wanted = (f64::from(iters) * scale).clamp(1.0, f64::from(u32::MAX / 2));
                return Ok(wanted as u32);
            }
            iters = iters.saturating_mul(4);
        }
        Err(CudaError::Overflow {
            what: "probe iteration count",
        })
    }

    /// `windows` back-to-back stretches of `window` each, every one a run of launches of `work_per_iter * iters`.
    fn windows(
        &self,
        window: Duration,
        windows: usize,
        work_per_iter: f64,
        mut launch: impl FnMut(u32) -> Result<(), CudaError>,
    ) -> Result<Vec<Window>, CudaError> {
        let iters = self.calibrate(&mut launch)?;
        let mut out = Vec::with_capacity(windows);
        for _ in 0..windows {
            let (begun, mut time, mut launches) = (Instant::now(), Duration::ZERO, 0_u32);
            while begun.elapsed() < window || launches == 0 {
                time = time.saturating_add(self.timed(iters, &mut launch)?);
                launches = launches.saturating_add(1);
            }
            out.push(Window {
                work: work_per_iter * f64::from(iters) * f64::from(launches),
                time,
            });
        }
        Ok(out)
    }

    /// Reads `bytes` of device memory over and over and reports bytes read per window. The data is all ones, so the
    /// sum of every read is known and checked.
    pub fn probe_read_bandwidth(
        &self,
        bytes: usize,
        window: Duration,
        windows: usize,
    ) -> Result<Vec<Window>, CudaError> {
        let quads_wanted = bytes / 16;
        let n =
            u32::try_from(quads_wanted.saturating_mul(4)).map_err(|_| CudaError::Unsupported {
                what: "probe buffer megabytes",
                max: 16_000,
                actual: bytes / 1_000_000,
            })?;
        if bytes < MIN_READ_BYTES {
            return Err(CudaError::Unsupported {
                what: "probe buffer megabytes",
                max: 16_000,
                actual: bytes / 1_000_000,
            });
        }
        let floats = usize::try_from(n).map_err(|_| CudaError::Overflow {
            what: "probe buffer",
        })?;
        let quads = n / 4;
        let pass_bytes = floats as f64 * 4.0;
        let (fill, read) = (
            self.probe_function("fill")?,
            self.probe_function("read_sum")?,
        );
        let blocks = self.blocks(READ_BLOCKS_PER_SM)?;
        let mut data = self.zeros(floats)?;
        let mut sink = self.stream.alloc_zeros::<f64>(1)?;
        let one = 1.0_f32;
        self.launch(&fill, grid(blocks), |a| {
            a.arg(&mut data.data).arg(&n).arg(&one);
        })?;
        let mut launches = 0_u64;
        let result = self.windows(window, windows, pass_bytes, |iters| {
            for _ in 0..iters {
                self.launch(&read, grid(blocks), |a| {
                    a.arg(&data.data).arg(&quads).arg(&mut sink);
                })?;
                launches = launches.saturating_add(1);
            }
            Ok(())
        })?;
        let total = self.stream.clone_dtoh(&sink)?;
        if total.first().copied() != Some(floats as f64 * launches as f64) {
            return Err(wrong("memory read"));
        }
        Ok(result)
    }

    /// Floating-point operations per window of eight independent fused multiply-add chains per thread; the
    /// result of one thread is checked bit for bit against the CPU.
    pub fn probe_fp32(&self, window: Duration, windows: usize) -> Result<Vec<Window>, CudaError> {
        let function = self.probe_function("fma_f32")?;
        let blocks = self.blocks(FMA_BLOCKS_PER_SM)?;
        let threads = f64::from(blocks) * f64::from(BLOCK);
        let sink_len = usize::try_from(blocks.saturating_mul(BLOCK))
            .map_err(|_| CudaError::Overflow { what: "probe grid" })?;
        let mut sink = self.zeros(sink_len)?;
        let mut run = |iters: u32| {
            self.launch(&function, grid(blocks), |a| {
                a.arg(&mut sink.data).arg(&iters).arg(&FMA_A).arg(&FMA_B);
            })
        };
        let result = self.windows(window, windows, threads * FMA_CHAINS * 2.0, &mut run)?;
        let checked = 5_000_u32;
        run(checked)?;
        let got = self.download(&sink)?;
        if got.first().map(|g| g.to_bits()) != Some(fma_reference(checked).to_bits()) {
            return Err(wrong("fp32 math"));
        }
        Ok(result)
    }

    /// Operations per window of bf16 tensor-core multiply-accumulates. `Err` when this GPU or toolkit cannot
    /// run them.
    pub fn probe_bf16_tensor(
        &self,
        window: Duration,
        windows: usize,
    ) -> Result<Vec<Window>, CudaError> {
        let function = self.probe_function("mma_bf16")?;
        let blocks = self.blocks(MMA_BLOCKS_PER_SM)?;
        let warps = f64::from(blocks) * f64::from(BLOCK) / 32.0;
        let sink_len = usize::try_from(blocks.saturating_mul(BLOCK))
            .map_err(|_| CudaError::Overflow { what: "probe grid" })?;
        let mut sink = self.zeros(sink_len)?;
        let mut run = |iters: u32| {
            self.launch(&function, grid(blocks), |a| {
                a.arg(&mut sink.data).arg(&iters);
            })
        };
        let result = self.windows(window, windows, warps * MMA_PER_ITER * MMA_FLOPS, &mut run)?;
        let checked = 1_000_u32;
        run(checked)?;
        let got = self.download(&sink)?;
        if got.first().copied() != Some(256.0 * checked as f32) {
            return Err(wrong("tensor-core math"));
        }
        Ok(result)
    }

    /// Times `repeats` downloads of `floats` values from an idle GPU, on the path the engine takes for logits.
    pub fn probe_download(&self, floats: usize, repeats: usize) -> Result<CopyTimes, CudaError> {
        let buffer = self.zeros(floats)?;
        self.synchronize()?;
        let mut times = Vec::with_capacity(repeats);
        for _ in 0..repeats.saturating_add(2) {
            let before = self.traffic();
            self.download(&buffer)?;
            times.push(self.traffic().since(&before).d2h_time);
        }
        times.drain(..2.min(times.len()));
        Ok(CopyTimes {
            bytes: floats.saturating_mul(std::mem::size_of::<f32>()),
            times,
        })
    }
}

/// Starting values of the eight chains in thread 0 of the fused multiply-add kernel, as offsets from 0.1.
const CHAIN_OFFSETS: [f32; 8] = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];

/// What thread 0 of the fused multiply-add kernel ends with after `iters` iterations, computed the same way.
fn fma_reference(iters: u32) -> f32 {
    let mut chains = CHAIN_OFFSETS.map(|offset| 0.1_f32 + offset);
    for _ in 0..iters {
        for chain in &mut chains {
            *chain = chain.mul_add(FMA_A, FMA_B);
        }
    }
    let [a, b, c, d, e, f, g, h] = chains;
    ((a + b) + (c + d)) + ((e + f) + (g + h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_chains_start_where_the_kernel_starts_and_settle_toward_one() {
        let start = CHAIN_OFFSETS.map(|offset| 0.1_f32 + offset);
        let [a, b, c, d, e, f, g, h] = start;
        assert_eq!(fma_reference(0), ((a + b) + (c + d)) + ((e + f) + (g + h)));
        // x -> x * 0.9999 + 0.0001 has the fixed point 1, and 5000 steps close about 40% of the distance to it.
        let after = fma_reference(5_000);
        assert!((3.4..3.9).contains(&after), "{after}");
        assert!(
            fma_reference(20_000) > after,
            "more steps get closer to the fixed point"
        );
    }

    #[test]
    fn a_window_reports_work_per_second_and_survives_no_time() {
        let window = Window {
            work: 3.0e9,
            time: Duration::from_millis(10),
        };
        assert!((window.per_second() - 3.0e11).abs() < 1.0);
        let none = Window {
            work: 1.0,
            time: Duration::ZERO,
        };
        assert!(none.per_second().is_finite());
    }
}
