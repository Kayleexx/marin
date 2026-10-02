use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchArgs, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::CudaError;
use crate::counters::Counters;

const KERNELS: &str = include_str!("kernels.cu");

pub(crate) struct Kernels {
    pub embed: CudaFunction,
    pub rms_norm: CudaFunction,
    pub matmul: CudaFunction,
    pub rope: CudaFunction,
    pub attention: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
}

pub struct Gpu {
    ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) kernels: Kernels,
    pub(crate) counters: Counters,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// Row-major bf16 weight matrix in GPU memory.
pub struct DeviceMatrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: CudaSlice<u16>,
}

/// A flat f32 array in GPU memory.
pub struct Buffer {
    pub(crate) len: usize,
    pub(crate) data: CudaSlice<f32>,
}

impl DeviceMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn libraries_present() -> Result<(), CudaError> {
    // SAFETY: both probes only try to dlopen the CUDA libraries and report whether that worked.
    let driver = unsafe { cudarc::driver::sys::is_culib_present() };
    // SAFETY: same as above, for the NVRTC compiler library.
    let nvrtc = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    match (driver, nvrtc) {
        (true, true) => Ok(()),
        (false, _) => Err(CudaError::Unavailable(
            "no NVIDIA driver found (libcuda)".into(),
        )),
        (true, false) => Err(CudaError::Unavailable(
            "the CUDA toolkit's libnvrtc is missing; install the CUDA toolkit".into(),
        )),
    }
}

impl Gpu {
    pub fn new() -> Result<Self, CudaError> {
        libraries_present()?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(compile_ptx(KERNELS)?)?;
        let kernels = Kernels {
            embed: module.load_function("embed")?,
            rms_norm: module.load_function("rms_norm")?,
            matmul: module.load_function("matmul_bf16")?,
            rope: module.load_function("rope")?,
            attention: module.load_function("attention")?,
            silu_mul: module.load_function("silu_mul")?,
            add: module.load_function("add_in_place")?,
        };
        Ok(Self {
            ctx,
            stream,
            kernels,
            counters: Counters::default(),
        })
    }

    pub fn info(&self) -> Result<GpuInfo, CudaError> {
        let (free_bytes, total_bytes) = self.ctx.mem_get_info()?;
        Ok(GpuInfo {
            name: self.ctx.name()?,
            compute_capability: self.ctx.compute_capability()?,
            free_bytes,
            total_bytes,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
    }

    pub fn upload_matrix(
        &self,
        rows: usize,
        cols: usize,
        bits: &[u16],
    ) -> Result<DeviceMatrix, CudaError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(CudaError::Overflow { what: "matrix" })?;
        if expected != bits.len() {
            return Err(CudaError::Shape {
                what: "matrix data",
                expected,
                actual: bits.len(),
            });
        }
        let started = Instant::now();
        let data = self.stream.clone_htod(bits)?;
        self.counters.upload(std::mem::size_of_val(bits), started);
        Ok(DeviceMatrix { rows, cols, data })
    }

    pub fn upload(&self, values: &[f32]) -> Result<Buffer, CudaError> {
        let started = Instant::now();
        let data = self.stream.clone_htod(values)?;
        self.counters.upload(std::mem::size_of_val(values), started);
        Ok(Buffer {
            len: values.len(),
            data,
        })
    }

    pub fn zeros(&self, len: usize) -> Result<Buffer, CudaError> {
        self.counters.allocation();
        Ok(Buffer {
            len,
            data: self.stream.alloc_zeros::<f32>(len)?,
        })
    }

    pub fn download(&self, buffer: &Buffer) -> Result<Vec<f32>, CudaError> {
        self.timed_download(buffer.len, |gpu| gpu.stream.clone_dtoh(&buffer.data))
    }

    /// Waits for queued GPU work first so the wait and the copy are timed apart.
    fn timed_download(
        &self,
        len: usize,
        copy: impl FnOnce(&Self) -> Result<Vec<f32>, cudarc::driver::DriverError>,
    ) -> Result<Vec<f32>, CudaError> {
        let started = Instant::now();
        self.stream.synchronize()?;
        let copying = Instant::now();
        let out = copy(self)?;
        self.counters.download(
            len.saturating_mul(std::mem::size_of::<f32>()),
            copying.duration_since(started),
            copying.elapsed(),
        );
        Ok(out)
    }

    /// Copies `len` values starting at `start` to the host.
    pub fn download_range(
        &self,
        buffer: &Buffer,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let end = start
            .checked_add(len)
            .filter(|end| *end <= buffer.len)
            .ok_or(CudaError::Shape {
                what: "download range",
                expected: buffer.len,
                actual: start.saturating_add(len),
            })?;
        self.timed_download(len, |gpu| {
            gpu.stream.clone_dtoh(&buffer.data.slice(start..end))
        })
    }

    /// Copies `len` values starting at `start` into a new buffer without leaving the GPU.
    pub fn copy_range(
        &self,
        buffer: &Buffer,
        start: usize,
        len: usize,
    ) -> Result<Buffer, CudaError> {
        let end = start
            .checked_add(len)
            .filter(|end| *end <= buffer.len)
            .ok_or(CudaError::Shape {
                what: "copy range",
                expected: buffer.len,
                actual: start.saturating_add(len),
            })?;
        let mut copy = self.zeros(len)?;
        self.stream
            .memcpy_dtod(&buffer.data.slice(start..end), &mut copy.data)?;
        Ok(copy)
    }

    pub fn duplicate(&self, buffer: &Buffer) -> Result<Buffer, CudaError> {
        let mut copy = self.zeros(buffer.len)?;
        self.stream.memcpy_dtod(&buffer.data, &mut copy.data)?;
        Ok(copy)
    }

    pub(crate) fn launch<'a>(
        &'a self,
        function: &'a CudaFunction,
        config: LaunchConfig,
        push_args: impl FnOnce(&mut LaunchArgs<'a>),
    ) -> Result<(), CudaError> {
        self.counters.launch();
        let mut args = self.stream.launch_builder(function);
        push_args(&mut args);
        // SAFETY: only the wrappers in `ops` call this. Each one verifies that every buffer
        // holds at least the number of elements its kernel indexes for the given sizes, that
        // scalar arguments fit in u32, and that the kernel's argument list matches `push_args`.
        unsafe { args.launch(config) }?;
        Ok(())
    }
}
