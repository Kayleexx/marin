use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::{CUdevice_attribute, cudaError_enum};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchArgs, LaunchConfig,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::CudaError;
use crate::KvFormat;
use crate::counters::Counters;

const KERNELS: &str = concat!(
    include_str!("kernels.cu"),
    include_str!("kv_format.cu"),
    include_str!("attention.cu"),
    include_str!("attention_mma.cu"),
    include_str!("mma.cu"),
    include_str!("quant.cu"),
    include_str!("qwen35.cu"),
    include_str!("granite.cu")
);

pub(crate) struct Kernels {
    pub embed: CudaFunction,
    pub rms_norm: CudaFunction,
    pub rms_norm_to: CudaFunction,
    pub matmul: CudaFunction,
    pub rope: CudaFunction,
    /// The kernels that touch the KV cache exist once per `KvFormat`, indexed by `KvFormat::index`.
    pub attention: Vec<CudaFunction>,
    pub attention_prefill_mma: CudaFunction,
    pub silu_mul: CudaFunction,
    pub add: CudaFunction,
    pub attention_partial: Vec<CudaFunction>,
    /// The few-token attention built for 8, 16 and 32 rows.
    pub attention_multi_partial: [Vec<CudaFunction>; 3],
    pub kv_store: Vec<CudaFunction>,
    pub attention_merge: CudaFunction,
    pub matmul_mma: CudaFunction,
    pub matmul_x2: CudaFunction,
    pub matmul_x3: CudaFunction,
    pub embed_q8: CudaFunction,
    pub matvec_q8: CudaFunction,
    pub matvec_q4: CudaFunction,
    pub matvec_q8_multi: Vec<CudaFunction>,
    pub matmul_q8: CudaFunction,
    pub matmul_q4: CudaFunction,
    pub matmul_q8_mma: CudaFunction,
    pub matmul_q4_mma: CudaFunction,
    pub q35_convolution: CudaFunction,
    pub q35_recurrent: CudaFunction,
    pub q35_split_query: CudaFunction,
    pub q35_partial_rope: CudaFunction,
    pub q35_gate_attention: CudaFunction,
    pub granite_scale: CudaFunction,
    pub granite_scaled_add: CudaFunction,
    pub granite_route: CudaFunction,
    pub granite_expert_input: CudaFunction,
    pub granite_silu: CudaFunction,
    pub granite_expert_output: CudaFunction,
    pub granite_mix: CudaFunction,
}

pub struct Gpu {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) kernels: Kernels,
    pub(crate) counters: Counters,
    pub(crate) sm_count: usize,
    /// Whether an f16 cache may use the fast attention kernels: tensor cores for prompt passes, 16-byte loads for one token.
    pub(crate) fast_attention: std::sync::atomic::AtomicBool,
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

/// Device memory for the token ids of one pass, so the embedding does not allocate.
pub struct Tokens {
    pub(crate) data: CudaSlice<u32>,
    pub(crate) capacity: usize,
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

fn per_format(module: &Arc<CudaModule>, name: &str) -> Result<Vec<CudaFunction>, CudaError> {
    KvFormat::ALL
        .iter()
        .map(|format| Ok(module.load_function(&format!("{name}_{}", format.label()))?))
        .collect()
}

impl Gpu {
    pub fn new() -> Result<Self, CudaError> {
        libraries_present()?;
        let ctx = CudaContext::new(0).map_err(|error| {
            if error.0 == cudaError_enum::CUDA_ERROR_NO_DEVICE {
                CudaError::Unavailable("no CUDA-capable device is visible".into())
            } else {
                CudaError::Driver(error)
            }
        })?;
        let stream = ctx.default_stream();
        // bf16 tensor-core instructions need at least compute_80; the driver compiles the PTX for the actual GPU.
        let options = CompileOptions {
            arch: Some("compute_80"),
            ..CompileOptions::default()
        };
        let module = ctx.load_module(compile_ptx_with_opts(KERNELS, options)?)?;
        let kernels = Kernels {
            embed: module.load_function("embed")?,
            rms_norm: module.load_function("rms_norm")?,
            rms_norm_to: module.load_function("rms_norm_to")?,
            matmul: module.load_function("matmul_bf16")?,
            rope: module.load_function("rope")?,
            attention: per_format(&module, "attention")?,
            attention_prefill_mma: module.load_function("attention_prefill_mma")?,
            silu_mul: module.load_function("silu_mul")?,
            add: module.load_function("add_in_place")?,
            attention_partial: per_format(&module, "attention_partial")?,
            attention_multi_partial: [
                per_format(&module, "attention_multi_partial_8")?,
                per_format(&module, "attention_multi_partial_16")?,
                per_format(&module, "attention_multi_partial_32")?,
            ],
            kv_store: per_format(&module, "kv_store")?,
            attention_merge: module.load_function("attention_merge")?,
            matmul_mma: module.load_function("matmul_bf16_mma")?,
            matmul_x2: module.load_function("matmul_bf16_x2")?,
            matmul_x3: module.load_function("matmul_bf16_x3")?,
            embed_q8: module.load_function("embed_q8")?,
            matvec_q8: module.load_function("matvec_q8")?,
            matvec_q4: module.load_function("matvec_q4")?,
            matvec_q8_multi: vec![
                module.load_function("matvec_q8_t2")?,
                module.load_function("matvec_q8_t3")?,
                module.load_function("matvec_q8_t4")?,
            ],
            matmul_q8: module.load_function("matmul_q8")?,
            matmul_q4: module.load_function("matmul_q4")?,
            matmul_q8_mma: module.load_function("matmul_q8_mma")?,
            matmul_q4_mma: module.load_function("matmul_q4_mma")?,
            q35_convolution: module.load_function("q35_convolution")?,
            q35_recurrent: module.load_function("q35_recurrent")?,
            q35_split_query: module.load_function("q35_split_query")?,
            q35_partial_rope: module.load_function("q35_partial_rope")?,
            q35_gate_attention: module.load_function("q35_gate_attention")?,
            granite_scale: module.load_function("granite_scale")?,
            granite_scaled_add: module.load_function("granite_scaled_add")?,
            granite_route: module.load_function("granite_route")?,
            granite_expert_input: module.load_function("granite_expert_input")?,
            granite_silu: module.load_function("granite_silu")?,
            granite_expert_output: module.load_function("granite_expert_output")?,
            granite_mix: module.load_function("granite_mix")?,
        };
        let sm_count = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            .max(1);
        Ok(Self {
            ctx,
            stream,
            kernels,
            counters: Counters::default(),
            sm_count: usize::try_from(sm_count).unwrap_or(1),
            fast_attention: std::sync::atomic::AtomicBool::new(
                std::env::var_os("MARIN_ATTENTION_SCALAR").is_none(),
            ),
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

    /// Makes this GPU's context current on the calling thread. Freeing device memory needs it, and a thread that has
    /// not made a GPU call yet does not have it.
    pub fn bind_to_thread(&self) -> Result<(), CudaError> {
        Ok(self.ctx.bind_to_thread()?)
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

    /// Copies `dst.len()` values starting at `start` into `dst` without leaving the GPU.
    pub fn copy_range_into(
        &self,
        buffer: &Buffer,
        start: usize,
        dst: &mut Buffer,
    ) -> Result<(), CudaError> {
        let end = start
            .checked_add(dst.len)
            .filter(|end| *end <= buffer.len)
            .ok_or(CudaError::Shape {
                what: "copy range",
                expected: buffer.len,
                actual: start.saturating_add(dst.len),
            })?;
        self.stream
            .memcpy_dtod(&buffer.data.slice(start..end), &mut dst.data)?;
        Ok(())
    }

    /// Copies all of `src` into `dst` starting at `start` without leaving the GPU.
    pub fn copy_into_range(
        &self,
        src: &Buffer,
        dst: &mut Buffer,
        start: usize,
    ) -> Result<(), CudaError> {
        let end = start
            .checked_add(src.len)
            .filter(|end| *end <= dst.len)
            .ok_or(CudaError::Shape {
                what: "copy range",
                expected: dst.len,
                actual: start.saturating_add(src.len),
            })?;
        self.stream
            .memcpy_dtod(&src.data, &mut dst.data.slice_mut(start..end))?;
        Ok(())
    }

    /// Room for `capacity` token ids on the GPU, reused by every `embed`.
    pub fn tokens(&self, capacity: usize) -> Result<Tokens, CudaError> {
        self.counters.allocation();
        Ok(Tokens {
            data: self.stream.alloc_zeros::<u32>(capacity)?,
            capacity,
        })
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
        // SAFETY: only the wrappers in `ops`, `attention` and `probe` call this. Each one verifies that every buffer
        // holds at least the number of elements its kernel indexes for the given sizes, that
        // scalar arguments fit in u32, and that the kernel's argument list matches `push_args`. The attention
        // kernels also follow `DeviceKv::page_ptrs`: their wrappers verify that every page the kernel can reach
        // is mapped, and `DeviceKv` keeps each mapped page allocated for as long as its address is in the table.
        unsafe { args.launch(config) }?;
        Ok(())
    }
}
