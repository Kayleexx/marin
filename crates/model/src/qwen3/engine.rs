use cuda::{CudaError, Gpu, GpuInfo, KvFormat, Traffic};
use kv::KvCache;
use loader::ModelDir;
use metrics::{Probe, StageTimes};
use runtime::{CausalLm, RollbackError, SequenceCache};
use tensor::QuantScheme;

use super::Observer;
use super::config::Qwen3Config;
use super::fit::{GPU_SLACK_BYTES, GpuFit, gpu_fit};
use super::gpu::{GpuCache, Qwen3Gpu};
use super::model::Qwen3;
use crate::ModelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// Use the GPU when CUDA works and the model fits, otherwise the CPU.
    Auto,
    Cpu,
    Cuda,
}

/// The backend that was picked, and why if it is not what was asked for.
pub struct Selected {
    pub backend: Backend,
    pub note: Option<String>,
}

pub enum Backend {
    Cpu,
    Cuda(Box<Gpu>),
}

impl Backend {
    /// `startup_bytes` is the VRAM the model and the first slab of the cache need to start, `window_bytes` what the
    /// model and the whole `--ctx` window would take. Only the first has to fit; when the second does not, the GPU
    /// is still used and the note says a conversation that fills the window can run out of memory.
    pub fn select(
        choice: BackendChoice,
        startup_bytes: u64,
        window_bytes: u64,
    ) -> Result<Selected, ModelError> {
        let try_gpu = || -> Result<(Gpu, Option<String>), ModelError> {
            let gpu = Gpu::new()?;
            let free = u64::try_from(gpu.info()?.free_bytes).unwrap_or(u64::MAX);
            match gpu_fit(free, startup_bytes, window_bytes) {
                GpuFit::No => Err(ModelError::GpuMemory {
                    needed: startup_bytes.saturating_add(GPU_SLACK_BYTES),
                    free,
                }),
                GpuFit::Startup => Ok((
                    gpu,
                    Some(format!(
                        "the GPU has {} MB free but the whole --ctx window needs about {} MB: a conversation that fills it will stop with an out-of-memory error",
                        free / 1_000_000,
                        window_bytes.saturating_add(GPU_SLACK_BYTES) / 1_000_000
                    )),
                )),
                GpuFit::Window => Ok((gpu, None)),
            }
        };
        match choice {
            BackendChoice::Cpu => Ok(Selected {
                backend: Backend::Cpu,
                note: None,
            }),
            BackendChoice::Cuda => {
                let (gpu, note) = try_gpu()?;
                Ok(Selected {
                    backend: Backend::Cuda(Box::new(gpu)),
                    note,
                })
            }
            BackendChoice::Auto => Ok(match try_gpu() {
                Ok((gpu, note)) => Selected {
                    backend: Backend::Cuda(Box::new(gpu)),
                    note,
                },
                Err(ModelError::Cuda(CudaError::Unavailable(why))) => Selected {
                    backend: Backend::Cpu,
                    note: Some(format!("no usable GPU ({why}); using the CPU")),
                },
                Err(e @ ModelError::GpuMemory { .. }) => Selected {
                    backend: Backend::Cpu,
                    note: Some(format!("{e}; using the CPU")),
                },
                Err(e) => return Err(e),
            }),
        }
    }
}

/// A loaded model on whichever backend was selected.
pub enum Engine {
    Cpu(Box<Qwen3>),
    Cuda(Box<Qwen3Gpu>),
}

pub enum EngineCache {
    Cpu(KvCache),
    Cuda(Box<GpuCache>),
}

impl Engine {
    /// Loads the weights onto `backend`, quantizing the layer projections with `quant` if given (INT8 on CPU).
    pub fn load(
        dir: &ModelDir,
        backend: Backend,
        quant: Option<QuantScheme>,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        Ok(match backend {
            Backend::Cpu => Self::Cpu(Box::new(Qwen3::load_quantized(dir, quant, progress)?)),
            Backend::Cuda(gpu) => Self::Cuda(Box::new(Qwen3Gpu::load(dir, *gpu, quant, progress)?)),
        })
    }

    /// Stores the KV cache of caches made from now on in `format`. Only the GPU cache has more than one.
    pub fn with_kv_format(self, format: KvFormat) -> Result<Self, ModelError> {
        match self {
            Self::Cuda(m) => Ok(Self::Cuda(Box::new(m.with_kv_format(format)))),
            Self::Cpu(_) if format == KvFormat::default() => Ok(self),
            Self::Cpu(_) => Err(ModelError::Unsupported(format!(
                "a {} KV cache needs the GPU backend",
                format.label()
            ))),
        }
    }

    pub fn kv_format(&self) -> KvFormat {
        match self {
            Self::Cpu(_) => KvFormat::default(),
            Self::Cuda(m) => m.kv_format(),
        }
    }

    /// The scheme the layer projections were quantized with; `None` for bf16 weights.
    pub fn quantization(&self) -> Option<QuantScheme> {
        match self {
            Self::Cpu(m) => m.quantization(),
            Self::Cuda(m) => m.quantization(),
        }
    }

    pub fn config(&self) -> &Qwen3Config {
        match self {
            Self::Cpu(m) => m.config(),
            Self::Cuda(m) => m.config(),
        }
    }

    /// Short description of where the math runs, e.g. `cpu` or `cuda (NVIDIA GeForce ...)`.
    pub fn backend_name(&self) -> String {
        match self {
            Self::Cpu(_) => "cpu".to_string(),
            Self::Cuda(m) => match m.gpu().info() {
                Ok(info) => format!("cuda ({})", info.name),
                Err(_) => "cuda".to_string(),
            },
        }
    }

    pub fn new_cache(&self, capacity: usize) -> Result<EngineCache, ModelError> {
        Ok(match self {
            Self::Cpu(m) => EngineCache::Cpu(m.new_cache(capacity)?),
            Self::Cuda(m) => EngineCache::Cuda(Box::new(m.new_cache(capacity)?)),
        })
    }

    /// What the GPU path has launched, allocated and copied so far; `None` on the CPU.
    pub fn traffic(&self) -> Option<Traffic> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(m) => Some(m.gpu().traffic()),
        }
    }

    /// The GPU's name, compute capability and memory; `None` on the CPU or when the driver cannot say.
    pub fn gpu_info(&self) -> Option<GpuInfo> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(m) => m.gpu().info().ok(),
        }
    }

    /// Free and total GPU memory in bytes; `None` on the CPU.
    pub fn gpu_memory(&self) -> Option<(u64, u64)> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(m) => {
                let info = m.gpu().info().ok()?;
                Some((
                    u64::try_from(info.free_bytes).ok()?,
                    u64::try_from(info.total_bytes).ok()?,
                ))
            }
        }
    }

    /// `forward` that adds the time of each stage of the pass to `times`.
    pub fn forward_profiled(
        &self,
        tokens: &[u32],
        cache: &mut EngineCache,
        times: &mut StageTimes,
    ) -> Result<Vec<f32>, ModelError> {
        match (self, cache) {
            (Self::Cpu(m), EngineCache::Cpu(kv)) => {
                m.forward_with(tokens, kv, None, &mut Probe::on(times))
            }
            (Self::Cuda(m), EngineCache::Cuda(kv)) => m.forward_with(tokens, kv, None, Some(times)),
            _ => Err(wrong_backend()),
        }
    }

    pub fn forward_observed(
        &self,
        tokens: &[u32],
        cache: &mut EngineCache,
        observe: Observer<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        match (self, cache) {
            (Self::Cpu(m), EngineCache::Cpu(kv)) => m.forward_observed(tokens, kv, observe),
            (Self::Cuda(m), EngineCache::Cuda(kv)) => m.forward_observed(tokens, kv, observe),
            _ => Err(wrong_backend()),
        }
    }
}

fn wrong_backend() -> ModelError {
    ModelError::InvalidConfig("cache belongs to the other backend".into())
}

impl EngineCache {
    /// Pages and bytes of KV memory held; `None` on the CPU, which reserves the whole window up front.
    pub fn kv_held(&self) -> Option<(usize, usize)> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(c) => Some(c.kv_held()),
        }
    }
}

impl SequenceCache for EngineCache {
    fn used(&self) -> usize {
        match self {
            Self::Cpu(c) => SequenceCache::used(c),
            Self::Cuda(c) => c.used(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Cpu(c) => SequenceCache::capacity(c),
            Self::Cuda(c) => c.capacity(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Cpu(c) => SequenceCache::clear(c),
            Self::Cuda(c) => c.clear(),
        }
    }

    fn truncate(&mut self, len: usize) -> Result<(), RollbackError> {
        match self {
            Self::Cpu(c) => SequenceCache::truncate(c, len),
            Self::Cuda(c) => c.truncate(len),
        }
    }
}

impl CausalLm for Engine {
    type Error = ModelError;
    type Cache = EngineCache;

    fn forward(&self, tokens: &[u32], cache: &mut EngineCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }

    /// The GPU checks up to 8 tokens in one pass; the CPU has no such pass, so it never guesses ahead.
    fn max_rows(&self) -> usize {
        match self {
            Self::Cpu(_) => 0,
            Self::Cuda(m) => m.max_rows(),
        }
    }

    fn forward_rows(
        &self,
        tokens: &[u32],
        cache: &mut EngineCache,
    ) -> Result<Vec<f32>, ModelError> {
        match (self, cache) {
            (Self::Cuda(m), EngineCache::Cuda(kv)) => m.forward_rows(tokens, kv),
            (Self::Cpu(m), EngineCache::Cpu(kv)) => m.forward_rows(tokens, kv),
            _ => Err(wrong_backend()),
        }
    }
}
