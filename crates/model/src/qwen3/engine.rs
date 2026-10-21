use cuda::{CudaError, Gpu, GpuInfo, KvFormat, Traffic};
use kv::KvCache;
use loader::ModelDir;
use metrics::{Probe, StageTimes};

use tensor::QuantScheme;

use super::Observer;
use super::config::Qwen3Config;
use super::fit::{GPU_SLACK_BYTES, GpuFit, gpu_fit};
use super::gpu::{GpuCache, Qwen3Gpu};
use super::model::Qwen3;
use crate::ModelError;
#[path = "engine_session.rs"]
mod session;

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
        let try_gpu = |require_window: bool| -> Result<(Gpu, Option<String>), ModelError> {
            let gpu = Gpu::new()?;
            let free = u64::try_from(gpu.info()?.free_bytes).unwrap_or(u64::MAX);
            match gpu_fit(free, startup_bytes, window_bytes) {
                GpuFit::No => Err(ModelError::GpuMemory {
                    needed: startup_bytes.saturating_add(GPU_SLACK_BYTES),
                    free,
                }),
                GpuFit::Startup if require_window => Err(ModelError::GpuMemory {
                    needed: window_bytes.saturating_add(GPU_SLACK_BYTES),
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
                let (gpu, note) = try_gpu(false)?;
                Ok(Selected {
                    backend: Backend::Cuda(Box::new(gpu)),
                    note,
                })
            }
            BackendChoice::Auto => Ok(match try_gpu(true) {
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
    Granite(Box<crate::granite::Granite>),
    GraniteCuda(Box<crate::granite::GraniteGpu>),
    Qwen35(Box<crate::qwen35::Qwen35>),
    Qwen35Cuda(Box<crate::qwen35::Qwen35Gpu>),
    Cuda(Box<Qwen3Gpu>),
}

pub enum EngineCache {
    Cpu(KvCache),
    GraniteCuda(Box<crate::granite::GpuCache>),
    Qwen35(Box<crate::qwen35::Cache>),
    Qwen35Cuda(Box<crate::qwen35::GpuCache>),
    Cuda(Box<GpuCache>),
}

impl Engine {
    pub fn parse_config(dir: &ModelDir) -> Result<Qwen3Config, ModelError> {
        if crate::granite::is_granite(&dir.config_json) {
            Ok(crate::granite::Config::parse(&dir.config_json)?.summary())
        } else if crate::qwen35::is_qwen35(&dir.config_json) {
            Ok(crate::qwen35::TextConfig::parse(
                &dir.config_json,
                dir.generation_config_json.as_deref(),
            )?
            .summary())
        } else {
            Qwen3Config::parse(&dir.config_json, dir.generation_config_json.as_deref())
        }
    }
    /// Loads the weights onto `backend`, quantizing the layer projections with `quant` if given (INT8 on CPU).
    pub fn load(
        dir: &ModelDir,
        backend: Backend,
        quant: Option<QuantScheme>,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        if crate::granite::is_granite(&dir.config_json) {
            return match backend {
                Backend::Cpu => Ok(Self::Granite(Box::new(
                    crate::granite::Granite::load_quantized(dir, quant, progress)?,
                ))),
                Backend::Cuda(gpu) if quant.is_none() => Ok(Self::GraniteCuda(Box::new(
                    crate::granite::GraniteGpu::load(dir, *gpu, progress)?,
                ))),
                Backend::Cuda(_) => Err(ModelError::Unsupported(
                    "Granite CUDA currently supports BF16 weights only".into(),
                )),
            };
        }
        if crate::qwen35::is_qwen35(&dir.config_json) {
            return match backend {
                Backend::Cpu => Ok(Self::Qwen35(Box::new(
                    crate::qwen35::Qwen35::load_quantized(dir, quant, progress)?,
                ))),
                Backend::Cuda(gpu) => Ok(Self::Qwen35Cuda(Box::new(
                    crate::qwen35::Qwen35Gpu::load(dir, *gpu, quant, progress)?,
                ))),
            };
        }
        Ok(match backend {
            Backend::Cpu => Self::Cpu(Box::new(Qwen3::load_quantized(dir, quant, progress)?)),
            Backend::Cuda(gpu) => Self::Cuda(Box::new(Qwen3Gpu::load(dir, *gpu, quant, progress)?)),
        })
    }

    /// Stores the KV cache of caches made from now on in `format`. Only the GPU cache has more than one.
    pub fn with_kv_format(self, format: KvFormat) -> Result<Self, ModelError> {
        match self {
            Self::Cuda(m) => Ok(Self::Cuda(Box::new(m.with_kv_format(format)))),
            Self::Qwen35Cuda(_) if format == KvFormat::F32 => Ok(self),
            Self::GraniteCuda(_) if format == KvFormat::F32 => Ok(self),
            Self::Qwen35Cuda(_) => Err(ModelError::Unsupported(
                "Qwen3.5 CUDA requires FP32 state".into(),
            )),
            Self::GraniteCuda(_) => Err(ModelError::Unsupported(
                "Granite CUDA requires FP32 KV".into(),
            )),
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) if format == KvFormat::default() => {
                Ok(self)
            }
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) => Err(ModelError::Unsupported(
                format!("a {} KV cache needs the GPU backend", format.label()),
            )),
        }
    }

    pub fn kv_format(&self) -> KvFormat {
        match self {
            Self::Cpu(_)
            | Self::Qwen35(_)
            | Self::Qwen35Cuda(_)
            | Self::Granite(_)
            | Self::GraniteCuda(_) => KvFormat::default(),
            Self::Cuda(m) => m.kv_format(),
        }
    }

    /// The scheme the layer projections were quantized with; `None` for bf16 weights.
    pub fn quantization(&self) -> Option<QuantScheme> {
        match self {
            Self::Cpu(m) => m.quantization(),
            Self::Granite(m) => m.quantization(),
            Self::GraniteCuda(_) => None,
            Self::Qwen35(m) => m.quantization(),
            Self::Qwen35Cuda(m) => m.quantization(),
            Self::Cuda(m) => m.quantization(),
        }
    }

    pub fn config(&self) -> &Qwen3Config {
        match self {
            Self::Cpu(m) => m.config(),
            Self::Granite(m) => m.config(),
            Self::GraniteCuda(m) => m.config(),
            Self::Qwen35(m) => m.config(),
            Self::Qwen35Cuda(m) => m.config(),
            Self::Cuda(m) => m.config(),
        }
    }

    /// Short description of where the math runs, e.g. `cpu` or `cuda (NVIDIA GeForce ...)`.
    pub fn backend_name(&self) -> String {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) => "cpu".to_string(),
            Self::Cuda(_) | Self::Qwen35Cuda(_) | Self::GraniteCuda(_) => self
                .gpu_info()
                .map_or_else(|| "cuda".into(), |info| format!("cuda ({})", info.name)),
        }
    }

    pub fn new_cache(&self, capacity: usize) -> Result<EngineCache, ModelError> {
        Ok(match self {
            Self::Cpu(m) => EngineCache::Cpu(m.new_cache(capacity)?),
            Self::Granite(m) => EngineCache::Cpu(m.new_cache(capacity)?),
            Self::GraniteCuda(m) => EngineCache::GraniteCuda(Box::new(m.new_cache(capacity)?)),
            Self::Qwen35(m) => EngineCache::Qwen35(Box::new(m.new_cache(capacity)?)),
            Self::Qwen35Cuda(m) => EngineCache::Qwen35Cuda(Box::new(m.new_cache(capacity)?)),
            Self::Cuda(m) => EngineCache::Cuda(Box::new(m.new_cache(capacity)?)),
        })
    }

    /// What the GPU path has launched, allocated and copied so far; `None` on the CPU.
    pub fn traffic(&self) -> Option<Traffic> {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) => None,
            Self::GraniteCuda(m) => Some(m.gpu().traffic()),
            Self::Cuda(m) => Some(m.gpu().traffic()),
            Self::Qwen35Cuda(m) => Some(m.gpu().traffic()),
        }
    }

    /// The GPU's name, compute capability and memory; `None` on the CPU or when the driver cannot say.
    pub fn gpu_info(&self) -> Option<GpuInfo> {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) | Self::Granite(_) => None,
            Self::GraniteCuda(m) => m.gpu().info().ok(),
            Self::Cuda(m) => m.gpu().info().ok(),
            Self::Qwen35Cuda(m) => m.gpu().info().ok(),
        }
    }

    /// Free and total GPU memory in bytes; `None` on the CPU.
    pub fn gpu_memory(&self) -> Option<(u64, u64)> {
        let info = self.gpu_info()?;
        Some((
            u64::try_from(info.free_bytes).ok()?,
            u64::try_from(info.total_bytes).ok()?,
        ))
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
            (Self::GraniteCuda(m), EngineCache::GraniteCuda(kv)) => {
                m.forward_profiled(tokens, kv, times)
            }
            (Self::Qwen35(_), _) => Err(ModelError::Unsupported(
                "Qwen3.5 per-stage profiling is not supported yet".into(),
            )),
            (Self::Granite(_), _) => Err(ModelError::Unsupported(
                "Granite per-stage profiling is not supported yet".into(),
            )),
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv)) => {
                m.forward_profiled(tokens, kv, times)
            }
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
            (Self::Granite(m), EngineCache::Cpu(kv)) => m.forward_observed(tokens, kv, observe),
            (Self::GraniteCuda(m), EngineCache::GraniteCuda(kv)) => {
                m.forward_observed(tokens, kv, observe)
            }
            (Self::Cuda(m), EngineCache::Cuda(kv)) => m.forward_observed(tokens, kv, observe),
            (Self::Qwen35(m), EngineCache::Qwen35(kv)) => m.forward_observed(tokens, kv, observe),
            (Self::Qwen35Cuda(m), EngineCache::Qwen35Cuda(kv)) => {
                m.forward_observed(tokens, kv, observe)
            }
            _ => Err(wrong_backend()),
        }
    }
}

pub(super) fn wrong_backend() -> ModelError {
    ModelError::InvalidConfig("cache belongs to the other backend".into())
}

impl EngineCache {
    /// Physical GPU pages and bytes; CPU storage has no page table.
    pub fn kv_held(&self) -> Option<(usize, usize)> {
        match self {
            Self::Cpu(_) | Self::Qwen35(_) => None,
            Self::GraniteCuda(c) => Some(c.kv_held()),
            Self::Qwen35Cuda(c) => Some(c.kv_held()),
            Self::Cuda(c) => Some(c.kv_held()),
        }
    }

    pub fn kv_bytes_held(&self) -> usize {
        match self {
            Self::Cpu(c) => c.bytes_held(),
            Self::Qwen35(c) => c.bytes_held(),
            Self::Qwen35Cuda(c) => c.bytes_held(),
            Self::GraniteCuda(c) => c.kv_held().1,
            Self::Cuda(c) => c.kv_held().1,
        }
    }
}
