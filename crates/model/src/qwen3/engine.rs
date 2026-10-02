use cuda::{CudaError, Gpu, Traffic};
use kv::KvCache;
use loader::ModelDir;
use metrics::{Probe, StageTimes};
use runtime::{CausalLm, SequenceCache};

use super::Observer;
use super::config::Qwen3Config;
use super::gpu::{GpuCache, Qwen3Gpu};
use super::model::Qwen3;
use crate::ModelError;

/// Memory the GPU needs beyond weights and the KV cache: activations and kernel code.
const GPU_SLACK_BYTES: u64 = 512 * 1024 * 1024;

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
    Cuda(Gpu),
}

impl Backend {
    /// `gpu_bytes_needed` is the VRAM a GPU run would use for this model and context.
    pub fn select(choice: BackendChoice, gpu_bytes_needed: u64) -> Result<Selected, ModelError> {
        let gpu_needed = gpu_bytes_needed.saturating_add(GPU_SLACK_BYTES);
        let try_gpu = || -> Result<Gpu, ModelError> {
            let gpu = Gpu::new()?;
            let free = u64::try_from(gpu.info()?.free_bytes).unwrap_or(u64::MAX);
            if free < gpu_needed {
                return Err(ModelError::GpuMemory {
                    needed: gpu_needed,
                    free,
                });
            }
            Ok(gpu)
        };
        match choice {
            BackendChoice::Cpu => Ok(Selected {
                backend: Backend::Cpu,
                note: None,
            }),
            BackendChoice::Cuda => Ok(Selected {
                backend: Backend::Cuda(try_gpu()?),
                note: None,
            }),
            BackendChoice::Auto => Ok(match try_gpu() {
                Ok(gpu) => Selected {
                    backend: Backend::Cuda(gpu),
                    note: None,
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
    Cuda(GpuCache),
}

impl Engine {
    pub fn load(
        dir: &ModelDir,
        backend: Backend,
        progress: impl FnMut(u64, u64),
    ) -> Result<Self, ModelError> {
        Ok(match backend {
            Backend::Cpu => Self::Cpu(Box::new(Qwen3::load(dir, progress)?)),
            Backend::Cuda(gpu) => Self::Cuda(Box::new(Qwen3Gpu::load(dir, gpu, progress)?)),
        })
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
            Self::Cuda(m) => EngineCache::Cuda(m.new_cache(capacity)?),
        })
    }

    /// What the GPU path has launched, allocated and copied so far; `None` on the CPU.
    pub fn traffic(&self) -> Option<Traffic> {
        match self {
            Self::Cpu(_) => None,
            Self::Cuda(m) => Some(m.gpu().traffic()),
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
}

impl CausalLm for Engine {
    type Error = ModelError;
    type Cache = EngineCache;

    fn forward(&self, tokens: &[u32], cache: &mut EngineCache) -> Result<Vec<f32>, ModelError> {
        self.forward_observed(tokens, cache, None)
    }
}
