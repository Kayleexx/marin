use super::{CpuMark, GpuMark};

/// A point in the history of a Qwen3.5 cache that it can be returned to exactly; see `Engine::mark`.
pub enum Mark {
    Gpu(GpuMark),
    Cpu(CpuMark),
}
