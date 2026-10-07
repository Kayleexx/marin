use cuda::KvFormat;
use tensor::QuantScheme;

/// How the numbers a run keeps are stored: the layer weights and the GPU KV cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Precision {
    pub weights: Option<QuantScheme>,
    pub kv: KvFormat,
}
