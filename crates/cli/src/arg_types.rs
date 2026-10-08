use clap::{Args, ValueEnum};

use model::{DEFAULT_GROUP, KvFormat, Precision, QuantBits, QuantScheme};

use crate::CliError;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartModeArg {
    Save,
    Restore,
    Cold,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendArg {
    /// Use the GPU when CUDA works and the model fits, otherwise the CPU
    Auto,
    Cpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantBitsArg {
    Int8,
    Int4,
}

#[derive(Args, Debug, Clone, Copy)]
pub struct QuantArgs {
    /// Keep the layer weights as 8-bit (int8) or 4-bit (int4) numbers on the GPU: less memory to read per token, a
    /// little accuracy lost (the embedding and output head stay bf16)
    #[arg(long, value_enum)]
    pub quantize: Option<QuantBitsArg>,
    /// Weights that share one scale when quantizing, a multiple of 32: smaller is more accurate and takes more memory
    #[arg(long, default_value_t = DEFAULT_GROUP, requires = "quantize")]
    pub quant_group: usize,
    #[command(flatten)]
    pub kv: KvArgs,
}

#[derive(Args, Debug, Clone, Copy)]
pub struct KvArgs {
    /// How the GPU KV cache stores keys and values: f32, or f16 for half the memory (attention math stays f32)
    #[arg(long, value_enum, default_value_t = KvTypeArg::F32)]
    pub kv_type: KvTypeArg,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvTypeArg {
    F32,
    F16,
}

impl QuantArgs {
    pub fn precision(&self) -> Result<Precision, CliError> {
        let bits = match self.quantize {
            None => None,
            Some(QuantBitsArg::Int8) => Some(QuantBits::Int8),
            Some(QuantBitsArg::Int4) => Some(QuantBits::Int4),
        };
        let weights = bits
            .map(|bits| QuantScheme::new(bits, self.quant_group))
            .transpose()
            .map_err(|e| CliError::BadOption(e.to_string()))?;
        Ok(Precision {
            weights,
            kv: self.kv.format(),
        })
    }
}

impl KvArgs {
    pub fn format(&self) -> KvFormat {
        match self.kv_type {
            KvTypeArg::F32 => KvFormat::F32,
            KvTypeArg::F16 => KvFormat::F16,
        }
    }
}
