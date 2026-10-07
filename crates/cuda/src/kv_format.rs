/// How the cache stores keys and values. Attention always computes in f32; only the stored numbers differ. The
/// order must match the `KV_*` constants in `kv_format.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvFormat {
    #[default]
    F32,
    F16,
}

impl KvFormat {
    pub const ALL: [KvFormat; 2] = [Self::F32, Self::F16];

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
        }
    }

    /// Bytes of the keys (or values) of one position for every kv head.
    pub fn row_bytes(self, kv_heads: usize, head_dim: usize) -> Option<usize> {
        let values = kv_heads.checked_mul(head_dim)?;
        match self {
            Self::F32 => values.checked_mul(size_of::<f32>()),
            Self::F16 => values.checked_mul(size_of::<u16>()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_have_the_documented_sizes() {
        assert_eq!(KvFormat::F32.row_bytes(8, 128), Some(4096));
        assert_eq!(KvFormat::F16.row_bytes(8, 128), Some(2048));
        assert_eq!(KvFormat::F32.row_bytes(usize::MAX, 2), None);
    }

    #[test]
    fn the_order_matches_the_kernel_constants() {
        let kernels = include_str!("kv_format.cu");
        for (format, name) in KvFormat::ALL.iter().zip(["F32", "F16"]) {
            assert!(
                kernels.contains(&format!("#define KV_{name} {}\n", format.index())),
                "{name}"
            );
        }
    }
}
