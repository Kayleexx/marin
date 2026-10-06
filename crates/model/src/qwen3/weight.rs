use cuda::{Buffer, CudaError, DeviceMatrix, DeviceQuantMatrix, Gpu, Tokens};

/// A weight matrix on the GPU, as bf16 or quantized.
pub(crate) enum Weight {
    Bf16(DeviceMatrix),
    Quant(DeviceQuantMatrix),
}

impl Weight {
    /// `out[t] = row ids[t]` of this matrix, as the token embedding.
    pub(crate) fn embed(
        &self,
        gpu: &Gpu,
        ids: &[u32],
        scratch: &mut Tokens,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        match self {
            Self::Bf16(table) => gpu.embed(table, ids, scratch, out),
            Self::Quant(table) => gpu.embed_quant(table, ids, scratch, out),
        }
    }

    pub(crate) fn matmul(
        &self,
        gpu: &Gpu,
        x: &Buffer,
        tokens: usize,
        out: &mut Buffer,
    ) -> Result<(), CudaError> {
        match self {
            Self::Bf16(w) => gpu.matmul(w, x, tokens, out),
            Self::Quant(w) => gpu.matmul_quant(w, x, tokens, out),
        }
    }
}

/// Two matrices over the same input; one launch when both are bf16.
pub(crate) fn matmul_pair(
    gpu: &Gpu,
    (a, b): (&Weight, &Weight),
    x: &Buffer,
    tokens: usize,
    (out_a, out_b): (&mut Buffer, &mut Buffer),
) -> Result<(), CudaError> {
    match (a, b) {
        (Weight::Bf16(a), Weight::Bf16(b)) => gpu.matmul_pair((a, b), x, tokens, (out_a, out_b)),
        _ => {
            a.matmul(gpu, x, tokens, out_a)?;
            b.matmul(gpu, x, tokens, out_b)
        }
    }
}

/// Three matrices over the same input; one launch when all are bf16.
pub(crate) fn matmul_triple(
    gpu: &Gpu,
    (a, b, c): (&Weight, &Weight, &Weight),
    x: &Buffer,
    tokens: usize,
    (out_a, out_b, out_c): (&mut Buffer, &mut Buffer, &mut Buffer),
) -> Result<(), CudaError> {
    match (a, b, c) {
        (Weight::Bf16(a), Weight::Bf16(b), Weight::Bf16(c)) => {
            gpu.matmul_triple((a, b, c), x, tokens, (out_a, out_b, out_c))
        }
        _ => {
            a.matmul(gpu, x, tokens, out_a)?;
            b.matmul(gpu, x, tokens, out_b)?;
            c.matmul(gpu, x, tokens, out_c)
        }
    }
}
