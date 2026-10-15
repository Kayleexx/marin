// Granite's selected experts stay in device memory; routing and accumulation never leave the stream.
extern "C" __global__ void granite_scale(float* x, unsigned int n, float scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

extern "C" __global__ void granite_scaled_add(float* x, const float* update, unsigned int n, float scale) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], __fmul_rn(scale, update[i]));
}

extern "C" __global__ void granite_route(const float* logits, unsigned int* ids, float* gates,
                                          unsigned int tokens, unsigned int experts, unsigned int top_k) {
    unsigned int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    float scores[32], probs[32];
    unsigned int choices[32];
    for (unsigned int rank = 0; rank < top_k; ++rank) {
        scores[rank] = -INF_F;
        choices[rank] = experts;
    }
    for (unsigned int id = 0; id < experts; ++id) {
        float value = logits[(size_t)token * experts + id];
        if (!isfinite(value)) value = -INF_F;
        for (unsigned int rank = 0; rank < top_k; ++rank) {
            if (value > scores[rank] || (value == scores[rank] && id < choices[rank])) {
                for (unsigned int tail = top_k - 1; tail > rank; --tail) {
                    scores[tail] = scores[tail - 1];
                    choices[tail] = choices[tail - 1];
                }
                scores[rank] = value;
                choices[rank] = id;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (unsigned int rank = 0; rank < top_k; ++rank) {
        probs[rank] = expf(scores[rank] - scores[0]);
        sum += probs[rank];
    }
    for (unsigned int rank = 0; rank < top_k; ++rank) probs[rank] /= sum;
    // HF index_add visits grouped experts in ascending ID order.
    for (unsigned int rank = 1; rank < top_k; ++rank) {
        unsigned int id = choices[rank];
        float prob = probs[rank];
        unsigned int slot = rank;
        while (slot > 0 && choices[slot - 1] > id) {
            choices[slot] = choices[slot - 1];
            probs[slot] = probs[slot - 1];
            --slot;
        }
        choices[slot] = id;
        probs[slot] = prob;
    }
    for (unsigned int rank = 0; rank < top_k; ++rank) {
        ids[(size_t)token * top_k + rank] = choices[rank];
        gates[(size_t)token * top_k + rank] = probs[rank];
    }
}

__device__ float granite_dot(const bf16_bits* weight, const float* input, unsigned int cols) {
    unsigned int lane = threadIdx.x & 31;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += 32)
        sum += __uint_as_float((unsigned int)weight[c] << 16) * input[c];
    return warp_sum(sum);
}

extern "C" __global__ void granite_expert_input(const bf16_bits* weights, const float* input,
                                                   const unsigned int* ids, float* out,
                                                   unsigned int tokens, unsigned int top_k,
                                                   unsigned int hidden, unsigned int intermediate) {
    unsigned int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    unsigned int rows = intermediate * 2;
    if (row >= tokens * top_k * rows) return;
    unsigned int assignment = row / rows;
    unsigned int expert = ids[assignment];
    const bf16_bits* weight = weights + ((size_t)expert * rows + row % rows) * hidden;
    float value = granite_dot(weight, input + (size_t)(assignment / top_k) * hidden, hidden);
    if ((threadIdx.x & 31) == 0) out[row] = value;
}

extern "C" __global__ void granite_silu(float* input, unsigned int assignments,
                                         unsigned int intermediate) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= assignments * intermediate) return;
    unsigned int assignment = i / intermediate, col = i % intermediate;
    float* row = input + (size_t)assignment * intermediate * 2;
    float gate = row[col];
    row[col] = gate / (1.0f + expf(-gate)) * row[intermediate + col];
}

extern "C" __global__ void granite_expert_output(const bf16_bits* weights, const float* input,
                                                    const unsigned int* ids, float* out,
                                                    unsigned int assignments, unsigned int hidden,
                                                    unsigned int intermediate) {
    unsigned int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= assignments * hidden) return;
    unsigned int assignment = row / hidden;
    unsigned int expert = ids[assignment];
    const bf16_bits* weight = weights + ((size_t)expert * hidden + row % hidden) * intermediate;
    float value = granite_dot(weight, input + (size_t)assignment * intermediate * 2, intermediate);
    if ((threadIdx.x & 31) == 0) out[row] = value;
}

extern "C" __global__ void granite_mix(const float* experts, const float* gates, float* out,
                                        unsigned int tokens, unsigned int top_k, unsigned int hidden) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * hidden) return;
    unsigned int token = i / hidden, col = i % hidden;
    float total = 0.0f;
    for (unsigned int rank = 0; rank < top_k; ++rank) {
        unsigned int assignment = token * top_k + rank;
        total = __fadd_rn(total, __fmul_rn(gates[assignment], experts[(size_t)assignment * hidden + col]));
    }
    out[i] = total;
}
