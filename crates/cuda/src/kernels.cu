// Kernels for decoder-only transformer inference: bf16 weights, f32 activations.
// Compiled at startup with NVRTC; every launch goes through the checked wrappers in ops.rs.

typedef unsigned short bf16_bits;
#define INF_F __int_as_float(0x7f800000)

__device__ __forceinline__ float bf16_lo(unsigned int pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float bf16_hi(unsigned int pair) { return __uint_as_float(pair & 0xffff0000u); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// Sums `v` over the whole block. `shared` needs one float per warp. Result valid in every thread.
__device__ float block_sum(float v, float* shared) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total += shared[w];
    __syncthreads();
    return total;
}

__device__ float block_max(float v, float* shared) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float best = -INF_F;
    for (int w = 0; w < warps; ++w) best = fmaxf(best, shared[w]);
    __syncthreads();
    return best;
}

extern "C" __global__ void embed(const bf16_bits* table, const unsigned int* ids, float* out,
                                 unsigned int hidden) {
    const bf16_bits* row = table + (size_t)ids[blockIdx.x] * hidden;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[(size_t)blockIdx.x * hidden + i] = __uint_as_float((unsigned int)row[i] << 16);
}

// One block per row of `dim` values; `src` and `dst` may be the same row.
__device__ void rms_norm_row(const float* src, float* dst, const float* weight, unsigned int dim, float eps) {
    __shared__ float shared[32];
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) sum += src[i] * src[i];
    float total = block_sum(sum, shared);
    float scale = 1.0f / sqrtf(total / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i] * scale * weight[i];
}

extern "C" __global__ void rms_norm(float* x, const float* weight, unsigned int dim, float eps) {
    float* row = x + (size_t)blockIdx.x * dim;
    rms_norm_row(row, row, weight, dim, eps);
}

extern "C" __global__ void rms_norm_to(const float* x, float* out, const float* weight, unsigned int dim, float eps) {
    rms_norm_row(x + (size_t)blockIdx.x * dim, out + (size_t)blockIdx.x * dim, weight, dim, eps);
}

#define WARPS_PER_BLOCK 4
#define TOKEN_TILE 8

// y[t][r] = dot(w[r], x[t]). One warp per (weight row, tile of TOKEN_TILE tokens); the row is read once per tile.
__device__ void matmul_row(const bf16_bits* w, const float* x, float* y, unsigned int rows, unsigned int cols,
                           unsigned int tokens, unsigned int row) {
    unsigned int lane = threadIdx.x & 31;
    unsigned int first_token = blockIdx.y * TOKEN_TILE;
    float acc[TOKEN_TILE];
    for (int j = 0; j < TOKEN_TILE; ++j) acc[j] = 0.0f;
    const bf16_bits* wrow = w + (size_t)row * cols;

    if ((cols & 7u) == 0u) {
        for (unsigned int c = lane * 8; c < cols; c += 256) {
            uint4 packed = *reinterpret_cast<const uint4*>(wrow + c);
            float wf[8] = {bf16_lo(packed.x), bf16_hi(packed.x), bf16_lo(packed.y), bf16_hi(packed.y),
                           bf16_lo(packed.z), bf16_hi(packed.z), bf16_lo(packed.w), bf16_hi(packed.w)};
            for (int j = 0; j < TOKEN_TILE; ++j) {
                unsigned int t = first_token + j;
                if (t >= tokens) break;
                const float4* xv = reinterpret_cast<const float4*>(x + (size_t)t * cols + c);
                float4 a = xv[0], b = xv[1];
                acc[j] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w +
                          wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
            }
        }
    } else {
        for (unsigned int c = lane; c < cols; c += 32) {
            float wf = __uint_as_float((unsigned int)wrow[c] << 16);
            for (int j = 0; j < TOKEN_TILE; ++j) {
                unsigned int t = first_token + j;
                if (t >= tokens) break;
                acc[j] += wf * x[(size_t)t * cols + c];
            }
        }
    }
    for (int j = 0; j < TOKEN_TILE; ++j) {
        float total = warp_sum(acc[j]);
        unsigned int t = first_token + j;
        if (lane == 0 && t < tokens) y[(size_t)t * rows + row] = total;
    }
}

extern "C" __global__ void matmul_bf16(const bf16_bits* w, const float* x, float* y,
                                       unsigned int rows, unsigned int cols, unsigned int tokens) {
    unsigned int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row < rows) matmul_row(w, x, y, rows, cols, tokens, row);
}

// Two or three matrices that multiply the same input (query/key/value, gate/up) in one launch: the grid covers
// their rows one after the other, so the small ones no longer leave the GPU idle between launches.
extern "C" __global__ void matmul_bf16_x2(const bf16_bits* w0, const bf16_bits* w1, const float* x, float* y0,
                                          float* y1, unsigned int rows0, unsigned int rows1, unsigned int cols,
                                          unsigned int tokens) {
    unsigned int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row < rows0) {
        matmul_row(w0, x, y0, rows0, cols, tokens, row);
    } else if (row - rows0 < rows1) {
        matmul_row(w1, x, y1, rows1, cols, tokens, row - rows0);
    }
}

extern "C" __global__ void matmul_bf16_x3(const bf16_bits* w0, const bf16_bits* w1, const bf16_bits* w2,
                                          const float* x, float* y0, float* y1, float* y2, unsigned int rows0,
                                          unsigned int rows1, unsigned int rows2, unsigned int cols,
                                          unsigned int tokens) {
    unsigned int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row < rows0) {
        matmul_row(w0, x, y0, rows0, cols, tokens, row);
    } else if (row - rows0 < rows1) {
        matmul_row(w1, x, y1, rows1, cols, tokens, row - rows0);
    } else if (row - rows0 - rows1 < rows2) {
        matmul_row(w2, x, y2, rows2, cols, tokens, row - rows0 - rows1);
    }
}


// x is [token][head][head_dim]; one block per (token, head), rotating pairs (i, i + head_dim / 2).
extern "C" __global__ void rope(float* x, unsigned int heads, unsigned int head_dim,
                                unsigned int start, float theta) {
    unsigned int token = blockIdx.x / heads;
    float* head = x + (size_t)blockIdx.x * head_dim;
    unsigned int half_dim = head_dim / 2;
    float position = (float)(start + token);
    for (unsigned int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float exponent = (float)(2 * i) / (float)head_dim;
        float angle = position * powf(theta, -exponent);
        float s, c;
        sincosf(angle, &s, &c);
        float a = head[i], b = head[i + half_dim];
        head[i] = a * c - b * s;
        head[i + half_dim] = b * c + a * s;
    }
}

// Causal grouped-query attention with an online softmax. One block per (token, head);
// blockDim.x must be >= head_dim and a multiple of 32. Shared memory: head_dim + blockDim.x + 32 floats.
extern "C" __global__ void attention(const float* q, const float* keys, const float* values, float* out,
                                     unsigned int heads, unsigned int kv_heads, unsigned int head_dim,
                                     unsigned int start, float scale) {
    extern __shared__ float shared[];
    float* q_s = shared;
    float* scores = shared + head_dim;
    float* reduce = scores + blockDim.x;

    unsigned int token = blockIdx.x / heads, head = blockIdx.x % heads;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned int kv_stride = kv_heads * head_dim;
    unsigned int visible = start + token + 1;
    unsigned int tid = threadIdx.x;

    if (tid < head_dim) q_s[tid] = q[(size_t)blockIdx.x * head_dim + tid];
    __syncthreads();

    float running_max = -INF_F, running_sum = 0.0f, acc = 0.0f;
    for (unsigned int base = 0; base < visible; base += blockDim.x) {
        unsigned int p = base + tid;
        float s = -INF_F;
        if (p < visible) {
            const float* k = keys + (size_t)p * kv_stride + (size_t)kv_head * head_dim;
            float dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d) dot += q_s[d] * k[d];
            s = dot * scale;
        }
        scores[tid] = s;
        float chunk_max = block_max(s, reduce);
        float new_max = fmaxf(running_max, chunk_max);
        float correction = expf(running_max - new_max);
        acc *= correction;
        running_sum *= correction;
        unsigned int count = min(blockDim.x, visible - base);
        for (unsigned int j = 0; j < count; ++j) {
            float weight = expf(scores[j] - new_max);
            running_sum += weight;
            if (tid < head_dim)
                acc += weight * values[(size_t)(base + j) * kv_stride + (size_t)kv_head * head_dim + tid];
        }
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) out[(size_t)blockIdx.x * head_dim + tid] = acc / running_sum;
}

// Decode attention for ONE query token, split over the context so every multiprocessor has work.
// Grid (splits, kv_heads), 128 threads. A block takes `chunk` consecutive positions for one kv head and for every query
// head that shares it, so each key and value row is read once. It writes that chunk's partial softmax for each query
// head: the maximum score, the sum of exponentials, and the exponential-weighted sum of values.
// Shared memory: group * (head_dim + chunk) floats. group = heads / kv_heads must be at most MAX_GROUP.
#define MAX_GROUP 8
extern "C" __global__ void attention_partial(const float* q, const float* keys, const float* values,
                                             float* partial_acc, float* partial_ml, unsigned int heads,
                                             unsigned int kv_heads, unsigned int head_dim, unsigned int visible,
                                             unsigned int chunk, unsigned int splits, float scale) {
    extern __shared__ float shared[];
    const unsigned int group = heads / kv_heads;
    float* q_s = shared;                      // [group][head_dim]
    float* weights = shared + group * head_dim;  // [group][chunk]
    const unsigned int split = blockIdx.x, kv_head = blockIdx.y;
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
    const unsigned int first = split * chunk;
    const unsigned int count = min(chunk, visible - first);
    const unsigned int kv_stride = kv_heads * head_dim;
    const float* q_head = q + (size_t)kv_head * group * head_dim;

    for (unsigned int i = tid; i < group * head_dim; i += blockDim.x) q_s[i] = q_head[i];
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced.
    for (unsigned int i = warp; i < count; i += warps) {
        const float* k = keys + (size_t)(first + i) * kv_stride + (size_t)kv_head * head_dim;
        float dots[MAX_GROUP];
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) dots[g] = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            const float kd = k[d];
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g)
                if (g < group) dots[g] += q_s[g * head_dim + d] * kd;
        }
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) {
            float sum = dots[g];
            for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0 && g < group) weights[g * chunk + i] = sum * scale;
        }
    }
    __syncthreads();

    // Per query head: the chunk's maximum, then the weights exp(score - max) and their sum.
    for (unsigned int g = warp; g < group; g += warps) {
        float maximum = -INF_F;
        for (unsigned int i = lane; i < count; i += 32) maximum = fmaxf(maximum, weights[g * chunk + i]);
        for (int offset = 16; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        float total = 0.0f;
        for (unsigned int i = lane; i < count; i += 32) {
            const float w = expf(weights[g * chunk + i] - maximum);
            weights[g * chunk + i] = w;
            total += w;
        }
        for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) {
            const size_t slot = ((size_t)(kv_head * group + g) * splits + split) * 2;
            partial_ml[slot] = maximum;
            partial_ml[slot + 1] = total;
        }
    }
    __syncthreads();

    // Weighted values: threads stride across the head, each row read coalesced and used for every query head.
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc[MAX_GROUP];
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) acc[g] = 0.0f;
        for (unsigned int i = 0; i < count; ++i) {
            const float v = values[(size_t)(first + i) * kv_stride + (size_t)kv_head * head_dim + d];
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g)
                if (g < group) acc[g] += weights[g * chunk + i] * v;
        }
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g)
            if (g < group) partial_acc[((size_t)(kv_head * group + g) * splits + split) * head_dim + d] = acc[g];
    }
}

// Attention for a few new tokens at once (checking guessed tokens), split over the context like the decode kernel above.
// Grid (splits, kv_heads), 128 threads. A block takes `chunk` consecutive positions for one kv head and for every
// (token, query head) row that uses it, so each key and value row is read once however many tokens are checked. Row
// `token * group + g` is causal: it sees positions up to `start + token`. It writes the chunk's partial softmax for
// each row at `(token * heads + head) * splits + split`, which `attention_merge` combines with one block per row.
// Shared memory: rows * (head_dim + chunk) floats, rows = tokens * group <= MAX_ROWS.
// The kernel is built for row capacities of 8, 16 and 32 so a pass of few rows does not pay for the loops of many.
template <int MAX_ROWS>
__device__ void attention_multi_body(const float* q, const float* keys, const float* values, float* partial_acc,
                                     float* partial_ml, unsigned int heads, unsigned int kv_heads,
                                     unsigned int head_dim, unsigned int start, unsigned int tokens,
                                     unsigned int chunk, unsigned int splits, float scale) {
    extern __shared__ float shared[];
    const unsigned int group = heads / kv_heads;
    const unsigned int rows = tokens * group;
    float* q_s = shared;                      // [rows][head_dim]
    float* weights = shared + rows * head_dim;  // [rows][chunk]
    const unsigned int split = blockIdx.x, kv_head = blockIdx.y;
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
    const unsigned int first = split * chunk;
    const unsigned int count = min(chunk, start + tokens - first);
    const unsigned int kv_stride = kv_heads * head_dim;

    for (unsigned int i = tid; i < rows * head_dim; i += blockDim.x) {
        const unsigned int r = i / head_dim, d = i % head_dim;
        const unsigned int token = r / group, g = r % group;
        q_s[i] = q[((size_t)token * heads + kv_head * group + g) * head_dim + d];
    }
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced and used for every row.
    for (unsigned int i = warp; i < count; i += warps) {
        const float* k = keys + (size_t)(first + i) * kv_stride + (size_t)kv_head * head_dim;
        float dots[MAX_ROWS];
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) dots[r] = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            const float kd = k[d];
#pragma unroll
            for (int r = 0; r < MAX_ROWS; ++r)
                if (r < rows) dots[r] += q_s[r * head_dim + d] * kd;
        }
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) {
            if (r < rows) {
                float sum = dots[r];
                for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
                // Row r may only see the positions up to start + its token.
                if (lane == 0) weights[r * chunk + i] = (first + i <= start + r / group) ? sum * scale : -INF_F;
            }
        }
    }
    __syncthreads();

    // Per row: the chunk's maximum, then the weights exp(score - max) and their sum. A row that sees nothing in this
    // chunk has no maximum: its weights are zero and the merge gives it no share.
    for (unsigned int r = warp; r < rows; r += warps) {
        float maximum = -INF_F;
        for (unsigned int i = lane; i < count; i += 32) maximum = fmaxf(maximum, weights[r * chunk + i]);
        for (int offset = 16; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        float total = 0.0f;
        for (unsigned int i = lane; i < count; i += 32) {
            const float w = maximum == -INF_F ? 0.0f : expf(weights[r * chunk + i] - maximum);
            weights[r * chunk + i] = w;
            total += w;
        }
        for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_down_sync(0xffffffffu, total, offset);
        if (lane == 0) {
            const unsigned int token = r / group, g = r % group;
            const size_t slot = (((size_t)token * heads + kv_head * group + g) * splits + split) * 2;
            partial_ml[slot] = maximum;
            partial_ml[slot + 1] = total;
        }
    }
    __syncthreads();

    // Weighted values: threads stride across the head, each value row read once and used for every row.
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc[MAX_ROWS];
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) acc[r] = 0.0f;
#pragma unroll 4
        for (unsigned int i = 0; i < count; ++i) {
            const float v = values[(size_t)(first + i) * kv_stride + (size_t)kv_head * head_dim + d];
#pragma unroll
            for (int r = 0; r < MAX_ROWS; ++r)
                if (r < rows) acc[r] += weights[r * chunk + i] * v;
        }
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) {
            if (r < rows) {
                const unsigned int token = r / group, g = r % group;
                partial_acc[(((size_t)token * heads + kv_head * group + g) * splits + split) * head_dim + d] = acc[r];
            }
        }
    }
}

#define MULTI_ARGS                                                                                                    \
    const float *q, const float *keys, const float *values, float *partial_acc, float *partial_ml,                    \
        unsigned int heads, unsigned int kv_heads, unsigned int head_dim, unsigned int start, unsigned int tokens,    \
        unsigned int chunk, unsigned int splits, float scale
#define MULTI_CALL q, keys, values, partial_acc, partial_ml, heads, kv_heads, head_dim, start, tokens, chunk, splits, scale
extern "C" __global__ void attention_multi_partial_8(MULTI_ARGS) { attention_multi_body<8>(MULTI_CALL); }
extern "C" __global__ void attention_multi_partial_16(MULTI_ARGS) { attention_multi_body<16>(MULTI_CALL); }
extern "C" __global__ void attention_multi_partial_32(MULTI_ARGS) { attention_multi_body<32>(MULTI_CALL); }

// Combines the partial softmaxes of one query head. Grid: heads; blockDim.x >= head_dim.
extern "C" __global__ void attention_merge(const float* partial_acc, const float* partial_ml, float* out,
                                           unsigned int head_dim, unsigned int splits) {
    const unsigned int head = blockIdx.x, d = threadIdx.x;
    float maximum = -INF_F;
    for (unsigned int s = 0; s < splits; ++s) maximum = fmaxf(maximum, partial_ml[((size_t)head * splits + s) * 2]);
    float total = 0.0f, acc = 0.0f;
    for (unsigned int s = 0; s < splits; ++s) {
        const size_t slot = ((size_t)head * splits + s) * 2;
        const float scale = expf(partial_ml[slot] - maximum);
        total += partial_ml[slot + 1] * scale;
        if (d < head_dim) acc += partial_acc[((size_t)head * splits + s) * head_dim + d] * scale;
    }
    if (d < head_dim) out[(size_t)head * head_dim + d] = acc / total;
}

extern "C" __global__ void silu_mul(float* gate, const float* up, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

extern "C" __global__ void add_in_place(float* a, const float* b, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
