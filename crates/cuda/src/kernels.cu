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

// One block per row of `dim` values, normalized in place.
extern "C" __global__ void rms_norm(float* x, const float* weight, unsigned int dim, float eps) {
    __shared__ float shared[32];
    float* row = x + (size_t)blockIdx.x * dim;
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) sum += row[i] * row[i];
    float total = block_sum(sum, shared);
    float scale = 1.0f / sqrtf(total / (float)dim + eps);
    for (unsigned int i = threadIdx.x; i < dim; i += blockDim.x) row[i] = row[i] * scale * weight[i];
}

#define WARPS_PER_BLOCK 4
#define TOKEN_TILE 8

// y[t][r] = dot(w[r], x[t]). One warp per (weight row, tile of TOKEN_TILE tokens); the row is read once per tile.
extern "C" __global__ void matmul_bf16(const bf16_bits* w, const float* x, float* y,
                                       unsigned int rows, unsigned int cols, unsigned int tokens) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) return;
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
