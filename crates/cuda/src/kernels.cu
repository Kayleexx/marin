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
