// Matrix times vector (and small token tiles) for int8 and int4 weight codes with one f32 scale per group of `group`
// columns. One warp per weight row; a lane takes 8 consecutive columns per step, which always lie inside one group
// because the group size is a multiple of 8. The codes are summed against the activations unscaled and the group's
// scale is applied to that partial sum.


// Integer to float without the slow conversion instruction: a float with the bits 0x4B0000nn is exactly 2^23 + nn, so
// subtracting 2^23 (plus the offset that turns the stored code into the signed one) leaves the code.
// `packed` must already have its bytes XORed with 0x80 for int8 (turning two's complement into offset binary).
__device__ __forceinline__ float code_q8(unsigned int packed, int i) {
    return __uint_as_float(__byte_perm(packed, 0x4B000000u, 0x7440 | i)) - 8388736.0f;  // 2^23 + 128
}

__device__ __forceinline__ float code_q4(unsigned int packed, int i) {
    return __uint_as_float(((packed >> (4 * i)) & 15u) | 0x4B000000u) - 8388616.0f;  // 2^23 + 8
}

__device__ __forceinline__ void decode_q8(uint2 packed, float* w) {
    const unsigned int lo = packed.x ^ 0x80808080u, hi = packed.y ^ 0x80808080u;
    for (int i = 0; i < 4; ++i) { w[i] = code_q8(lo, i); w[i + 4] = code_q8(hi, i); }
}

__device__ __forceinline__ void decode_q4(unsigned int packed, float* w) {
    for (int i = 0; i < 8; ++i) w[i] = code_q4(packed, i);
}

__device__ __forceinline__ float dot8(const float* w, const float* x) {
    return w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + w[5] * x[5] + w[6] * x[6] +
           w[7] * x[7];
}

template <int FORMAT>
__device__ void matvec_row(const void* codes, const float* scales, const float* x, float* y, unsigned int rows,
                           unsigned int cols, unsigned int group, unsigned int tokens, unsigned int row) {
    const unsigned int lane = threadIdx.x & 31, first_token = blockIdx.y * TOKEN_TILE;
    const size_t base = (size_t)row * cols;
    const float* row_scales = scales + (size_t)row * (cols / group);
    float acc[TOKEN_TILE];
    for (int j = 0; j < TOKEN_TILE; ++j) acc[j] = 0.0f;
    for (unsigned int c = lane * 8; c < cols; c += 256) {
        float w[8];
        if (FORMAT == FORMAT_Q8) {
            decode_q8(*reinterpret_cast<const uint2*>(reinterpret_cast<const signed char*>(codes) + base + c), w);
        } else {
            decode_q4(*reinterpret_cast<const unsigned int*>(reinterpret_cast<const unsigned char*>(codes) + (base + c) / 2),
                      w);
        }
        const float scale = row_scales[c / group];
        for (int j = 0; j < TOKEN_TILE; ++j) {
            const unsigned int t = first_token + j;
            if (t >= tokens) break;
            const float4 xa = *reinterpret_cast<const float4*>(x + (size_t)t * cols + c);
            const float4 xb = *reinterpret_cast<const float4*>(x + (size_t)t * cols + c + 4);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            acc[j] += scale * dot8(w, xv);
        }
    }
    for (int j = 0; j < TOKEN_TILE; ++j) {
        const float total = warp_sum(acc[j]);
        const unsigned int t = first_token + j;
        if (lane == 0 && t < tokens) y[(size_t)t * rows + row] = total;
    }
}

extern "C" __global__ void matmul_q8(const signed char* codes, const float* scales, const float* x, float* y,
                                     unsigned int rows, unsigned int cols, unsigned int tokens, unsigned int group) {
    const unsigned int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row < rows) matvec_row<FORMAT_Q8>(codes, scales, x, y, rows, cols, group, tokens, row);
}

extern "C" __global__ void matmul_q4(const unsigned char* codes, const float* scales, const float* x, float* y,
                                     unsigned int rows, unsigned int cols, unsigned int tokens, unsigned int group) {
    const unsigned int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row < rows) matvec_row<FORMAT_Q4>(codes, scales, x, y, rows, cols, group, tokens, row);
}

// One-token version: a warp takes ROWS_PER_WARP rows and reads the activations once for all of them, and each step
// covers UNROLL strides of 256 columns with every load issued before any is used, so each lane keeps
// ROWS_PER_WARP * UNROLL weight loads in flight. Rows past the end read the last row and are not written; columns
// past the end add nothing.
#define ROWS_PER_WARP 4
#define UNROLL 2

template <int FORMAT>
__device__ void matvec_rows(const void* codes, const float* scales, const float* x, float* y, unsigned int rows,
                            unsigned int cols, unsigned int group, unsigned int first_row) {
    const unsigned int lane = threadIdx.x & 31, groups = cols / group;
    float acc[ROWS_PER_WARP];
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;
    for (unsigned int c0 = lane * 8; c0 < cols; c0 += 256 * UNROLL) {
        uint2 raw[UNROLL][ROWS_PER_WARP];
        float scale[UNROLL][ROWS_PER_WARP];
        float4 xa[UNROLL], xb[UNROLL];
        bool valid[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            valid[u] = c0 + 256 * u < cols;
            const unsigned int c = valid[u] ? c0 + 256 * u : c0;
            xa[u] = *reinterpret_cast<const float4*>(x + c);
            xb[u] = *reinterpret_cast<const float4*>(x + c + 4);
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const unsigned int row = min(first_row + r, rows - 1);
                const size_t base = (size_t)row * cols + c;
                if (FORMAT == FORMAT_Q8) {
                    raw[u][r] = *reinterpret_cast<const uint2*>(reinterpret_cast<const signed char*>(codes) + base);
                } else {
                    raw[u][r] = make_uint2(
                        *reinterpret_cast<const unsigned int*>(reinterpret_cast<const unsigned char*>(codes) + base / 2), 0);
                }
                scale[u][r] = scales[(size_t)row * groups + c / group];
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float xv[8] = {xa[u].x, xa[u].y, xa[u].z, xa[u].w, xb[u].x, xb[u].y, xb[u].z, xb[u].w};
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float w[8];
                if (FORMAT == FORMAT_Q8) decode_q8(raw[u][r], w); else decode_q4(raw[u][r].x, w);
                if (valid[u]) acc[r] += scale[u][r] * dot8(w, xv);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const float total = warp_sum(acc[r]);
        if (lane == 0 && first_row + r < rows) y[first_row + r] = total;
    }
}

extern "C" __global__ void matvec_q8(const signed char* codes, const float* scales, const float* x, float* y,
                                     unsigned int rows, unsigned int cols, unsigned int group) {
    const unsigned int first_row = (blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5)) * ROWS_PER_WARP;
    if (first_row < rows) matvec_rows<FORMAT_Q8>(codes, scales, x, y, rows, cols, group, first_row);
}

extern "C" __global__ void matvec_q4(const unsigned char* codes, const float* scales, const float* x, float* y,
                                     unsigned int rows, unsigned int cols, unsigned int group) {
    const unsigned int first_row = (blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5)) * ROWS_PER_WARP;
    if (first_row < rows) matvec_rows<FORMAT_Q4>(codes, scales, x, y, rows, cols, group, first_row);
}

// out[t] = dequantized row ids[t] of an int8 table; one block per token.
extern "C" __global__ void embed_q8(const signed char* codes, const float* scales, const unsigned int* ids, float* out,
                                    unsigned int hidden, unsigned int group) {
    const size_t row = ids[blockIdx.x];
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[(size_t)blockIdx.x * hidden + i] = (float)codes[row * hidden + i] * scales[row * (hidden / group) + i / group];
}
