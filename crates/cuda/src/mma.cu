// Tensor-core matrix multiply for prompt passes, for bf16 weights and for int8/int4 weight codes.
// One block takes MMA_ROWS weight rows and MMA_TOKENS tokens; each of its 4 warps owns 16 rows. The activations are
// f32, so each is split into a bf16 high part and a bf16 low part (about 16 mantissa bits together) and multiplied
// separately: two multiply-accumulates per weight tile keep the result as accurate as the f32 kernel needs, which a
// single bf16 rounding of the activations would not. Quantized codes are small integers, which bf16 holds exactly, so
// they go through the tensor cores as they are; the products of one group are summed unscaled and the group's scale is
// applied to that partial sum, so the result is the same as scaling every weight first. cols must be a multiple of
// 16, and for quantized weights of the group size. With SPLITS = 3 the activations are split into three bf16 parts
// (8 + 8 + 8 mantissa bits, the whole f32 mantissa), so the products are exact and only the f32 accumulation rounds.
#define MMA_ROWS 64
#define MMA_TOKENS 64
#define MMA_PAD 24  // shared row length in bf16 values: 16 plus padding so fragment reads do not share banks
#define FORMAT_BF16 0
#define FORMAT_Q8 1
#define FORMAT_Q4 2

__device__ __forceinline__ bf16_bits round_bf16(float v) {
    unsigned int bits = __float_as_uint(v);
    return (bf16_bits)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void mma_16816(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two small integers as a pair of bf16 values (exact: an 8-bit integer fits in bf16's mantissa).
__device__ __forceinline__ unsigned int pair_bf16(int low, int high) {
    return (__float_as_uint((float)low) >> 16) | (__float_as_uint((float)high) & 0xffff0000u);
}

// The 8 weights of `row` at columns k0 + k_offset.. as bf16 pairs (codes for the quantized formats).
template <int FORMAT>
__device__ __forceinline__ uint4 stage_weights(const void* w, size_t first) {
    if (FORMAT == FORMAT_BF16) return *reinterpret_cast<const uint4*>(reinterpret_cast<const bf16_bits*>(w) + first);
    if (FORMAT == FORMAT_Q8) {
        const uint2 c = *reinterpret_cast<const uint2*>(reinterpret_cast<const signed char*>(w) + first);
        return make_uint4(pair_bf16((signed char)c.x, (signed char)(c.x >> 8)),
                          pair_bf16((signed char)(c.x >> 16), (signed char)(c.x >> 24)),
                          pair_bf16((signed char)c.y, (signed char)(c.y >> 8)),
                          pair_bf16((signed char)(c.y >> 16), (signed char)(c.y >> 24)));
    }
    const unsigned int c = *reinterpret_cast<const unsigned int*>(reinterpret_cast<const unsigned char*>(w) + first / 2);
    return make_uint4(pair_bf16((int)(c & 15) - 8, (int)((c >> 4) & 15) - 8),
                      pair_bf16((int)((c >> 8) & 15) - 8, (int)((c >> 12) & 15) - 8),
                      pair_bf16((int)((c >> 16) & 15) - 8, (int)((c >> 20) & 15) - 8),
                      pair_bf16((int)((c >> 24) & 15) - 8, (int)(c >> 28) - 8));
}

template <int FORMAT, int SPLITS>
__device__ void matmul_mma(const void* w, const float* scales, const float* x, float* y, unsigned int rows,
                           unsigned int cols, unsigned int tokens, unsigned int group) {
    __shared__ __align__(16) bf16_bits w_s[MMA_ROWS * MMA_PAD];
    __shared__ __align__(16) bf16_bits x_hi[MMA_TOKENS * MMA_PAD];
    __shared__ __align__(16) bf16_bits x_lo[MMA_TOKENS * MMA_PAD];
    __shared__ __align__(16) bf16_bits x_mid[SPLITS == 3 ? MMA_TOKENS * MMA_PAD : 1];
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const unsigned int row0 = blockIdx.x * MMA_ROWS, token0 = blockIdx.y * MMA_TOKENS;
    const unsigned int load_row = tid >> 1, load_k = (tid & 1) * 8;
    const unsigned int groups = FORMAT == FORMAT_BF16 ? 1 : cols / group;
    float acc[MMA_TOKENS / 8][4], part[MMA_TOKENS / 8][4];
    for (int n = 0; n < MMA_TOKENS / 8; ++n)
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; part[n][i] = 0.0f; }

    for (unsigned int k0 = 0; k0 < cols; k0 += 16) {
        uint4 weights = make_uint4(0, 0, 0, 0);
        if (row0 + load_row < rows)
            weights = stage_weights<FORMAT>(w, (size_t)(row0 + load_row) * cols + k0 + load_k);
        *reinterpret_cast<uint4*>(w_s + load_row * MMA_PAD + load_k) = weights;

        float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (token0 + load_row < tokens) {
            const float4* src = reinterpret_cast<const float4*>(x + (size_t)(token0 + load_row) * cols + k0 + load_k);
            const float4 a = src[0], b = src[1];
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        }
        for (int j = 0; j < 8; ++j) {
            const bf16_bits high = round_bf16(v[j]);
            const float rest = v[j] - __uint_as_float((unsigned int)high << 16);
            x_hi[load_row * MMA_PAD + load_k + j] = high;
            if (SPLITS == 3) {
                const bf16_bits middle = round_bf16(rest);
                x_mid[load_row * MMA_PAD + load_k + j] = middle;
                x_lo[load_row * MMA_PAD + load_k + j] = round_bf16(rest - __uint_as_float((unsigned int)middle << 16));
            } else {
                x_lo[load_row * MMA_PAD + load_k + j] = round_bf16(rest);
            }
        }
        __syncthreads();

        const bf16_bits* wa = w_s + (warp * 16 + g) * MMA_PAD;
        const unsigned int a[4] = {*reinterpret_cast<const unsigned int*>(wa + 2 * t),
                                   *reinterpret_cast<const unsigned int*>(wa + 8 * MMA_PAD + 2 * t),
                                   *reinterpret_cast<const unsigned int*>(wa + 2 * t + 8),
                                   *reinterpret_cast<const unsigned int*>(wa + 8 * MMA_PAD + 2 * t + 8)};
        for (int n = 0; n < MMA_TOKENS / 8; ++n) {
            const unsigned int offset = (n * 8 + g) * MMA_PAD + 2 * t;
            float* c = FORMAT == FORMAT_BF16 ? acc[n] : part[n];
            float step[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float* into = SPLITS == 3 ? step : c;
            mma_16816(into, a, *reinterpret_cast<const unsigned int*>(x_hi + offset),
                      *reinterpret_cast<const unsigned int*>(x_hi + offset + 8));
            if (SPLITS == 3)
                mma_16816(into, a, *reinterpret_cast<const unsigned int*>(x_mid + offset),
                          *reinterpret_cast<const unsigned int*>(x_mid + offset + 8));
            mma_16816(into, a, *reinterpret_cast<const unsigned int*>(x_lo + offset),
                      *reinterpret_cast<const unsigned int*>(x_lo + offset + 8));
            if (SPLITS == 3)
                for (int i = 0; i < 4; ++i) c[i] += step[i];
        }
        __syncthreads();

        if (FORMAT != FORMAT_BF16 && (k0 + 16) % group == 0) {
            const unsigned int row_a = row0 + warp * 16 + g, row_b = row_a + 8, which = k0 / group;
            const float scale_a = row_a < rows ? scales[(size_t)row_a * groups + which] : 0.0f;
            const float scale_b = row_b < rows ? scales[(size_t)row_b * groups + which] : 0.0f;
            for (int n = 0; n < MMA_TOKENS / 8; ++n) {
                acc[n][0] += part[n][0] * scale_a; acc[n][1] += part[n][1] * scale_a;
                acc[n][2] += part[n][2] * scale_b; acc[n][3] += part[n][3] * scale_b;
                part[n][0] = 0.0f; part[n][1] = 0.0f; part[n][2] = 0.0f; part[n][3] = 0.0f;
            }
        }
    }

    for (int n = 0; n < MMA_TOKENS / 8; ++n) {
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = row0 + warp * 16 + g + half * 8;
            if (row >= rows) continue;
            const unsigned int token = token0 + n * 8 + 2 * t;
            if (token < tokens) y[(size_t)token * rows + row] = acc[n][half * 2];
            if (token + 1 < tokens) y[(size_t)(token + 1) * rows + row] = acc[n][half * 2 + 1];
        }
    }
}

extern "C" __global__ void matmul_bf16_mma(const bf16_bits* w, const float* x, float* y, unsigned int rows,
                                           unsigned int cols, unsigned int tokens) {
    matmul_mma<FORMAT_BF16, 2>(w, nullptr, x, y, rows, cols, tokens, 0);
}

extern "C" __global__ void matmul_q8_mma(const signed char* codes, const float* scales, const float* x, float* y,
                                         unsigned int rows, unsigned int cols, unsigned int tokens,
                                         unsigned int group) {
    matmul_mma<FORMAT_Q8, 2>(codes, scales, x, y, rows, cols, tokens, group);
}

extern "C" __global__ void matmul_q8_mma_exact(const signed char* codes, const float* scales, const float* x, float* y,
                                               unsigned int rows, unsigned int cols, unsigned int tokens,
                                               unsigned int group) {
    matmul_mma<FORMAT_Q8, 3>(codes, scales, x, y, rows, cols, tokens, group);
}

extern "C" __global__ void matmul_q4_mma(const unsigned char* codes, const float* scales, const float* x, float* y,
                                         unsigned int rows, unsigned int cols, unsigned int tokens,
                                         unsigned int group) {
    matmul_mma<FORMAT_Q4, 2>(codes, scales, x, y, rows, cols, tokens, group);
}

// Exact-integer tensor-core multiply for int8 weight codes. The activations of one token and one quantization group are
// written as fixed point with a shared power-of-two scale (22 bits below the group's largest magnitude), split into
// three signed int8 limbs. Each limb is multiplied by the codes on the int8 tensor cores with int32 sums, which are
// exact, and the limbs are recombined in f32 once per group, where the weight scale and the token scale are applied.
#define IMMA_TOKENS 32
#define IMMA_MAX_GROUP 128
#define IMMA_STRIDE (IMMA_MAX_GROUP + 16)  // bytes per shared row: padding keeps fragment reads off shared banks

__device__ __forceinline__ void mma_s8(int* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void matmul_q8_imma(const signed char* codes, const float* scales, const float* x, float* y,
                                          unsigned int rows, unsigned int cols, unsigned int tokens,
                                          unsigned int group) {
    __shared__ __align__(16) signed char w_s[MMA_ROWS * IMMA_STRIDE];
    __shared__ __align__(16) signed char limb[3][IMMA_TOKENS * IMMA_STRIDE];
    __shared__ float token_scale[IMMA_TOKENS];
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const unsigned int row0 = blockIdx.x * MMA_ROWS, token0 = blockIdx.y * IMMA_TOKENS;
    const unsigned int groups = cols / group;
    const unsigned int per_thread = group / 4;
    const unsigned int tl = tid >> 2, sub = tid & 3;
    float acc[IMMA_TOKENS / 8][4];
    for (int n = 0; n < IMMA_TOKENS / 8; ++n)
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;

    for (unsigned int which = 0; which < groups; ++which) {
        const unsigned int k0 = which * group;
        for (unsigned int chunk = tid; chunk < MMA_ROWS * group / 16; chunk += 128) {
            const unsigned int r = chunk / (group / 16), c = (chunk % (group / 16)) * 16;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row0 + r < rows) v = *reinterpret_cast<const uint4*>(codes + (size_t)(row0 + r) * cols + k0 + c);
            *reinterpret_cast<uint4*>(w_s + r * IMMA_STRIDE + c) = v;
        }

        float v[IMMA_MAX_GROUP / 4];
        float peak = 0.0f;
        const bool live = token0 + tl < tokens;
        for (unsigned int i = 0; i < per_thread; i += 4) {
            float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (live) q = *reinterpret_cast<const float4*>(x + (size_t)(token0 + tl) * cols + k0 + sub * per_thread + i);
            v[i] = q.x; v[i + 1] = q.y; v[i + 2] = q.z; v[i + 3] = q.w;
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fmaxf(fabsf(q.z), fabsf(q.w))));
        }
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1));
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2));
        const int top = peak > 0.0f ? max((int)((__float_as_uint(peak) >> 23) & 255u) - 127, -100) : 0;
        const float up = ldexpf(1.0f, 21 - top);
        if (sub == 0) token_scale[tl] = ldexpf(1.0f, top - 21);
        for (unsigned int i = 0; i < per_thread; i += 4) {
            int packed[3] = {0, 0, 0};
            for (int j = 0; j < 4; ++j) {
                const int fixed = (int)rintf(v[i + j] * up);
                const int d0 = ((fixed + 128) & 255) - 128;
                const int rest = (fixed - d0) >> 8;
                const int d1 = ((rest + 128) & 255) - 128;
                const int d2 = (rest - d1) >> 8;
                packed[0] |= (d0 & 255) << (8 * j);
                packed[1] |= (d1 & 255) << (8 * j);
                packed[2] |= (d2 & 255) << (8 * j);
            }
            for (int l = 0; l < 3; ++l)
                *reinterpret_cast<int*>(limb[l] + tl * IMMA_STRIDE + sub * per_thread + i) = packed[l];
        }
        __syncthreads();

        int part[3][IMMA_TOKENS / 8][4];
        for (int l = 0; l < 3; ++l)
            for (int n = 0; n < IMMA_TOKENS / 8; ++n)
                for (int i = 0; i < 4; ++i) part[l][n][i] = 0;
        for (unsigned int ks = 0; ks < group; ks += 32) {
            const signed char* wa = w_s + (warp * 16 + g) * IMMA_STRIDE + ks + 4 * t;
            const unsigned int a[4] = {*reinterpret_cast<const unsigned int*>(wa),
                                       *reinterpret_cast<const unsigned int*>(wa + 8 * IMMA_STRIDE),
                                       *reinterpret_cast<const unsigned int*>(wa + 16),
                                       *reinterpret_cast<const unsigned int*>(wa + 8 * IMMA_STRIDE + 16)};
            for (int n = 0; n < IMMA_TOKENS / 8; ++n)
                for (int l = 0; l < 3; ++l) {
                    const signed char* xb = limb[l] + (n * 8 + g) * IMMA_STRIDE + ks + 4 * t;
                    mma_s8(part[l][n], a, *reinterpret_cast<const unsigned int*>(xb),
                           *reinterpret_cast<const unsigned int*>(xb + 16));
                }
        }
        const unsigned int row_a = row0 + warp * 16 + g, row_b = row_a + 8;
        const float scale_a = row_a < rows ? scales[(size_t)row_a * groups + which] : 0.0f;
        const float scale_b = row_b < rows ? scales[(size_t)row_b * groups + which] : 0.0f;
        for (int n = 0; n < IMMA_TOKENS / 8; ++n)
            for (int i = 0; i < 4; ++i) {
                const float joined = fmaf((float)part[2][n][i], 65536.0f, fmaf((float)part[1][n][i], 256.0f, (float)part[0][n][i]));
                const float scale = (i < 2 ? scale_a : scale_b) * token_scale[n * 8 + 2 * t + (i & 1)];
                acc[n][i] = fmaf(scale, joined, acc[n][i]);
            }
        __syncthreads();
    }

    for (int n = 0; n < IMMA_TOKENS / 8; ++n)
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = row0 + warp * 16 + g + half * 8;
            if (row >= rows) continue;
            const unsigned int token = token0 + n * 8 + 2 * t;
            if (token < tokens) y[(size_t)token * rows + row] = acc[n][half * 2];
            if (token + 1 < tokens) y[(size_t)(token + 1) * rows + row] = acc[n][half * 2 + 1];
        }
}
