// Tensor-core attention for prompt passes over an f16 cache (head size 128), in the style of FlashAttention-2: one block
// takes ATT_ROWS query positions of one query head, each of its 4 warps owning 16 of them. Keys and values are staged
// through shared memory ATT_KEYS positions at a time (read once per block, not once per query), scores and the weighted
// values go through mma.sync with f16 inputs and f32 accumulation, and the softmax is online, in registers. The queries
// and the softmax weights are rounded to f16 for the multiplies; the sums are f32.
// Needs kv_row and the page constants from kv_format.cu.
#define ATT_ROWS 64
#define ATT_KEYS 64
#define ATT_DIM 128
#define ATT_STRIDE 136  // shared row length in f16 values: 128 plus 8 of padding so ldmatrix rows use distinct banks

__device__ __forceinline__ void mma_f16_16816(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two f32 values as f16 pair: `low` in the low half.
__device__ __forceinline__ unsigned int pack_f16x2(float low, float high) {
    unsigned int bits;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(high), "f"(low));
    return bits;
}

// Two 8x8 matrices of 16-bit values from shared memory, each row address given by one of lanes 0..7 (first matrix) and
// 8..15 (second); thread (g, t) receives elements (g, 2t) and (g, 2t + 1) of each.
__device__ __forceinline__ void ldmatrix_x2(unsigned int* r, const unsigned short* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"((unsigned int)__cvta_generic_to_shared(row)));
}

// The same, transposed: thread (g, t) receives elements (2t, g) and (2t + 1, g).
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned int* r, const unsigned short* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"((unsigned int)__cvta_generic_to_shared(row)));
}

// Grid: (ceil(tokens / ATT_ROWS), heads), 128 threads. `q` and `out` are f32 rows of heads * ATT_DIM, one per token;
// `scale_log2` is the usual 1/sqrt(head size) times log2(e), so the softmax can use exp2.
extern "C" __global__ void attention_prefill_mma(const float* q, const unsigned long long* pages,
                                                 unsigned int key_offset, unsigned int row_bytes, float* out,
                                                 unsigned int heads, unsigned int kv_heads, unsigned int tokens,
                                                 unsigned int start, float scale_log2) {
    __shared__ __align__(16) unsigned short k_s[ATT_KEYS * ATT_STRIDE];
    __shared__ __align__(16) unsigned short v_s[ATT_KEYS * ATT_STRIDE];
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const unsigned int head = blockIdx.y, kv_head = head / (heads / kv_heads);
    // The tiles with the most keys go first.
    const unsigned int q0 = (gridDim.x - 1 - blockIdx.x) * ATT_ROWS;
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * row_bytes;
    const unsigned int row_a = q0 + warp * 16 + g, row_b = row_a + 8;

    // This thread's share of the queries, as the A operand of the 8 steps of 16 dimensions.
    unsigned int qa[ATT_DIM / 16][4];
    #pragma unroll
    for (int ks = 0; ks < ATT_DIM / 16; ++ks) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const unsigned int row = half == 0 ? row_a : row_b;
            float2 low = make_float2(0.0f, 0.0f), high = make_float2(0.0f, 0.0f);
            if (row < tokens) {
                const float* source = q + ((size_t)row * heads + head) * ATT_DIM + ks * 16 + 2 * t;
                low = *reinterpret_cast<const float2*>(source);
                high = *reinterpret_cast<const float2*>(source + 8);
            }
            qa[ks][half] = pack_f16x2(low.x, low.y);
            qa[ks][half + 2] = pack_f16x2(high.x, high.y);
        }
    }

    float o[ATT_DIM / 8][4];
    #pragma unroll
    for (int d = 0; d < ATT_DIM / 8; ++d) { o[d][0] = 0.0f; o[d][1] = 0.0f; o[d][2] = 0.0f; o[d][3] = 0.0f; }
    float max_a = -INF_F, max_b = -INF_F, sum_a = 0.0f, sum_b = 0.0f;

    const unsigned int last_row = min(q0 + ATT_ROWS, tokens);
    const unsigned int visible = start + last_row;  // positions below this are read by some row of the block
    const unsigned int warp_last = start + min(q0 + warp * 16 + 15, tokens - 1);  // newest position this warp sees

    for (unsigned int key0 = 0; key0 < visible; key0 += ATT_KEYS) {
        __syncthreads();
        for (int i = 0; i < ATT_KEYS * 16 / 128; ++i) {
            const unsigned int index = tid + i * 128, key = index >> 4, vec = index & 15, position = key0 + key;
            uint4 k = make_uint4(0, 0, 0, 0), v = make_uint4(0, 0, 0, 0);
            if (position < visible) {
                k = *reinterpret_cast<const uint4*>(kv_row(pages, key_offset, position, row_bytes) +
                                                    (size_t)kv_head * ATT_DIM * 2 + vec * 16);
                v = *reinterpret_cast<const uint4*>(kv_row(pages, value_offset, position, row_bytes) +
                                                    (size_t)kv_head * ATT_DIM * 2 + vec * 16);
            }
            *reinterpret_cast<uint4*>(k_s + key * ATT_STRIDE + vec * 8) = k;
            *reinterpret_cast<uint4*>(v_s + key * ATT_STRIDE + vec * 8) = v;
        }
        __syncthreads();
        if (key0 > warp_last) continue;

        float s[ATT_KEYS / 8][4];
        #pragma unroll
        for (int n = 0; n < ATT_KEYS / 8; ++n) { s[n][0] = 0.0f; s[n][1] = 0.0f; s[n][2] = 0.0f; s[n][3] = 0.0f; }
        #pragma unroll
        for (int ks = 0; ks < ATT_DIM / 16; ++ks) {
            #pragma unroll
            for (int n = 0; n < ATT_KEYS / 8; ++n) {
                unsigned int b[2];
                ldmatrix_x2(b, k_s + (n * 8 + (lane & 7)) * ATT_STRIDE + ks * 16 + ((lane >> 3) & 1) * 8);
                mma_f16_16816(s[n], qa[ks], b[0], b[1]);
            }
        }

        // Scale, mask the positions after each row's own, and update the running maxima.
        float block_a = -INF_F, block_b = -INF_F;
        #pragma unroll
        for (int n = 0; n < ATT_KEYS / 8; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned int position = key0 + n * 8 + 2 * t + (i & 1), row = (i < 2) ? row_a : row_b;
                float value = s[n][i] * scale_log2;
                if (position > start + row || position >= visible) value = -INF_F;
                s[n][i] = value;
                if (i < 2) block_a = fmaxf(block_a, value); else block_b = fmaxf(block_b, value);
            }
        }
        block_a = fmaxf(block_a, __shfl_xor_sync(0xffffffffu, block_a, 1));
        block_a = fmaxf(block_a, __shfl_xor_sync(0xffffffffu, block_a, 2));
        block_b = fmaxf(block_b, __shfl_xor_sync(0xffffffffu, block_b, 1));
        block_b = fmaxf(block_b, __shfl_xor_sync(0xffffffffu, block_b, 2));
        const float new_a = fmaxf(max_a, block_a), new_b = fmaxf(max_b, block_b);
        const float base_a = new_a == -INF_F ? 0.0f : new_a, base_b = new_b == -INF_F ? 0.0f : new_b;
        const float fix_a = exp2f(max_a - base_a), fix_b = exp2f(max_b - base_b);
        max_a = new_a;
        max_b = new_b;
        sum_a *= fix_a;
        sum_b *= fix_b;
        #pragma unroll
        for (int d = 0; d < ATT_DIM / 8; ++d) {
            o[d][0] *= fix_a; o[d][1] *= fix_a; o[d][2] *= fix_b; o[d][3] *= fix_b;
        }

        unsigned int pa[ATT_KEYS / 8], pb[ATT_KEYS / 8];
        #pragma unroll
        for (int n = 0; n < ATT_KEYS / 8; ++n) {
            const float p0 = exp2f(s[n][0] - base_a), p1 = exp2f(s[n][1] - base_a);
            const float p2 = exp2f(s[n][2] - base_b), p3 = exp2f(s[n][3] - base_b);
            sum_a += p0 + p1;
            sum_b += p2 + p3;
            pa[n] = pack_f16x2(p0, p1);
            pb[n] = pack_f16x2(p2, p3);
        }

        // O += P V, 16 positions at a time.
        #pragma unroll
        for (int c = 0; c < ATT_KEYS / 16; ++c) {
            const unsigned int a[4] = {pa[2 * c], pb[2 * c], pa[2 * c + 1], pb[2 * c + 1]};
            #pragma unroll
            for (int d = 0; d < ATT_DIM / 8; ++d) {
                unsigned int b[2];
                ldmatrix_x2_trans(b, v_s + (c * 16 + (lane & 15)) * ATT_STRIDE + d * 8);
                mma_f16_16816(o[d], a, b[0], b[1]);
            }
        }
    }

    sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 1);
    sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 2);
    sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 1);
    sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 2);
    const float inv_a = 1.0f / sum_a, inv_b = 1.0f / sum_b;
    #pragma unroll
    for (int d = 0; d < ATT_DIM / 8; ++d) {
        if (row_a < tokens)
            *reinterpret_cast<float2*>(out + ((size_t)row_a * heads + head) * ATT_DIM + d * 8 + 2 * t) =
                make_float2(o[d][0] * inv_a, o[d][1] * inv_a);
        if (row_b < tokens)
            *reinterpret_cast<float2*>(out + ((size_t)row_b * heads + head) * ATT_DIM + d * 8 + 2 * t) =
                make_float2(o[d][2] * inv_b, o[d][3] * inv_b);
    }
}
