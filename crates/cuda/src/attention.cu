// Attention kernels: the decode kernel for one token, the few-token kernel for checking guesses, and the merge.
// Helpers (`block_max`, `INF_F`) come from kernels.cu, which is compiled first.

// Causal grouped-query attention with an online softmax. One block per (token, head);
// blockDim.x must be >= head_dim and a multiple of 32. Shared memory: head_dim + blockDim.x + 32 floats.
template <int F>
__device__ void attention_body(const float* q, const unsigned long long* pages, unsigned int key_offset,
                               unsigned int row_bytes, float* out, unsigned int heads, unsigned int kv_heads,
                               unsigned int head_dim, unsigned int start, float scale) {
    extern __shared__ float shared[];
    float* q_s = shared;
    float* scores = shared + head_dim;
    float* reduce = scores + blockDim.x;

    unsigned int token = blockIdx.x / heads, head = blockIdx.x % heads;
    unsigned int kv_head = head / (heads / kv_heads);
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * row_bytes;
    unsigned int visible = start + token + 1;
    unsigned int tid = threadIdx.x;

    if (tid < head_dim) q_s[tid] = q[(size_t)blockIdx.x * head_dim + tid];
    __syncthreads();

    // Each thread scores one position, reading its key row four values per load (rows start on 16 bytes when
    // head_dim is a multiple of 4). Sums keep the order of a one-value-at-a-time loop.
    const bool vector = (head_dim & 3u) == 0;
    float running_max = -INF_F, running_sum = 0.0f, acc = 0.0f;
    for (unsigned int base = 0; base < visible; base += blockDim.x) {
        unsigned int p = base + tid;
        float s = -INF_F;
        if (p < visible) {
            const unsigned char* k = kv_row(pages, key_offset, p, row_bytes);
            float dot = 0.0f;
            if (vector) {
                for (unsigned int d = 0; d < head_dim; d += 4) {
                    const float4 kv = kv_at4<F>(k, kv_head * head_dim + d);
                    const float4 qv = *reinterpret_cast<const float4*>(q_s + d);
                    dot += qv.x * kv.x;
                    dot += qv.y * kv.y;
                    dot += qv.z * kv.z;
                    dot += qv.w * kv.w;
                }
            } else {
                for (unsigned int d = 0; d < head_dim; ++d)
                    dot += q_s[d] * kv_at<F>(k, kv_head * head_dim + d);
            }
            s = dot * scale;
        }
        scores[tid] = s;
        float chunk_max = block_max(s, reduce);
        float new_max = fmaxf(running_max, chunk_max);
        float correction = expf(running_max - new_max);
        acc *= correction;
        running_sum *= correction;
        unsigned int count = min(blockDim.x, visible - base);
        // Each position's weight is computed once, by the thread that scored it, then read by all.
        scores[tid] = expf(s - new_max);
        __syncthreads();
        const size_t column = (size_t)kv_head * head_dim + tid;
        unsigned int j = 0;
        // One page lookup per page of rows. Four value rows are loaded before they are used; the sums still add the
        // rows in order.
        while (j < count) {
            const unsigned int first = j;
            const unsigned int end = min(count, j + PAGE_TOKENS - ((base + j) & (PAGE_TOKENS - 1u)));
            const unsigned char* rows = tid < head_dim ? kv_row(pages, value_offset, base + j, row_bytes) : nullptr;
            for (; j + 4 <= end; j += 4) {
                const float w0 = scores[j], w1 = scores[j + 1], w2 = scores[j + 2], w3 = scores[j + 3];
                running_sum += w0;
                running_sum += w1;
                running_sum += w2;
                running_sum += w3;
                if (tid < head_dim) {
                    const unsigned char* r = rows + (size_t)(j - first) * row_bytes;
                    const float v0 = kv_at<F>(r, column);
                    const float v1 = kv_at<F>(r + row_bytes, column);
                    const float v2 = kv_at<F>(r + 2 * (size_t)row_bytes, column);
                    const float v3 = kv_at<F>(r + 3 * (size_t)row_bytes, column);
                    acc += w0 * v0;
                    acc += w1 * v1;
                    acc += w2 * v2;
                    acc += w3 * v3;
                }
            }
            for (; j < end; ++j) {
                const float weight = scores[j];
                running_sum += weight;
                if (tid < head_dim)
                    acc += weight * kv_at<F>(rows + (size_t)(j - first) * row_bytes, column);
            }
        }
        running_max = new_max;
        __syncthreads();
    }
    if (tid < head_dim) out[(size_t)blockIdx.x * head_dim + tid] = acc / running_sum;
}

#define ATTENTION_ARGS                                                                                            \
    const float *q, const unsigned long long *pages, unsigned int key_offset, unsigned int row_bytes, float *out, \
        unsigned int heads, unsigned int kv_heads, unsigned int head_dim, unsigned int start, float scale
#define ATTENTION_CALL q, pages, key_offset, row_bytes, out, heads, kv_heads, head_dim, start, scale

// Decode attention for ONE query token, split over the context so every multiprocessor has work.
// Grid (splits, kv_heads), 128 threads. A block takes `chunk` consecutive positions for one kv head and for every query
// head that shares it, so each key and value row is read once. It writes that chunk's partial softmax for each query
// head: the maximum score, the sum of exponentials, and the exponential-weighted sum of values.
// Shared memory: group * (head_dim + chunk) floats. group = heads / kv_heads must be at most MAX_GROUP and head_dim
// at most 32 * HEAD_STEPS.
#define MAX_GROUP 8
#define HEAD_STEPS 8
#define SCORE_BATCH 4
// Decode attention over an f16 cache with head size 128, loading 16 bytes per lane: a half-warp (16 lanes) covers one
// 256-byte row, so one load instruction moves two rows where the general path moves a quarter of one. The sums add in
// a different order from the general path.
#define VEC_LANES 16
#define VEC_STREAMS 8  // half-warps in a block of 128 threads

__device__ __forceinline__ void unpack_f16x8(const uint4& v, float* f) {
    f[0] = f16_to_f32(v.x & 0xffffu); f[1] = f16_to_f32(v.x >> 16);
    f[2] = f16_to_f32(v.y & 0xffffu); f[3] = f16_to_f32(v.y >> 16);
    f[4] = f16_to_f32(v.z & 0xffffu); f[5] = f16_to_f32(v.z >> 16);
    f[6] = f16_to_f32(v.w & 0xffffu); f[7] = f16_to_f32(v.w >> 16);
}

// Scores of the chunk's positions for every query head of the group into `weights[g * chunk + position]`.
__device__ void scores_f16_vec(const float* q_s, float* weights, const unsigned long long* pages,
                               unsigned int key_offset, unsigned int row_bytes, unsigned int kv_head,
                               unsigned int first, unsigned int count, unsigned int chunk, unsigned int group,
                               float scale) {
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned int half = lane >> 4, slot = lane & (VEC_LANES - 1);
    for (unsigned int base = warp * 2 * SCORE_BATCH; base < count; base += warps * 2 * SCORE_BATCH) {
        uint4 keys[SCORE_BATCH];
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p) {
            const unsigned int pos = base + p * 2 + half;
            keys[p] = make_uint4(0, 0, 0, 0);
            if (pos < count)
                keys[p] = *reinterpret_cast<const uint4*>(kv_row(pages, key_offset, first + pos, row_bytes) +
                                                          (size_t)kv_head * 256 + slot * 16);
        }
        float dots[SCORE_BATCH][MAX_GROUP];
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p) {
            float k[8];
            unpack_f16x8(keys[p], k);
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g) {
                dots[p][g] = 0.0f;
                if (g < group) {
                    const float4 a = *reinterpret_cast<const float4*>(q_s + g * 128 + slot * 8);
                    const float4 b = *reinterpret_cast<const float4*>(q_s + g * 128 + slot * 8 + 4);
                    dots[p][g] = a.x * k[0] + a.y * k[1] + a.z * k[2] + a.w * k[3] + b.x * k[4] + b.y * k[5] +
                                 b.z * k[6] + b.w * k[7];
                }
            }
        }
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p)
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g) {
                if (g >= group) continue;
                float sum = dots[p][g];
                for (int offset = VEC_LANES / 2; offset > 0; offset >>= 1)
                    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                const unsigned int pos = base + p * 2 + half;
                if (slot == 0 && pos < count) weights[g * chunk + pos] = sum * scale;
            }
    }
}

// Weighted sum of the chunk's value rows for every query head. Each half-warp sums every 8th position; `partial`
// ([stream][group][128] floats of shared memory) holds the 8 sums, which are added in a fixed order.
__device__ void values_f16_vec(const float* weights, float* partial, float* partial_acc,
                               const unsigned long long* pages, size_t value_offset, unsigned int row_bytes,
                               unsigned int kv_head, unsigned int first, unsigned int count, unsigned int chunk,
                               unsigned int group, size_t out_base) {
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int half = lane >> 4, slot = lane & (VEC_LANES - 1), stream = warp * 2 + half;
    float acc[MAX_GROUP][8];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g][j] = 0.0f;
    for (unsigned int i = stream; i < count; i += VEC_STREAMS * 4) {
        uint4 rows[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned int pos = i + u * VEC_STREAMS;
            rows[u] = make_uint4(0, 0, 0, 0);
            if (pos < count)
                rows[u] = *reinterpret_cast<const uint4*>(kv_row(pages, value_offset, first + pos, row_bytes) +
                                                          (size_t)kv_head * 256 + slot * 16);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned int pos = i + u * VEC_STREAMS;
            if (pos >= count) continue;
            float f[8];
            unpack_f16x8(rows[u], f);
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g)
                if (g < group) {
                    const float w = weights[g * chunk + pos];
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[g][j] += w * f[j];
                }
        }
    }
#pragma unroll
    for (int g = 0; g < MAX_GROUP; ++g)
        if (g < group)
#pragma unroll
            for (int j = 0; j < 8; ++j) partial[(stream * group + g) * 128 + slot * 8 + j] = acc[g][j];
    __syncthreads();
    for (unsigned int d = threadIdx.x; d < 128; d += blockDim.x)
        for (unsigned int g = 0; g < group; ++g) {
            float sum = 0.0f;
            for (unsigned int s = 0; s < VEC_STREAMS; ++s) sum += partial[(s * group + g) * 128 + d];
            partial_acc[out_base + (size_t)g * gridDim.x * 128 + d] = sum;
        }
}

template <int F>
__device__ void attention_partial_body(const float* q, const unsigned long long* pages, unsigned int key_offset,
                                       unsigned int row_bytes, float* partial_acc, float* partial_ml,
                                       unsigned int heads, unsigned int kv_heads, unsigned int head_dim,
                                       unsigned int visible, unsigned int chunk, unsigned int splits, float scale,
                                       unsigned int vector) {
    extern __shared__ float shared[];
    const unsigned int group = heads / kv_heads;
    float* q_s = shared;                      // [group][head_dim]
    float* weights = shared + group * head_dim;  // [group][chunk]
    const unsigned int split = blockIdx.x, kv_head = blockIdx.y;
    const unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
    const unsigned int first = split * chunk;
    const unsigned int count = min(chunk, visible - first);
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * row_bytes;
    const float* q_head = q + (size_t)kv_head * group * head_dim;

    for (unsigned int i = tid; i < group * head_dim; i += blockDim.x) q_s[i] = q_head[i];
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced. A warp takes
    // SCORE_BATCH positions at a time and loads all their rows before using any, so enough loads are in flight to
    // hide the memory latency. Each score is summed in the same order as for a single position.
    const bool wide = F == KV_F16 && vector != 0;
    if (wide)
        scores_f16_vec(q_s, weights, pages, key_offset, row_bytes, kv_head, first, count, chunk, group, scale);
    for (unsigned int base = warp; !wide && base < count; base += SCORE_BATCH * warps) {
        float key[SCORE_BATCH][HEAD_STEPS];
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p) {
            const unsigned int pos = base + p * warps;
            const unsigned char* k = pos < count ? kv_row(pages, key_offset, first + pos, row_bytes) : nullptr;
#pragma unroll
            for (int j = 0; j < HEAD_STEPS; ++j) {
                const unsigned int d = lane + 32 * j;
                key[p][j] = (pos < count && d < head_dim) ? kv_at<F>(k, kv_head * head_dim + d) : 0.0f;
            }
        }
        float dots[SCORE_BATCH][MAX_GROUP];
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p)
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g) dots[p][g] = 0.0f;
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p)
#pragma unroll
            for (int j = 0; j < HEAD_STEPS; ++j) {
                const unsigned int d = lane + 32 * j;
                if (d < head_dim) {
#pragma unroll
                    for (int g = 0; g < MAX_GROUP; ++g)
                        if (g < group) dots[p][g] += q_s[g * head_dim + d] * key[p][j];
                }
            }
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p) {
            const unsigned int pos = base + p * warps;
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g) {
                if (g >= group) continue;
                float sum = dots[p][g];
                for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
                if (lane == 0 && pos < count) weights[g * chunk + pos] = sum * scale;
            }
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

    // Weighted values: threads stride across the head, each row read coalesced and used for every query head. Four
    // rows are loaded before they are used; the sums still add the rows in order.
    if (wide) {
        values_f16_vec(weights, weights + group * chunk, partial_acc, pages, value_offset, row_bytes, kv_head, first,
                       count, chunk, group, ((size_t)(kv_head * group) * splits + split) * head_dim);
        return;
    }
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc[MAX_GROUP];
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) acc[g] = 0.0f;
        unsigned int i = 0;
        for (; i + 4 <= count; i += 4) {
            const size_t column = (size_t)kv_head * head_dim + d;
            const float v0 = kv_at<F>(kv_row(pages, value_offset, first + i, row_bytes), column);
            const float v1 = kv_at<F>(kv_row(pages, value_offset, first + i + 1, row_bytes), column);
            const float v2 = kv_at<F>(kv_row(pages, value_offset, first + i + 2, row_bytes), column);
            const float v3 = kv_at<F>(kv_row(pages, value_offset, first + i + 3, row_bytes), column);
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g)
                if (g < group) {
                    acc[g] += weights[g * chunk + i] * v0;
                    acc[g] += weights[g * chunk + i + 1] * v1;
                    acc[g] += weights[g * chunk + i + 2] * v2;
                    acc[g] += weights[g * chunk + i + 3] * v3;
                }
        }
        for (; i < count; ++i) {
            const float v = kv_at<F>(kv_row(pages, value_offset, first + i, row_bytes), (size_t)kv_head * head_dim + d);
#pragma unroll
            for (int g = 0; g < MAX_GROUP; ++g)
                if (g < group) acc[g] += weights[g * chunk + i] * v;
        }
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g)
            if (g < group) partial_acc[((size_t)(kv_head * group + g) * splits + split) * head_dim + d] = acc[g];
    }
}

#define PARTIAL_ARGS                                                                                                \
    const float *q, const unsigned long long *pages, unsigned int key_offset, unsigned int row_bytes,               \
        float *partial_acc, float *partial_ml, unsigned int heads, unsigned int kv_heads, unsigned int head_dim,    \
        unsigned int visible, unsigned int chunk, unsigned int splits, float scale, unsigned int vector
#define PARTIAL_CALL \
    q, pages, key_offset, row_bytes, partial_acc, partial_ml, heads, kv_heads, head_dim, visible, chunk, splits, scale, vector

// Attention for a few new tokens at once (checking guessed tokens), split over the context like the decode kernel above.
// Grid (splits, kv_heads), 128 threads. A block takes `chunk` consecutive positions for one kv head and for every
// (token, query head) row that uses it, so each key and value row is read once however many tokens are checked. Row
// `token * group + g` is causal: it sees positions up to `start + token`. It writes the chunk's partial softmax for
// each row at `(token * heads + head) * splits + split`, which `attention_merge` combines with one block per row.
// Shared memory: rows * (head_dim + chunk) floats, rows = tokens * group <= MAX_ROWS.
// The kernel is built for row capacities of 8, 16 and 32 so a pass of few rows does not pay for the loops of many.
template <int MAX_ROWS, int F>
__device__ void attention_multi_body(const float* q, const unsigned long long* pages, unsigned int key_offset,
                                     unsigned int row_bytes, float* partial_acc, float* partial_ml, unsigned int heads, unsigned int kv_heads,
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
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * row_bytes;

    for (unsigned int i = tid; i < rows * head_dim; i += blockDim.x) {
        const unsigned int r = i / head_dim, d = i % head_dim;
        const unsigned int token = r / group, g = r % group;
        q_s[i] = q[((size_t)token * heads + kv_head * group + g) * head_dim + d];
    }
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced and used for every row.
    for (unsigned int i = warp; i < count; i += warps) {
        const unsigned char* k = kv_row(pages, key_offset, first + i, row_bytes);
        float dots[MAX_ROWS];
#pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) dots[r] = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32) {
            const float kd = kv_at<F>(k, kv_head * head_dim + d);
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
            const float v = kv_at<F>(kv_row(pages, value_offset, first + i, row_bytes), kv_head * head_dim + d);
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

#define MULTI_ARGS                                                                                            \
    const float *q, const unsigned long long *pages, unsigned int key_offset, unsigned int row_bytes,         \
        float *partial_acc, float *partial_ml, unsigned int heads, unsigned int kv_heads, unsigned int head_dim, \
        unsigned int start, unsigned int tokens, unsigned int chunk, unsigned int splits, float scale
#define MULTI_CALL \
    q, pages, key_offset, row_bytes, partial_acc, partial_ml, heads, kv_heads, head_dim, start, tokens, chunk, splits, scale

// Every kernel is built once per cache format; the name ends in _f32 or _f16.
#define PER_FORMAT(name, args, call, body)                                                                   \
    extern "C" __global__ void name##_f32(args) { body<KV_F32>(call); }                                     \
    extern "C" __global__ void name##_f16(args) { body<KV_F16>(call); }
#define PER_FORMAT_ROWS(name, rows)                                                                          \
    extern "C" __global__ void name##_f32(MULTI_ARGS) { attention_multi_body<rows, KV_F32>(MULTI_CALL); }   \
    extern "C" __global__ void name##_f16(MULTI_ARGS) { attention_multi_body<rows, KV_F16>(MULTI_CALL); }
PER_FORMAT(attention, ATTENTION_ARGS, ATTENTION_CALL, attention_body)
PER_FORMAT(attention_partial, PARTIAL_ARGS, PARTIAL_CALL, attention_partial_body)
PER_FORMAT_ROWS(attention_multi_partial_8, 8)
PER_FORMAT_ROWS(attention_multi_partial_16, 16)
PER_FORMAT_ROWS(attention_multi_partial_32, 32)

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
