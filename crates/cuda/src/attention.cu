// Attention kernels: the decode kernel for one token, the few-token kernel for checking guesses, and the merge.
// Helpers (`block_max`, `INF_F`) come from kernels.cu, which is compiled first.

// `pages[position >> PAGE_SHIFT]` is the address of the page holding `position` inside its slab of SLAB_PAGES pages.
// A layer's keys start `key_offset` floats on from it and its values SLAB_PAGES * PAGE_TOKENS * kv_stride floats after
// them. PAGE_TOKENS and SLAB_PAGES must match pages.rs.
#define PAGE_SHIFT 6
#define PAGE_TOKENS (1u << PAGE_SHIFT)
#define SLAB_PAGES 8
__device__ __forceinline__ const float* kv_row(const unsigned long long* pages, size_t offset, unsigned int position,
                                               unsigned int kv_stride) {
    return reinterpret_cast<const float*>(pages[position >> PAGE_SHIFT]) + offset +
           (size_t)(position & (PAGE_TOKENS - 1u)) * kv_stride;
}

// Causal grouped-query attention with an online softmax. One block per (token, head);
// blockDim.x must be >= head_dim and a multiple of 32. Shared memory: head_dim + blockDim.x + 32 floats.
extern "C" __global__ void attention(const float* q, const unsigned long long* pages, unsigned int key_offset,
                                     float* out, unsigned int heads, unsigned int kv_heads, unsigned int head_dim,
                                     unsigned int start, float scale) {
    extern __shared__ float shared[];
    float* q_s = shared;
    float* scores = shared + head_dim;
    float* reduce = scores + blockDim.x;

    unsigned int token = blockIdx.x / heads, head = blockIdx.x % heads;
    unsigned int kv_head = head / (heads / kv_heads);
    unsigned int kv_stride = kv_heads * head_dim;
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * kv_stride;
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
            const float* k = kv_row(pages, key_offset, p, kv_stride) + (size_t)kv_head * head_dim;
            float dot = 0.0f;
            if (vector) {
                for (unsigned int d = 0; d < head_dim; d += 4) {
                    const float4 kv = *reinterpret_cast<const float4*>(k + d);
                    const float4 qv = *reinterpret_cast<const float4*>(q_s + d);
                    dot += qv.x * kv.x;
                    dot += qv.y * kv.y;
                    dot += qv.z * kv.z;
                    dot += qv.w * kv.w;
                }
            } else {
                for (unsigned int d = 0; d < head_dim; ++d) dot += q_s[d] * k[d];
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
            const float* rows = tid < head_dim ? kv_row(pages, value_offset, base + j, kv_stride) + column : nullptr;
            for (; j + 4 <= end; j += 4) {
                const float w0 = scores[j], w1 = scores[j + 1], w2 = scores[j + 2], w3 = scores[j + 3];
                running_sum += w0;
                running_sum += w1;
                running_sum += w2;
                running_sum += w3;
                if (tid < head_dim) {
                    const float* r = rows + (size_t)(j - first) * kv_stride;
                    const float v0 = r[0], v1 = r[kv_stride];
                    const float v2 = r[2 * (size_t)kv_stride], v3 = r[3 * (size_t)kv_stride];
                    acc += w0 * v0;
                    acc += w1 * v1;
                    acc += w2 * v2;
                    acc += w3 * v3;
                }
            }
            for (; j < end; ++j) {
                const float weight = scores[j];
                running_sum += weight;
                if (tid < head_dim) acc += weight * rows[(size_t)(j - first) * kv_stride];
            }
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
// Shared memory: group * (head_dim + chunk) floats. group = heads / kv_heads must be at most MAX_GROUP and head_dim
// at most 32 * HEAD_STEPS.
#define MAX_GROUP 8
#define HEAD_STEPS 8
#define SCORE_BATCH 4
extern "C" __global__ void attention_partial(const float* q, const unsigned long long* pages, unsigned int key_offset,
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
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * kv_stride;
    const float* q_head = q + (size_t)kv_head * group * head_dim;

    for (unsigned int i = tid; i < group * head_dim; i += blockDim.x) q_s[i] = q_head[i];
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced. A warp takes
    // SCORE_BATCH positions at a time and loads all their rows before using any, so enough loads are in flight to
    // hide the memory latency. Each score is summed in the same order as for a single position.
    for (unsigned int base = warp; base < count; base += SCORE_BATCH * warps) {
        float key[SCORE_BATCH][HEAD_STEPS];
#pragma unroll
        for (int p = 0; p < SCORE_BATCH; ++p) {
            const unsigned int pos = base + p * warps;
            const float* k = pos < count ? kv_row(pages, key_offset, first + pos, kv_stride) + (size_t)kv_head * head_dim
                                         : nullptr;
#pragma unroll
            for (int j = 0; j < HEAD_STEPS; ++j) {
                const unsigned int d = lane + 32 * j;
                key[p][j] = (pos < count && d < head_dim) ? k[d] : 0.0f;
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
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc[MAX_GROUP];
#pragma unroll
        for (int g = 0; g < MAX_GROUP; ++g) acc[g] = 0.0f;
        unsigned int i = 0;
        for (; i + 4 <= count; i += 4) {
            const size_t column = (size_t)kv_head * head_dim + d;
            const float v0 = kv_row(pages, value_offset, first + i, kv_stride)[column];
            const float v1 = kv_row(pages, value_offset, first + i + 1, kv_stride)[column];
            const float v2 = kv_row(pages, value_offset, first + i + 2, kv_stride)[column];
            const float v3 = kv_row(pages, value_offset, first + i + 3, kv_stride)[column];
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
            const float v = kv_row(pages, value_offset, first + i, kv_stride)[(size_t)kv_head * head_dim + d];
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
__device__ void attention_multi_body(const float* q, const unsigned long long* pages, unsigned int key_offset,
                                     float* partial_acc, float* partial_ml, unsigned int heads, unsigned int kv_heads,
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
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * kv_stride;

    for (unsigned int i = tid; i < rows * head_dim; i += blockDim.x) {
        const unsigned int r = i / head_dim, d = i % head_dim;
        const unsigned int token = r / group, g = r % group;
        q_s[i] = q[((size_t)token * heads + kv_head * group + g) * head_dim + d];
    }
    __syncthreads();

    // Scores: one warp per position, lanes stride across the head so the row is read coalesced and used for every row.
    for (unsigned int i = warp; i < count; i += warps) {
        const float* k = kv_row(pages, key_offset, first + i, kv_stride) + (size_t)kv_head * head_dim;
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
            const float v = kv_row(pages, value_offset, first + i, kv_stride)[(size_t)kv_head * head_dim + d];
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
    const float *q, const unsigned long long *pages, unsigned int key_offset, float *partial_acc,                     \
        float *partial_ml, unsigned int heads, unsigned int kv_heads, unsigned int head_dim, unsigned int start,      \
        unsigned int tokens, unsigned int chunk, unsigned int splits, float scale
#define MULTI_CALL \
    q, pages, key_offset, partial_acc, partial_ml, heads, kv_heads, head_dim, start, tokens, chunk, splits, scale
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
