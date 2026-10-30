// Qwen3.5 hybrid text operations. FP32 state stays on the device.

extern "C" __global__ void q35_convolution(const float* qkv, const float* weight,
                                             float* history, float* mixed,
                                             unsigned int channels, unsigned int kernel,
                                             unsigned int tokens) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float* old = history + (size_t)channel * kernel;
    const float* w = weight + (size_t)channel * kernel;
    for (unsigned int t = 0; t < tokens; ++t) {
        for (unsigned int i = 1; i < kernel; ++i) old[i - 1] = old[i];
        old[kernel - 1] = qkv[(size_t)t * channels + channel];
        float sum = 0.0f;
        for (unsigned int i = 0; i < kernel; ++i) sum += old[i] * w[i];
        mixed[(size_t)t * channels + channel] = sum / (1.0f + expf(-sum));
    }
}

// One block per value head, one thread per value column. A thread keeps its column of the recurrent state in
// registers for the whole call and writes it back once; KD is the key dimension rounded up to 64, 128 or 256.
template <int KD>
__device__ void q35_recurrent_body(const float* mixed, const float* z,
                                   const float* a, const float* b,
                                   const float* a_log, const float* dt_bias,
                                   const float* norm, float* state, float* core,
                                   unsigned int key_heads, unsigned int value_heads,
                                   unsigned int key_dim, unsigned int value_dim,
                                   unsigned int channels, unsigned int tokens,
                                   float eps) {
    unsigned int h = blockIdx.x, v = threadIdx.x;
    unsigned int kd_total = key_heads * key_dim;
    unsigned int vd_total = value_heads * value_dim;
    unsigned int key_head = h / (value_heads / key_heads);
    float* s = state + (size_t)h * key_dim * value_dim;
    __shared__ float q[256], k[256], partial[32], scale_qk[2];
    float col[KD];
#pragma unroll
    for (unsigned int i = 0; i < KD; ++i) col[i] = (v < value_dim && i < key_dim) ? s[(size_t)i * value_dim + v] : 0.0f;
    for (unsigned int t = 0; t < tokens; ++t) {
        const float* src = mixed + (size_t)t * channels;
        for (unsigned int i = v; i < key_dim; i += blockDim.x) {
            q[i] = src[key_head * key_dim + i];
            k[i] = src[kd_total + key_head * key_dim + i];
        }
        __syncthreads();
        if (v == 0) {
            float qsum = 0.0f, ksum = 0.0f;
            for (unsigned int i = 0; i < key_dim; ++i) {
                qsum += q[i] * q[i];
                ksum += k[i] * k[i];
            }
            scale_qk[0] = rsqrtf(qsum + 1.0e-6f) * rsqrtf((float)key_dim);
            scale_qk[1] = rsqrtf(ksum + 1.0e-6f);
        }
        __syncthreads();
        for (unsigned int i = v; i < key_dim; i += blockDim.x) { q[i] *= scale_qk[0]; k[i] *= scale_qk[1]; }
        __syncthreads();
        float value = 0.0f;
        if (v < value_dim) {
            float av = a[(size_t)t * value_heads + h] + dt_bias[h];
            float soft = av > 20.0f ? av : log1pf(expf(av));
            float decay = expf(-expf(a_log[h]) * soft);
            float bv = b[(size_t)t * value_heads + h];
            float beta = 1.0f / (1.0f + expf(-bv));
            float pred = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < KD; ++i) {
                if (i < key_dim) {
                    col[i] *= decay;
                    pred += col[i] * k[i];
                }
            }
            float delta = (src[2 * kd_total + h * value_dim + v] - pred) * beta;
            float out = 0.0f;
#pragma unroll
            for (unsigned int i = 0; i < KD; ++i) {
                if (i < key_dim) {
                    col[i] += k[i] * delta;
                    out += col[i] * q[i];
                }
            }
            value = out;
        }
        float total = block_sum(value * value, partial);
        if (v < value_dim) {
            float scale = rsqrtf(total / (float)value_dim + eps);
            float gated = value * scale * norm[v];
            float zv = z[(size_t)t * vd_total + h * value_dim + v];
            core[(size_t)t * vd_total + h * value_dim + v] = gated * zv / (1.0f + expf(-zv));
        }
        __syncthreads();
    }
    if (v < value_dim) {
#pragma unroll
        for (unsigned int i = 0; i < KD; ++i)
            if (i < key_dim) s[(size_t)i * value_dim + v] = col[i];
    }
}

extern "C" __global__ void q35_recurrent(const float* mixed, const float* z,
                                           const float* a, const float* b,
                                           const float* a_log, const float* dt_bias,
                                           const float* norm, float* state, float* core,
                                           unsigned int key_heads, unsigned int value_heads,
                                           unsigned int key_dim, unsigned int value_dim,
                                           unsigned int channels, unsigned int tokens,
                                           float eps) {
#define RECURRENT(KD)                                                                                          \
    q35_recurrent_body<KD>(mixed, z, a, b, a_log, dt_bias, norm, state, core, key_heads, value_heads, key_dim, \
                           value_dim, channels, tokens, eps)
    if (key_dim <= 64) RECURRENT(64);
    else if (key_dim <= 128) RECURRENT(128);
    else RECURRENT(256);
#undef RECURRENT
}

extern "C" __global__ void q35_split_query(const float* projected, float* query,
                                              float* gate, unsigned int heads,
                                              unsigned int head_dim, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int head = i / head_dim;
    unsigned int off = i % head_dim;
    size_t src = (size_t)head * 2 * head_dim + off;
    query[i] = projected[src];
    gate[i] = projected[src + head_dim];
}

extern "C" __global__ void q35_partial_rope(float* x, unsigned int heads,
                                               unsigned int head_dim, unsigned int rotary,
                                               unsigned int start, float theta) {
    unsigned int token = blockIdx.x / heads;
    float* head = x + (size_t)blockIdx.x * head_dim;
    unsigned int half = rotary / 2;
    float position = (float)(start + token);
    for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = position * powf(theta, -(float)(2 * i) / (float)rotary);
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float a = head[i], b = head[i + half];
        head[i] = a * cs - b * sn;
        head[i + half] = b * cs + a * sn;
    }
}

extern "C" __global__ void q35_gate_attention(float* context, const float* gate,
                                                 unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) context[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
