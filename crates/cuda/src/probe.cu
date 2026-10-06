// Kernels that measure what this GPU can do, with no model in the way. Each one does work whose result is known,
// so the host can check it and a wrong answer cannot pass for a fast one.

extern "C" __global__ void fill(float* data, unsigned int n, float value) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        data[i] = value;
    }
}

// Streams `n4` float4 values from memory and adds them up: all reads, almost no arithmetic. Four independent loads
// per thread per step keep enough requests in flight to saturate the memory system.
extern "C" __global__ void read_sum(const float4* data, unsigned int n4, double* sink) {
    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (; i + 3u * stride < n4; i += 4u * stride) {
        float4 p = data[i];
        float4 q = data[i + stride];
        float4 r = data[i + 2u * stride];
        float4 s = data[i + 3u * stride];
        a0 += (p.x + p.y) + (p.z + p.w);
        a1 += (q.x + q.y) + (q.z + q.w);
        a2 += (r.x + r.y) + (r.z + r.w);
        a3 += (s.x + s.y) + (s.z + s.w);
    }
    for (; i < n4; i += stride) {
        float4 v = data[i];
        a0 += (v.x + v.y) + (v.z + v.w);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(sink, (double)acc);
    }
}

// Eight independent chains of fused multiply-adds per thread: all arithmetic, no memory.
extern "C" __global__ void fma_f32(float* sink, unsigned int iters, float a, float b) {
    float x0 = threadIdx.x * 1.0e-3f + 0.1f;
    float x1 = x0 + 0.01f, x2 = x0 + 0.02f, x3 = x0 + 0.03f;
    float x4 = x0 + 0.04f, x5 = x0 + 0.05f, x6 = x0 + 0.06f, x7 = x0 + 0.07f;
    for (unsigned int i = 0; i < iters; ++i) {
        x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
        x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
    }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
}

// Back-to-back tensor-core multiply-accumulates on bf16 ones: every accumulator grows by 16 per instruction.
extern "C" __global__ void mma_bf16(float* sink, unsigned int iters) {
    const unsigned int one_pair = 0x3F803F80u;  // two bf16 values of 1.0
    float c[4][4];
    for (int j = 0; j < 4; ++j) { c[j][0] = 0.0f; c[j][1] = 0.0f; c[j][2] = 0.0f; c[j][3] = 0.0f; }
    for (unsigned int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(one_pair), "r"(one_pair), "r"(one_pair), "r"(one_pair), "r"(one_pair), "r"(one_pair));
        }
    }
    float total = 0.0f;
    for (int j = 0; j < 4; ++j) { total += (c[j][0] + c[j][1]) + (c[j][2] + c[j][3]); }
    sink[blockIdx.x * blockDim.x + threadIdx.x] = total;
}
