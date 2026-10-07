// How keys and values sit in the paged cache. A row is the keys (or values) of one position for every kv head. Rows are
// addressed in bytes: `pages[position >> PAGE_SHIFT]` is the address of the page holding `position` inside its slab of
// SLAB_PAGES pages, a layer's keys start `key_offset` bytes on from it and its values SLAB_PAGES * PAGE_TOKENS *
// row_bytes bytes after them. PAGE_TOKENS, SLAB_PAGES and the formats must match pages.rs and kv_format.rs.
#define PAGE_SHIFT 6
#define PAGE_TOKENS (1u << PAGE_SHIFT)
#define SLAB_PAGES 8
#define KV_F32 0
#define KV_F16 1

__device__ __forceinline__ unsigned char* kv_row(const unsigned long long* pages, size_t offset, unsigned int position,
                                                 unsigned int row_bytes) {
    return reinterpret_cast<unsigned char*>(pages[position >> PAGE_SHIFT]) + offset +
           (size_t)(position & (PAGE_TOKENS - 1u)) * row_bytes;
}

__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

// Values beyond the f16 range saturate to its largest number instead of becoming infinite.
__device__ __forceinline__ unsigned short f32_to_f16(float value) {
    unsigned short bits;
    asm("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

// One value at `index` = kv_head * head_dim + d.
template <int F>
__device__ __forceinline__ float kv_at(const unsigned char* row, unsigned int index) {
    if (F == KV_F32) return reinterpret_cast<const float*>(row)[index];
    return f16_to_f32(reinterpret_cast<const unsigned short*>(row)[index]);
}

// Four consecutive values from `index`, a multiple of four.
template <int F>
__device__ __forceinline__ float4 kv_at4(const unsigned char* row, unsigned int index) {
    if (F == KV_F32) return *reinterpret_cast<const float4*>(row + (size_t)index * 4);
    const uint2 u = *reinterpret_cast<const uint2*>(row + (size_t)index * 2);
    return make_float4(f16_to_f32(u.x & 0xffffu), f16_to_f32(u.x >> 16), f16_to_f32(u.y & 0xffffu),
                       f16_to_f32(u.y >> 16));
}

// Stores the `tokens` new positions from `start` for one layer: grid (tokens, kv_heads), blockDim.x a multiple of 32 and
// at least head_dim. `keys` and `values` are f32 rows of kv_heads * head_dim.
template <int F>
__device__ void kv_store_body(const float* keys, const float* values, const unsigned long long* pages,
                              unsigned int key_offset, unsigned int row_bytes, unsigned int start,
                              unsigned int kv_heads, unsigned int head_dim) {
    const unsigned int token = blockIdx.x, kv_head = blockIdx.y, d = threadIdx.x;
    const unsigned int kv_stride = kv_heads * head_dim;
    const unsigned int index = kv_head * head_dim + d;
    const size_t value_offset = (size_t)key_offset + (size_t)SLAB_PAGES * PAGE_TOKENS * row_bytes;
    for (int which = 0; which < 2; ++which) {
        const float* source = which == 0 ? keys : values;
        const float x = d < head_dim ? source[(size_t)token * kv_stride + index] : 0.0f;
        unsigned char* row = kv_row(pages, which == 0 ? key_offset : value_offset, start + token, row_bytes);
        if (d < head_dim) {
            if (F == KV_F32) reinterpret_cast<float*>(row)[index] = x;
            if (F == KV_F16) reinterpret_cast<unsigned short*>(row)[index] = f32_to_f16(x);
        }
    }
}

#define KV_STORE_ARGS                                                                                        \
    const float *keys, const float *values, const unsigned long long *pages, unsigned int key_offset,         \
        unsigned int row_bytes, unsigned int start, unsigned int kv_heads, unsigned int head_dim
#define KV_STORE_CALL keys, values, pages, key_offset, row_bytes, start, kv_heads, head_dim
extern "C" __global__ void kv_store_f32(KV_STORE_ARGS) { kv_store_body<KV_F32>(KV_STORE_CALL); }
extern "C" __global__ void kv_store_f16(KV_STORE_ARGS) { kv_store_body<KV_F16>(KV_STORE_CALL); }
