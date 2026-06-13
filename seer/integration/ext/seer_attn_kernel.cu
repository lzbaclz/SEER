// SEER per-block attention proxy — CUDA kernel.
//
// Computes, for each of L Attention layers in one launch:
//   K_centroid[l, b, :] = mean over (block_size, n_kv_heads) of
//                         K_cache[l][b, :, :, :]                    (shape [d])
//   score[l, b]       = <K_centroid[l, b, :], q_pooled> / sqrt(d)
//
// Inputs (all fp16):
//   k_caches: vector of L tensors, each shape
//             [n_blocks, block_size, n_kv_heads, head_dim] = [B, S, H, D]
//   q_pooled: [D] — already pooled query (amax over tokens + mean over heads)
// Output (fp16):
//   scores:  [L, B] — per-layer per-block attention magnitude proxy
//
// Single launch covers all L layers via 2D grid (L * B blocks, D threads).
// All k_caches must share (S, H, D). B can vary but we use the max across L
// and zero-fill the slack (in practice all layers share B because vLLM
// allocates a single block-table for the whole model).
//
// Phase 2.5 long-form: this replaces the Python per-layer mean() + matmul
// chain that Step 0c hit a 100+ ms P999 ceiling on. One fused kernel
// brings the per-engine.step compute overhead from ~3 ms to <100 us.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <vector>

namespace {

constexpr int kMaxLayers   = 64;   // generous; Llama-2-7B has 32
constexpr int kMaxHeadDim  = 256;  // Llama: 128

// One CUDA block per (layer, kv_block). Each thread handles one head_dim
// position. Reduction over (block_size * n_kv_heads) is done by all
// threads cooperatively via shared memory.
//
// Block dim: (D,) where D = head_dim
// Grid dim:  (B, L)
__global__ void seer_attn_proxy_kernel(
    const __half* __restrict__ k_ptrs_packed,  // [L] pointers cast to __half*
    const uint64_t* __restrict__ k_ptrs,        // [L] device pointers
    const int* __restrict__ n_blocks_per_layer, // [L]
    int block_size,
    int n_kv_heads,
    int head_dim,
    const __half* __restrict__ q_pooled,        // [D]
    float inv_sqrt_d,
    __half* __restrict__ scores,                // [L, max_B]
    int max_blocks
) {
    int b = blockIdx.x;          // block id within layer
    int l = blockIdx.y;          // layer id
    int d = threadIdx.x;         // head_dim position

    if (b >= n_blocks_per_layer[l] || d >= head_dim) {
        return;
    }

    // Pointer to this layer's k cache.
    const __half* k_layer = reinterpret_cast<const __half*>(k_ptrs[l]);

    // K_centroid[l, b, d] = mean over (s, h) of k_layer[b, s, h, d].
    // Memory layout assumed: [B, S, H, D] contiguous.
    // Index of k_layer[b, s, h, d] = ((b * S + s) * H + h) * D + d
    float acc = 0.f;
    int n_avg = block_size * n_kv_heads;
    for (int s = 0; s < block_size; ++s) {
        for (int h = 0; h < n_kv_heads; ++h) {
            int idx = ((b * block_size + s) * n_kv_heads + h) * head_dim + d;
            acc += __half2float(k_layer[idx]);
        }
    }
    float centroid_d = acc / static_cast<float>(n_avg);

    // Multiply by q_pooled[d] in fp32, accumulate via warp reduction.
    float prod = centroid_d * __half2float(q_pooled[d]);

    // Warp/block reduction to sum across d.
    __shared__ float smem[256];
    smem[d] = prod;
    __syncthreads();
    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            smem[d] += smem[d + stride];
        }
        __syncthreads();
    }
    if (d == 0) {
        scores[l * max_blocks + b] = __float2half(smem[0] * inv_sqrt_d);
    }
}

}  // namespace

at::Tensor seer_attn_proxy(
    std::vector<at::Tensor> k_caches,   // L tensors [B, S, H, D] fp16 cuda
    at::Tensor q_pooled                  // [D] fp16 cuda
) {
    TORCH_CHECK(!k_caches.empty(), "k_caches empty");
    int L = static_cast<int>(k_caches.size());
    TORCH_CHECK(L <= kMaxLayers, "too many layers");
    TORCH_CHECK(q_pooled.is_cuda(), "q_pooled must be CUDA");
    TORCH_CHECK(q_pooled.scalar_type() == at::kHalf,
                "q_pooled must be fp16");

    auto k0 = k_caches[0];
    TORCH_CHECK(k0.is_cuda() && k0.scalar_type() == at::kHalf,
                "k_caches[0] must be fp16 CUDA");
    TORCH_CHECK(k0.dim() == 4, "k_caches[0] must be 4-D");
    int block_size = static_cast<int>(k0.size(1));
    int n_kv_heads = static_cast<int>(k0.size(2));
    int head_dim   = static_cast<int>(k0.size(3));
    TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim too large");
    TORCH_CHECK(q_pooled.numel() == head_dim,
                "q_pooled head_dim mismatch");

    // Build per-layer pointer + n_blocks tables on the device.
    std::vector<uint64_t> ptrs(L);
    std::vector<int>      n_blocks(L);
    int max_blocks = 0;
    for (int i = 0; i < L; ++i) {
        auto kc = k_caches[i];
        TORCH_CHECK(kc.is_cuda() && kc.scalar_type() == at::kHalf,
                    "k_caches[", i, "] must be fp16 CUDA");
        TORCH_CHECK(kc.size(1) == block_size && kc.size(2) == n_kv_heads
                    && kc.size(3) == head_dim,
                    "k_caches[", i, "] shape mismatch");
        TORCH_CHECK(kc.is_contiguous(), "k_caches[", i, "] not contiguous");
        ptrs[i] = reinterpret_cast<uint64_t>(kc.data_ptr());
        n_blocks[i] = static_cast<int>(kc.size(0));
        if (n_blocks[i] > max_blocks) max_blocks = n_blocks[i];
    }

    auto opts_u64 = at::TensorOptions().dtype(at::kLong)
                                        .device(q_pooled.device());
    auto opts_i32 = at::TensorOptions().dtype(at::kInt)
                                        .device(q_pooled.device());
    auto opts_f16 = at::TensorOptions().dtype(at::kHalf)
                                        .device(q_pooled.device());

    // Copy ptr table + n_blocks to device. Small (L=32) so cheap.
    auto ptrs_t = torch::from_blob(
        ptrs.data(),
        {static_cast<long>(L)},
        at::TensorOptions().dtype(at::kLong)
    ).to(opts_u64);
    auto nbk_t = torch::from_blob(
        n_blocks.data(),
        {static_cast<long>(L)},
        at::TensorOptions().dtype(at::kInt)
    ).to(opts_i32);

    auto scores = at::zeros({L, max_blocks}, opts_f16);

    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    dim3 grid(max_blocks, L);
    dim3 block(head_dim);  // <= 256

    auto stream = at::cuda::getCurrentCUDAStream();
    seer_attn_proxy_kernel<<<grid, block, 0, stream>>>(
        /*k_ptrs_packed*/ nullptr,
        reinterpret_cast<const uint64_t*>(ptrs_t.data_ptr<int64_t>()),
        nbk_t.data_ptr<int>(),
        block_size,
        n_kv_heads,
        head_dim,
        reinterpret_cast<const __half*>(q_pooled.data_ptr<at::Half>()),
        inv_sqrt_d,
        reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
        max_blocks
    );

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("seer_attn_proxy", &seer_attn_proxy,
          "SEER per-block attention proxy: all layers in one launch");
}
