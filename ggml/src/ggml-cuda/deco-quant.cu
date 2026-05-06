/*
 * DecoQuant fused dequant × GeMM CUDA kernel (arXiv:2405.12591).
 *
 * Reconstructs K from its MPO factors T_L and T_S:
 *
 *   K[d, t] = Σ_{r=0}^{R-1}  TL[r, t]  ×  TS[r, d]
 *
 * Memory layout (ggml, contiguous):
 *   TL : FP16 [R=8, kv_size]   → flat: tl_mem[r + t*R]  (R is the fast dim)
 *   TS : FP16 [R=8, D]         → flat: ts_mem[r + d*R]
 *   K  : FP16 [D,   kv_size]   → flat: k_mem [d + t*D]
 *
 * The inner dimension R = DECO_INNER_RANK = 8 is compile-time constant,
 * so the inner loop is fully unrolled and expressed as 4 half2 FMAs.
 *
 * Access pattern:
 *   - For a fixed t, ALL d-threads share the same TL[0..7, t] (16 bytes).
 *     → broadcast read from L2, very cheap.
 *   - For a fixed d, TS[0..7, d] is at ts_mem + d*8 (16 contiguous bytes).
 *     → consecutive threads load consecutive 16-byte aligned chunks.
 *     → coalesced read.
 *
 * This replaces the CPU triple-loop write-back from Stage 2 with a
 * fast GPU reconstruction, matching the spirit of the official authors'
 * fused dequantisation + GeMM kernel from the DecoQuant paper.
 */

#include "deco-quant.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Compile-time constant: DECO_INNER_RANK = 8.
// Must match the value in ggml-common.h.
static constexpr int DECO_R = 8;

// Each CUDA thread reconstructs one element K[d, t].
// blockIdx.y = token index t  (one block row per token)
// blockIdx.x * blockDim.x + threadIdx.x = embedding index d
static __global__ void deco_recon_kernel(
    const half * __restrict__ tl,   // [R, kv_size]
    const half * __restrict__ ts,   // [R, D]
    half       * __restrict__ k,    // [D, kv_size]  — written in place
    int T, int D
) {
    const int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int t = (int)blockIdx.y;
    if (d >= D || t >= T) return;

    // Load TL[0..7, t] as 4 × half2 (16 contiguous bytes at tl + t*8).
    // All threads in this row share the same 16 bytes → L2 broadcast.
    const half2 * tl_row = reinterpret_cast<const half2 *>(tl + t * DECO_R);
    const half2 tl0 = tl_row[0];
    const half2 tl1 = tl_row[1];
    const half2 tl2 = tl_row[2];
    const half2 tl3 = tl_row[3];

    // Load TS[0..7, d] as 4 × half2 (16 contiguous bytes at ts + d*8).
    // Consecutive d threads load consecutive 16-byte chunks → coalesced.
    const half2 * ts_col = reinterpret_cast<const half2 *>(ts + d * DECO_R);
    const half2 ts0 = ts_col[0];
    const half2 ts1 = ts_col[1];
    const half2 ts2 = ts_col[2];
    const half2 ts3 = ts_col[3];

    // Dot product: 4 half2 FMAs → 8 scalar FMAs.
    half2 acc = __hmul2(tl0, ts0);
    acc = __hfma2(tl1, ts1, acc);
    acc = __hfma2(tl2, ts2, acc);
    acc = __hfma2(tl3, ts3, acc);

    // Horizontal add of the two half2 lanes.
    k[d + t * D] = __hadd(acc.x, acc.y);
}

void ggml_deco_reconstruct_cuda(
    const void * tl_gpu,
    const void * ts_gpu,
    void       * k_gpu,
    int T, int D, int /*kv_size*/)
{
    // Grid: ceil(D/256) blocks along X, T blocks along Y.
    const dim3 threads(256);
    const dim3 blocks((D + 255) / 256, T);

    deco_recon_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half *>(tl_gpu),
        reinterpret_cast<const half *>(ts_gpu),
        reinterpret_cast<half *>(k_gpu),
        T, D);

    // Synchronise so the reconstructed K is visible to subsequent ggml ops.
    cudaDeviceSynchronize();
}