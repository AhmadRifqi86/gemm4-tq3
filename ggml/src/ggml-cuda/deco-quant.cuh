#pragma once

// DecoQuant fused dequant × GeMM kernel (arXiv:2405.12591).
//
// Reconstructs K[D, T] = TS[R, D]^T × TL[R, T]  on the GPU.
//
// TL and TS are FP16 tensors already resident on the CUDA device.
// This replaces the CPU triple-loop write-back that Stage 2 used.
//
// Exposed as plain C++ (void *) so llama-kv-cache.cpp can include
// this header without the CUDA compiler.

// Launch the reconstruction kernel.
//   tl_gpu   : device pointer, FP16 [R=8, kv_size]
//   ts_gpu   : device pointer, FP16 [R=8, D]
//   k_gpu    : device pointer, FP16 [D, kv_size]  — written in place
//   T        : number of token positions to reconstruct (≤ kv_size)
//   D        : n_embd_k_gqa (embedding dimension)
//   kv_size  : total cache slots (stride for tl along the T dimension)
void ggml_deco_reconstruct_cuda(
    const void * tl_gpu,
    const void * ts_gpu,
    void       * k_gpu,
    int T, int D, int kv_size);