/*
 * DecoQuant: data-free KV cache compression via MPO tensor decomposition
 * Paper: arXiv:2405.12591 — "Unlocking Data-free Low-bit Quantization with
 *         Matrix Decomposition for KV Cache Compression"
 *
 * Core idea:
 *   K ∈ R^(T×D)  →  T_L ∈ R^(T×R)  ×  T_S ∈ R^(R×D)
 *   T_L: large factor (~99% of params), narrow value range → quantized to B bits
 *   T_S: small factor (~1%  of params), outliers tolerated → kept FP16
 *   Reconstruction: K ≈ dequant(T_L) × T_S
 *
 * Stage 1: CPU reference — types DECO4_L (4-bit T_L) and DECO8_L (8-bit T_L)
 * Stage 2: KV cache integration (staging buffer + lazy decomposition)
 * Stage 3: CUDA fused dequant×GeMM kernel
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/* 4-bit T_L quantization (block_deco4_l)                             */
/*   Symmetric per-block absmax, 4-bit signed (-8..7), nibble packed  */
/*   scale d = absmax / 7                                             */
/* ------------------------------------------------------------------ */

void quantize_row_deco4_l_ref(const float * GGML_RESTRICT x,
                               block_deco4_l * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_DECO == 0);
    const int nb = (int)(k / QK_DECO);

    for (int b = 0; b < nb; b++) {
        const float * xb = x + b * QK_DECO;
        block_deco4_l * yb = y + b;

        /* find absmax */
        float amax = 0.0f;
        for (int i = 0; i < QK_DECO; i++) {
            const float v = fabsf(xb[i]);
            if (v > amax) amax = v;
        }

        const float d     = amax / 7.0f;
        const float id    = (d > 1e-9f) ? (1.0f / d) : 0.0f;
        yb->d = GGML_FP32_TO_FP16(d);

        /* quantize and nibble-pack: q ∈ [-8, 7], stored as q+8 ∈ [0, 15] */
        for (int i = 0; i < QK_DECO / 2; i++) {
            const int q0 = (int)(xb[2*i  ] * id + 0.5f);
            const int q1 = (int)(xb[2*i+1] * id + 0.5f);
            const uint8_t qi0 = (uint8_t)((q0 < -8 ? -8 : q0 > 7 ? 7 : q0) + 8);
            const uint8_t qi1 = (uint8_t)((q1 < -8 ? -8 : q1 > 7 ? 7 : q1) + 8);
            yb->qs[i] = (qi0) | (qi1 << 4);
        }
    }
}

void dequantize_row_deco4_l(const block_deco4_l * GGML_RESTRICT x,
                             float * GGML_RESTRICT y,
                             int64_t k) {
    assert(k % QK_DECO == 0);
    const int nb = (int)(k / QK_DECO);

    for (int b = 0; b < nb; b++) {
        const block_deco4_l * xb = x + b;
        float * yb = y + b * QK_DECO;
        const float d = GGML_FP16_TO_FP32(xb->d);

        for (int i = 0; i < QK_DECO / 2; i++) {
            const uint8_t packed = xb->qs[i];
            yb[2*i  ] = ((int)(packed & 0x0F) - 8) * d;
            yb[2*i+1] = ((int)(packed >>   4) - 8) * d;
        }
    }
}

size_t quantize_deco4_l(const float * GGML_RESTRICT src,
                         void       * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row,
                         const float * imatrix) {
    (void)imatrix; /* data-free: no imatrix needed */
    assert(n_per_row % QK_DECO == 0);
    const size_t row_size = (n_per_row / QK_DECO) * sizeof(block_deco4_l);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_deco4_l_ref(src + r * n_per_row,
                                  (block_deco4_l *)((char *)dst + r * row_size),
                                  n_per_row);
    }
    return nrows * row_size;
}

/* ------------------------------------------------------------------ */
/* 8-bit T_L quantization (block_deco8_l)                             */
/*   Symmetric per-block absmax, 8-bit signed (-127..127)             */
/*   scale d = absmax / 127                                           */
/* ------------------------------------------------------------------ */

void quantize_row_deco8_l_ref(const float * GGML_RESTRICT x,
                               block_deco8_l * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_DECO == 0);
    const int nb = (int)(k / QK_DECO);

    for (int b = 0; b < nb; b++) {
        const float * xb = x + b * QK_DECO;
        block_deco8_l * yb = y + b;

        float amax = 0.0f;
        for (int i = 0; i < QK_DECO; i++) {
            const float v = fabsf(xb[i]);
            if (v > amax) amax = v;
        }

        const float d  = amax / 127.0f;
        const float id = (d > 1e-9f) ? (1.0f / d) : 0.0f;
        yb->d = GGML_FP32_TO_FP16(d);

        for (int i = 0; i < QK_DECO; i++) {
            int q = (int)(xb[i] * id + 0.5f);
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            yb->qs[i] = (int8_t)q;
        }
    }
}

void dequantize_row_deco8_l(const block_deco8_l * GGML_RESTRICT x,
                             float * GGML_RESTRICT y,
                             int64_t k) {
    assert(k % QK_DECO == 0);
    const int nb = (int)(k / QK_DECO);

    for (int b = 0; b < nb; b++) {
        const block_deco8_l * xb = x + b;
        float * yb = y + b * QK_DECO;
        const float d = GGML_FP16_TO_FP32(xb->d);
        for (int i = 0; i < QK_DECO; i++) {
            yb[i] = xb->qs[i] * d;
        }
    }
}

size_t quantize_deco8_l(const float * GGML_RESTRICT src,
                         void       * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row,
                         const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_DECO == 0);
    const size_t row_size = (n_per_row / QK_DECO) * sizeof(block_deco8_l);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_deco8_l_ref(src + r * n_per_row,
                                  (block_deco8_l *)((char *)dst + r * row_size),
                                  n_per_row);
    }
    return nrows * row_size;
}

/* ------------------------------------------------------------------ */
/* MPO decomposition via Alternating Least Squares (ALS)              */
/*                                                                    */
/* Finds T_L[T×R] and T_S[R×D] such that T_L × T_S ≈ K[T×D]         */
/*                                                                    */
/* ALS update rules (standard low-rank matrix factorization):         */
/*   T_L = K × T_S^T × (T_S × T_S^T)^{-1}                           */
/*   T_S = (T_L^T × T_L)^{-1} × T_L^T × K                           */
/*                                                                    */
/* After convergence, T_L has a narrow value distribution (the MPO   */
/* property: outliers migrate to T_S) making it easy to quantize.    */
/* ------------------------------------------------------------------ */

/* Solve R×R linear system A*x = b via Gaussian elimination (in-place).
   A is row-major [R×R], b is [R]. Returns 0 on success, -1 if singular. */
static int solve_rr(float * A, float * b, int R) {
    for (int col = 0; col < R; col++) {
        /* find pivot */
        int pivot = col;
        for (int row = col+1; row < R; row++) {
            if (fabsf(A[row*R + col]) > fabsf(A[pivot*R + col])) pivot = row;
        }
        if (pivot != col) {
            for (int k = 0; k < R; k++) {
                float tmp = A[col*R+k]; A[col*R+k] = A[pivot*R+k]; A[pivot*R+k] = tmp;
            }
            float tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }
        if (fabsf(A[col*R + col]) < 1e-10f) return -1;
        const float inv = 1.0f / A[col*R + col];
        for (int row = col+1; row < R; row++) {
            const float factor = A[row*R + col] * inv;
            for (int k = col; k < R; k++) A[row*R+k] -= factor * A[col*R+k];
            b[row] -= factor * b[col];
        }
    }
    /* back-substitution */
    for (int row = R-1; row >= 0; row--) {
        for (int k = row+1; k < R; k++) b[row] -= A[row*R+k] * b[k];
        b[row] /= A[row*R + row];
    }
    return 0;
}

void deco_mpo_decompose(
        const float     * GGML_RESTRICT K,   /* [T * D] */
        float           * GGML_RESTRICT TL,  /* [T * R] */
        ggml_fp16_t     * GGML_RESTRICT TS,  /* [R * D] */
        int T, int D, int R, int n_iter)
{
    assert(R > 0 && R <= D && R <= T);

    /* --- workspace --- */
    float * ts_f  = (float *)malloc(R * D * sizeof(float)); /* T_S in FP32 for computation */
    float * gram  = (float *)malloc(R * R * sizeof(float)); /* R×R Gram matrix */
    float * rhs   = (float *)malloc(R     * sizeof(float)); /* R right-hand side */
    float * tmp   = (float *)malloc(R * R * sizeof(float)); /* scratch */

    assert(ts_f && gram && rhs && tmp);

    /* --- initialise T_S with first R rows of K (simple, deterministic) --- */
    for (int r = 0; r < R; r++) {
        for (int d = 0; d < D; d++) {
            ts_f[r*D + d] = K[r*D + d];
        }
        /* normalise row so ALS doesn't diverge */
        float norm = 0.0f;
        for (int d = 0; d < D; d++) norm += ts_f[r*D+d] * ts_f[r*D+d];
        norm = sqrtf(norm) + 1e-8f;
        for (int d = 0; d < D; d++) ts_f[r*D+d] /= norm;
    }

    /* --- ALS iterations --- */
    for (int iter = 0; iter < n_iter; iter++) {

        /* Update T_L: T_L = K × T_S^T × (T_S × T_S^T)^{-1}
           Gram = T_S × T_S^T  [R×R] */
        memset(gram, 0, R * R * sizeof(float));
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < R; s++) {
                float v = 0.0f;
                for (int d = 0; d < D; d++) v += ts_f[r*D+d] * ts_f[s*D+d];
                gram[r*R+s] = v;
            }
        }

        /* For each row t of K, solve gram * TL[t,:] = K[t,:] × T_S^T */
        for (int t = 0; t < T; t++) {
            /* rhs = K[t,:] × T_S^T  [R] */
            for (int r = 0; r < R; r++) {
                float v = 0.0f;
                for (int d = 0; d < D; d++) v += K[t*D+d] * ts_f[r*D+d];
                rhs[r] = v;
            }
            /* solve gram * x = rhs */
            memcpy(tmp, gram, R * R * sizeof(float));
            if (solve_rr(tmp, rhs, R) == 0) {
                for (int r = 0; r < R; r++) TL[t*R+r] = rhs[r];
            } else {
                /* fallback: pseudoinverse via scaling */
                for (int r = 0; r < R; r++) TL[t*R+r] = rhs[r] / (gram[r*R+r] + 1e-8f);
            }
        }

        /* Update T_S: T_S = (T_L^T × T_L)^{-1} × T_L^T × K
           Gram = T_L^T × T_L  [R×R] */
        memset(gram, 0, R * R * sizeof(float));
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < R; s++) {
                float v = 0.0f;
                for (int t = 0; t < T; t++) v += TL[t*R+r] * TL[t*R+s];
                gram[r*R+s] = v;
            }
        }

        /* For each column d of K, solve gram * TS[:,d] = T_L^T × K[:,d] */
        for (int d = 0; d < D; d++) {
            for (int r = 0; r < R; r++) {
                float v = 0.0f;
                for (int t = 0; t < T; t++) v += TL[t*R+r] * K[t*D+d];
                rhs[r] = v;
            }
            memcpy(tmp, gram, R * R * sizeof(float));
            if (solve_rr(tmp, rhs, R) == 0) {
                for (int r = 0; r < R; r++) ts_f[r*D+d] = rhs[r];
            } else {
                for (int r = 0; r < R; r++) ts_f[r*D+d] = rhs[r] / (gram[r*R+r] + 1e-8f);
            }
        }
    }

    /* --- convert T_S to FP16 for storage --- */
    for (int i = 0; i < R * D; i++) {
        TS[i] = GGML_FP32_TO_FP16(ts_f[i]);
    }

    free(ts_f);
    free(gram);
    free(rhs);
    free(tmp);
}