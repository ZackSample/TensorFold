// Learning in the weights on Flash Next (Qwen3.8): the input gradients of its hyper-connections, its 513-row router and
// SwiGLU experts (the shared expert and the pocket's neurons behind a sigmoid gate), its gated attention and its
// group-32 4-bit products, on train.cu's conventions (fp32 gradients, bf16 kept activations, the MLX affine layout).
// The Gated DeltaNet layer's backward is flashnext_train.cu's.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float wide(float v) { return v; }
__device__ __forceinline__ float wide(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// The sum over a block of `groups` warps, the same in every thread.
__device__ __forceinline__ float block_sum(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_sum(lane < groups ? scratch[lane] : 0.0f);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max(float v, float* scratch, int groups) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    v = warp_max(lane < groups ? scratch[lane] : -INFINITY);
    __syncthreads();
    return v;
}

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float silu_grad(float v) {
    const float s = sigm(v);
    return s * (1.0f + v * (1.0f - s));
}

// ---- group-32 4-bit products ----------------------------------------------------------------------------------------

// C [M, N] (+)= A [M, K] W, W a 4-bit [K, N] matrix in MLX's layout (8 codes a word, a bf16 scale and bias every 32 of a
// row): a linear's input gradient from its output's (W its weight, K outputs by N inputs). tf_train_gemm's 64x64 tiles
// and m16n8k16 bf16 products, A rounded to bf16 and W dequantized to bf16 as each tile loads. Split K (grid z): split z
// takes K rows [z kchunk, (z + 1) kchunk) into its own fp32 partial C + z M N (rows N apart), which tf_fn_sum_parts adds.
template <typename TA>
__device__ __forceinline__ void gemm_q4(const TA* __restrict__ A, const unsigned* __restrict__ W, const bf16* __restrict__ S,
                                        const bf16* __restrict__ B, float* __restrict__ C, int M, int N, int K, int lda, int ldc,
                                        int accumulate, int kchunk) {
    __shared__ __align__(16) bf16 As[64][40];
    __shared__ __align__(16) bf16 Bs[64][40];  // W's tile transposed: [n][k]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int kb = blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int words = N / 8, groups = N / 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    const bf16 zero = b16(0.0f);
    const bool wide_a = sizeof(TA) == 2 && (lda & 7) == 0 && ((size_t)A & 15) == 0;
    for (int k0 = kb; k0 < ke; k0 += 32) {
        for (int e = tid; e < 64 * 4; e += 128) {
            const int r = e >> 2, c = (e & 3) * 8, gm = m0 + r, gk = k0 + c;
            if (wide_a && gm < M && gk + 8 <= ke) {
                *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(A + (size_t)gm * lda + gk);
            } else {
                for (int i = 0; i < 8; i++) As[r][c + i] = (gm < M && gk + i < ke) ? b16(wide(A[(size_t)gm * lda + gk + i])) : zero;
            }
        }
        for (int e = tid; e < 32 * 8; e += 128) {
            const int r = e >> 3, c = e & 7, gk = k0 + r, gn = n0 + c * 8;
            if (gk < ke && gn < N) {
                const unsigned q = W[(size_t)gk * words + gn / 8];
                const float s = f32(S[(size_t)gk * groups + gn / 32]), b = f32(B[(size_t)gk * groups + gn / 32]);
#pragma unroll
                for (int i = 0; i < 8; i++) Bs[c * 8 + i][r] = b16(fmaf(s, (float)((q >> (4 * i)) & 0xfu), b));
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) Bs[c * 8 + i][r] = zero;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            unsigned af[2][4], bq[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm + i * 16 + (lane >> 2), c = kk + (lane & 3) * 2;
                af[i][0] = *reinterpret_cast<const unsigned*>(&As[r][c]);
                af[i][1] = *reinterpret_cast<const unsigned*>(&As[r + 8][c]);
                af[i][2] = *reinterpret_cast<const unsigned*>(&As[r][c + 8]);
                af[i][3] = *reinterpret_cast<const unsigned*>(&As[r + 8][c + 8]);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = wn + j * 8 + (lane >> 2), k = kk + (lane & 3) * 2;
                bq[j][0] = *reinterpret_cast<const unsigned*>(&Bs[n][k]);
                bq[j][1] = *reinterpret_cast<const unsigned*>(&Bs[n][k + 8]);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bq[j][0]), "r"(bq[j][1]));
        }
        __syncthreads();
    }
    const bool split = gridDim.z > 1;
    float* out0 = split ? C + (size_t)blockIdx.z * M * N : C;
    const int ld = split ? N : ldc;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = m0 + wm + i * 16 + (lane >> 2) + h * 8, c = n0 + wn + j * 8 + (lane & 3) * 2;
                if (r >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    if (c + e >= N) continue;
                    float* out = out0 + (size_t)r * ld + c + e;
                    const float v = acc[i][j][h * 2 + e];
                    *out = (accumulate && !split) ? *out + v : v;
                }
            }
}

extern "C" __global__ void __launch_bounds__(128) tf_fn_gemm_q4(const float* A, const unsigned* W, const bf16* S, const bf16* B,
                                                               float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                               int kchunk) {
    gemm_q4<float>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk);
}

// tf_fn_gemm_q4 with bf16 A: the head's gradient from tf_train_softmax's bf16 rows.
extern "C" __global__ void __launch_bounds__(128) tf_fn_gemm_q4_bf(const bf16* A, const unsigned* W, const bf16* S, const bf16* B,
                                                                  float* C, int M, int N, int K, int lda, int ldc, int accumulate,
                                                                  int kchunk) {
    gemm_q4<bf16>(A, W, S, B, C, M, N, K, lda, ldc, accumulate, kchunk);
}

// C [M, N] (rows ldc apart) (+)= the split products' partials [splits, M, N], summed in split order.
extern "C" __global__ void tf_fn_sum_parts(const float* parts, int splits, int M, int N, float* C, int ldc, int accumulate) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * N) return;
    float s = 0;
    for (int z = 0; z < splits; z++) s += parts[(size_t)z * M * N + i];
    float* out = C + (size_t)(i / N) * ldc + i % N;
    *out = accumulate ? *out + s : s;
}

// The 4-bit matrix of expert e (e below `routed`: the stack w [routed, n, k]; e == routed: the shared expert's w2).
__device__ __forceinline__ void expert_mat(int e, int routed, size_t per_w, size_t per_s, const unsigned*& w, const bf16*& s,
                                           const bf16*& b, const unsigned* w2, const bf16* s2, const bf16* b2) {
    if (e == routed && w2 != nullptr) {
        w = w2, s = s2, b = b2;
    } else {
        w += (size_t)e * per_w, s += (size_t)e * per_s, b += (size_t)e * per_s;
    }
}

// y[s] (+)= x[s] w[e] (fp32): tf_train_experts_mm with group-32 scales, a shared expert (id `routed`) apart from the
// stack, x rows `ldx` apart and, when `order` is set, x[order[s] / slots] (a pass's row for each sorted pair).
extern "C" __global__ void __launch_bounds__(64) tf_fn_experts_mm(const float* x, int ldx, const int* order, int slots,
                                                                 const unsigned* w, const bf16* scales, const bf16* biases,
                                                                 const unsigned* w2, const bf16* s2, const bf16* b2,
                                                                 const unsigned* tiles, const int* offsets, float* y, int pairs,
                                                                 int n_in, int k_out, int routed, int accumulate) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][64];
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int k0 = blockIdx.x * 64, words = k_out / 8, groups = k_out / 32, t = threadIdx.x, tr = t / 16, tc = t % 16;
    expert_mat(e, routed, (size_t)n_in * words, (size_t)n_in * groups, w, scales, biases, w2, s2, b2);
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int n0 = 0; n0 < n_in; n0 += 32) {
        // rows past the expert's last read its last row again, and are dropped
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1), row = order ? order[s] / slots : s;
            xs[i / 32][i % 32] = x[(size_t)row * ldx + n0 + i % 32];
        }
        for (int i = t; i < 32 * 8; i += 64) {
            const size_t n = (size_t)n0 + i / 8;
            const unsigned q = w[n * words + k0 / 8 + i % 8];
            const int g = (k0 + (i % 8) * 8) / 32;
            const float sc = f32(scales[n * groups + g]), bi = f32(biases[n * groups + g]);
            for (int j = 0; j < 8; j++) ws[i / 8][(i % 8) * 8 + j] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int nn = 0; nn < 32; nn++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[nn][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][nn];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r >= last) continue;
        float4* out = reinterpret_cast<float4*>(y + (size_t)r * k_out + k0 + tc * 4);
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (accumulate) {
            const float4 o = *out;
            v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
        }
        *out = v;
    }
}

// y[s, col0 + n] = w[e][n] . x[order[s] / slots] (fp32), n below n_out: each sorted pair's expert rows (gate or up)
// again from its pass row (bf16, `ldx` apart), w [n_out, k_in] group-32 along k; tiles of 16 pairs by 64 outputs.
extern "C" __global__ void __launch_bounds__(64) tf_fn_experts_nt(const bf16* x, int ldx, const int* order, int slots,
                                                                 const unsigned* w, const bf16* scales, const bf16* biases,
                                                                 const unsigned* w2, const bf16* s2, const bf16* b2,
                                                                 const unsigned* tiles, const int* offsets, float* y, int ldy,
                                                                 int col0, int pairs, int k_in, int n_out, int routed) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][68];  // [k][n]
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int n0 = blockIdx.x * 64, words = k_in / 8, groups = k_in / 32, t = threadIdx.x, tr = t / 16, tc = t % 16;
    expert_mat(e, routed, (size_t)n_out * words, (size_t)n_out * groups, w, scales, biases, w2, s2, b2);
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int k0 = 0; k0 < k_in; k0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1);
            xs[i / 32][i % 32] = f32(x[(size_t)(order[s] / slots) * ldx + k0 + i % 32]);
        }
        for (int i = t; i < 64 * 4; i += 64) {
            const int nn = i / 4, c = i % 4;
            const size_t n = (size_t)n0 + nn;
            const unsigned q = w[n * words + k0 / 8 + c];
            const float sc = f32(scales[n * groups + k0 / 32]), bi = f32(biases[n * groups + k0 / 32]);
            for (int j = 0; j < 8; j++) ws[c * 8 + j][nn] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][kk];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r < last) *reinterpret_cast<float4*>(y + (size_t)r * ldy + col0 + n0 + tc * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

// ---- the MoE layer --------------------------------------------------------------------------------------------------

// Sorted pair s's SwiGLU (act = silu(g) u) undone from dact, its down projection's input gradient before the routing
// weight: a[p] = dact . act (dL/d its weight, p = order[s]) and dgu = wt[p] (dact u silu'(g), dact silu(g)); gu and
// dgu [pairs, 2 width] (g then u); grid pairs, block 128.
extern "C" __global__ void __launch_bounds__(128) tf_fn_swiglu_back(const float* gu, const float* dact, const int* order,
                                                                   const float* wts, float* a, float* dgu, int width) {
    __shared__ float scratch[32];
    const int s = blockIdx.x, t = threadIdx.x, p = order[s];
    const float wt = wts[p];
    const float* g = gu + (size_t)s * 2 * width;
    const float* da = dact + (size_t)s * width;
    float* out = dgu + (size_t)s * 2 * width;
    float dot = 0;
    for (int i = t; i < width; i += 128) {
        const float gv = g[i], uv = g[width + i], d = da[i], sg = silu(gv);
        dot += d * sg * uv;
        out[i] = wt * d * uv * silu_grad(gv);
        out[width + i] = wt * d * sg;
    }
    dot = block_sum(dot, scratch, 4);
    if (t == 0) a[p] = dot;
}

// dx[r] += sum_k dl_k router[ids_k] + dl_s router[shared]: the routing weights w_k = softmax over the top_k logits and
// the shared expert's gate sigmoid(logit `shared`) (slot top_k) back to the router's input, from each slot's a = dL/dw;
// dlog [rows, slots] keeps the logits' gradients when set. logits [rows, nl] fp32; grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_route_back(const float* logits, int nl, const int* ids, int slots,
                                                                  const float* a, const bf16* router, float* dx, int top_k,
                                                                  int shared, int dim, float* dlog) {
    __shared__ float dl[17];
    __shared__ int at[17];
    const int r = blockIdx.x, t = threadIdx.x;
    if (t == 0) {
        const float* l = logits + (size_t)r * nl;
        float top = -INFINITY;
        for (int k = 0; k < top_k; k++) top = fmaxf(top, l[ids[r * slots + k]]);
        float e[16], total = 0;
        for (int k = 0; k < top_k; k++) {
            e[k] = expf(l[ids[r * slots + k]] - top);
            total += e[k];
        }
        float mix = 0;
        for (int k = 0; k < top_k; k++) mix += e[k] / total * a[r * slots + k];
        for (int k = 0; k < top_k; k++) {
            dl[k] = e[k] / total * (a[r * slots + k] - mix);
            at[k] = ids[r * slots + k];
        }
        const float sg = sigm(l[shared]);
        dl[top_k] = a[r * slots + top_k] * sg * (1.0f - sg);
        at[top_k] = shared;
        if (dlog)
            for (int k = 0; k <= top_k; k++) dlog[(size_t)r * slots + k] = dl[k];
    }
    __syncthreads();
    for (int i = t; i < dim; i += 256) {
        float s = 0;
        for (int k = 0; k <= top_k; k++) s += dl[k] * f32(router[(size_t)at[k] * dim + i]);
        dx[(size_t)r * dim + i] += s;
    }
}

// ---- the pocket's neurons (SwiGLU, beside the shared expert, behind its gate) ---------------------------------------

#define POCKET_MAX 128

// The pocket's dot products on row y (d values) for neuron j: z = (gate_j . y, up_j . y) and, when g is set, e = down_j . g.
__device__ __forceinline__ void pocket_dots(const bf16* y, const float* g, int d, const float* gate, const float* up,
                                            const float* down, int j, float& zg, float& zu, float& e) {
    const int lane = threadIdx.x & 31;
    const float* gj = gate + (size_t)j * d;
    const float* uj = up + (size_t)j * d;
    const float* dj = down + (size_t)j * d;
    float a = 0, b = 0, c = 0;
    for (int i = lane; i < d; i += 32) {
        const float v = f32(y[i]);
        a += gj[i] * v;
        b += uj[i] * v;
        if (g) c += dj[i] * g[i];
    }
    zg = warp_sum(a), zu = warp_sum(b), e = g ? warp_sum(c) : 0.0f;
}

// Neuron j's scale on row r: the row's own (rs[r], a pass's sequence scale) from neuron `from` on, else 1.
__device__ __forceinline__ float pocket_scale(const float* rs, int from, int j, int r) {
    return (rs != nullptr && j >= from) ? rs[r] : 1.0f;
}

// out[r] (+)= s_r sum_j f_j silu(gate_j . y_r) (up_j . y_r) down_j, s_r = wts[r slots + at] (the shared expert's gate;
// 1 when wts is null), f_j = rs[r] for neurons from `from` on (1 when rs is null); the neurons fp32 [n, d], y bf16
// rows, out bf16 or fp32 rows; grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_rows(const bf16* y, int d, const float* gate, const float* up,
                                                                   const float* down, int n, const float* wts, int slots, int at,
                                                                   void* out, int out_f32, const float* rs, int from) {
    __shared__ float act[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    const float s = wts ? wts[(size_t)r * slots + at] : 1.0f;
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, nullptr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) act[j] = s * pocket_scale(rs, from, j, r) * silu(zg) * zu;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += act[j] * down[(size_t)j * d + k];
        if (out_f32) {
            static_cast<float*>(out)[(size_t)r * d + k] += v;
        } else {
            bf16* o = static_cast<bf16*>(out) + (size_t)r * d + k;
            *o = b16(f32(*o) + v);
        }
    }
}

// The pocket undone on row r from g (dL/d the MoE output): e_j = down_j . g, dx[r] += s sum_j f_j e_j (silu'(zg) zu
// gate_j + silu(zg) up_j), and the shared slot's a[r slots + at] += sum_j f_j silu(zg) zu e_j (its neurons ride the
// shared gate); f_j as tf_fn_pocket_rows scales it; grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_back(const bf16* y, int d, const float* gate, const float* up,
                                                                   const float* down, int n, const float* wts, int slots, int at,
                                                                   const float* g, float* dx, float* a, const float* rs, int from) {
    __shared__ float cg[POCKET_MAX], cu[POCKET_MAX], ca[POCKET_MAX];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const bf16* yr = y + (size_t)r * d;
    const float* gr = g + (size_t)r * d;
    const float s = wts[(size_t)r * slots + at];
    for (int j = w; j < n; j += 8) {
        float zg, zu, e;
        pocket_dots(yr, gr, d, gate, up, down, j, zg, zu, e);
        if (lane == 0) {
            const float f = pocket_scale(rs, from, j, r);
            cg[j] = s * f * e * silu_grad(zg) * zu;
            cu[j] = s * f * e * silu(zg);
            ca[j] = f * e * silu(zg) * zu;
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < d; k += 256) {
        float v = 0;
        for (int j = 0; j < n; j++) v += cg[j] * gate[(size_t)j * d + k] + cu[j] * up[(size_t)j * d + k];
        dx[(size_t)r * d + k] += v;
    }
    if (threadIdx.x == 0) {
        float v = 0;
        for (int j = 0; j < n; j++) v += ca[j];
        a[(size_t)r * slots + at] += v;
    }
}

// grad[j] += w s act_j push[row_j]: neuron j's output vector's gradient at its row row_j = rows[j] (none when negative),
// act_j = silu(gate_j . y) (up_j . y), s = wts[row_j slots + at] (the shared gate), push = dL/d the MoE output there;
// grid neurons, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_pocket_grad(const bf16* y, int d, const float* gate, const float* up,
                                                                   int n, const float* wts, int slots, int at, const float* push,
                                                                   const int* rows, float w, float* grad) {
    __shared__ float scratch[32];
    const int j = blockIdx.x, t = threadIdx.x, r = rows[j];
    if (j >= n || r < 0) return;
    const bf16* yr = y + (size_t)r * d;
    float a = 0, b = 0;
    for (int i = t; i < d; i += 256) {
        const float v = f32(yr[i]);
        a += gate[(size_t)j * d + i] * v;
        b += up[(size_t)j * d + i] * v;
    }
    a = block_sum(a, scratch, 8);
    b = block_sum(b, scratch, 8);
    const float k = w * wts[(size_t)r * slots + at] * silu(a) * b;
    for (int i = t; i < d; i += 256) grad[(size_t)j * d + i] += k * push[(size_t)r * d + i];
}

// ---- the hyper-connections ------------------------------------------------------------------------------------------

// A block's write-back h' = h + bf16(branch inject_s) undone: dbranch = sum_s g_s inject_s and dinject_s = g_s . branch,
// g = dL/dh' [rows, streams d]; grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_branch_back(const float* g, const bf16* inject, const bf16* branch,
                                                                      float* dbranch, float* dinject, int d, int streams) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, t = threadIdx.x;
    const float* gr = g + (size_t)r * streams * d;
    float inj[8], part[8];
    for (int s = 0; s < streams; s++) inj[s] = f32(inject[r * streams + s]), part[s] = 0;
    for (int i = t; i < d; i += 256) {
        const float b = f32(branch[(size_t)r * d + i]);
        float v = 0;
        for (int s = 0; s < streams; s++) {
            const float gs = gr[(size_t)s * d + i];
            v += gs * inj[s];
            part[s] += gs * b;
        }
        dbranch[(size_t)r * d + i] = v;
    }
    for (int s = 0; s < streams; s++) {
        const float v = block_sum(part[s], scratch, 8);
        if (t == 0) dinject[r * streams + s] = v;
    }
}

// The read-out's mix = mean_s sigmoid(up_s) normed_s undone from dmix [rows, d]: dn_s = dmix sigmoid(up_s) / S (into
// dnormed) and dup_s = dmix normed_s sigmoid'(up_s) / S, up and normed bf16 [rows, S d]; grid (d / 256, rows).
extern "C" __global__ void tf_fn_hc_mix_back(const float* dmix, const bf16* up, const bf16* normed, float* dnormed, float* dup,
                                             int d, int streams) {
    const int i = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (i >= d) return;
    const float g = dmix[(size_t)r * d + i] / (float)streams;
    for (int s = 0; s < streams; s++) {
        const size_t at = ((size_t)r * streams + s) * d + i;
        const float sg = sigm(f32(up[at]));
        dnormed[at] = g * sg;
        dup[at] = g * f32(normed[at]) * sg * (1.0f - sg);
    }
}

// The down projection's activations undone: v [rows, nv] its fp32 output (low SiLU rows, then the inject rows when
// dinject is set), act = silu(v / S) and inject = 2 sigmoid(v / S): dv = dact silu'(v / S) / S and dinject 2
// sigmoid'(v / S) / S; grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_act_back(const float* dact, const float* v, const float* dinject,
                                                                   float* dv, int low, int streams, int nv) {
    const int r = blockIdx.x;
    const float inv = 1.0f / (float)streams;
    for (int i = threadIdx.x; i < nv; i += 256) {
        const float x = v[(size_t)r * nv + i] * inv;
        float g = 0;
        if (i < low) {
            g = dact[(size_t)r * low + i] * silu_grad(x) * inv;
        } else if (dinject) {
            const float sg = sigm(x);
            g = dinject[r * streams + i - low] * 2.0f * sg * (1.0f - sg) * inv;
        }
        dv[(size_t)r * nv + i] = g;
    }
}

// g[r, s] += the stream's RMS norm (scale fp32 1 + w) undone from dnormed: rinv scale dn - rinv^3 h (scale dn . h) / d;
// grid (streams, rows), block 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_norm_back(const bf16* h, const float* scale, const float* dnormed,
                                                                    float* g, int d, int streams, float eps) {
    __shared__ float scratch[32];
    const int s = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const size_t base = ((size_t)r * streams + s) * d;
    const float* w = scale + (size_t)s * d;
    float sq = 0, dot = 0;
    for (int i = t; i < d; i += 256) {
        const float v = f32(h[base + i]);
        sq += v * v;
        dot += w[i] * dnormed[base + i] * v;
    }
    sq = block_sum(sq, scratch, 8);
    dot = block_sum(dot, scratch, 8);
    const float inv = rsqrtf(sq / (float)d + eps), k = inv * inv * inv * dot / (float)d;
    for (int i = t; i < d; i += 256) g[base + i] += inv * w[i] * dnormed[base + i] - k * f32(h[base + i]);
}

// ---- the attention layer --------------------------------------------------------------------------------------------

// Attention's shape (train.cu's Heads): rows, heads, kv heads, head dim, q|k|v row width, scale, prefix keys.
struct Heads {
    int rows, heads, kv_heads, dim, qkv;
    float scale;
    int prefix;
};

// The rotate-half RoPE angle of pair i at position pos.
__device__ __forceinline__ void rope_cs(int pos, const float* inv, int i, float& c, float& s) {
    const float ang = (float)pos * inv[i];
    c = cosf(ang), s = sinf(ang);
}

// The attention's prepared heads as glue.attn_prep writes them, into qkv [rows, (heads + 2 kv) dim] (q, then k, then
// v, train.cu's attention layout): q and k heads RMS-normed (scale fp32) to bf16, the first 2 half dims rotated
// (rotate-half, position pos0 + r, inv [half]) to bf16; v copied. pq [rows, heads 2 dim] (q | gate a head), pk and pv
// [rows, kv dim]; grid (heads + 2 kv, rows), block dim.
extern "C" __global__ void __launch_bounds__(256) tf_fn_attn_prep(const bf16* pq, const bf16* pk, const bf16* pv,
                                                                 const float* qscale, const float* kscale, const float* inv,
                                                                 int pos0, bf16* qkv, int heads, int kv_heads, int dim, int half,
                                                                 float eps) {
    __shared__ float xn[256], scratch[32];
    const int hd = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int qw = heads * dim, kw = kv_heads * dim, width = qw + 2 * kw;
    if (hd >= heads + kv_heads) {
        const int k = hd - heads - kv_heads;
        qkv[(size_t)r * width + qw + kw + k * dim + d] = pv[(size_t)r * kw + k * dim + d];
        return;
    }
    const bool q = hd < heads;
    const float x = q ? f32(pq[(size_t)r * 2 * qw + hd * 2 * dim + d]) : f32(pk[(size_t)r * kw + (hd - heads) * dim + d]);
    const float* w = q ? qscale : kscale;
    const float rinv = 1.0f / sqrtf(block_sum(x * x, scratch, dim / 32) / (float)dim + eps);
    xn[d] = f32(b16(x * rinv * w[d]));
    __syncthreads();
    float out = xn[d];
    if (d < 2 * half) {
        float c, s;
        rope_cs(pos0 + r, inv, d % half, c, s);
        out = d < half ? xn[d] * c - xn[d + half] * s : xn[d - half] * s + xn[d] * c;
    }
    qkv[(size_t)r * width + (q ? hd * dim : qw + (hd - heads) * dim) + d] = b16(out);
}

#define FN_KEYS 2048

// tf_train_attn_q for up to FN_KEYS keys: one query row i (position prefix + i) of head h, its probabilities and score
// gradients kept [heads, rows, keys] for tf_train_attn_kv, dq; a warp a key for the dot products; block 256, dim <= 256.
extern "C" __global__ void __launch_bounds__(256) tf_fn_attn_q(const bf16* qkv, const float* dout, float* probs, float* dscores,
                                                              float* dq, Heads a, const bf16* pk, const bf16* pv) {
    __shared__ float qi[256], gi[256], sc[FN_KEYS], dp[FN_KEYS], scratch[32];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, h = blockIdx.x, i = blockIdx.y;
    const int kv = h / (a.heads / a.kv_heads), qw = a.heads * a.dim, kw = a.kv_heads * a.dim;
    const int keys = a.prefix + a.rows, at = a.prefix + i;
    const bf16* k = qkv + qw;
    const bf16* v = qkv + qw + kw;
    for (int d = t; d < a.dim; d += 256) {
        qi[d] = f32(qkv[(size_t)i * a.qkv + h * a.dim + d]);
        gi[d] = dout[(size_t)i * qw + h * a.dim + d];
    }
    __syncthreads();
    for (int j = w; j <= at; j += 8) {
        const bf16* kr = (j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        const bf16* vr = (j < a.prefix ? pv + (size_t)j * kw : v + (size_t)(j - a.prefix) * a.qkv) + kv * a.dim;
        float s = 0, g = 0;
        for (int d = lane; d < a.dim; d += 32) {
            s += qi[d] * f32(kr[d]);
            g += gi[d] * f32(vr[d]);
        }
        s = warp_sum(s);
        g = warp_sum(g);
        if (lane == 0) sc[j] = s * a.scale, dp[j] = g;
    }
    __syncthreads();
    float top = -INFINITY;
    for (int j = t; j <= at; j += 256) top = fmaxf(top, sc[j]);
    top = block_max(top, scratch, 8);
    float total = 0;
    for (int j = t; j <= at; j += 256) {
        const float e = expf(sc[j] - top);
        sc[j] = e;
        total += e;
    }
    total = block_sum(total, scratch, 8);
    float mix = 0;
    for (int j = t; j <= at; j += 256) {
        sc[j] /= total;
        mix += sc[j] * dp[j];
    }
    mix = block_sum(mix, scratch, 8);
    const size_t row = ((size_t)h * a.rows + i) * keys;
    for (int j = t; j < keys; j += 256) {
        const bool live = j <= at;
        const float p = live ? sc[j] : 0.0f, g = live ? p * (dp[j] - mix) : 0.0f;
        if (live) dp[j] = g;
        probs[row + j] = p;
        dscores[row + j] = g;
    }
    __syncthreads();
    for (int d = t; d < a.dim; d += 256) {
        float acc = 0;
        for (int j = 0; j <= at; j++) {
            const bf16* kr = j < a.prefix ? pk + (size_t)j * kw : k + (size_t)(j - a.prefix) * a.qkv;
            acc += dp[j] * f32(kr[kv * a.dim + d]);
        }
        dq[(size_t)i * qw + h * a.dim + d] = acc * a.scale;
    }
}

// The output gate gated = o sigmoid(gate) undone from dgated [rows, heads dim]: dout = dgated sigmoid(gate) (the
// attention output's gradient) and the gate's dgated gated (1 - sigmoid(gate)) into dpq's gate half [rows, heads 2 dim]
// (gated: o_proj's bf16 input, so o itself is not needed); grid (heads dim / 256, rows).
extern "C" __global__ void tf_fn_attn_gate_back(const float* dgated, const bf16* gated, const bf16* pq, float* dout, float* dpq,
                                                int heads, int dim) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y, qw = heads * dim;
    if (c >= qw) return;
    const int h = c / dim, d = c % dim;
    const size_t at = (size_t)r * 2 * qw + h * 2 * dim + dim + d;
    const float sg = sigm(f32(pq[at])), g = dgated[(size_t)r * qw + c];
    dout[(size_t)r * qw + c] = g * sg;
    dpq[at] = g * f32(gated[(size_t)r * qw + c]) * (1.0f - sg);
}

// tf_fn_attn_prep undone for the q and k heads: dq [rows, heads dim] and dk [rows, kv dim] (the prepared heads'
// gradients) rotated back and through each head's RMS norm onto the projections, dpq's q half and dpk [rows, kv dim];
// grid (heads + kv, rows), block dim.
extern "C" __global__ void __launch_bounds__(256) tf_fn_attn_prep_back(const bf16* pq, const bf16* pk, const float* qscale,
                                                                      const float* kscale, const float* inv, int pos0,
                                                                      const float* dq, const float* dk, float* dpq, float* dpk,
                                                                      int heads, int kv_heads, int dim, int half, float eps) {
    __shared__ float go[256], scratch[32];
    const int hd = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int qw = heads * dim, kw = kv_heads * dim;
    const bool q = hd < heads;
    const float x = q ? f32(pq[(size_t)r * 2 * qw + hd * 2 * dim + d]) : f32(pk[(size_t)r * kw + (hd - heads) * dim + d]);
    go[d] = q ? dq[(size_t)r * qw + hd * dim + d] : dk[(size_t)r * kw + (hd - heads) * dim + d];
    __syncthreads();
    float gn = go[d];
    if (d < 2 * half) {
        float c, s;
        rope_cs(pos0 + r, inv, d % half, c, s);
        gn = d < half ? go[d] * c + go[d + half] * s : go[d] * c - go[d - half] * s;
    }
    const float w = (q ? qscale : kscale)[d];
    const float sq = block_sum(x * x, scratch, dim / 32), dot = block_sum(w * gn * x, scratch, dim / 32);
    const float rinv = 1.0f / sqrtf(sq / (float)dim + eps);
    const float g = rinv * w * gn - rinv * rinv * rinv * x * dot / (float)dim;
    if (q)
        dpq[(size_t)r * 2 * qw + hd * 2 * dim + d] = g;
    else
        dpk[(size_t)r * kw + (hd - heads) * dim + d] = g;
}

// ---- the engine's packed experts ------------------------------------------------------------------------------------

// The engine keeps every layer's experts (512 routed, the shared one at 512) in experts_pack.cu's layout, group 32:
// [E, R / 32, K / 32] blocks of 160 words, 128 words of codes (each word's nibbles in the order 0 2 4 6 1 3 5 7, the
// words of a block's 32 rows x 4 laid out as (i % 32, i / 32) for i = row 4 + word) then 32 words of bf16 scale and
// bias pairs; `stack` tables block by block side by side (gate | up: 2, `half` picks one). The backward reads them
// there: no second copy of the experts.
__device__ __forceinline__ const unsigned* packed_block(const unsigned* w, int stack, int half, size_t e, int R, int K, int r,
                                                        int k) {
    const size_t b = (e * (size_t)(R / 32) + (size_t)(r / 32)) * (size_t)(K / 32) + (size_t)(k / 32);
    return w + (b * stack + half) * 160;
}

// Row rr's word kk (codes 8 kk .. 8 kk + 7 of the block's 32 inputs) in MLX's order: code j at bits 4 j.
__device__ __forceinline__ unsigned packed_word(const unsigned* blk, int rr, int kk) {
    const int i = rr * 4 + kk;
    const unsigned p = blk[(i % 32) * 4 + i / 32];
    unsigned out = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) out |= (((p >> (4 * j)) & 0xfu) << (8 * j)) | (((p >> (16 + 4 * j)) & 0xfu) << (8 * j + 4));
    return out;
}

// Row rr's scale and bias in its block.
__device__ __forceinline__ void packed_sb(const unsigned* blk, int rr, float& s, float& b) {
    const int t = rr / 8, p = (rr % 8) / 2, hi = rr % 2;
    const unsigned ws = blk[128 + p * 8 + t], wb = blk[128 + p * 8 + 4 + t];
    s = __uint_as_float((hi ? ws >> 16 : ws & 0xffffu) << 16);
    b = __uint_as_float((hi ? wb >> 16 : wb & 0xffffu) << 16);
}

// tf_fn_experts_nt over the packed table w [E, n_out, k_in]: y[s, col0 + n] = w[e][n] . x[order[s] / slots].
extern "C" __global__ void __launch_bounds__(64) tf_fn_experts_nt_packed(const bf16* x, int ldx, const int* order, int slots,
                                                                        const unsigned* w, int stack, int half,
                                                                        const unsigned* tiles, const int* offsets, float* y, int ldy,
                                                                        int col0, int pairs, int k_in, int n_out, int routed) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][68];  // [k][n]
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int n0 = blockIdx.x * 64, t = threadIdx.x, tr = t / 16, tc = t % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int k0 = 0; k0 < k_in; k0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1);
            xs[i / 32][i % 32] = f32(x[(size_t)(order[s] / slots) * ldx + k0 + i % 32]);
        }
        for (int i = t; i < 64 * 4; i += 64) {
            const int nn = i / 4, c = i % 4, n = n0 + nn;
            const unsigned* blk = packed_block(w, stack, half, e, n_out, k_in, n, k0);
            const unsigned q = packed_word(blk, n % 32, c);
            float sc, bi;
            packed_sb(blk, n % 32, sc, bi);
            for (int j = 0; j < 8; j++) ws[c * 8 + j][nn] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int kk = 0; kk < 32; kk++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][kk];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r < last) *reinterpret_cast<float4*>(y + (size_t)r * ldy + col0 + n0 + tc * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

// tf_fn_experts_mm over the packed table w [E, n_in, k_out]: y[s] (+)= x[s] w[e] (x rows ldx apart; x[order[s] / slots]
// when order is set).
extern "C" __global__ void __launch_bounds__(64) tf_fn_experts_mm_packed(const float* x, int ldx, const int* order, int slots,
                                                                        const unsigned* w, int stack, int half,
                                                                        const unsigned* tiles, const int* offsets, float* y, int pairs,
                                                                        int n_in, int k_out, int routed, int accumulate) {
    __shared__ float xs[16][33];
    __shared__ __align__(16) float ws[32][64];
    const int e = (int)tiles[2 * blockIdx.y], r0 = (int)tiles[2 * blockIdx.y + 1];
    if (e > routed) return;
    const int last = e < routed ? offsets[e + 1] : pairs;
    const int k0 = blockIdx.x * 64, t = threadIdx.x, tr = t / 16, tc = t % 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (int n0 = 0; n0 < n_in; n0 += 32) {
        for (int i = t; i < 16 * 32; i += 64) {
            const int s = min(r0 + i / 32, last - 1), row = order ? order[s] / slots : s;
            xs[i / 32][i % 32] = x[(size_t)row * ldx + n0 + i % 32];
        }
        for (int i = t; i < 32 * 8; i += 64) {
            const int n = n0 + i / 8, kw = i % 8, k = k0 + kw * 8;
            const unsigned* blk = packed_block(w, stack, half, e, n_in, k_out, n, k);
            const unsigned q = packed_word(blk, n % 32, (k % 32) / 8);
            float sc, bi;
            packed_sb(blk, n % 32, sc, bi);
            for (int j = 0; j < 8; j++) ws[i / 8][kw * 8 + j] = fmaf(sc, (float)((q >> (4 * j)) & 0xfu), bi);
        }
        __syncthreads();
        for (int nn = 0; nn < 32; nn++) {
            const float4 b = *reinterpret_cast<const float4*>(&ws[nn][tc * 4]);
            for (int i = 0; i < 4; i++) {
                const float a = xs[tr * 4 + i][nn];
                acc[i][0] = fmaf(a, b.x, acc[i][0]);
                acc[i][1] = fmaf(a, b.y, acc[i][1]);
                acc[i][2] = fmaf(a, b.z, acc[i][2]);
                acc[i][3] = fmaf(a, b.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int r = r0 + tr * 4 + i;
        if (r >= last) continue;
        float4* out = reinterpret_cast<float4*>(y + (size_t)r * k_out + k0 + tc * 4);
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (accumulate) {
            const float4 o = *out;
            v.x += o.x, v.y += o.y, v.z += o.z, v.w += o.w;
        }
        *out = v;
    }
}

// y[i] += x[i] for n floats: a nudge onto its row of the MoE output.
extern "C" __global__ void tf_fn_axpy(float* y, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] += x[i];
}
