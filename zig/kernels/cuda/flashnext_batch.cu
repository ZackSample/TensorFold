// Flash Next's batched prompt pass (cuda_batch.zig): the small steps between the captured kernels that the row-by-row
// path did on the host or one row at a time. Built without FMA contraction: every fma below is written out.

#include <cuda_bf16.h>
#include <stdint.h>

// out[i] = part[0][i] + part[1][i] + ... + part[sk - 1][i], slices in order and one rounding at the end (qmm.cu's
// reduce_kernel and _hc_reduce_act's sum): bf16 when out_bf16, else fp32. n values a slice.
extern "C" __global__ void fn_sum_slices(const float* __restrict__ part, void* __restrict__ out, int out_bf16,
                                         long long n, int sk) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = part[i];
    for (int s = 1; s < sk; ++s) acc = acc + part[static_cast<long long>(s) * n + i];
    if (out_bf16) reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
    else reinterpret_cast<float*>(out)[i] = acc;
}

// A window of the last `taps` rows [taps, c] (in place) after `rows` more rows, each the first c values of a row
// `stride` apart: the DeltaNet conv history (taps 3) and the n-gram tail (taps 9) as the row-by-row shifts leave them.
extern "C" __global__ void fn_shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ rows,
                                           long long stride, int n, int taps, int c) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    __nv_bfloat16 next[16];
    for (int j = 0; j < taps; ++j) {
        const int src = j + n;                       // row src of [window (taps rows); new rows]
        next[j] = src < taps ? win[static_cast<long long>(src) * c + ch]
                             : rows[static_cast<long long>(src - taps) * stride + ch];
    }
    for (int j = 0; j < taps; ++j) win[static_cast<long long>(j) * c + ch] = next[j];
}

// The MoE output a row's write-back adds (glue._hc_writeback's MODE 2 sum: slots in order, fp32 fma, one bf16
// rounding), plus an optional fp32 nudge [rows, d] before the rounding. Program (row, 256 columns).
extern "C" __global__ void fn_combine(const float* __restrict__ y, const float* __restrict__ wts,
                                      const float* __restrict__ nudge, __nv_bfloat16* __restrict__ out, int d,
                                      int slots) {
    const int r = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    float acc = 0.0f;
    for (int k = 0; k < slots; ++k)
        acc = __fmaf_rn(y[(static_cast<long long>(r) * slots + k) * d + col], wts[r * slots + k], acc);
    if (nudge != nullptr) acc = acc + nudge[static_cast<long long>(r) * d + col];
    out[static_cast<long long>(r) * d + col] = __float2bfloat16_rn(acc);
}

// rows x width 16-bit values from src (rows src_pitch values apart) to dst (dst_pitch apart): a column range of a
// projection, or (with `index`) the rows index[0..rows) of src gathered into consecutive dst rows.
extern "C" __global__ void fn_copy2d(const uint16_t* __restrict__ src, long long src_pitch, uint16_t* __restrict__ dst,
                                     long long dst_pitch, int width, int rows, const int* __restrict__ index) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (col >= width || r >= rows) return;
    const long long from = index != nullptr ? index[r] : r;
    dst[r * dst_pitch + col] = src[from * src_pitch + col];
}

// The same for 32-bit values (fp32 routing weights, int32 expert ids).
extern "C" __global__ void fn_copy2d32(const uint32_t* __restrict__ src, long long src_pitch,
                                       uint32_t* __restrict__ dst, long long dst_pitch, int width, int rows,
                                       const int* __restrict__ index) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (col >= width || r >= rows) return;
    const long long from = index != nullptr ? index[r] : r;
    dst[r * dst_pitch + col] = src[from * src_pitch + col];
}

// ---- the DeltaNet chain over a chunk ------------------------------------------------------------------------
// flashnext_gdn.cu's chain (one 1024-thread block a value head, row after row) cut in three: fn_gdn_front computes
// every row's conv, q and k norms and gates at once, fn_gdn_rows runs the recurrence (each warp 4 state rows, no
// block barrier) and stores each row's readout, fn_gdn_norm runs the gated RMS norm per row and head. Every value is
// computed by the same expression in the same order as the one-block chain: the outputs, states and sums are its bits.

namespace fn_gdn {
constexpr int NK = 16, NV = 48, DK = 128, DV = 128, TAPS = 4;
constexpr int C = 2 * NK * DK + NV * DV;
constexpr int PW = C + NV * DV + 2 * NV;

static __device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

static __device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

static __device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

static __device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static __device__ __forceinline__ float conv_act(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
                                                 const __nv_bfloat16* __restrict__ cw, int r, int c) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + tap;
        const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                      : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}
}  // namespace fn_gdn

// Program (row, unit), 256 threads: unit u < NK is key head u's q and k (conv, SiLU, then the L2 norm, q's scaled by
// 1/sqrt(dk)) into qn, kn [rows, NK, DK]; unit NK + hv is value head hv's 128 v (conv, SiLU) into v [rows, NV DV]
// and its decay g and beta into gb [rows, NV, 2]. A pure function of the projection rows and the conv history, so
// it runs for every row at once; each value comes from the chain's expression.
extern "C" __global__ void __launch_bounds__(256) fn_gdn_front(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ qn,
        float* __restrict__ kn, float* __restrict__ v, float* __restrict__ gb) {
    using namespace fn_gdn;
    const int r = blockIdx.x, u = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    if (u < NK) {
        __shared__ float x[2][DK];
        const int hk = u;
        x[t / DK][t % DK] = conv_act(P, cs, cw, r, t < DK ? hk * DK + t : NK * DK + hk * DK + (t - DK));
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v4[i] = x[warp][lane * 4 + i];
                ss = ss + v4[i] * v4[i];
            }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* dst = (warp == 0 ? qn : kn) + ((size_t)r * NK + hk) * DK;
#pragma unroll
            for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = v4[i] * inv;
        }
    } else {
        const int hv = u - NK;
        if (t < DV) v[(size_t)r * NV * DV + hv * DV + t] = conv_act(P, cs, cw, r, 2 * NK * DK + hv * DV + t);
        else if (t == DV) {
            const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
            const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
            gb[((size_t)r * NV + hv) * 2] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gb[((size_t)r * NV + hv) * 2 + 1] = bf(sigmoidf_(b));
        }
    }
}

// The chain through the chunk: a warp holds 4 state rows of value head blockIdx.x (16 a block), needs only the row's
// q, k, its own 4 v and the gates, and so runs with no block barrier; the next row's inputs load while this row's
// update and readout run. ys [rows, NV, DV] gets each readout as the chain rounds it.
// The state is read from state_in and the chunk's end state written to state_out (the same buffer in place, or
// another so the state before the chunk stays: a drafted window's rollback, cuda_mtp.zig).
static __device__ __forceinline__ void gdn_rows(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, const float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    using namespace fn_gdn;
    const int hv = blockIdx.x, hk = hv / (NV / NK), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.y * 16 + warp * 4;
    const size_t sbase = (size_t)hv * DV * DK;
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i];
    float kk[4], qq[4], vv[4], g = 0.0f, beta = 0.0f;
    auto load = [&](int r, float (&k_)[4], float (&q_)[4], float (&v_)[4], float& g_, float& b_) {
        const float4 k4 = *reinterpret_cast<const float4*>(kn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float4 q4 = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float4 v4 = *reinterpret_cast<const float4*>(v + (size_t)r * NV * DV + hv * DV + row0);
        k_[0] = k4.x; k_[1] = k4.y; k_[2] = k4.z; k_[3] = k4.w;
        q_[0] = q4.x; q_[1] = q4.y; q_[2] = q4.z; q_[3] = q4.w;
        v_[0] = v4.x; v_[1] = v4.y; v_[2] = v4.z; v_[3] = v4.w;
        g_ = gb[((size_t)r * NV + hv) * 2];
        b_ = gb[((size_t)r * NV + hv) * 2 + 1];
    };
    if (rows > 0) load(0, kk, qq, vv, g, beta);
    for (int r = 0; r < rows; ++r) {
        float nk[4], nq[4], nv[4], ng = 0.0f, nb = 0.0f;
        if (r + 1 < rows) load(r + 1, nk, nq, nv, ng, nb);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
            kv = warp_sum(kv);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        float o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o4[j] = bf(warp_sum(o));
        }
        if (lane == 0) *reinterpret_cast<float4*>(ys_out + ((size_t)r * NV + hv) * DV + row0) = make_float4(o4[0], o4[1], o4[2], o4[3]);
        if (r + 1 < rows) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kk[i] = nk[i];
                qq[i] = nq[i];
                vv[i] = nv[i];
            }
            g = ng;
            beta = nb;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, float* state, int rows, float* __restrict__ ys_out) {
    gdn_rows(qn, kn, v, gb, state, state, rows, ys_out);
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows_to(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ v,
        const float* __restrict__ gb, const float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    gdn_rows(qn, kn, v, gb, state_in, state_out, rows, ys_out);
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_norm(
        const float* __restrict__ ys_in, const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ norm_w,
        float eps, __nv_bfloat16* __restrict__ out, float* __restrict__ xs) {
    using namespace fn_gdn;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = ys_in[((size_t)r * NV + hv) * DV + t];
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float y = ys[lane * 4 + i];
            ss = ss + y * y;
        }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}
