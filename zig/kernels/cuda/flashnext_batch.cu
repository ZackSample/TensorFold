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
