// Living Weights on Flash Next: the Gated DeltaNet layer's forward for training and its input-gradient backward.
//
// The forward is the Python 0.6.1 engine's (families/qwen4_exp/cuda/gdn.cu chain_kernel, gdn_io.cu front_kernel and
// back_kernel), per row t, value head h and its key head h / (nv / nk), with S [DV][DK] (row j a value channel):
//   a_c   = bf16(silu(sum_tap cw[c][tap] x[t - 3 + tap][c]))       x: [conv history (3 rows) | projection rows]
//   q     = a_q / sqrt(|a_q|^2 + 1e-6) / sqrt(DK),   k = a_k / sqrt(|a_k|^2 + 1e-6),   v = a_v
//   g     = exp(-exp(A_log) softplus(a + dt_bias)),   beta = bf16(sigmoid(b))
//   S'    = g S;   delta_j = beta (v_j - S'_j . k);   S_j = S'_j + delta_j k
//   y_j   = bf16(S_j . q)
//   out   = bf16(bf16(bf16(y rsqrt(mean(y^2) + eps)) w) sigmoid(z))        over the head's DV channels
// built from the same operations in the same order (this file builds with --fmad=false), so y equals the serving bits.
//
// The backward takes the bf16 roundings as identities and runs in fp32 from bf16 inputs. Gradients reach the projection
// row [qkv | z | b | a] (dproj, fp32); the state's and the conv history's gradients into the kept prefix are dropped, as
// Nemotron's Mamba backward (train_mixers.cuh) drops them. The chain keeps its state every CK rows and replays a
// chunk's states into a work area (L2-sized at CK 4) before walking it backward: one replay of every row a pass.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

// The layer's shape: rows, key and value heads, conv channels (2 nk DK + nv DV) and projection row width.
struct GdnShape {
    int rows, nk, nv, conv, proj;
};

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;
constexpr int CK = 4;      // scan rows between state checkpoints (gdn_train.zig's ck)
constexpr int SPLIT = 4;   // blocks a value head's chain runs in (gdn_train.zig's split)
constexpr int ROWS = DV / SPLIT;  // state rows a block holds: 8 warps of 4

__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 b16(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bf(float x) { return f32(b16(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
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

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float silu_grad(float v) {
    const float s = sigmoidf_(v);
    return s * (1.0f + v * (1.0f - s));
}

__device__ __forceinline__ int chunks(int rows) { return (rows + CK - 1) / CK; }

// Channel c's conv sum for row r over [history (zeros when null) | projection rows], the chain kernel's tap order.
__device__ __forceinline__ float conv_acc(const bf16* P, const bf16* hist, const bf16* cw, int r, int c, const GdnShape& s) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r - (TAPS - 1) + tap;
        const float x = at >= 0 ? f32(P[(size_t)at * s.proj + c])
                                : (hist != nullptr ? f32(hist[(size_t)(at + TAPS - 1) * s.conv + c]) : 0.0f);
        acc = acc + f32(cw[c * TAPS + tap]) * x;
    }
    return acc;
}

// This thread's 4 x 4 block of a [DV][DK] state: rows 4 warp .. 4 warp + 3, columns 4 lane .. 4 lane + 3.
__device__ __forceinline__ void load_state(float (&s)[4][4], const float* base, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 v = base != nullptr ? *reinterpret_cast<const float4*>(base + (warp * 4 + j) * DK + lane * 4)
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        s[j][0] = v.x; s[j][1] = v.y; s[j][2] = v.z; s[j][3] = v.w;
    }
}

__device__ __forceinline__ void store_state(const float (&s)[4][4], float* base, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(base + (warp * 4 + j) * DK + lane * 4) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

// gdn.cu's update: one delta-rule step on this thread's block of the state (decay, read along k, correct toward v).
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp, float g,
                                       float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

}  // namespace

// ---- the forward for training --------------------------------------------------------------------------------------

// Block (row, head), gdn_io.cu's front: heads < nk conv q and k and L2-norm them (q times DK^-0.5); the others conv v
// and make g and beta. act [rows, conv] bf16 keeps every conv output for the backward; hist [3, conv] or null.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_front(const bf16* __restrict__ P, const bf16* __restrict__ hist,
                                                               const bf16* __restrict__ cw, const float* __restrict__ a_log,
                                                               const float* __restrict__ dt_bias, bf16* __restrict__ act,
                                                               float* __restrict__ q, float* __restrict__ k,
                                                               float* __restrict__ g, float* __restrict__ beta, GdnShape s) {
    const int r = blockIdx.x, head = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float xs[2][DK];
    if (head < s.nk) {
        const int cq = head * DK + t, ck = s.nk * DK + head * DK + t;
        const float aq = conv_acc(P, hist, cw, r, cq, s), ak = conv_acc(P, hist, cw, r, ck, s);
        xs[0][t] = bf(aq / (1.0f + expf(-aq)));
        xs[1][t] = bf(ak / (1.0f + expf(-ak)));
        act[(size_t)r * s.conv + cq] = b16(xs[0][t]);
        act[(size_t)r * s.conv + ck] = b16(xs[1][t]);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* out = (warp == 0 ? q : k) + ((size_t)r * s.nk + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = v4[i] * inv;
        }
        return;
    }
    const int hv = head - s.nk, c = 2 * s.nk * DK + hv * DV + t;
    const float av = conv_acc(P, hist, cw, r, c, s);
    act[(size_t)r * s.conv + c] = b16(av / (1.0f + expf(-av)));
    if (t == 0) {
        const size_t at = (size_t)r * s.proj + s.conv + s.nv * DV;
        const float b = f32(P[at + hv]), a = f32(P[at + s.nv + hv]);
        g[r * s.nv + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[r * s.nv + hv] = bf(sigmoidf_(b));
    }
}

// One value head's chain from init [nv][DV][DK] (null: zeros), SPLIT blocks of 256 threads a head, each its 32 state
// rows (the rows never mix: a row needs only k, q, g, beta and its own v): y [rows, nv, DV] bf16 and the state before
// every CK-th row into ckpt [nv][chunks][DV][DK]. gdn.cu's update and read-out on gdn.cu's lane layout, 8 warps a block.
extern "C" __global__ void __launch_bounds__(256) tf_gdn_scan_fwd(const bf16* __restrict__ act, const float* __restrict__ q,
                                                                  const float* __restrict__ k, const float* __restrict__ g,
                                                                  const float* __restrict__ beta, const float* __restrict__ init,
                                                                  bf16* __restrict__ y, float* __restrict__ ckpt, GdnShape s) {
    const int h = blockIdx.x / SPLIT, rb = (blockIdx.x % SPLIT) * ROWS, hk = h / (s.nv / s.nk);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const size_t size = (size_t)DV * DK;
    const int v0 = 2 * s.nk * DK + h * DV + rb;
    __shared__ float qs[DK], ks[DK], vs[ROWS];
    float st[4][4];
    load_state(st, init != nullptr ? init + h * size + rb * DK : nullptr, warp, lane);
    float* mine = ckpt + (size_t)h * chunks(s.rows) * size + rb * DK;
    for (int r = 0; r < s.rows; ++r) {
        if (r % CK == 0) store_state(st, mine + (r / CK) * size, warp, lane);
        if (t < DK) qs[t] = q[((size_t)r * s.nk + hk) * DK + t];
        else ks[t - DK] = k[((size_t)r * s.nk + hk) * DK + t - DK];
        if (t < ROWS) vs[t] = f32(act[(size_t)r * s.conv + v0 + t]);
        __syncthreads();
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
        update(st, kk, vs, warp, g[r * s.nv + h], beta[r * s.nv + h]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + st[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) y[((size_t)r * s.nv + h) * DV + rb + warp * 4 + j] = b16(o);
        }
        __syncthreads();
    }
}

// ---- the backward ---------------------------------------------------------------------------------------------------

// Block (row, value head): the gated RMS norm undone. dout [rows, nv DV] (the out projection's input gradient) gives
// dy [rows, nv, DV] (the scan's output gradient) and dz into dproj's z columns; y, rms and yn as the forward made them.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_norm_back(const bf16* __restrict__ y, const bf16* __restrict__ P,
                                                                   const bf16* __restrict__ norm_w, const float* __restrict__ dout,
                                                                   float* __restrict__ dy, float* __restrict__ dproj, GdnShape s,
                                                                   float eps) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV], scratch[32];
    __shared__ float rinv;
    const size_t row = ((size_t)r * s.nv + h) * DV;
    ys[t] = f32(y[row + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = ys[lane * 4 + i]; ss = ss + v * v; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float w = f32(norm_w[t]), yv = ys[t], ri = rinv;
    const float yn = bf(bf(yv * ri) * w);
    const size_t zc = (size_t)r * s.proj + s.conv + h * DV + t;
    const float sg = sigmoidf_(f32(P[zc])), d = dout[row + t];
    dproj[zc] = d * yn * sg * (1.0f - sg);
    const float dn = d * sg * w;
    const float dot = block_sum(yv * dn, scratch, 4);
    dy[row + t] = ri * dn - ri * ri * ri * yv * dot / (float)DV;
}

// One value head's chain backward, SPLIT blocks of 256 threads a head as the forward (each its 32 state rows), last
// chunk first: the chunk's states replayed from its checkpoint into work [nv][SPLIT][CK][ROWS][DK], then its rows walked
// back. Per row, with ds the state's gradient carried:
//   ds += dy_j q;  dq = sum_j dy_j S_j;  dd_j = ds_j . k;  dv_j = beta dd_j;  dbeta = sum_j dd_j (v_j - S'_j . k)
//   dk = sum_j (delta_j ds_j - beta dd_j S'_j);  ds' = ds - beta dd k^T;  dg = sum ds' . S_prev;  ds = g ds'
// The sums over j are this block's rows' share: dq and dk into dqp, dkp [rows, nv, SPLIT, DK], dg and dbeta into
// [rows, nv, SPLIT]; dv into dact's v columns.
extern "C" __global__ void __launch_bounds__(256) tf_gdn_scan_back(const bf16* __restrict__ act, const float* __restrict__ q,
                                                                   const float* __restrict__ k, const float* __restrict__ g,
                                                                   const float* __restrict__ beta, const float* __restrict__ ckpt,
                                                                   const float* __restrict__ dy, float* __restrict__ work,
                                                                   float* __restrict__ dqp, float* __restrict__ dkp,
                                                                   float* __restrict__ dact, float* __restrict__ dg,
                                                                   float* __restrict__ dbeta, GdnShape s) {
    constexpr int WARPS = ROWS / 4;
    __shared__ float qs[DK], ks[DK], vs[ROWS], dos[ROWS];
    __shared__ float red_q[WARPS][DK], red_k[WARPS][DK];
    __shared__ float red_g[WARPS], red_b[WARPS];
    const int h = blockIdx.x / SPLIT, part = blockIdx.x % SPLIT, rb = part * ROWS, hk = h / (s.nv / s.nk);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const size_t size = (size_t)DV * DK, mine_size = (size_t)ROWS * DK;
    const int v0 = 2 * s.nk * DK + h * DV + rb, nch = chunks(s.rows);
    const float* saved = ckpt + (size_t)h * nch * size + rb * DK;
    float* mine = work + (size_t)blockIdx.x * CK * mine_size;
    float ds[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) ds[j][i] = 0.0f;
    for (int c = nch - 1; c >= 0; --c) {
        const int first = c * CK, last = min(first + CK, s.rows);
        {
            // the state before each of the chunk's rows, each thread its own elements
            float st[4][4];
            load_state(st, saved + c * size, warp, lane);
            float in_kv = 0.0f, in_g = 0.0f, in_b = 0.0f;
            auto fetch = [&](int r) {
                if (t < DK) in_kv = k[((size_t)r * s.nk + hk) * DK + t];
                else if (t < DK + ROWS) in_kv = f32(act[(size_t)r * s.conv + v0 + t - DK]);
                in_g = g[r * s.nv + h];
                in_b = beta[r * s.nv + h];
            };
            if (first + 1 < last) fetch(first);
            for (int r = first; r < last; ++r) {
                store_state(st, mine + (r - first) * mine_size, warp, lane);
                if (r + 1 == last) break;
                if (t < DK) ks[t] = in_kv;
                else if (t < DK + ROWS) vs[t - DK] = in_kv;
                const float gg = in_g, bb = in_b;
                __syncthreads();
                if (r + 2 < last) fetch(r + 1);
                float kk[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) kk[i] = ks[lane * 4 + i];
                update(st, kk, vs, warp, gg, bb);
                __syncthreads();
            }
        }
        // each row's inputs fetched into registers while the row before it in the walk computes
        float in_qk, in_vd = 0.0f, in_g, in_b;
        auto fetch = [&](int r) {
            in_qk = t < DK ? q[((size_t)r * s.nk + hk) * DK + t] : k[((size_t)r * s.nk + hk) * DK + t - DK];
            if (t < ROWS) in_vd = f32(act[(size_t)r * s.conv + v0 + t]);
            else if (t < 2 * ROWS) in_vd = dy[((size_t)r * s.nv + h) * DV + rb + t - ROWS];
            in_g = g[r * s.nv + h];
            in_b = beta[r * s.nv + h];
        };
        fetch(last - 1);
        for (int r = last - 1; r >= first; --r) {
            if (t < DK) qs[t] = in_qk;
            else ks[t - DK] = in_qk;
            if (t < ROWS) vs[t] = in_vd;
            else if (t < 2 * ROWS) dos[t - ROWS] = in_vd;
            const float gg = in_g, bb = in_b;
            __syncthreads();
            if (r > first) fetch(r - 1);
            float kk[4], qq[4], pq[4], pk[4], pg = 0.0f, pb = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                kk[i] = ks[lane * 4 + i];
                qq[i] = qs[lane * 4 + i];
                pq[i] = 0.0f;
                pk[i] = 0.0f;
            }
            const float* prev = mine + (r - first) * mine_size;
            float4 p4[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) p4[j] = *reinterpret_cast<const float4*>(prev + (warp * 4 + j) * DK + lane * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = warp * 4 + j;
                const float sp[4] = {p4[j].x, p4[j].y, p4[j].z, p4[j].w};
                float sd[4], kv = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sd[i] = sp[i] * gg;
                    kv = kv + sd[i] * kk[i];
                }
                kv = warp_sum(kv);
                const float err = vs[row] - kv, delta = err * bb, dj = dos[row];
                float dd = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pq[i] += dj * (sd[i] + kk[i] * delta);
                    ds[j][i] += dj * qq[i];
                    dd += ds[j][i] * kk[i];
                }
                dd = warp_sum(dd);
                const float dkv = -bb * dd;
                if (lane == 0) {
                    dact[(size_t)r * s.conv + v0 + row] = bb * dd;
                    pb += dd * err;
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pk[i] += delta * ds[j][i] + dkv * sd[i];
                    const float dsp = ds[j][i] + dkv * kk[i];
                    pg += dsp * sp[i];
                    ds[j][i] = gg * dsp;
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                red_q[warp][lane * 4 + i] = pq[i];
                red_k[warp][lane * 4 + i] = pk[i];
            }
            pg = warp_sum(pg);
            if (lane == 0) {
                red_g[warp] = pg;
                red_b[warp] = pb;
            }
            __syncthreads();
            const size_t out = ((size_t)r * s.nv + h) * SPLIT + part;
            if (t < DK) {
                float x = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) x += red_q[w][t];
                dqp[out * DK + t] = x;
                if (t == 0) {
                    float e = 0.0f;
#pragma unroll
                    for (int w = 0; w < WARPS; ++w) e += red_g[w];
                    dg[out] = e;
                }
            } else {
                float x = 0.0f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) x += red_k[w][t - DK];
                dkp[out * DK + t - DK] = x;
                if (t == DK) {
                    float e = 0.0f;
#pragma unroll
                    for (int w = 0; w < WARPS; ++w) e += red_b[w];
                    dbeta[out] = e;
                }
            }
        }
    }
}

// Block (row, key head): the value heads' dq and dk shares summed in head and block order, then each L2 norm undone into dact's
// q and k columns: dx = sc r (d - r^2 x (x . d)), r = 1 / sqrt(|x|^2 + 1e-6), sc = DK^-0.5 for q and 1 for k.
extern "C" __global__ void __launch_bounds__(128) tf_gdn_qk_back(const bf16* __restrict__ act, const float* __restrict__ dqp,
                                                                 const float* __restrict__ dkp, float* __restrict__ dact,
                                                                 GdnShape s) {
    __shared__ float scratch[32];
    const int r = blockIdx.x, hk = blockIdx.y, t = threadIdx.x, per = s.nv / s.nk;
    for (int which = 0; which < 2; ++which) {
        const float* part = which == 0 ? dqp : dkp;
        float d = 0.0f;
        for (int h = hk * per; h < (hk + 1) * per; ++h)
            for (int p = 0; p < SPLIT; ++p) d += part[(((size_t)r * s.nv + h) * SPLIT + p) * DK + t];
        const int c = which * s.nk * DK + hk * DK + t;
        const float x = f32(act[(size_t)r * s.conv + c]);
        const float ss = block_sum(x * x, scratch, 4);
        const float dot = block_sum(x * d, scratch, 4);
        const float inv = 1.0f / sqrtf(ss + 1e-6f), sc = which == 0 ? 1.0f / sqrtf((float)DK) : 1.0f;
        dact[(size_t)r * s.conv + c] = sc * inv * (d - inv * inv * x * dot);
    }
}

// Block r, thread h: dg and dbeta (the chain blocks' shares summed) back to dproj's a and b columns (beta's bf16 rounding an identity, softplus's slope 1
// past 20 as the forward's switch).
extern "C" __global__ void tf_gdn_gates_back(const bf16* __restrict__ P, const float* __restrict__ a_log,
                                             const float* __restrict__ dt_bias, const float* __restrict__ g,
                                             const float* __restrict__ dg, const float* __restrict__ dbeta,
                                             float* __restrict__ dproj, GdnShape s) {
    const int h = threadIdx.x, r = blockIdx.x;
    if (h >= s.nv || r >= s.rows) return;
    const size_t at = (size_t)r * s.proj + s.conv + s.nv * DV;
    const float b = f32(P[at + h]), x = f32(P[at + s.nv + h]) + dt_bias[h];
    const float sb = sigmoidf_(b), sx = x > 20.0f ? 1.0f : sigmoidf_(x);
    float db = 0.0f, dgs = 0.0f;
    for (int p = 0; p < SPLIT; ++p) {
        db += dbeta[(r * s.nv + h) * SPLIT + p];
        dgs += dg[(r * s.nv + h) * SPLIT + p];
    }
    dproj[at + h] = db * sb * (1.0f - sb);
    dproj[at + s.nv + h] = -dgs * g[r * s.nv + h] * expf(a_log[h]) * sx;
}

// The depthwise causal conv and its SiLU undone into dproj's qkv columns (the pre-activations recomputed from P and
// hist); rows past the pass and the history get nothing. Grid (conv / 256, rows).
extern "C" __global__ void tf_gdn_conv_back(const bf16* __restrict__ P, const bf16* __restrict__ hist,
                                            const bf16* __restrict__ cw, const float* __restrict__ dact,
                                            float* __restrict__ dproj, GdnShape s) {
    const int c = blockIdx.x * 256 + threadIdx.x, r = blockIdx.y;
    if (c >= s.conv || r >= s.rows) return;
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + TAPS - 1 - tap;
        if (at >= s.rows) continue;
        const float pre = conv_acc(P, hist, cw, at, c, s);
        acc += f32(cw[c * TAPS + tap]) * dact[(size_t)at * s.conv + c] * silu_grad(pre);
    }
    dproj[(size_t)r * s.proj + c] = acc;
}

// A 4-bit [N, K] MLX affine matrix (8 codes a word, a bf16 scale and bias every `group`) as bf16 [N, K], in order:
// train.cu's tf_train_dequant for any group size (Flash Next's checkpoints use 32).
extern "C" __global__ void tf_gdn_dequant(const unsigned* __restrict__ w, const bf16* __restrict__ scales,
                                          const bf16* __restrict__ biases, bf16* __restrict__ out, long long count, int group) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned qv = (w[i / 8] >> (4 * (i % 8))) & 0xfu;
    out[i] = b16(f32(scales[i / group]) * (float)qv + f32(biases[i / group]));
}
