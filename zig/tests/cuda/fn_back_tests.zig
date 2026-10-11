//! Flash Next's input-gradient kernels (flashnext_train_mix.cu, nemotron.fn_back) at the model's shapes against float64
//! references, each reference first checked against central differences of a float64 forward: the hyper-connections
//! (`fn-hc-back`), the MoE layer with its shared expert and pocket (`fn-moe-back`), the gated attention
//! (`fn-attn-back`) and the loss through the 4-bit head into the final mixer (`fn-head-back`), at 64 and 256 rows.
//! Every gradient reports its worst column (the largest difference in a column over that column's largest reference
//! value) and its relative L2 error; synthetic weights, a few GB of GPU memory at most.
const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const fb = nemotron.fn_back;
const Ops = fb.Ops;

const D = 2560;
const S = 4;
const WIDE = S * D;
const LOW = 320;
const NV = LOW + S;
const eps: f64 = 1e-6;
const threads = 16;
const row_counts = [_]usize{ 64, 256 };

// the bf16 products' tolerance: the worst column within 3% of its largest value, 1% in L2
const tol_col = 3e-2;
const tol_l2 = 1e-2;
// a float64 reference against central differences
const tol_fd = 1e-5;

// ---- numbers ----------------------------------------------------------------------------------------------------------

fn bf16(x: f64) u16 {
    const f: f32 = @floatCast(x);
    const v: u32 = @bitCast(f);
    return @truncate((v + 0x7fff + ((v >> 16) & 1)) >> 16);
}

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// x rounded to bf16, or to fp32.
fn rb(x: f64) f64 {
    return bf(bf16(x));
}

fn rf(x: f64) f64 {
    return @as(f32, @floatCast(x));
}

fn sig(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

fn silu(x: f64) f64 {
    return x * sig(x);
}

fn siluGrad(x: f64) f64 {
    const s = sig(x);
    return s * (1 + x * (1 - s));
}

fn dot(a: []const f64, b: []const f64) f64 {
    const Vec = @Vector(8, f64);
    var acc: Vec = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) acc += @as(Vec, a[i..][0..8].*) * @as(Vec, b[i..][0..8].*);
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

fn axpy(y: []f64, c: f64, x: []const f64) void {
    const Vec = @Vector(8, f64);
    const cv: Vec = @splat(c);
    var i: usize = 0;
    while (i + 8 <= y.len) : (i += 8) y[i..][0..8].* = @as(Vec, y[i..][0..8].*) + cv * @as(Vec, x[i..][0..8].*);
    while (i < y.len) : (i += 1) y[i] += c * x[i];
}

fn dotF(a: []const f32, b: []const f64) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// A 4-bit [n, k] matrix in MLX's layout on the host: 8 codes a word, a bf16 scale and bias every 32 of a row.
const HQ4 = struct {
    n: usize,
    k: usize,
    words: []u32,
    sc: []u16,
    bi: []u16,

    /// Row i . x.
    fn rowDot(m: *const HQ4, i: usize, x: []const f64) f64 {
        const g = m.k / 32;
        const words = m.words[i * (m.k / 8) ..][0 .. m.k / 8];
        var total: f64 = 0;
        for (0..g) |gi| {
            var sq: f64 = 0;
            var sx: f64 = 0;
            for (0..4) |wi| {
                const w = words[gi * 4 + wi];
                const xs = x[gi * 32 + wi * 8 ..][0..8];
                inline for (0..8) |j| {
                    const q: f64 = @floatFromInt((w >> (4 * j)) & 0xf);
                    sq += q * xs[j];
                    sx += xs[j];
                }
            }
            total += bf(m.sc[i * g + gi]) * sq + bf(m.bi[i * g + gi]) * sx;
        }
        return total;
    }

    /// y += c row i.
    fn rowAxpy(m: *const HQ4, i: usize, c: f64, y: []f64) void {
        const g = m.k / 32;
        const words = m.words[i * (m.k / 8) ..][0 .. m.k / 8];
        for (0..g) |gi| {
            const cs = c * bf(m.sc[i * g + gi]);
            const cb = c * bf(m.bi[i * g + gi]);
            for (0..4) |wi| {
                const w = words[gi * 4 + wi];
                const ys = y[gi * 32 + wi * 8 ..][0..8];
                inline for (0..8) |j| ys[j] += cs * @as(f64, @floatFromInt((w >> (4 * j)) & 0xf)) + cb;
            }
        }
    }

    fn row(m: *const HQ4, i: usize, out: []f64) void {
        @memset(out, 0);
        m.rowAxpy(i, 1, out);
    }

    fn dev(m: *const HQ4, r: *Rig) !fb.Q4 {
        return .{ .w = try r.dev(u32, m.words), .s = try r.dev(u16, m.sc), .b = try r.dev(u16, m.bi), .n = m.n, .k = m.k };
    }

    fn bytes(m: *const HQ4) usize {
        return m.words.len * 4 + m.sc.len * 4;
    }
};

// ---- the rig ----------------------------------------------------------------------------------------------------------

const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0, // device bytes allocated, for the memory lines
    bad: usize = 0,
    checks: usize = 0,

    fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        r.bytes += b.len;
        return b.ptr;
    }

    fn zeros(r: *Rig, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(n, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, r.s.handle);
        r.bytes += b.len;
        return b.ptr;
    }

    fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    fn release(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
        r.bytes = 0;
    }

    fn uni(r: *Rig, lo: f64, hi: f64) f64 {
        return lo + r.rng.float(f64) * (hi - lo);
    }

    /// A random 4-bit matrix whose values lie in about [-amp, amp]: codes uniform, each group's scale and bias jittered.
    fn q4(r: *Rig, n: usize, k: usize, amp: f64) !HQ4 {
        const m: HQ4 = .{ .n = n, .k = k, .words = try r.a.alloc(u32, n * k / 8), .sc = try r.a.alloc(u16, n * k / 32), .bi = try r.a.alloc(u16, n * k / 32) };
        r.rng.bytes(std.mem.sliceAsBytes(m.words));
        for (m.sc, m.bi) |*s, *b| {
            const sv = 2 * amp / 15 * r.uni(0.6, 1.4);
            s.* = bf16(sv);
            b.* = bf16(-7.5 * sv * r.uni(0.85, 1.15));
        }
        return m;
    }

    fn f64s(r: *Rig, n: usize, lo: f64, hi: f64, round: enum { none, bf, f }) ![]f64 {
        const out = try r.a.alloc(f64, n);
        for (out) |*v| {
            const x = r.uni(lo, hi);
            v.* = switch (round) {
                .none => x,
                .bf => rb(x),
                .f => rf(x),
            };
        }
        return out;
    }

    fn f32s(r: *Rig, n: usize, lo: f64, hi: f64) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = @floatCast(r.uni(lo, hi));
        return out;
    }

    /// `got` against `want` [rows, cols]: the worst column's largest difference over its largest reference value (a
    /// column's scale floored at 1e-3 of the tensor's) and the relative L2 error, within tolerance.
    fn compare(r: *Rig, name: []const u8, got: []const f32, want: []const f64, cols: usize, tc: f64, tl: f64) void {
        const rows = want.len / cols;
        var top: f64 = 0;
        for (want) |w| top = @max(top, @abs(w));
        var worst: f64 = 0;
        var at: usize = 0;
        var dd: f64 = 0;
        var ww: f64 = 0;
        for (0..cols) |c| {
            var num: f64 = 0;
            var den: f64 = 0;
            for (0..rows) |i| {
                const w = want[i * cols + c];
                const diff = @as(f64, got[i * cols + c]) - w;
                num = if (std.math.isNan(diff)) std.math.inf(f64) else @max(num, @abs(diff));
                den = @max(den, @abs(w));
                dd += diff * diff;
                ww += w * w;
            }
            const e = num / @max(den, 1e-3 * top, 1e-300);
            if (e > worst) {
                worst = e;
                at = c;
            }
        }
        const l2 = @sqrt(dd / @max(ww, 1e-300));
        const ok = worst <= tc and l2 <= tl;
        r.checks += 1;
        if (!ok) r.bad += 1;
        std.debug.print("RESULT {s}: worst column {e:.2} (column {d} of {d}), relative L2 {e:.2}, largest |ref| {e:.2}{s}\n", .{ name, worst, at, cols, l2, top, if (ok) "" else "  <- FAIL" });
    }

    /// Central differences against a reference gradient's coordinates: each difference over max(|ref|, 5% of the
    /// gradient's RMS `scale`).
    fn fd(r: *Rig, name: []const u8, num: []const f64, ref: []const f64, scale: f64) void {
        var worst: f64 = 0;
        for (num, ref) |n, g| {
            const e = @abs(n - g) / @max(@abs(g), 0.05 * scale, 1e-300);
            worst = if (std.math.isNan(e)) std.math.inf(f64) else @max(worst, e);
        }
        const ok = worst <= tol_fd;
        r.checks += 1;
        if (!ok) r.bad += 1;
        std.debug.print("RESULT fd {s}: float64 reference against central differences at {d} coordinates, worst relative {e:.2}{s}\n", .{ name, num.len, worst, if (ok) "" else "  <- FAIL" });
    }
};

fn rms(xs: []const f64) f64 {
    return @sqrt(dot(xs, xs) / @as(f64, @floatFromInt(@max(xs.len, 1))));
}

fn toF32(a: std.mem.Allocator, xs: []const f64) ![]f32 {
    const out = try a.alloc(f32, xs.len);
    for (out, xs) |*o, x| o.* = @floatCast(x);
    return out;
}

fn toBf16(a: std.mem.Allocator, xs: []const f64) ![]u16 {
    const out = try a.alloc(u16, xs.len);
    for (out, xs) |*o, x| o.* = bf16(x);
    return out;
}

/// f(ctx, i, thread) for i below n, on `threads` threads (inline if one cannot start).
fn parallel(n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
    const C = @TypeOf(ctx);
    const Run = struct {
        fn run(c: C, first: usize, count: usize) void {
            var i = first;
            while (i < count) : (i += threads) f(c, i, first);
        }
    };
    var ts: [threads]?std.Thread = @splat(null);
    for (&ts, 0..) |*t, i| t.* = std.Thread.spawn(.{}, Run.run, .{ ctx, i, n }) catch null;
    for (ts, 0..) |t, i| if (t) |th| th.join() else Run.run(ctx, i, n);
}

/// GPU milliseconds between two events on the rig's stream.
const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    fn init(r: *Rig) !Timer {
        return .{ .a = try cuda.Event.init(r.gpu.d, true), .b = try cuda.Event.init(r.gpu.d, true) };
    }
    fn deinit(t: *Timer) void {
        t.a.deinit();
        t.b.deinit();
    }
    fn start(t: *Timer, r: *Rig) !void {
        try t.a.record(r.s);
    }
    fn stop(t: *Timer, r: *Rig) !f32 {
        try t.b.record(r.s);
        try t.b.synchronize();
        return cuda.Event.elapsedMs(t.a, t.b);
    }
};

fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}

// ---- the hyper-connections -------------------------------------------------------------------------------------------

const Hc = struct {
    down: HQ4, // [LOW, WIDE]
    inj: ?HQ4, // [S, WIDE]
    up: HQ4, // [WIDE, LOW]
    scale: []f32, // [WIDE] 1 + w

    fn nv(h: *const Hc) usize {
        return LOW + if (h.inj != null) @as(usize, S) else 0;
    }
};

fn makeHc(r: *Rig, inject: bool) !Hc {
    const scale = try r.a.alloc(f32, WIDE);
    for (scale) |*v| v.* = @floatCast(1 + r.uni(-0.3, 0.3));
    return .{
        .down = try r.q4(LOW, WIDE, 4 * @sqrt(3.0 / @as(f64, WIDE))),
        .inj = if (inject) try r.q4(S, WIDE, 4 * @sqrt(3.0 / @as(f64, WIDE))) else null,
        .up = try r.q4(WIDE, LOW, 3 * @sqrt(3.0 / @as(f64, LOW))),
        .scale = scale,
    };
}

/// One row's read-out as glue.py computes it (rounded where it stores bf16 or fp32 when `round`).
fn hcForward(hc: *const Hc, h: []const f64, round: bool, normed: []f64, v: []f64, u: []f64, inj: []f64, mixed: []f64) void {
    for (0..S) |s| {
        const hs = h[s * D ..][0..D];
        const rinv = 1 / @sqrt(dot(hs, hs) / D + eps);
        for (0..D) |i| {
            const x = hs[i] * rinv * hc.scale[s * D + i];
            normed[s * D + i] = if (round) rb(x) else x;
        }
    }
    var act: [LOW]f64 = undefined;
    for (0..LOW) |i| {
        const x = hc.down.rowDot(i, normed);
        v[i] = if (round) rf(x) else x;
        act[i] = if (round) rb(silu(rb(v[i] / S))) else silu(v[i] / S);
    }
    if (hc.inj) |m| for (0..S) |s| {
        const x = m.rowDot(s, normed);
        v[LOW + s] = if (round) rf(x) else x;
        inj[s] = if (round) rb(2 * rb(sig(rb(v[LOW + s] / S)))) else 2 * sig(v[LOW + s] / S);
    };
    for (0..WIDE) |j| {
        const x = hc.up.rowDot(j, &act);
        u[j] = if (round) rb(x) else x;
    }
    for (0..D) |i| {
        var t: f64 = 0;
        for (0..S) |s| t += if (round) rb(rb(sig(u[s * D + i])) * normed[s * D + i]) else sig(u[s * D + i]) * normed[s * D + i];
        mixed[i] = if (round) rb(t / S) else t / S;
    }
}

/// The reference: dL/dh from G = dL/dh' (the write-back's output, null on the final mixer), the branch b and gm =
/// dL/dmixed; db and dinj when the block injects.
fn hcBack(hc: *const Hc, h: []const f64, normed: []const f64, v: []const f64, u: []const f64, inj: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64, db: []f64, dinj: []f64, dh: []f64) void {
    var dij: [S]f64 = @splat(0);
    if (G) |g| {
        for (0..D) |d| {
            var t: f64 = 0;
            for (0..S) |s| t += g[s * D + d] * inj[s];
            db[d] = t;
        }
        for (0..S) |s| {
            dij[s] = dot(g[s * D ..][0..D], b);
            dinj[s] = dij[s];
        }
    }
    var dn: [WIDE]f64 = undefined;
    var du: [WIDE]f64 = undefined;
    for (0..S) |s| for (0..D) |d| {
        const sg = sig(u[s * D + d]);
        dn[s * D + d] = gm[d] * sg / S;
        du[s * D + d] = gm[d] * normed[s * D + d] * sg * (1 - sg) / S;
    };
    var dact: [LOW]f64 = @splat(0);
    for (0..WIDE) |j| hc.up.rowAxpy(j, du[j], &dact);
    for (0..LOW) |i| hc.down.rowAxpy(i, dact[i] * siluGrad(v[i] / S) / S, &dn);
    if (hc.inj) |m| for (0..S) |s| {
        const sg = sig(v[LOW + s] / S);
        m.rowAxpy(s, dij[s] * 2 * sg * (1 - sg) / S, &dn);
    };
    for (0..S) |s| {
        const hs = h[s * D ..][0..D];
        const rinv = 1 / @sqrt(dot(hs, hs) / D + eps);
        var t: f64 = 0;
        for (0..D) |i| t += hc.scale[s * D + i] * dn[s * D + i] * hs[i];
        for (0..D) |i| {
            const base = if (G) |g| g[s * D + i] else 0;
            dh[s * D + i] = base + rinv * hc.scale[s * D + i] * dn[s * D + i] - rinv * rinv * rinv * hs[i] * t / D;
        }
    }
}

/// L = G . (h + b inject) + gm . mixed for one row, a float64 forward without rounding.
fn hcLoss(hc: *const Hc, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) f64 {
    const normed = std.heap.page_allocator.alloc(f64, 2 * WIDE + NV + S + D) catch unreachable;
    defer std.heap.page_allocator.free(normed);
    const u = normed[WIDE..][0..WIDE];
    const v = normed[2 * WIDE ..][0..NV];
    const inj = normed[2 * WIDE + NV ..][0..S];
    const mixed = normed[2 * WIDE + NV + S ..][0..D];
    hcForward(hc, h, false, normed[0..WIDE], v, u, inj, mixed);
    var l = dot(gm, mixed);
    if (G) |g| for (0..S) |s| for (0..D) |d| {
        l += g[s * D + d] * (h[s * D + d] + b[d] * inj[s]);
    };
    return l;
}

/// Kept tensors and reference gradients of `rows` rows (inputs h, b, G, gm).
const HcCase = struct {
    hc: *const Hc,
    rows: usize,
    round: bool,
    h: []const f64,
    b: []const f64,
    G: ?[]const f64,
    gm: []const f64,
    normed: []f64,
    v: []f64,
    u: []f64,
    inj: []f64,
    mixed: []f64,
    db: []f64,
    dinj: []f64,
    dh: []f64,

    fn init(a: std.mem.Allocator, hc: *const Hc, rows: usize, round: bool, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) !HcCase {
        var c: HcCase = .{ .hc = hc, .rows = rows, .round = round, .h = h, .b = b, .G = G, .gm = gm, .normed = try a.alloc(f64, rows * WIDE), .v = try a.alloc(f64, rows * NV), .u = try a.alloc(f64, rows * WIDE), .inj = try a.alloc(f64, rows * S), .mixed = try a.alloc(f64, rows * D), .db = try a.alloc(f64, rows * D), .dinj = try a.alloc(f64, rows * S), .dh = try a.alloc(f64, rows * WIDE) };
        @memset(c.inj, 0);
        @memset(c.db, 0);
        @memset(c.dinj, 0);
        parallel(rows, @as(*const HcCase, &c), row);
        return c;
    }

    fn row(c: *const HcCase, r: usize, _: usize) void {
        const nv = c.hc.nv();
        const h = c.h[r * WIDE ..][0..WIDE];
        hcForward(c.hc, h, c.round, c.normed[r * WIDE ..][0..WIDE], c.v[r * NV ..][0..nv], c.u[r * WIDE ..][0..WIDE], c.inj[r * S ..][0..S], c.mixed[r * D ..][0..D]);
        hcBack(c.hc, h, c.normed[r * WIDE ..][0..WIDE], c.v[r * NV ..][0..nv], c.u[r * WIDE ..][0..WIDE], c.inj[r * S ..][0..S], c.b[r * D ..][0..D], if (c.G) |g| g[r * WIDE ..][0..WIDE] else null, c.gm[r * D ..][0..D], c.db[r * D ..][0..D], c.dinj[r * S ..][0..S], c.dh[r * WIDE ..][0..WIDE]);
    }
};

/// The reference's gradients in h and b at a few coordinates of two rows, against central differences.
fn hcFd(r: *Rig, name: []const u8, hc: *const Hc, h: []const f64, b: []const f64, G: ?[]const f64, gm: []const f64) !void {
    const rows = 64;
    const c = try HcCase.init(r.a, hc, rows, false, h[0 .. rows * WIDE], b[0 .. rows * D], if (G) |g| g[0 .. rows * WIDE] else null, gm[0 .. rows * D]);
    var num: std.ArrayList(f64) = .empty;
    var ref: std.ArrayList(f64) = .empty;
    var numb: std.ArrayList(f64) = .empty;
    var refb: std.ArrayList(f64) = .empty;
    const step = 1e-4;
    for ([_]usize{ 0, 37 }) |row| {
        const hr = try r.a.dupe(f64, h[row * WIDE ..][0..WIDE]);
        const br = try r.a.dupe(f64, b[row * D ..][0..D]);
        const gr: ?[]const f64 = if (G) |g| g[row * WIDE ..][0..WIDE] else null;
        const gmr = gm[row * D ..][0..D];
        for ([_]usize{ 5, D + 1000, 2 * D + 77, 3 * D + 2559, 2 * D + 1500 }) |i| {
            const x = hr[i];
            hr[i] = x + step;
            const lp = hcLoss(hc, hr, br, gr, gmr);
            hr[i] = x - step;
            const lm = hcLoss(hc, hr, br, gr, gmr);
            hr[i] = x;
            try num.append(r.a, (lp - lm) / (2 * step));
            try ref.append(r.a, c.dh[row * WIDE + i]);
        }
        if (G != null) for ([_]usize{ 3, 1290, 2500 }) |i| {
            const x = br[i];
            br[i] = x + step;
            const lp = hcLoss(hc, hr, br, gr, gmr);
            br[i] = x - step;
            const lm = hcLoss(hc, hr, br, gr, gmr);
            br[i] = x;
            try numb.append(r.a, (lp - lm) / (2 * step));
            try refb.append(r.a, c.db[row * D + i]);
        };
    }
    var buf: [96]u8 = undefined;
    r.fd(try std.fmt.bufPrint(&buf, "{s} dh", .{name}), num.items, ref.items, rms(c.dh));
    if (G != null) r.fd(try std.fmt.bufPrint(&buf, "{s} dbranch", .{name}), numb.items, refb.items, rms(c.db));
}

fn hcKept(r: *Rig, c: *const HcCase, rows: usize) !Ops.HcKept {
    const nv = c.hc.nv();
    const v = try r.a.alloc(f32, rows * nv);
    for (0..rows) |i| for (0..nv) |j| {
        v[i * nv + j] = @floatCast(c.v[i * NV + j]);
    };
    return .{
        .h = try r.dev(u16, try toBf16(r.a, c.h[0 .. rows * WIDE])),
        .normed = try r.dev(u16, try toBf16(r.a, c.normed[0 .. rows * WIDE])),
        .v = try r.dev(f32, v),
        .up = try r.dev(u16, try toBf16(r.a, c.u[0 .. rows * WIDE])),
        .inject = try r.dev(u16, try toBf16(r.a, c.inj[0 .. rows * S])),
    };
}

fn hcDev(r: *Rig, hc: *const Hc) !Ops.Hc {
    return .{ .down = try hc.down.dev(r), .inject = if (hc.inj) |*m| try m.dev(r) else null, .up = try hc.up.dev(r), .scale = try r.dev(f32, hc.scale), .streams = S, .d = D, .low = LOW, .eps = @floatCast(eps) };
}

fn hcScratch(r: *Rig, rows: usize) !Ops.HcScratch {
    return .{ .dn = try r.zeros(rows * WIDE * 4), .du = try r.zeros(rows * WIDE * 4), .dact = try r.zeros(rows * LOW * 4), .dv = try r.zeros(rows * NV * 4) };
}

/// The GPU read-out backward (and write-back's, when the block injects) on `rows` rows against the reference case.
fn hcGpu(r: *Rig, o: Ops, name: []const u8, dhc: Ops.Hc, c: *const HcCase, rows: usize) !void {
    const kept = try hcKept(r, c, rows);
    const before = r.bytes;
    const g = if (c.G) |gg| try r.dev(f32, try toF32(r.a, gg[0 .. rows * WIDE])) else try r.zeros(rows * WIDE * 4);
    const gm = try r.dev(f32, try toF32(r.a, c.gm[0 .. rows * D]));
    const db = try r.zeros(rows * D * 4);
    const dinj = try r.zeros(rows * S * 4);
    const t = try hcScratch(r, rows);
    const scratch = r.bytes - before;
    const branch = try r.dev(u16, try toBf16(r.a, c.b[0 .. rows * D]));
    var tm = try Timer.init(r);
    defer tm.deinit();
    try tm.start(r);
    if (c.G != null) try o.hcBranchBack(dhc, g, kept.inject, branch, db, dinj, rows);
    try o.hcReadoutBack(dhc, kept, gm, if (c.G != null) dinj else 0, g, t, rows);
    const ms = try tm.stop(r);
    var buf: [128]u8 = undefined;
    if (c.G != null) {
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dbranch", .{ name, rows }), try r.back(f32, db, rows * D), c.db[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dinject", .{ name, rows }), try r.back(f32, dinj, rows * S), c.dinj[0 .. rows * S], S, tol_col, tol_l2);
    }
    const got = try r.back(f32, g, rows * WIDE);
    r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dh", .{ name, rows }), got, c.dh[0 .. rows * WIDE], WIDE, tol_col, tol_l2);
    if (c.G) |gg| {
        // the read-out's own part, the residual's gradient (exact) taken out
        const want = try r.a.alloc(f64, rows * WIDE);
        for (got, want, 0..) |*x, *w, i| {
            x.* = @floatCast(@as(f64, x.*) - rf(gg[i]));
            w.* = c.dh[i] - rf(gg[i]);
        }
        r.compare(try std.fmt.bufPrint(&buf, "{s} {d} rows dh less the residual", .{ name, rows }), got, want, WIDE, tol_col, tol_l2);
    }
    std.debug.print("RESULT {s} {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ name, rows, ms, mib(scratch) });
}

fn hcTest(r: *Rig, o: Ops) !void {
    const block = try makeHc(r, true);
    const mixer = try makeHc(r, false);
    const R = row_counts[row_counts.len - 1];
    const h = try r.a.alloc(f64, R * WIDE);
    for (0..R) |i| for (0..S) |s| for (0..D) |d| {
        h[(i * S + s) * D + d] = rb(r.uni(-1, 1) * @as(f64, @floatFromInt(@as(usize, 1) << @intCast(s))) * 0.5);
    };
    const b = try r.f64s(R * D, -1, 1, .bf);
    const G = try r.f64s(R * WIDE, -1, 1, .f);
    const gm = try r.f64s(R * D, -1, 1, .f);
    try hcFd(r, "hc block", &block, h, b, G, gm);
    try hcFd(r, "hc final mixer", &mixer, h, b, null, gm);
    const cb = try HcCase.init(r.a, &block, R, true, h, b, G, gm);
    const cm = try HcCase.init(r.a, &mixer, R, true, h, b, null, gm);
    const db = try hcDev(r, &block);
    const dm = try hcDev(r, &mixer);
    std.debug.print("RESULT hc weights: {d:.1} MiB a read-out\n", .{mib(block.down.bytes() + block.inj.?.bytes() + block.up.bytes() + WIDE * 4)});
    for (row_counts) |rows| {
        try hcGpu(r, o, "hc block", db, &cb, rows);
        try hcGpu(r, o, "hc final mixer", dm, &cm, rows);
    }
}

// ---- the MoE layer ---------------------------------------------------------------------------------------------------

const E = 512;
const W = 640;
const TOPK = 10;
const SLOTS = TOPK + 1;
const P = 16;

const Moe = struct {
    router: []u16, // [E + 1, D] bf16: the routed experts' rows, then the shared expert's gate
    gate: HQ4, // [E W, D]
    up: HQ4, // [E W, D]
    down: HQ4, // [E D, W]
    sgate: HQ4, // [W, D]
    sup: HQ4,
    sdown: HQ4, // [D, W]
    pg: []f32, // the pocket [P, D]
    pu: []f32,
    pd: []f32,

    const Mats = struct { g: *const HQ4, u: *const HQ4, d: *const HQ4, go: usize, do: usize };

    fn expert(m: *const Moe, e: usize) Mats {
        if (e == E) return .{ .g = &m.sgate, .u = &m.sup, .d = &m.sdown, .go = 0, .do = 0 };
        return .{ .g = &m.gate, .u = &m.up, .d = &m.down, .go = e * W, .do = e * D };
    }

    fn logit(m: *const Moe, e: usize, x: []const f64) f64 {
        var s: f64 = 0;
        for (0..D) |i| s += bf(m.router[e * D + i]) * x[i];
        return s;
    }
};

/// One row's slot weights as moe.py's _topk_rows makes them: top_k by logit (lower id on ties), softmax over them, then
/// the shared gate sigmoid(logit E); rounded to bf16 when `round`.
fn route(m: *const Moe, x: []const f64, round: bool, ids: []i32, wts: []f64) void {
    var l: [E]f64 = undefined;
    for (0..E) |e| l[e] = m.logit(e, x);
    var taken: [E]bool = @splat(false);
    for (0..TOPK) |k| {
        var best: usize = E;
        for (0..E) |e| if (!taken[e] and (best == E or l[e] > l[best])) {
            best = e;
        };
        taken[best] = true;
        ids[k] = @intCast(best);
    }
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[@intCast(ids[k])] - l[@intCast(ids[0])]);
    for (0..TOPK) |k| {
        const w = @exp(l[@intCast(ids[k])] - l[@intCast(ids[0])]) / total;
        wts[k] = if (round) rb(w) else w;
    }
    ids[TOPK] = E;
    const ls = m.logit(E, x);
    wts[TOPK] = if (round) rb(sig(rb(ls))) else sig(ls);
}

/// The reference on one row: dx from g = dL/d the MoE output (the slots' weights wts as kept), each slot's a = dL/dw,
/// the logits' gradients and the pocket's forward output.
fn moeBack(m: *const Moe, x: []const f64, g: []const f64, ids: []const i32, wts: []const f64, dx: []f64, a: []f64, dlog: []f64, pout: []f64) void {
    @memset(dx, 0);
    var gv: [W]f64 = undefined;
    var uv: [W]f64 = undefined;
    var dact: [W]f64 = undefined;
    for (0..SLOTS) |k| {
        const mt = m.expert(@intCast(ids[k]));
        for (0..W) |i| {
            gv[i] = mt.g.rowDot(mt.go + i, x);
            uv[i] = mt.u.rowDot(mt.go + i, x);
        }
        @memset(&dact, 0);
        for (0..D) |d| mt.d.rowAxpy(mt.do + d, g[d], &dact);
        var s: f64 = 0;
        for (0..W) |i| {
            s += dact[i] * silu(gv[i]) * uv[i];
            mt.g.rowAxpy(mt.go + i, wts[k] * dact[i] * uv[i] * siluGrad(gv[i]), dx);
            mt.u.rowAxpy(mt.go + i, wts[k] * dact[i] * silu(gv[i]), dx);
        }
        a[k] = s;
    }
    const sg = wts[TOPK];
    @memset(pout, 0);
    for (0..P) |j| {
        const zg = dotF(m.pg[j * D ..][0..D], x);
        const zu = dotF(m.pu[j * D ..][0..D], x);
        const e = dotF(m.pd[j * D ..][0..D], g);
        const cg = sg * e * siluGrad(zg) * zu;
        const cu = sg * e * silu(zg);
        for (0..D) |d| {
            dx[d] += cg * m.pg[j * D + d] + cu * m.pu[j * D + d];
            pout[d] += sg * silu(zg) * zu * m.pd[j * D + d];
        }
        a[TOPK] += silu(zg) * zu * e;
    }
    var l: [SLOTS]f64 = undefined;
    for (0..SLOTS) |k| l[k] = m.logit(@intCast(ids[k]), x);
    var top: f64 = -std.math.inf(f64);
    for (0..TOPK) |k| top = @max(top, l[k]);
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[k] - top);
    var mix: f64 = 0;
    for (0..TOPK) |k| mix += @exp(l[k] - top) / total * a[k];
    for (0..TOPK) |k| dlog[k] = @exp(l[k] - top) / total * (a[k] - mix);
    const ss = sig(l[TOPK]);
    dlog[TOPK] = a[TOPK] * ss * (1 - ss);
    for (0..SLOTS) |k| {
        const e: usize = @intCast(ids[k]);
        for (0..D) |i| dx[i] += dlog[k] * bf(m.router[e * D + i]);
    }
}

/// L = g . the MoE output on one row (its experts fixed by ids), a float64 forward without rounding.
fn moeLoss(m: *const Moe, x: []const f64, g: []const f64, ids: []const i32) f64 {
    var l: [SLOTS]f64 = undefined;
    for (0..SLOTS) |k| l[k] = m.logit(@intCast(ids[k]), x);
    var top: f64 = -std.math.inf(f64);
    for (0..TOPK) |k| top = @max(top, l[k]);
    var total: f64 = 0;
    for (0..TOPK) |k| total += @exp(l[k] - top);
    const sg = sig(l[TOPK]);
    var loss: f64 = 0;
    var act: [W]f64 = undefined;
    for (0..SLOTS) |k| {
        const mt = m.expert(@intCast(ids[k]));
        for (0..W) |i| act[i] = silu(mt.g.rowDot(mt.go + i, x)) * mt.u.rowDot(mt.go + i, x);
        var y: f64 = 0;
        for (0..D) |d| y += g[d] * mt.d.rowDot(mt.do + d, &act);
        loss += y * if (k < TOPK) @exp(l[k] - top) / total else sg;
    }
    for (0..P) |j| loss += sg * silu(dotF(m.pg[j * D ..][0..D], x)) * dotF(m.pu[j * D ..][0..D], x) * dotF(m.pd[j * D ..][0..D], g);
    return loss;
}

const MoeCase = struct {
    m: *const Moe,
    x: []const f64,
    g: []const f64,
    ids: []const i32,
    wts: []const f64,
    dx: []f64,
    a: []f64,
    dlog: []f64,
    pout: []f64,

    fn init(al: std.mem.Allocator, m: *const Moe, rows: usize, x: []const f64, g: []const f64, ids: []const i32, wts: []const f64) !MoeCase {
        var c: MoeCase = .{ .m = m, .x = x, .g = g, .ids = ids, .wts = wts, .dx = try al.alloc(f64, rows * D), .a = try al.alloc(f64, rows * SLOTS), .dlog = try al.alloc(f64, rows * SLOTS), .pout = try al.alloc(f64, rows * D) };
        parallel(rows, @as(*const MoeCase, &c), row);
        return c;
    }

    fn row(c: *const MoeCase, r: usize, _: usize) void {
        moeBack(c.m, c.x[r * D ..][0..D], c.g[r * D ..][0..D], c.ids[r * SLOTS ..][0..SLOTS], c.wts[r * SLOTS ..][0..SLOTS], c.dx[r * D ..][0..D], c.a[r * SLOTS ..][0..SLOTS], c.dlog[r * SLOTS ..][0..SLOTS], c.pout[r * D ..][0..D]);
    }
};

const RouteCtx = struct { m: *const Moe, x: []const f64, round: bool, ids: []i32, wts: []f64 };
fn routeRow(c: *const RouteCtx, r: usize, _: usize) void {
    route(c.m, c.x[r * D ..][0..D], c.round, c.ids[r * SLOTS ..][0..SLOTS], c.wts[r * SLOTS ..][0..SLOTS]);
}

fn moeTest(r: *Rig, o: Ops) !void {
    const R = row_counts[row_counts.len - 1];
    const amp_in = @sqrt(3.0 / @as(f64, D));
    var m: Moe = .{
        .router = try r.a.alloc(u16, (E + 1) * D),
        .gate = try r.q4(E * W, D, 2 * amp_in),
        .up = try r.q4(E * W, D, 2 * amp_in),
        .down = try r.q4(E * D, W, @sqrt(3.0 / @as(f64, W))),
        .sgate = try r.q4(W, D, 2 * amp_in),
        .sup = try r.q4(W, D, 2 * amp_in),
        .sdown = try r.q4(D, W, @sqrt(3.0 / @as(f64, W))),
        .pg = try r.f32s(P * D, -3 * amp_in, 3 * amp_in),
        .pu = try r.f32s(P * D, -3 * amp_in, 3 * amp_in),
        .pd = try r.f32s(P * D, -0.05, 0.05),
    };
    for (m.router) |*v| v.* = bf16(r.uni(-6, 6) / @sqrt(@as(f64, D)));
    const x = try r.f64s(R * D, -1, 1, .bf);
    const g = try r.f64s(R * D, -1, 1, .f);
    const ids = try r.a.alloc(i32, R * SLOTS);
    const wts = try r.a.alloc(f64, R * SLOTS);
    parallel(R, &RouteCtx{ .m = &m, .x = x, .round = true, .ids = ids, .wts = wts }, routeRow);
    // the float64 reference on two rows of smooth weights, against central differences in x and the pocket's outputs
    {
        const sw = try r.a.alloc(f64, R * SLOTS);
        parallel(64, &RouteCtx{ .m = &m, .x = x, .round = false, .ids = try r.a.alloc(i32, R * SLOTS), .wts = sw }, routeRow);
        const c = try MoeCase.init(r.a, &m, 64, x, g, ids, sw);
        var num: std.ArrayList(f64) = .empty;
        var ref: std.ArrayList(f64) = .empty;
        var nump: std.ArrayList(f64) = .empty;
        var refp: std.ArrayList(f64) = .empty;
        const step = 1e-4;
        for ([_]usize{ 1, 50 }) |row| {
            const xr = try r.a.dupe(f64, x[row * D ..][0..D]);
            const gr = g[row * D ..][0..D];
            const ir = ids[row * SLOTS ..][0..SLOTS];
            for ([_]usize{ 0, 777, 1600, 2559 }) |i| {
                const v = xr[i];
                xr[i] = v + step;
                const lp = moeLoss(&m, xr, gr, ir);
                xr[i] = v - step;
                const lm = moeLoss(&m, xr, gr, ir);
                xr[i] = v;
                try num.append(r.a, (lp - lm) / (2 * step));
                try ref.append(r.a, c.dx[row * D + i]);
            }
            // a pocket neuron's output vector: dL/dpd_j[i] = gate act_j g[i]
            const j = row % P;
            for ([_]usize{ 9, 2000 }) |i| {
                const v = m.pd[j * D + i];
                const hi: f32 = v + 1e-2;
                const lo: f32 = v - 1e-2;
                m.pd[j * D + i] = hi;
                const lp = moeLoss(&m, xr, gr, ir);
                m.pd[j * D + i] = lo;
                const lm = moeLoss(&m, xr, gr, ir);
                m.pd[j * D + i] = v;
                try nump.append(r.a, (lp - lm) / (@as(f64, hi) - @as(f64, lo)));
                try refp.append(r.a, sw[row * SLOTS + TOPK] * silu(dotF(m.pg[j * D ..][0..D], xr)) * dotF(m.pu[j * D ..][0..D], xr) * gr[i]);
            }
        }
        r.fd("moe dx", num.items, ref.items, rms(c.dx));
        r.fd("moe pocket output vectors", nump.items, refp.items, 0);
    }
    const c = try MoeCase.init(r.a, &m, R, x, g, ids, wts);
    // the device side
    const e: fb.Experts = blk: {
        const gq = try m.gate.dev(r);
        const uq = try m.up.dev(r);
        const dq = try m.down.dev(r);
        const sg = try m.sgate.dev(r);
        const su = try m.sup.dev(r);
        const sd = try m.sdown.dev(r);
        break :blk .{ .gate = .{ gq.w, gq.s, gq.b }, .up = .{ uq.w, uq.s, uq.b }, .down = .{ dq.w, dq.s, dq.b }, .shared_gate = .{ sg.w, sg.s, sg.b }, .shared_up = .{ su.w, su.s, su.b }, .shared_down = .{ sd.w, sd.s, sd.b }, .routed = E, .width = W, .dim = D };
    };
    const router = try r.dev(u16, m.router);
    const pocket: Ops.Pocket = .{ .gate = try r.dev(f32, m.pg), .up = try r.dev(f32, m.pu), .down = try r.dev(f32, m.pd), .n = P };
    std.debug.print("RESULT moe weights: {d:.1} MiB (512 routed experts, the shared expert, the router, a pocket of {d})\n", .{ mib(r.bytes), P });
    for (row_counts) |rows| {
        const pairs = rows * SLOTS;
        const wts32 = try toF32(r.a, wts[0 .. rows * SLOTS]);
        const kept: Ops.MoeKept = .{ .x = try r.dev(u16, try toBf16(r.a, x[0 .. rows * D])), .ids = try r.dev(i32, ids[0 .. rows * SLOTS]), .wts = try r.dev(f32, wts32) };
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const before = r.bytes;
        const dx = try r.zeros(rows * D * 4);
        const t: Ops.MoeScratch = .{
            .logits = try r.zeros(rows * (E + 1) * 4),
            .order = try r.zeros(pairs * 4),
            .inv = try r.zeros(pairs * 4),
            .offsets = try r.zeros((E + 1) * 4),
            .tiles = try r.zeros(2 * fb.tileCap(pairs, E + 1) * 4),
            .gu = try r.zeros(pairs * 2 * W * 4),
            .dact = try r.zeros(pairs * W * 4),
            .dgu = try r.zeros(pairs * 2 * W * 4),
            .dxp = try r.zeros(pairs * D * 4),
            .a = try r.zeros(pairs * 4),
            .dlog = try r.zeros(rows * SLOTS * 4),
        };
        const scratch = r.bytes - before;
        const row_of = try r.a.alloc(i32, P);
        for (row_of, 0..) |*v, j| v.* = if (j == 3) -1 else @intCast((j * 37 + 5) % rows);
        const rows_dev = try r.dev(i32, row_of);
        const grad = try r.zeros(P * D * 4);
        const pout = try r.zeros(rows * D * 4);
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        try o.moeBack(e, router, kept, pocket, gd, dx, t, rows, TOPK);
        try o.pocketGrad(pocket, kept, gd, rows_dev, 1.0, grad, D, TOPK);
        const ms = try tm.stop(r);
        try o.pocketRows(pocket, kept, pout, true, rows, D, TOPK);
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows dx", .{rows}), try r.back(f32, dx, rows * D), c.dx[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows slot scores a = dL/dw", .{rows}), try r.back(f32, t.a, pairs), c.a[0..pairs], SLOTS, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows router logits' gradients", .{rows}), try r.back(f32, t.dlog, pairs), c.dlog[0..pairs], SLOTS, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows pocket forward", .{rows}), try r.back(f32, pout, rows * D), c.pout[0 .. rows * D], D, tol_col, tol_l2);
        // the pocket's output vectors at their rows
        const want = try r.a.alloc(f64, P * D);
        @memset(want, 0);
        for (row_of, 0..) |row, j| if (row >= 0) {
            const ri: usize = @intCast(row);
            const xr = x[ri * D ..][0..D];
            const k = @as(f64, wts32[ri * SLOTS + TOPK]) * silu(dotF(m.pg[j * D ..][0..D], xr)) * dotF(m.pu[j * D ..][0..D], xr);
            for (0..D) |i| want[j * D + i] = k * g[ri * D + i];
        };
        r.compare(try std.fmt.bufPrint(&buf, "moe {d} rows pocket output vectors' gradients", .{rows}), try r.back(f32, grad, P * D), want, D, tol_col, tol_l2);
        std.debug.print("RESULT moe {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ rows, ms, mib(scratch) });
    }
}

// ---- the attention layer ---------------------------------------------------------------------------------------------

const NQ = 24;
const NKV = 2;
const HD = 256;
const HALF = 32;
const QW = NQ * HD;
const KW = NKV * HD;
const PQ = 2 * QW;
const QKV = QW + 2 * KW;
const theta: f64 = 1e7;

const Attn = struct {
    q: HQ4, // [PQ, D]
    k: HQ4, // [KW, D]
    v: HQ4,
    o: HQ4, // [D, QW]
    qscale: []f32,
    kscale: []f32,
    inv: []f32,
};

/// A q or k head as glue.attn_prep prepares it: RMS norm (scale 1 + w), rotate-half RoPE on 2 HALF dims at `pos`;
/// rounded to bf16 at both steps when `round` (the angle then in fp32, as the kernel takes it).
fn prepHead(x: []const f64, w: []const f32, pos: usize, inv: []const f32, round: bool, out: []f64) void {
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var xn: [HD]f64 = undefined;
    for (0..HD) |d| xn[d] = if (round) rb(x[d] * rinv * w[d]) else x[d] * rinv * w[d];
    for (0..HD) |d| {
        var v = xn[d];
        if (d < 2 * HALF) {
            const i = d % HALF;
            const ang: f64 = if (round) @as(f32, @floatFromInt(pos)) * inv[i] else @as(f64, @floatFromInt(pos)) * inv[i];
            const cs = @cos(ang);
            const sn = @sin(ang);
            v = if (d < HALF) xn[d] * cs - xn[d + HALF] * sn else xn[d - HALF] * sn + xn[d] * cs;
        }
        out[d] = if (round) rb(v) else v;
    }
}

/// prepHead undone: dx from dout through the rotation and the norm.
fn prepHeadBack(x: []const f64, w: []const f32, pos: usize, inv: []const f32, dout: []const f64, dx: []f64) void {
    var gn: [HD]f64 = undefined;
    for (0..HD) |d| {
        gn[d] = dout[d];
        if (d < 2 * HALF) {
            const ang = @as(f64, @floatFromInt(pos)) * inv[d % HALF];
            const cs = @cos(ang);
            const sn = @sin(ang);
            gn[d] = if (d < HALF) dout[d] * cs + dout[d + HALF] * sn else dout[d] * cs - dout[d - HALF] * sn;
        }
    }
    const rinv = 1 / @sqrt(dot(x, x) / HD + eps);
    var t: f64 = 0;
    for (0..HD) |d| t += w[d] * gn[d] * x[d];
    for (0..HD) |d| dx[d] = rinv * w[d] * gn[d] - rinv * rinv * rinv * x[d] * t / HD;
}

/// A case's projections, prepared heads, attention outputs and o_proj's input (rounded where the engine stores bf16).
const AttnCase = struct {
    at: *const Attn,
    rows: usize,
    round: bool,
    x: []const f64,
    pq: []f64,
    pk: []f64,
    pv: []f64,
    qkv: []f64,
    o: []f64,
    gated: []f64,

    fn init(al: std.mem.Allocator, at: *const Attn, rows: usize, round: bool, x: []const f64) !AttnCase {
        var c: AttnCase = .{ .at = at, .rows = rows, .round = round, .x = x, .pq = try al.alloc(f64, rows * PQ), .pk = try al.alloc(f64, rows * KW), .pv = try al.alloc(f64, rows * KW), .qkv = try al.alloc(f64, rows * QKV), .o = try al.alloc(f64, rows * QW), .gated = try al.alloc(f64, rows * QW) };
        parallel(rows, @as(*const AttnCase, &c), project);
        parallel(NQ, @as(*const AttnCase, &c), attend);
        for (0..rows * QW) |i| c.gated[i] = c.gate(i / QW, i % QW);
        return c;
    }

    fn gate(c: *const AttnCase, r: usize, col: usize) f64 {
        const v = c.o[r * QW + col] * sig(c.pq[r * PQ + (col / HD) * 2 * HD + HD + col % HD]);
        return if (c.round) rb(v) else v;
    }

    fn project(c: *const AttnCase, r: usize, _: usize) void {
        c.projectRow(r);
    }

    fn projectRow(c: *const AttnCase, r: usize) void {
        const x = c.x[r * D ..][0..D];
        for (0..PQ) |i| c.pq[r * PQ + i] = c.rnd(c.at.q.rowDot(i, x));
        for (0..KW) |i| {
            c.pk[r * KW + i] = c.rnd(c.at.k.rowDot(i, x));
            c.pv[r * KW + i] = c.rnd(c.at.v.rowDot(i, x));
        }
        c.prepRow(r);
    }

    fn prepRow(c: *const AttnCase, r: usize) void {
        for (0..NQ) |h| prepHead(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, c.round, c.qkv[r * QKV + h * HD ..][0..HD]);
        for (0..NKV) |h| {
            prepHead(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, c.round, c.qkv[r * QKV + QW + h * HD ..][0..HD]);
            @memcpy(c.qkv[r * QKV + QW + KW + h * HD ..][0..HD], c.pv[r * KW + h * HD ..][0..HD]);
        }
    }

    fn rnd(c: *const AttnCase, v: f64) f64 {
        return if (c.round) rb(v) else v;
    }

    fn attend(c: *const AttnCase, h: usize, _: usize) void {
        for (0..c.rows) |i| c.attendRow(h, i);
    }

    /// Row i of head h: softmax over keys 0..i of q . k / 16, times v.
    fn attendRow(c: *const AttnCase, h: usize, i: usize) void {
        const kv = h / (NQ / NKV);
        const q = c.qkv[i * QKV + h * HD ..][0..HD];
        var p: [2048]f64 = undefined;
        var top: f64 = -std.math.inf(f64);
        for (0..i + 1) |j| {
            p[j] = dot(q, c.qkv[j * QKV + QW + kv * HD ..][0..HD]) / 16;
            top = @max(top, p[j]);
        }
        var total: f64 = 0;
        for (0..i + 1) |j| {
            p[j] = @exp(p[j] - top);
            total += p[j];
        }
        const o = c.o[i * QW + h * HD ..][0..HD];
        @memset(o, 0);
        for (0..i + 1) |j| axpy(o, p[j] / total, c.qkv[j * QKV + QW + KW + kv * HD ..][0..HD]);
    }
};

/// The reference: dx [rows, D] from g = dL/d the attention output, through o_proj, the gate, causal attention, the
/// heads' preparation and the three projections; dpq, dpk, dv kept for the intermediate checks.
const AttnRef = struct {
    c: *const AttnCase,
    g: []const f64,
    dg: []f64,
    dout: []f64,
    dpq: []f64,
    dq: []f64,
    dkh: []f64, // [NQ, rows, HD] each head's part
    dvh: []f64,
    dk: []f64,
    dv: []f64,
    dpk: []f64,
    dx: []f64,

    fn init(al: std.mem.Allocator, c: *const AttnCase, g: []const f64) !AttnRef {
        const R = c.rows;
        var a: AttnRef = .{ .c = c, .g = g, .dg = try al.alloc(f64, R * QW), .dout = try al.alloc(f64, R * QW), .dpq = try al.alloc(f64, R * PQ), .dq = try al.alloc(f64, R * QW), .dkh = try al.alloc(f64, NQ * R * HD), .dvh = try al.alloc(f64, NQ * R * HD), .dk = try al.alloc(f64, R * KW), .dv = try al.alloc(f64, R * KW), .dpk = try al.alloc(f64, R * KW), .dx = try al.alloc(f64, R * D) };
        @memset(a.dg, 0);
        @memset(a.dkh, 0);
        @memset(a.dvh, 0);
        @memset(a.dk, 0);
        @memset(a.dv, 0);
        parallel(R, @as(*const AttnRef, &a), gateRow);
        parallel(NQ, @as(*const AttnRef, &a), head);
        for (0..NQ) |h| {
            const kv = h / (NQ / NKV);
            for (0..R) |j| {
                axpy(a.dk[j * KW + kv * HD ..][0..HD], 1, a.dkh[(h * R + j) * HD ..][0..HD]);
                axpy(a.dv[j * KW + kv * HD ..][0..HD], 1, a.dvh[(h * R + j) * HD ..][0..HD]);
            }
        }
        parallel(R, @as(*const AttnRef, &a), inputRow);
        return a;
    }

    fn gateRow(a: *const AttnRef, r: usize, _: usize) void {
        const c = a.c;
        const dg = a.dg[r * QW ..][0..QW];
        for (0..D) |d| c.at.o.rowAxpy(d, a.g[r * D + d], dg);
        for (0..QW) |col| {
            const at = r * PQ + (col / HD) * 2 * HD + HD + col % HD;
            const sg = sig(c.pq[at]);
            a.dout[r * QW + col] = dg[col] * sg;
            a.dpq[at] = dg[col] * c.gated[r * QW + col] * (1 - sg);
        }
    }

    fn head(a: *const AttnRef, h: usize, _: usize) void {
        const c = a.c;
        const R = c.rows;
        const kv = h / (NQ / NKV);
        var p: [2048]f64 = undefined;
        var dp: [2048]f64 = undefined;
        for (0..R) |i| {
            const q = c.qkv[i * QKV + h * HD ..][0..HD];
            const go = a.dout[i * QW + h * HD ..][0..HD];
            var top: f64 = -std.math.inf(f64);
            for (0..i + 1) |j| {
                p[j] = dot(q, c.qkv[j * QKV + QW + kv * HD ..][0..HD]) / 16;
                dp[j] = dot(go, c.qkv[j * QKV + QW + KW + kv * HD ..][0..HD]);
                top = @max(top, p[j]);
            }
            var total: f64 = 0;
            for (0..i + 1) |j| {
                p[j] = @exp(p[j] - top);
                total += p[j];
            }
            var mix: f64 = 0;
            for (0..i + 1) |j| {
                p[j] /= total;
                mix += p[j] * dp[j];
            }
            const dq = a.dq[i * QW + h * HD ..][0..HD];
            @memset(dq, 0);
            for (0..i + 1) |j| {
                const ds = p[j] * (dp[j] - mix);
                axpy(dq, ds / 16, c.qkv[j * QKV + QW + kv * HD ..][0..HD]);
                axpy(a.dkh[(h * R + j) * HD ..][0..HD], ds / 16, q);
                axpy(a.dvh[(h * R + j) * HD ..][0..HD], p[j], go);
            }
        }
    }

    fn inputRow(a: *const AttnRef, r: usize, _: usize) void {
        const c = a.c;
        for (0..NQ) |h| prepHeadBack(c.pq[r * PQ + h * 2 * HD ..][0..HD], c.at.qscale, r, c.at.inv, a.dq[r * QW + h * HD ..][0..HD], a.dpq[r * PQ + h * 2 * HD ..][0..HD]);
        for (0..NKV) |h| prepHeadBack(c.pk[r * KW + h * HD ..][0..HD], c.at.kscale, r, c.at.inv, a.dk[r * KW + h * HD ..][0..HD], a.dpk[r * KW + h * HD ..][0..HD]);
        const dx = a.dx[r * D ..][0..D];
        @memset(dx, 0);
        for (0..PQ) |i| c.at.q.rowAxpy(i, a.dpq[r * PQ + i], dx);
        for (0..KW) |i| {
            c.at.k.rowAxpy(i, a.dpk[r * KW + i], dx);
            c.at.v.rowAxpy(i, a.dv[r * KW + i], dx);
        }
    }
};

/// sum over rows i >= from of dg_i . gated_i with row `from` re-projected from x: the part of L the row moves.
fn attnLoss(c: *AttnCase, dg: []const f64, from: usize) f64 {
    c.projectRow(from);
    var l: f64 = 0;
    for (from..c.rows) |i| {
        for (0..NQ) |h| c.attendRow(h, i);
        for (0..QW) |col| l += dg[i * QW + col] * c.gate(i, col);
    }
    return l;
}

fn attnTest(r: *Rig, o: Ops) !void {
    const amp = @sqrt(3.0 / @as(f64, D));
    const at: Attn = blk: {
        const qs = try r.a.alloc(f32, HD);
        const ks = try r.a.alloc(f32, HD);
        for (qs, ks) |*a, *b| {
            a.* = @floatCast(1 + r.uni(-0.3, 0.3));
            b.* = @floatCast(1 + r.uni(-0.3, 0.3));
        }
        const inv = try r.a.alloc(f32, HALF);
        for (inv, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, theta, -@as(f64, @floatFromInt(i)) / HALF));
        break :blk .{ .q = try r.q4(PQ, D, 2 * amp), .k = try r.q4(KW, D, 2 * amp), .v = try r.q4(KW, D, amp), .o = try r.q4(D, QW, @sqrt(3.0 / @as(f64, QW))), .qscale = qs, .kscale = ks, .inv = inv };
    };
    const R = row_counts[row_counts.len - 1];
    const x = try r.f64s(R * D, -1, 1, .bf);
    const g = try r.f64s(R * D, -1, 1, .f);
    // the float64 reference on 64 smooth rows against central differences in x (rows 3 and 40 move every later row)
    {
        var c = try AttnCase.init(r.a, &at, 64, false, x[0 .. 64 * D]);
        const ref = try AttnRef.init(r.a, &c, g[0 .. 64 * D]);
        const xm = try r.a.dupe(f64, x[0 .. 64 * D]);
        c.x = xm;
        var num: std.ArrayList(f64) = .empty;
        var want: std.ArrayList(f64) = .empty;
        const step = 1e-4;
        for ([_]usize{ 3, 40 }) |row| for ([_]usize{ 11, 1234, 2558 }) |i| {
            const v = xm[row * D + i];
            xm[row * D + i] = v + step;
            const lp = attnLoss(&c, ref.dg, row);
            xm[row * D + i] = v - step;
            const lm = attnLoss(&c, ref.dg, row);
            xm[row * D + i] = v;
            _ = attnLoss(&c, ref.dg, row);
            try num.append(r.a, (lp - lm) / (2 * step));
            try want.append(r.a, ref.dx[row * D + i]);
        };
        r.fd("attn dx", num.items, want.items, rms(ref.dx));
    }
    const before_w = r.bytes;
    const da: Ops.Attn = .{ .q = try at.q.dev(r), .k = try at.k.dev(r), .v = try at.v.dev(r), .o = try at.o.dev(r), .qscale = try r.dev(f32, at.qscale), .kscale = try r.dev(f32, at.kscale), .inv = try r.dev(f32, at.inv), .heads = NQ, .kv_heads = NKV, .dim = HD, .half = HALF, .eps = @floatCast(eps) };
    std.debug.print("RESULT attn weights: {d:.1} MiB\n", .{mib(r.bytes - before_w)});
    for (row_counts) |rows| {
        var c = try AttnCase.init(r.a, &at, rows, true, x[0 .. rows * D]);
        const kept: Ops.AttnKept = .{ .pq = try r.dev(u16, try toBf16(r.a, c.pq)), .pk = try r.dev(u16, try toBf16(r.a, c.pk)), .pv = try r.dev(u16, try toBf16(r.a, c.pv)), .gated = try r.dev(u16, try toBf16(r.a, c.gated)), .pos0 = 0 };
        const gd = try r.dev(f32, try toF32(r.a, g[0 .. rows * D]));
        const before = r.bytes;
        const dx = try r.zeros(rows * D * 4);
        const t: Ops.AttnScratch = .{
            .qkv = try r.zeros(rows * QKV * 2),
            .dg = try r.zeros(rows * QW * 4),
            .dout = try r.zeros(rows * QW * 4),
            .dpq = try r.zeros(rows * PQ * 4),
            .dq = try r.zeros(rows * QW * 4),
            .dk = try r.zeros(rows * KW * 4),
            .dv = try r.zeros(rows * KW * 4),
            .dpk = try r.zeros(rows * KW * 4),
            .probs = try r.zeros(NQ * rows * rows * 4),
            .dscores = try r.zeros(NQ * rows * rows * 4),
        };
        const scratch = r.bytes - before;
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        try o.attnBack(da, kept, gd, dx, t, rows, false);
        const ms = try tm.stop(r);
        // the prepared heads against the host's: a value may move by a bf16 step of its rotation partner (the norm's
        // fp32 rinv against float64 flips a rounding), so within 2^-5 max(1, |value|) on these unit-RMS heads
        const qkv = try r.back(u16, t.qkv, rows * QKV);
        var off: usize = 0;
        var most: f64 = 0;
        var where: usize = 0;
        for (qkv, c.qkv, 0..) |got, want, i| {
            if (got != bf16(want)) off += 1;
            const e = @abs(bf(got) - want) / @max(1, @abs(want));
            if (e > most) {
                most = e;
                where = i;
            }
        }
        const prep_ok = most <= 1.0 / 32.0;
        r.checks += 1;
        if (!prep_ok) r.bad += 1;
        std.debug.print("RESULT attn {d} rows prepared heads: {d} of {d} values differ from the host's bf16, the largest by {e:.2} of max(1, |value|) (row {d}, column {d}){s}\n", .{ rows, off, qkv.len, most, where / QKV, where % QKV, if (prep_ok) "" else "  <- FAIL" });
        // the reference from the GPU's prepared heads (the same bf16 inputs)
        for (qkv, c.qkv) |got, *want| want.* = bf(got);
        const ref = try AttnRef.init(r.a, &c, g[0 .. rows * D]);
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpq (q and gate)", .{rows}), try r.back(f32, t.dpq, rows * PQ), ref.dpq, PQ, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dpk", .{rows}), try r.back(f32, t.dpk, rows * KW), ref.dpk, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dv", .{rows}), try r.back(f32, t.dv, rows * KW), ref.dv, KW, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "attn {d} rows dx", .{rows}), try r.back(f32, dx, rows * D), ref.dx, D, tol_col, tol_l2);
        std.debug.print("RESULT attn {d} rows: {d:.3} ms on the GPU, {d:.1} MiB of gradients and scratch\n", .{ rows, ms, mib(scratch) });
    }
}

// ---- the head --------------------------------------------------------------------------------------------------------

const V = 248320;

const HeadCtx = struct {
    w: *const HQ4,
    logits: []const u16,
    targets: []const u32,
    weights: []const f32,
    rows: usize,
    top: []const f64,
    total: []const f64,
    acc: [][]f64, // a thread's partial [rows, D]
};

fn headChunk(c: *const HeadCtx, i: usize, _: usize) void {
    const chunk = (V + threads - 1) / threads;
    const acc = c.acc[i];
    @memset(acc, 0);
    var wrow: [D]f64 = undefined;
    var v = i * chunk;
    while (v < @min(V, (i + 1) * chunk)) : (v += 1) {
        c.w.row(v, &wrow);
        for (0..c.rows) |r| {
            const p = @exp(bf(c.logits[r * V + v]) - c.top[r]) / c.total[r];
            const dl = (p - @as(f64, if (c.targets[r] == v) 1 else 0)) * c.weights[r];
            axpy(acc[r * D ..][0..D], dl, &wrow);
        }
    }
}

fn headTest(r: *Rig, o: Ops) !void {
    const R = row_counts[row_counts.len - 1];
    const head = try r.q4(V, D, @sqrt(3.0 / @as(f64, D)));
    const logits = try r.a.alloc(u16, R * V);
    for (logits) |*l| l.* = bf16(r.uni(-3, 3));
    const targets = try r.a.alloc(u32, R);
    const weights = try r.a.alloc(f32, R);
    for (targets, weights, 0..) |*t, *w, i| {
        t.* = r.rng.uintLessThan(u32, V);
        w.* = if (i % 7 == 3) 0 else @floatCast(1.0 / @as(f64, @floatFromInt(R)) * r.uni(0.5, 1.5));
        logits[i * V + t.*] = bf16(r.uni(2, 6)); // the target a little likelier
    }
    // the softmax's float64 statistics, then dmixed = dlogits head on threads over the vocabulary
    const top = try r.a.alloc(f64, R);
    const total = try r.a.alloc(f64, R);
    for (0..R) |i| {
        top[i] = -std.math.inf(f64);
        for (logits[i * V ..][0..V]) |l| top[i] = @max(top[i], bf(l));
        total[i] = 0;
        for (logits[i * V ..][0..V]) |l| total[i] += @exp(bf(l) - top[i]);
    }
    // the loss's gradient against central differences of the float64 cross-entropy (row 0's target and others)
    {
        var num: [4]f64 = undefined;
        var want: [4]f64 = undefined;
        const ce = struct {
            fn f(row: []const u16, t: u32, at: usize, add: f64) f64 {
                var m: f64 = -std.math.inf(f64);
                for (row, 0..) |l, i| m = @max(m, bf(l) + if (i == at) add else 0);
                var s: f64 = 0;
                for (row, 0..) |l, i| s += @exp(bf(l) + (if (i == at) add else 0) - m);
                return @log(s) + m - (bf(row[t]) + if (t == at) add else 0);
            }
        };
        const row = logits[0..V];
        for ([_]usize{ targets[0], 5, 100000, V - 1 }, 0..) |at, j| {
            num[j] = (ce.f(row, targets[0], at, 1e-5) - ce.f(row, targets[0], at, -1e-5)) / 2e-5;
            want[j] = @exp(bf(row[at]) - top[0]) / total[0] - @as(f64, if (at == targets[0]) 1 else 0);
        }
        r.fd("head cross-entropy dlogits", &num, &want, 1.0 / @sqrt(@as(f64, V)));
    }
    const accs = try r.a.alloc([]f64, threads);
    for (accs) |*a| a.* = try r.a.alloc(f64, R * D);
    parallel(threads, &HeadCtx{ .w = &head, .logits = logits, .targets = targets, .weights = weights, .rows = R, .top = top, .total = total, .acc = accs }, headChunk);
    const dmixed = try r.a.alloc(f64, R * D);
    @memset(dmixed, 0);
    for (accs) |a| axpy(dmixed, 1, a);
    // the final mixer after it, against the reference's dmixed
    const mixer = try makeHc(r, false);
    const h = try r.f64s(R * WIDE, -2, 2, .bf);
    try hcFd(r, "head final mixer", &mixer, h, h[0 .. R * D], null, dmixed);
    const cm = try HcCase.init(r.a, &mixer, R, true, h, h[0 .. R * D], null, dmixed);
    const before_w = r.bytes;
    const dh = try head.dev(r);
    const dm = try hcDev(r, &mixer);
    std.debug.print("RESULT head weights: {d:.1} MiB (vocab {d}) and the final mixer\n", .{ mib(r.bytes - before_w), V });
    for (row_counts) |rows| {
        const ld = try r.dev(u16, logits[0 .. rows * V]);
        const td = try r.dev(u32, targets[0..rows]);
        const wd = try r.dev(f32, weights[0..rows]);
        const kept = try hcKept(r, &cm, rows);
        const before = r.bytes;
        const stats = try r.zeros(rows * 8);
        const dmix = try r.zeros(rows * D * 4);
        const parts = try r.zeros(fb.splitsFor(rows, D, V) * rows * D * 4);
        const g = try r.zeros(rows * WIDE * 4);
        const t = try hcScratch(r, rows);
        const scratch = r.bytes - before;
        var tm = try Timer.init(r);
        defer tm.deinit();
        try tm.start(r);
        try o.softmax(ld, td, wd, stats, rows, V);
        try o.headBack(ld, dh, dmix, rows, parts);
        const ms_head = try tm.stop(r);
        try tm.start(r);
        try o.hcReadoutBack(dm, kept, dmix, 0, g, t, rows);
        const ms_mix = try tm.stop(r);
        // the loss statistics, the bf16 gradient rows, dmixed and the streams' gradient
        const st = try r.back(f32, stats, rows * 2);
        const want_st = try r.a.alloc(f64, rows * 2);
        for (0..rows) |i| {
            const lt = bf(logits[i * V + targets[i]]);
            want_st[2 * i] = @log(total[i]) + top[i] - lt;
            want_st[2 * i + 1] = @exp(lt - top[i]) / total[i];
        }
        var buf: [96]u8 = undefined;
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows (loss, target probability)", .{rows}), st, want_st, 2, 1e-5, 1e-5);
        const dl = try r.back(u16, ld, rows * V);
        var most: f64 = 0;
        var big: f64 = 0;
        for (0..rows) |i| for (0..V) |v| {
            const p = @exp(bf(logits[i * V + v]) - top[i]) / total[i];
            const want = (p - @as(f64, if (targets[i] == v) 1 else 0)) * weights[i];
            most = @max(most, @abs(bf(dl[i * V + v]) - want));
            big = @max(big, @abs(want));
        };
        const dl_ok = most <= 4e-3 * big;
        r.checks += 1;
        if (!dl_ok) r.bad += 1;
        std.debug.print("RESULT head {d} rows dlogits (bf16): largest difference {e:.2} of the largest |ref| {e:.2}{s}\n", .{ rows, most / big, big, if (dl_ok) "" else "  <- FAIL" });
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows dmixed", .{rows}), try r.back(f32, dmix, rows * D), dmixed[0 .. rows * D], D, tol_col, tol_l2);
        r.compare(try std.fmt.bufPrint(&buf, "head {d} rows final mixer dh", .{rows}), try r.back(f32, g, rows * WIDE), cm.dh[0 .. rows * WIDE], WIDE, tol_col, tol_l2);
        std.debug.print("RESULT head {d} rows: {d:.3} ms softmax and head, {d:.3} ms the final mixer, {d:.1} MiB of gradients and scratch (+ {d:.1} MiB bf16 logits)\n", .{ rows, ms_head, ms_mix, mib(scratch), mib(rows * V * 2) });
    }
}

// ---- entry -------------------------------------------------------------------------------------------------------------

pub const Case = enum { hc, moe, attn, head, all };

pub fn run(gpu: Gpu, case: Case) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x5eed_f1a5);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var k = try fb.Kernels.load(gpu.d);
    defer k.deinit();
    const o: Ops = .{ .k = &k, .s = s };
    const t0 = check.now(gpu.io);
    if (case == .hc or case == .all) {
        try hcTest(&r, o);
        r.release();
    }
    if (case == .moe or case == .all) {
        try moeTest(&r, o);
        r.release();
    }
    if (case == .attn or case == .all) {
        try attnTest(&r, o);
        r.release();
    }
    if (case == .head or case == .all) {
        try headTest(&r, o);
        r.release();
    }
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    try check.expect(r.bad == 0, "fn-back {t}: {d} of {d} checks outside tolerance", .{ case, r.bad, r.checks });
    check.pass("fn-back {t}: {d} checks within tolerance ({d:.1} s with the float64 references)", .{ case, r.checks, secs });
}
