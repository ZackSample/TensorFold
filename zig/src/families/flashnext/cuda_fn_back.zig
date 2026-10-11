//! Flash Next's input gradients for Living Weights (flashnext_train_mix.cu, with train.cu's loss, sort and attention
//! kernels) as calls on one stream: the hyper-connections, the MoE with its shared expert and pocket, the gated
//! attention and the 4-bit head, each from its output's gradient to its input's. Weights are MLX affine 4-bit, group 32.
const std = @import("std");
const cuda = @import("cuda");

/// Attention's shape (train.cu's Heads): rows, heads, kv heads, head dim, the q|k|v row width, scale, prefix keys.
pub const Heads = extern struct { rows: c_int, heads: c_int, kv_heads: c_int, dim: c_int, qkv: c_int, scale: f32, prefix: c_int = 0 };

/// A 4-bit matrix [n, k] in MLX's layout: words [n, k / 8], bf16 scales and biases [n, k / 32].
pub const Q4 = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
};

/// The routed experts' stacks [routed, n, k] (gate, up, down) and the shared expert's own matrices (expert `routed`).
pub const Experts = struct {
    gate: [3]u64,
    up: [3]u64,
    down: [3]u64,
    shared_gate: [3]u64,
    shared_up: [3]u64,
    shared_down: [3]u64,
    routed: usize,
    width: usize,
    dim: usize,
};

/// Keys tf_fn_attn_q reads at most (its shared score rows).
pub const max_keys = 2048;
/// Neurons a pocket kernel takes at most.
pub const max_pocket = 128;

const mix_names = [_][:0]const u8{
    "tf_fn_gemm_q4",        "tf_fn_gemm_q4_bf",   "tf_fn_sum_parts",      "tf_fn_experts_mm",    "tf_fn_experts_nt",
    "tf_fn_swiglu_back",    "tf_fn_route_back",   "tf_fn_pocket_rows",    "tf_fn_pocket_back",   "tf_fn_pocket_grad",
    "tf_fn_hc_branch_back", "tf_fn_hc_mix_back",  "tf_fn_hc_act_back",    "tf_fn_hc_norm_back",  "tf_fn_attn_prep",
    "tf_fn_attn_q",         "tf_fn_attn_gate_back", "tf_fn_attn_prep_back",
};

const train_names = [_][:0]const u8{ "tf_train_softmax", "tf_train_logits", "tf_train_sort_pairs", "tf_train_experts_tiles", "tf_train_pairs_out", "tf_train_attn_kv", "tf_train_zero" };

fn index(comptime list: []const [:0]const u8, comptime name: []const u8) usize {
    inline for (list, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

pub const Kernels = struct {
    mix: cuda.Module,
    train: cuda.Module,
    fm: [mix_names.len]cuda.Function,
    ft: [train_names.len]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .mix = try cuda.Module.load(d, cuda.kernels.flashnext_train_mix), .train = undefined, .fm = undefined, .ft = undefined };
        errdefer k.mix.unload();
        k.train = try cuda.Module.load(d, cuda.kernels.train);
        errdefer k.train.unload();
        for (mix_names, &k.fm) |n, *f| f.* = try k.mix.function(n);
        for (train_names, &k.ft) |n, *f| f.* = try k.train.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mix.unload();
        k.train.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

/// The tiles of 16 pairs `pairs` pairs cut into at most, each expert's last one part full.
pub fn tileCap(pairs: usize, experts: usize) usize {
    return pairs / 16 + experts;
}

/// K splits for a product of m rows by n columns over k: enough blocks to fill the GPU, each split 1024 rows or more.
pub fn splitsFor(m: usize, n: usize, k: usize) usize {
    const tiles = ((m + 63) / 64) * ((n + 63) / 64);
    const want = (192 + tiles - 1) / tiles;
    return @max(1, @min(want, k / 1024, 64));
}

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *cuda.Args) !void {
        const f = if (comptime std.mem.startsWith(u8, name, "tf_fn_")) o.k.fm[index(&mix_names, name)] else o.k.ft[index(&train_names, name)];
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    pub fn zero(o: Ops, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(n)));
        try o.go("tf_train_zero", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// c [m, w.k] (+)= a [m, w.n] (fp32, or bf16 when `a_bf16`; rows lda apart) times w: a linear's input gradient.
    /// `parts` holds splitsFor(m, w.k, w.n) m w.k fp32 partials when that is above 1 (else it may be 0).
    pub fn product(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, m: usize, accumulate: bool, parts: u64) !void {
        const n = w.k;
        const k = w.n;
        if (n % 32 != 0) return error.UnsupportedShape;
        const splits = if (parts != 0) splitsFor(m, n, k) else 1;
        const kchunk = (((k + splits - 1) / splits) + 31) / 32 * 32;
        const z = (k + kchunk - 1) / kchunk;
        var a: cuda.Args = .{};
        a.add(a_);
        for ([_]u64{ w.w, w.s, w.b, if (z > 1) parts else c }) |v| a.add(v);
        for ([_]usize{ m, n, k, lda, ldc }) |v| a.add(int(v));
        a.add(@as(c_int, @intFromBool(accumulate)));
        a.add(int(kchunk));
        if (a_bf16) try o.go("tf_fn_gemm_q4_bf", .{ up(n, 64), up(m, 64), u(z) }, 128, &a) else try o.go("tf_fn_gemm_q4", .{ up(n, 64), up(m, 64), u(z) }, 128, &a);
        if (z == 1) return;
        var b: cuda.Args = .{};
        b.add(parts);
        for ([_]usize{ z, m, n }) |v| b.add(int(v));
        b.add(c);
        b.add(int(ldc));
        b.add(@as(c_int, @intFromBool(accumulate)));
        try o.go("tf_fn_sum_parts", .{ up(m * n, 256), 1, 1 }, 256, &b);
    }

    // ---- the head ------------------------------------------------------------------------------------------------

    /// Each row's softmax over its bf16 logits: (loss, target probability) into stats, then (p - onehot) w in place.
    pub fn softmax(o: Ops, logits: u64, targets: u64, weights: u64, stats: u64, rows: usize, vocab: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, targets, weights, stats }) |v| a.add(v);
        a.add(int(vocab));
        try o.go("tf_train_softmax", .{ u(rows), 1, 1 }, 1024, &a);
    }

    /// dmixed [rows, head.k] = the softmax's bf16 rows [rows, head.n] times the 4-bit head (split over the vocabulary).
    pub fn headBack(o: Ops, dlogits: u64, head: Q4, dmixed: u64, rows: usize, parts: u64) !void {
        try o.product(dlogits, true, head.n, head, dmixed, head.k, rows, false, parts);
    }

    // ---- the hyper-connections -----------------------------------------------------------------------------------

    pub const Hc = struct {
        down: Q4, // input_mix_weight_down [low, S d]
        inject: ?Q4, // block_inject_weight [S, S d]; null on the final mixer
        up: Q4, // input_mix_weight_up [S d, low]
        scale: u64, // hc_norm's fp32 1 + w [S d]
        streams: usize,
        d: usize,
        low: usize,
        eps: f32,
    };

    /// What a read-out kept: h [rows, S d] bf16 (its input), normed bf16, v the down projection fp32 [rows, low (+ S)],
    /// up the up projection bf16 [rows, S d], inject bf16 [rows, S].
    pub const HcKept = struct { h: u64, normed: u64, v: u64, up: u64, inject: u64 };

    /// Scratch for one read-out's backward: dn and du [rows, S d], dact [rows, low], dv [rows, low + S] (fp32).
    pub const HcScratch = struct { dn: u64, du: u64, dact: u64, dv: u64 };

    /// A block's write-back h' = h + bf16(branch inject) undone from g = dL/dh': dbranch [rows, d], dinject [rows, S].
    pub fn hcBranchBack(o: Ops, hc: Hc, g: u64, inject: u64, branch: u64, dbranch: u64, dinject: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, inject, branch, dbranch, dinject }) |v| a.add(v);
        a.add(int(hc.d));
        a.add(int(hc.streams));
        try o.go("tf_fn_hc_branch_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// The read-out (mix, up, activations, down, the streams' norms) undone: g [rows, S d] += dL/dh from dmix [rows, d]
    /// and dinject [rows, S] (0 on the final mixer). g must already hold the residual's own gradient.
    pub fn hcReadoutBack(o: Ops, hc: Hc, kept: HcKept, dmix: u64, dinject: u64, g: u64, t: HcScratch, rows: usize) !void {
        const wide = hc.streams * hc.d;
        const nv = hc.low + if (hc.inject != null) hc.streams else 0;
        var a: cuda.Args = .{};
        for ([_]u64{ dmix, kept.up, kept.normed, t.dn, t.du }) |v| a.add(v);
        a.add(int(hc.d));
        a.add(int(hc.streams));
        try o.go("tf_fn_hc_mix_back", .{ up(hc.d, 256), u(rows), 1 }, 256, &a);
        try o.product(t.du, false, wide, hc.up, t.dact, hc.low, rows, false, 0);
        var b: cuda.Args = .{};
        for ([_]u64{ t.dact, kept.v, if (hc.inject != null) dinject else 0, t.dv }) |v| b.add(v);
        for ([_]usize{ hc.low, hc.streams, nv }) |v| b.add(int(v));
        try o.go("tf_fn_hc_act_back", .{ u(rows), 1, 1 }, 256, &b);
        try o.product(t.dv, false, nv, hc.down, t.dn, wide, rows, true, 0);
        if (hc.inject) |inj| try o.product(t.dv + hc.low * 4, false, nv, inj, t.dn, wide, rows, true, 0);
        var c: cuda.Args = .{};
        for ([_]u64{ kept.h, hc.scale, t.dn, g }) |v| c.add(v);
        c.add(int(hc.d));
        c.add(int(hc.streams));
        c.add(hc.eps);
        try o.go("tf_fn_hc_norm_back", .{ u(hc.streams), u(rows), 1 }, 256, &c);
    }

    // ---- the MoE layer -------------------------------------------------------------------------------------------

    /// The pocket: SwiGLU neurons gate, up, down fp32 [n, d] beside the shared expert, behind its gate.
    pub const Pocket = struct { gate: u64, up: u64, down: u64, n: usize };

    /// What the MoE layer kept: x [rows, d] bf16 (its input), ids [rows, slots] (top_k experts, then `routed` for the
    /// shared expert) and wts [rows, slots] fp32 (bf16 routing weights, then the shared gate).
    pub const MoeKept = struct { x: u64, ids: u64, wts: u64 };

    /// Scratch for the MoE backward of `rows` rows (pairs = rows slots).
    pub const MoeScratch = struct {
        logits: u64, // [rows, routed + 1] fp32
        order: u64, // [pairs] i32
        inv: u64, // [pairs] i32
        offsets: u64, // [routed + 1] i32
        tiles: u64, // [2 tileCap] u32
        gu: u64, // [pairs, 2 width] fp32
        dact: u64, // [pairs, width] fp32
        dgu: u64, // [pairs, 2 width] fp32
        dxp: u64, // [pairs, d] fp32
        a: u64, // [pairs] fp32
        dlog: u64, // [rows, slots] fp32 or 0
    };

    fn experts(o: Ops, comptime name: []const u8, x: u64, ldx: usize, order: u64, slots: usize, w: [3]u64, w2: [3]u64, t: MoeScratch, y: u64, extra: anytype, pairs: usize, grid_x: usize, n_experts: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(ldx));
        a.add(order);
        a.add(int(slots));
        for (w) |v| a.add(v);
        for (w2) |v| a.add(v);
        a.add(t.tiles);
        a.add(t.offsets);
        a.add(y);
        inline for (extra) |v| a.add(v);
        try o.go(name, .{ u(grid_x), u(tileCap(pairs, n_experts)), 1 }, 64, &a);
    }

    /// The MoE layer undone from g = dL/d its output [rows, d]: dx [rows, d] += the routed and shared experts' input
    /// gradients, the pocket's and the router's (softmax over the top_k logits, the shared gate's sigmoid).
    pub fn moeBack(o: Ops, e: Experts, router: u64, kept: MoeKept, pocket: ?Pocket, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        const slots = top_k + 1;
        const pairs = rows * slots;
        const total = e.routed + 1;
        const W = e.width;
        const D = e.dim;
        var a: cuda.Args = .{};
        for ([_]u64{ kept.x, router, t.logits }) |v| a.add(v);
        for ([_]usize{ rows, total, D }) |v| a.add(int(v));
        try o.go("tf_train_logits", .{ up(32 * total, 256), u(rows), 1 }, 256, &a);
        var b: cuda.Args = .{};
        b.add(kept.ids);
        for ([_]usize{ rows, slots, slots, total }) |v| b.add(int(v));
        for ([_]u64{ t.order, t.inv, t.offsets }) |v| b.add(v);
        try o.go("tf_train_sort_pairs", .{ 1, 1, 1 }, 256, &b);
        var c: cuda.Args = .{};
        c.add(t.offsets);
        c.add(t.tiles);
        for ([_]usize{ pairs, tileCap(pairs, total), total }) |v| c.add(int(v));
        try o.go("tf_train_experts_tiles", .{ 1, 1, 1 }, 1, &c);
        // each pair's gate and up rows again, then its down projection's input gradient (before its weight)
        inline for (.{ .{ e.gate, e.shared_gate, 0 }, .{ e.up, e.shared_up, W } }) |m|
            try o.experts("tf_fn_experts_nt", kept.x, D, t.order, slots, m[0], m[1], t, t.gu, .{ int(2 * W), int(m[2]), int(pairs), int(D), int(W), int(e.routed) }, pairs, W / 64, total);
        try o.experts("tf_fn_experts_mm", g, D, t.order, slots, e.down, e.shared_down, t, t.dact, .{ int(pairs), int(D), int(W), int(e.routed), @as(c_int, 0) }, pairs, W / 64, total);
        var s: cuda.Args = .{};
        for ([_]u64{ t.gu, t.dact, t.order, kept.wts, t.a, t.dgu }) |v| s.add(v);
        s.add(int(W));
        try o.go("tf_fn_swiglu_back", .{ u(pairs), 1, 1 }, 128, &s);
        try o.experts("tf_fn_experts_mm", t.dgu, 2 * W, 0, slots, e.gate, e.shared_gate, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 0) }, pairs, D / 64, total);
        try o.experts("tf_fn_experts_mm", t.dgu + W * 4, 2 * W, 0, slots, e.up, e.shared_up, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 1) }, pairs, D / 64, total);
        var p: cuda.Args = .{};
        for ([_]u64{ t.inv, t.dxp, dx }) |v| p.add(v);
        p.add(int(D));
        p.add(int(slots));
        try o.go("tf_train_pairs_out", .{ up(D, 256), u(rows), 1 }, 256, &p);
        if (pocket) |pk| try o.pocketBack(pk, kept, g, dx, t.a, rows, D, top_k);
        var r: cuda.Args = .{};
        r.add(t.logits);
        r.add(int(total));
        r.add(kept.ids);
        r.add(int(slots));
        for ([_]u64{ t.a, router, dx }) |v| r.add(v);
        for ([_]usize{ top_k, e.routed, D }) |v| r.add(int(v));
        r.add(t.dlog);
        try o.go("tf_fn_route_back", .{ u(rows), 1, 1 }, 256, &r);
    }

    /// The pocket undone: dx += its input gradient (scaled by the shared gate), a's shared slot += its part of dL/dgate.
    pub fn pocketBack(o: Ops, pk: Pocket, kept: MoeKept, g: u64, dx: u64, a_: u64, rows: usize, d: usize, top_k: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        for ([_]u64{ g, dx, a_ }) |v| a.add(v);
        try o.go("tf_fn_pocket_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// out [rows, d] (+)= the pocket's output, its neurons' sum times each row's shared gate (bf16 or fp32 rows).
    pub fn pocketRows(o: Ops, pk: Pocket, kept: MoeKept, out: u64, out_f32: bool, rows: usize, d: usize, top_k: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        a.add(out);
        a.add(@as(c_int, @intFromBool(out_f32)));
        try o.go("tf_fn_pocket_rows", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// grad [n, d] += w gate act_j push[row_j] for each neuron j at its row rows[j] (i32, negative: none): the output
    /// vectors' gradient, push = dL/d the MoE output.
    pub fn pocketGrad(o: Ops, pk: Pocket, kept: MoeKept, push: u64, row_of: u64, w: f32, grad: u64, d: usize, top_k: usize) !void {
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        a.add(pk.gate);
        a.add(pk.up);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        a.add(push);
        a.add(row_of);
        a.add(w);
        a.add(grad);
        try o.go("tf_fn_pocket_grad", .{ u(pk.n), 1, 1 }, 256, &a);
    }

    // ---- the attention layer -------------------------------------------------------------------------------------

    pub const Attn = struct {
        q: Q4, // q_proj [heads 2 dim, d]: each head's q then gate
        k: Q4, // k_proj [kv dim, d]
        v: Q4, // v_proj [kv dim, d]
        o: Q4, // o_proj [d, heads dim]
        qscale: u64, // q_norm's fp32 1 + w [dim]
        kscale: u64, // k_norm's fp32 1 + w [dim]
        inv: u64, // RoPE's fp32 inverse frequencies [half]
        heads: usize,
        kv_heads: usize,
        dim: usize,
        half: usize,
        eps: f32,
    };

    /// What attention kept: its projections pq [rows, heads 2 dim], pk, pv [rows, kv dim] (bf16) and o_proj's input
    /// gated [rows, heads dim] (bf16); pos0 the first row's position.
    pub const AttnKept = struct { pq: u64, pk: u64, pv: u64, gated: u64, pos0: usize };

    /// Scratch: qkv [rows, (heads + 2 kv) dim] bf16, dg and dout [rows, heads dim], dpq [rows, heads 2 dim], dq
    /// [rows, heads dim], dk, dv, dpk [rows, kv dim] fp32, probs and dscores [heads, rows, keys] fp32.
    pub const AttnScratch = struct { qkv: u64, dg: u64, dout: u64, dpq: u64, dq: u64, dk: u64, dv: u64, dpk: u64, probs: u64, dscores: u64 };

    /// The prepared heads (normed, rotated q and k; v) into qkv, as the forward made them.
    pub fn attnPrep(o: Ops, at: Attn, kept: AttnKept, qkv: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ kept.pq, kept.pk, kept.pv, at.qscale, at.kscale, at.inv }) |v| a.add(v);
        a.add(int(kept.pos0));
        a.add(qkv);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| a.add(int(v));
        a.add(at.eps);
        try o.go("tf_fn_attn_prep", .{ u(at.heads + 2 * at.kv_heads), u(rows), 1 }, u(at.dim), &a);
    }

    /// Causal attention undone on `rows` rows (no prefix keys: the indexer selects every key below 2052): dx [rows, d]
    /// += its input gradient from g = dL/d its output [rows, d].
    pub fn attnBack(o: Ops, at: Attn, kept: AttnKept, g: u64, dx: u64, t: AttnScratch, rows: usize, accumulate: bool) !void {
        if (rows > max_keys or at.dim > 256 or at.dim % 32 != 0) return error.UnsupportedShape;
        const qw = at.heads * at.dim;
        const kw = at.kv_heads * at.dim;
        const d = at.o.n;
        try o.attnPrep(at, kept, t.qkv, rows);
        try o.product(g, false, d, at.o, t.dg, qw, rows, false, 0);
        var a: cuda.Args = .{};
        for ([_]u64{ t.dg, kept.gated, kept.pq, t.dout, t.dpq }) |v| a.add(v);
        a.add(int(at.heads));
        a.add(int(at.dim));
        try o.go("tf_fn_attn_gate_back", .{ up(qw, 256), u(rows), 1 }, 256, &a);
        const h: Heads = .{ .rows = int(rows), .heads = int(at.heads), .kv_heads = int(at.kv_heads), .dim = int(at.dim), .qkv = int(qw + 2 * kw), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(at.dim)))) };
        var b: cuda.Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dq }) |v| b.add(v);
        b.add(h);
        b.add(t.qkv);
        b.add(t.qkv);
        try o.go("tf_fn_attn_q", .{ u(at.heads), u(rows), 1 }, 256, &b);
        var c: cuda.Args = .{};
        for ([_]u64{ t.qkv, t.dout, t.probs, t.dscores, t.dk, t.dv }) |v| c.add(v);
        c.add(h);
        try o.go("tf_train_attn_kv", .{ u(at.kv_heads), u(rows), 1 }, u(at.dim), &c);
        var p: cuda.Args = .{};
        for ([_]u64{ kept.pq, kept.pk, at.qscale, at.kscale, at.inv }) |v| p.add(v);
        p.add(int(kept.pos0));
        for ([_]u64{ t.dq, t.dk, t.dpq, t.dpk }) |v| p.add(v);
        for ([_]usize{ at.heads, at.kv_heads, at.dim, at.half }) |v| p.add(int(v));
        p.add(at.eps);
        try o.go("tf_fn_attn_prep_back", .{ u(at.heads + at.kv_heads), u(rows), 1 }, u(at.dim), &p);
        try o.product(t.dpq, false, 2 * qw, at.q, dx, d, rows, accumulate, 0);
        try o.product(t.dpk, false, kw, at.k, dx, d, rows, true, 0);
        try o.product(t.dv, false, kw, at.v, dx, d, rows, true, 0);
    }
};
