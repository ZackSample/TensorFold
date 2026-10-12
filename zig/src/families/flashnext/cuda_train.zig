//! Living Weights' trainer for Flash Next on CUDA: sequences side by side through the engine's batched pass (its
//! pocket in the forward, scaled per sequence), every row's loss on the answer tokens, and the loss's gradient back
//! from the LM head through the final mixer and each layer (both hyper-connection sites, the MoE with its shared
//! expert and pocket, the DeltaNet or attention block) down to the lowest nudged layer, where `pushAt` reads dL/d
//! the MoE output at each nudged row. The MoE backward reads the engine's packed experts where they sit; the small
//! linears' backward (hyper-connections, DeltaNet and attention projections, the head) read MLX copies of them.
const std = @import("std");
const cuda = @import("cuda");
const living = @import("living");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const fb = @import("cuda_fn_back.zig");

const batch = prompt.batch;
const gt = cuda.gdn_train;
const train_mode = living.train_mode;
const lesson_route = living.lesson_route;
const Count = living.pocket_clock.Count;

pub const Mode = train_mode.Mode;
pub const Result = train_mode.Result;
pub const Seq = train_mode.Seq;
pub const Span = train_mode.Span;
pub const Answer = train_mode.Answer;
pub const Turn = lesson_route.Turn;

const layers_n = wt.layers_n;
const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const low_rank = wt.low;
const vocab = wt.vocab;
const slots = wt.slots;
const top_k = slots - 1;
const moe_w = wt.moe_w;
const experts_n = wt.experts_n;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const out_k = wt.out_k;
const eps = wt.eps;

/// Rows a sequence learns from at most (a question, its answer and the template around them).
pub const max_rows = 512;
/// Rows a pass holds at most, its sequences side by side at even rows.
pub const pack_rows = 1024;
/// Sequences a pass holds at most.
pub const max_seqs = 32;
/// The lowest layer a pass keeps what a backward reads: a pocket's neurons sit at layers / 5 and up.
pub const keep_low = layers_n / 5;

const per_row_kept = 2 * (3 * wide * 2 + (low_rank + streams) * 4 + streams * 2) + 3 * dims * 2 + 2 * slots * 4;

fn up(n: usize, by: usize) usize {
    return (n + by - 1) / by * by;
}

/// The device bytes `init` takes at most, beside the engine (its batched pass's scratch is the engine's own).
pub fn memory() usize {
    const kept_layers = layers_n - keep_low;
    var n: usize = 0;
    n += pack_rows * (kept_layers * (per_row_kept + 33 * 1024) + layers_n * (dims * 2 + slots * 8) + 3 * wide * 2 + low_rank * 4);
    n += pack_rows * vocab * 2; // every row's logits, then their gradient
    n += scratchBytes();
    n += weightBytes();
    n += prompt.seqBytes(max_rows);
    return n + (256 << 20);
}

fn scratchBytes() usize {
    const pairs = pack_rows * slots;
    var n: usize = 0;
    n += pack_rows * wide * 4 * 3; // g, dn, du
    n += pack_rows * dims * 4 * 3 + pack_rows * (low_rank + streams) * 4 * 2 + pack_rows * streams * 4; // dbranch, dmix, dmixed; dact, dv; dinj
    n += pack_rows * (experts_n + 1) * 4 + pairs * 4 * 3 + 2 * fb.tileCap(pairs, experts_n + 1) * 4 + (experts_n + 2) * 4;
    n += pairs * 2 * moe_w * 4 * 2 + pairs * moe_w * 4 + pairs * dims * 4;
    n += max_rows * ((q_heads + 2 * kv_heads) * head_dim * 2 + q_heads * head_dim * 4 * 3 + q_heads * 2 * head_dim * 4 + 3 * kv_heads * head_dim * 4);
    n += 2 * q_heads * max_rows * max_rows * 4;
    n += gt.Scratch.bytes(max_rows, gdn.nk, gdn.nv, dims);
    n += (pack_rows + 512) * dims * 4; // the head's split partials
    return n;
}

fn weightBytes() usize {
    const q = struct {
        fn f(n: usize, k: usize) usize {
            return n * k / 2 + n * k / 16 + 1024;
        }
    }.f;
    var n: usize = q(vocab, dims) + q(low_rank, wide) + q(wide, low_rank);
    for (keep_low..layers_n) |l| {
        n += 2 * (q(low_rank, wide) + q(streams, wide) + q(wide, low_rank));
        n += if (wt.linearLayer(l)) q(gdn.proj_width, dims) + q(dims, gdn.value_dim) else q(q_heads * 2 * head_dim, dims) + 2 * q(kv_heads * head_dim, dims) + q(dims, out_k);
    }
    return n;
}

/// Host memory the GPU maps: kernels write it at `dev`, the host reads it through `slice`.
const Mapped = struct {
    host: cuda.HostBuffer,
    dev: u64,

    fn init(d: *const cuda.Driver, bytes: usize) !Mapped {
        var h = try cuda.HostBuffer.allocMapped(d, @max(bytes, 16));
        errdefer h.free();
        @memset(h.bytes, 0);
        return .{ .host = h, .dev = try h.device() };
    }

    fn deinit(m: *Mapped) void {
        m.host.free();
    }

    fn slice(m: Mapped, comptime T: type, n: usize) []T {
        return m.host.slice(T)[0..n];
    }
};

/// One layer's MLX copies its backward's products read, and the engine's own small tensors.
const LayerW = struct {
    attn_hc: fb.Ops.Hc,
    mlp_hc: fb.Ops.Hc,
    gdn: ?gt.Layer = null,
    attn: ?fb.Ops.Attn = null,
};

/// The trainer over the engine `e`, its captured kernels launched through `Tri`.
pub fn Trainer(comptime Tri: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        e: *prompt.Engine,
        tri: Tri,
        seq: prompt.Seq,
        k: fb.Kernels,
        gk: gt.Kernels,
        owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
        w: [layers_n]LayerW = undefined,
        mixer: fb.Ops.Hc = undefined,
        head: fb.Q4 = undefined,
        /// What the forward keeps: every layer's MoE input and weights, and from keep_low on all a backward reads.
        full: batch.Keep = .{},
        light: batch.Keep = .{},
        logits: u64 = 0, // bf16 [pack_rows, vocab]: the logits, then their gradient
        // the backward's scratch
        g: u64 = 0,
        dbranch: u64 = 0,
        dmix: u64 = 0,
        dinj: u64 = 0,
        parts: u64 = 0,
        hs: fb.Ops.HcScratch = undefined,
        ms: fb.Ops.MoeScratch = undefined,
        as: fb.Ops.AttnScratch = undefined,
        gs: gt.Scratch = undefined,
        // host-mapped
        host: Mapped = undefined, // targets u32, weights f32, stats [2]f32, scales f32: [pack_rows] each
        inputs: Mapped = undefined, // bf16 [layers, pack_rows, D]: each layer's MoE input as the last keys or target pass read it
        gates: Mapped = undefined, // f32 [layers, pack_rows, slots]: each layer's routing weights, the shared gate last
        delta: Mapped = undefined, // f32 [max_seqs, 3, layers, D]: each sequence's nudges after each layer it names
        push: Mapped = undefined, // f32 [max_seqs, 3, layers, D]: the loss's gradient at each nudged row after each of them
        resid: [dims]u16 = undefined,
        // the learner's view
        spans: [max_seqs]Span = undefined,
        answers: [max_seqs]Answer = undefined,
        seqs: [max_seqs]Seq = undefined,
        nseq: usize = 0,
        rows: usize = 0,
        pack: usize = pack_rows - max_seqs, // rows a pass may hold: room left for the rows between sequences
        count: Count = .{},
        prefixes: struct {
            pub const most = 0; // no prompt prefix is run once and kept: every pass runs its sequences whole
        } = .{},
        nudge: train_mode.Nudge = .{},
        from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),

        pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *prompt.Engine, tri: Tri) !*Self {
            const t = try gpa.create(Self);
            errdefer gpa.destroy(t);
            t.* = .{ .gpa = gpa, .e = e, .tri = tri, .seq = undefined, .k = undefined, .gk = undefined };
            const d = e.driver;
            t.k = try fb.Kernels.load(d);
            errdefer t.k.deinit();
            t.gk = try gt.Kernels.load(d);
            errdefer t.gk.deinit();
            t.seq = try e.newSeq(max_rows);
            errdefer t.seq.deinit(gpa);
            errdefer for (t.owned.items) |*b| b.free();
            errdefer t.owned.deinit(gpa);
            _ = try e.reserve(pack_rows);
            const t0 = std.Io.Clock.awake.now(io);
            try t.loadWeights(io);
            const t1 = std.Io.Clock.awake.now(io);
            try t.carve();
            t.host = try Mapped.init(d, pack_rows * 20);
            errdefer t.host.deinit();
            t.inputs = try Mapped.init(d, layers_n * pack_rows * dims * 2);
            errdefer t.inputs.deinit();
            t.gates = try Mapped.init(d, layers_n * pack_rows * slots * 4);
            errdefer t.gates.deinit();
            t.delta = try Mapped.init(d, max_seqs * 3 * layers_n * dims * 4);
            errdefer t.delta.deinit();
            t.push = try Mapped.init(d, max_seqs * 3 * layers_n * dims * 4);
            errdefer t.push.deinit();
            try e.stream.synchronize();
            var bytes: usize = 0;
            for (t.owned.items) |b| bytes += b.len;
            std.log.info("slide: the trainer's MLX copies of the small linears read in {d:.1} s; its buffers {d:.2} GiB in all", .{ seconds(t0, t1), @as(f64, @floatFromInt(bytes)) / (1 << 30) });
            return t;
        }

        fn seconds(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(b.toNanoseconds() - a.toNanoseconds())) / 1e9;
        }

        pub fn deinit(t: *Self) void {
            const gpa = t.gpa;
            t.e.stream.synchronize() catch {};
            inline for (.{ "host", "inputs", "gates", "delta", "push" }) |f| @field(t, f).deinit();
            for (t.owned.items) |*b| b.free();
            t.owned.deinit(gpa);
            t.seq.deinit(gpa);
            t.gk.deinit();
            t.k.deinit();
            gpa.destroy(t);
        }

        fn device(t: *Self, bytes: usize) !u64 {
            var b = try cuda.DeviceBuffer.alloc(t.e.driver, @max(bytes, 256));
            errdefer b.free();
            try t.owned.append(t.gpa, b);
            return b.ptr;
        }

        fn devBytes(t: *Self, bytes: []const u8) !u64 {
            var b = try cuda.DeviceBuffer.fromHost(t.e.driver, bytes);
            errdefer b.free();
            try t.owned.append(t.gpa, b);
            return b.ptr;
        }

        fn ops(t: *const Self) fb.Ops {
            return .{ .k = &t.k, .s = t.e.stream.* };
        }

        fn gops(t: *const Self) gt.Ops {
            return .{ .k = &t.gk, .s = t.e.stream.* };
        }

        /// `name` (.weight, .scales, .biases under `prefix`) in MLX's layout on the device, [n, k].
        fn q4(t: *Self, store: *wt.Store, prefix: []const u8, name: []const u8, n: usize, k: usize) !fb.Q4 {
            var buf: [200]u8 = undefined;
            const got = try wt.join3(t.gpa, store, &buf, prefix, name);
            defer t.gpa.free(got.w);
            defer t.gpa.free(got.s);
            defer t.gpa.free(got.b);
            if (got.w.len != n * k / 2 or got.s.len != n * k / 16) return error.UnexpectedTensor;
            return .{ .w = try t.devBytes(got.w), .s = try t.devBytes(got.s), .b = try t.devBytes(got.b), .n = n, .k = k };
        }

        fn hc(t: *Self, store: *wt.Store, prefix: []const u8, inject: bool, scale: u64) !fb.Ops.Hc {
            return .{
                .down = try t.q4(store, prefix, ".input_mix_weight_down", low_rank, wide),
                .inject = if (inject) try t.q4(store, prefix, ".block_inject_weight", streams, wide) else null,
                .up = try t.q4(store, prefix, ".input_mix_weight_up", wide, low_rank),
                .scale = scale,
                .streams = streams,
                .d = dims,
                .low = low_rank,
                .eps = eps,
            };
        }

        fn gq(m: fb.Q4) gt.Q4 {
            return .{ .w = m.w, .s = m.s, .b = m.b, .n = m.n, .k = m.k };
        }

        /// The small linears a backward reads, from the checkpoint the engine loaded, for layers keep_low and up.
        fn loadWeights(t: *Self, io: std.Io) !void {
            const e = t.e;
            var store = try wt.Store.open(t.gpa, io, e.model_dir);
            defer store.deinit();
            var name: [160]u8 = undefined;
            const inv = e.batch_s.?.inv;
            for (keep_low..layers_n) |l| {
                const lw = &e.layers[l];
                const root = try std.fmt.bufPrint(&name, "language_model.model.layers.{d}", .{l});
                var pre: [200]u8 = undefined;
                var lw_t: LayerW = .{
                    .attn_hc = try t.hc(&store, try std.fmt.bufPrint(&pre, "{s}.attn_hyper_connection", .{root}), true, lw.attn_hc.scale.ptr),
                    .mlp_hc = undefined,
                };
                lw_t.mlp_hc = try t.hc(&store, try std.fmt.bufPrint(&pre, "{s}.mlp_hyper_connection", .{root}), true, lw.mlp_hc.scale.ptr);
                if (wt.linearLayer(l)) {
                    const lp = try std.fmt.bufPrint(&pre, "{s}.linear_attn", .{root});
                    var parts: [4][]const u8 = .{ ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a" };
                    const st = try wt.stitch(t.gpa, &store, lp, &parts);
                    defer t.gpa.free(st.w);
                    defer t.gpa.free(st.s);
                    defer t.gpa.free(st.b);
                    if (st.w.len != gdn.proj_width * dims / 2) return error.UnexpectedTensor;
                    const in_proj: fb.Q4 = .{ .w = try t.devBytes(st.w), .s = try t.devBytes(st.s), .b = try t.devBytes(st.b), .n = gdn.proj_width, .k = dims };
                    const out = try t.q4(&store, lp, ".out_proj", dims, gdn.value_dim);
                    lw_t.gdn = .{ .nk = gdn.nk, .nv = gdn.nv, .hidden = dims, .eps = eps, .conv_w = lw.cw.ptr, .a_log = lw.a_log.ptr, .dt_bias = lw.dt.ptr, .norm_w = lw.norm.ptr, .in_proj = gq(in_proj), .out_proj = gq(out) };
                } else {
                    const ap = try std.fmt.bufPrint(&pre, "{s}.self_attn", .{root});
                    var buf2: [200]u8 = undefined;
                    const apc = buf2[0..ap.len];
                    @memcpy(apc, ap);
                    lw_t.attn = .{
                        .q = try t.q4(&store, apc, ".q_proj", q_heads * 2 * head_dim, dims),
                        .k = try t.q4(&store, apc, ".k_proj", kv_heads * head_dim, dims),
                        .v = try t.q4(&store, apc, ".v_proj", kv_heads * head_dim, dims),
                        .o = try t.q4(&store, apc, ".o_proj", dims, out_k),
                        .qscale = lw.q_scale.ptr,
                        .kscale = lw.k_scale.ptr,
                        .inv = inv,
                        .heads = q_heads,
                        .kv_heads = kv_heads,
                        .dim = head_dim,
                        .half = wt.half,
                        .eps = eps,
                    };
                }
                store.closeFile();
                t.w[l] = lw_t;
            }
            t.mixer = try t.hc(&store, "language_model.model.hyper_connection_mixer", false, e.mixer.scale.ptr);
            t.head = try t.q4(&store, "language_model.lm_head", "", vocab, dims);
        }

        /// The keep lists and the backward's scratch, every buffer zeroed (rows between sequences read zeros).
        fn carve(t: *Self) !void {
            const R = pack_rows;
            const pairs = R * slots;
            const z = struct {
                fn get(tt: *Self, n: usize) !u64 {
                    const p = try tt.device(n);
                    const d = tt.e.driver;
                    try d.check(d.api.cuMemsetD8_v2(p, 0, @max(n, 256)), "cuMemsetD8");
                    return p;
                }
            }.get;
            for (0..layers_n) |l| {
                var k: batch.LayerKeep = .{};
                k.moe_x = try z(t, R * dims * 2);
                k.moe_wts = try z(t, R * slots * 4);
                k.moe_ids = try z(t, R * slots * 4);
                t.light.layers[l] = k;
                if (l >= keep_low) {
                    inline for (.{ "attn_hc", "mlp_hc" }) |f| @field(k, f) = .{ .h = try z(t, R * wide * 2), .normed = try z(t, R * wide * 2), .v = try z(t, R * (low_rank + streams) * 4), .up = try z(t, R * wide * 2), .inject = try z(t, R * streams * 2) };
                    k.mix_branch = try z(t, R * dims * 2);
                    k.moe_branch = try z(t, R * dims * 2);
                    if (wt.linearLayer(l)) {
                        k.proj = try z(t, R * gdn.proj_width * 2);
                    } else {
                        k.pq = try z(t, R * q_heads * 2 * head_dim * 2);
                        k.pk = try z(t, R * kv_heads * head_dim * 2);
                        k.pv = try z(t, R * kv_heads * head_dim * 2);
                        k.gated = try z(t, R * out_k * 2);
                    }
                }
                t.full.layers[l] = k;
            }
            t.full.mixer = .{ .h = try z(t, R * wide * 2), .normed = try z(t, R * wide * 2), .v = try z(t, R * low_rank * 4), .up = try z(t, R * wide * 2) };
            t.logits = try z(t, R * vocab * 2);
            t.g = try z(t, R * wide * 4);
            t.dbranch = try z(t, R * dims * 4);
            t.dmix = try z(t, R * dims * 4);
            t.dinj = try z(t, R * streams * 4);
            t.parts = try z(t, (R + 512) * dims * 4);
            t.hs = .{ .dn = try z(t, R * wide * 4), .du = try z(t, R * wide * 4), .dact = try z(t, R * low_rank * 4), .dv = try z(t, R * (low_rank + streams) * 4) };
            const total = experts_n + 1;
            t.ms = .{
                .logits = try z(t, R * total * 4),
                .order = try z(t, pairs * 4),
                .inv = try z(t, pairs * 4),
                .offsets = try z(t, (total + 1) * 4),
                .tiles = try z(t, 2 * fb.tileCap(pairs, total) * 4),
                .gu = try z(t, pairs * 2 * moe_w * 4),
                .dact = try z(t, pairs * moe_w * 4),
                .dgu = try z(t, pairs * 2 * moe_w * 4),
                .dxp = try z(t, pairs * dims * 4),
                .a = try z(t, pairs * 4),
                .dlog = 0,
            };
            const M = max_rows;
            const qw = q_heads * head_dim;
            const kw = kv_heads * head_dim;
            t.as = .{
                .qkv = try z(t, M * (qw + 2 * kw) * 2),
                .dg = try z(t, M * qw * 4),
                .dout = try z(t, M * qw * 4),
                .dpq = try z(t, M * 2 * qw * 4),
                .dq = try z(t, M * qw * 4),
                .dk = try z(t, M * kw * 4),
                .dv = try z(t, M * kw * 4),
                .dpk = try z(t, M * kw * 4),
                .probs = try z(t, q_heads * M * M * 4),
                .dscores = try z(t, q_heads * M * M * 4),
            };
            t.gs = gt.Scratch.carve(try z(t, gt.Scratch.bytes(M, gdn.nk, gdn.nv, dims)), M, gdn.nk, gdn.nv, dims);
        }

        // ---- the learner's hooks ----------------------------------------------------------------------------------

        fn targets(t: *const Self) []u32 {
            return t.host.slice(u32, pack_rows);
        }

        fn weights(t: *const Self) []f32 {
            return t.host.slice(f32, 2 * pack_rows)[pack_rows..];
        }

        fn stats(t: *const Self) [][2]f32 {
            return t.host.slice([2]f32, 2 * pack_rows)[pack_rows..];
        }

        fn scales(t: *const Self) []f32 {
            return t.host.slice(f32, 5 * pack_rows)[4 * pack_rows ..];
        }

        /// Layer l's MoE input (the MLP hyper-connection's mixed row, what the shared expert and the pocket read) at
        /// the pass's `row`, bf16 [D], as the last keys or target pass read it.
        pub fn normed(t: *const Self, l: usize, row: usize) []const u16 {
            return t.inputs.slice(u16, layers_n * pack_rows * dims)[(l * pack_rows + row) * dims ..][0..dims];
        }

        /// The shared expert's gate (sigmoid of its router logit) at layer l, the pass's `row`.
        pub fn gateAt(t: *const Self, l: usize, row: usize) f32 {
            return t.gates.slice(f32, layers_n * pack_rows * slots)[(l * pack_rows + row) * slots + slots - 1];
        }

        /// The streams after layer l - 1's MoE write-back at `row`, in the MoE output's units: sum_s inject_s h_s /
        /// sum_s inject_s^2 with layer l - 1's inject gates, so a change v of that MoE output moves it by v (bf16 [D];
        /// l == layers: after the last). As the last target pass kept it, read from the device now.
        pub fn residual(t: *Self, l: usize, row: usize) []const u16 {
            @memset(&t.resid, 0);
            if (l == 0 or l > layers_n or l - 1 < keep_low) return &t.resid;
            const kp = t.full;
            const h_at = if (l < layers_n) kp.layers[l].attn_hc.h else kp.mixer.h;
            if (h_at == 0) return &t.resid;
            var h: [wide]u16 = undefined;
            var inj: [streams]u16 = undefined;
            const d = t.e.driver;
            t.e.stream.synchronize() catch return &t.resid;
            d.check(d.api.cuMemcpyDtoH_v2(&h, h_at + row * wide * 2, wide * 2), "cuMemcpyDtoH") catch return &t.resid;
            d.check(d.api.cuMemcpyDtoH_v2(&inj, kp.layers[l - 1].mlp_hc.inject + row * streams * 2, streams * 2), "cuMemcpyDtoH") catch return &t.resid;
            var a: [streams]f32 = undefined;
            var norm: f32 = 0;
            for (&a, inj) |*x, b| {
                x.* = wt.promote(b);
                norm += x.* * x.*;
            }
            if (norm == 0) return &t.resid;
            for (0..dims) |i| {
                var v: f32 = 0;
                for (0..streams) |s| v += a[s] * wt.promote(h[s * dims + i]);
                t.resid[i] = wt.toBf16(v / norm);
            }
            return &t.resid;
        }

        fn vec(s: usize, i: usize, l: usize) usize {
            return ((s * 3 + i) * layers_n + l) * dims;
        }

        /// The loss's gradient sequence s's target pass read at its key i's row after layer l (dL/d the MoE output).
        pub fn pushAt(t: *const Self, s: usize, i: usize, l: usize) []const f32 {
            return t.push.slice(f32, max_seqs * 3 * layers_n * dims)[vec(s, i, l)..][0..dims];
        }

        /// Sequence s's nudge for key i after layer l, which its target pass adds to that layer's MoE output.
        pub fn deltaAt(t: *const Self, s: usize, i: usize, l: usize) []f32 {
            return t.delta.slice(f32, max_seqs * 3 * layers_n * dims)[vec(s, i, l)..][0..dims];
        }

        /// Whether a pass may hold several sequences: always (each runs from its own start).
        pub fn packs(_: *const Self) bool {
            return true;
        }

        /// No prompt prefix is kept (prefixes.most is 0): nothing to make room for.
        pub fn prefixSlots(_: *Self, n: usize) !void {
            if (n > 0) return error.TooManyPrefixes;
        }

        pub fn step(t: *Self, ids: []const u32, start: usize, mode: Mode, turn: lesson_route.Turn) !Result {
            var out: [1]Result = undefined;
            try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
            return out[0];
        }

        // ---- the pass ---------------------------------------------------------------------------------------------

        /// `seqs` side by side in one pass: keys (each layer's MoE inputs and gates for the host), loss (each answer's
        /// mean loss and whether it comes back token for token), target (that, and dL/d each nudged layer's MoE
        /// output at its nudged rows). The open fact's pocket neurons are scaled per sequence by its `scale`.
        pub fn pass(t: *Self, seqs: []const Seq, mode: Mode, turn: lesson_route.Turn, out: []Result) !void {
            _ = turn;
            if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
            switch (mode) {
                .keys, .loss, .target => {},
                .prime => {
                    for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                    return;
                },
                else => return error.RoutedUnsupported,
            }
            const e = t.e;
            const d = e.driver;
            const stream = e.stream.*;
            var rows: usize = 0;
            for (seqs, t.spans[0..seqs.len]) |sq, *sp| {
                if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
                if (sq.skip != 0) return error.BadPrefix;
                rows = up(rows, 2);
                sp.* = .{ .at = rows, .rows = sq.ids.len - 1 };
                rows += sp.rows;
            }
            if (rows > pack_rows) return error.PassTooLong;
            const spans = t.spans[0..seqs.len];
            @memcpy(t.seqs[0..seqs.len], seqs);
            t.nseq = seqs.len;
            t.rows = rows;
            t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
            try stream.synchronize(); // the host rows below are memory the last pass's kernels may still read
            const ids = try t.gpa.alloc(u32, rows);
            defer t.gpa.free(ids);
            @memset(ids, @intCast(e.eos));
            @memset(t.scales()[0..rows], 1);
            @memset(t.targets()[0..rows], 0);
            @memset(t.weights()[0..rows], 0);
            _ = train_mode.answers(seqs, spans, t.answers[0..seqs.len]);
            for (seqs, spans, t.answers[0..seqs.len]) |sq, sp, a| {
                @memcpy(ids[sp.at..][0..sp.rows], sq.ids[0..sp.rows]);
                @memset(t.scales()[sp.at..][0..sp.rows], sq.scale);
                const share = 1 / @as(f32, @floatFromInt(a.rows));
                for (0..a.rows) |r| {
                    t.targets()[a.first + r] = sq.ids[sq.start + r];
                    t.weights()[a.first + r] = share;
                }
            }
            var low: usize = layers_n;
            if (mode == .target) {
                var mask: u64 = 0;
                for (seqs) |sq| mask |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
                if (mask != 0) low = @ctz(mask);
                if (low < keep_low) return error.NudgeBelowKept;
            }
            // the forward
            var keep = t.light;
            if (mode == .target) {
                // from the lowest nudge up; with none, every layer a host's residual reads
                keep = t.full;
                for (0..if (low == layers_n) keep_low else low) |l| keep.layers[l] = t.light.layers[l];
            }
            const s = &e.batch_s.?;
            var p: batch.Pass(Tri) = .{
                .gpa = t.gpa,
                .io = e.io,
                .stream = stream,
                .tri = t.tri,
                .k = &e.batch_k,
                .s = s,
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram },
                .seq = .{ .conv = t.seq.conv, .state = t.seq.state, .kc = t.seq.kc, .vc = t.seq.vc, .ikc = t.seq.ikc, .tail = t.seq.tail, .history = &t.seq.history },
                .hooks = .{ .logits = if (mode == .keys) .none else .all, .logits_out = t.logits, .keep = &keep, .moe_add = .{ .ctx = t, .add = moeAdd } },
            };
            var list: [max_seqs]batch.Span = undefined;
            for (spans, list[0..seqs.len]) |sp, *x| x.* = .{ .at = sp.at, .rows = sp.rows };
            try p.spans(ids, list[0..seqs.len], e.eos);
            const o = t.ops();
            if (mode == .keys or mode == .target) {
                // each layer's MoE inputs and routing weights where the host reads them
                for (0..layers_n) |l| {
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(t.inputs.dev + l * pack_rows * dims * 2, keep.layers[l].moe_x, rows * dims * 2, stream.handle), "cuMemcpyDtoDAsync");
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(t.gates.dev + l * pack_rows * slots * 4, keep.layers[l].moe_wts, rows * slots * 4, stream.handle), "cuMemcpyDtoDAsync");
                }
            }
            if (mode == .keys) {
                try stream.synchronize();
                for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                return;
            }
            // every row's softmax: the answer rows' (p - onehot) / their count, every other row's zero
            const hd = t.host.dev;
            try o.softmax(t.logits, hd, hd + pack_rows * 4, hd + pack_rows * 8, rows, vocab);
            if (mode == .target and low < layers_n) try t.backward(&keep, rows, low);
            try stream.synchronize();
            for (out[0..seqs.len], t.answers[0..seqs.len]) |*r, a| {
                r.* = .{ .loss = 0, .recalled = true };
                for (t.stats()[a.first..][0..a.rows], t.weights()[a.first..][0..a.rows]) |st, w| {
                    r.loss += st[0] * w;
                    r.recalled = r.recalled and st[1] > 0.5;
                }
            }
        }

        /// The pass's moe_add: each layer's pocket (the open fact's neurons scaled by their row's sequence), then
        /// each sequence's nudges at their rows.
        fn moeAdd(ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const o = t.ops();
            if (t.e.pocketOf(layer)) |pk0| {
                var pk = pk0;
                pk.scale = t.host.dev + 4 * pack_rows * 4;
                pk.from = t.from[layer];
                try o.pocketRows(pk, .{ .x = x, .ids = ids, .wts = wts }, out, true, rows, dims, top_k);
            }
            for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
                if (mask >> @intCast(layer) & 1 == 0) continue;
                try o.axpy(out + sp.row(row) * dims * 4, t.delta.dev + vec(si, i, layer) * 4, dims);
            };
        }

        fn copyPush(t: *Self, l: usize) !void {
            const d = t.e.driver;
            for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
                if (mask >> @intCast(l) & 1 == 0) continue;
                try d.check(d.api.cuMemcpyDtoDAsync_v2(t.push.dev + vec(si, i, l) * 4, t.dbranch + sp.row(row) * dims * 4, dims * 4, t.e.stream.handle), "cuMemcpyDtoDAsync");
            };
        }

        /// The loss back from the head to layer `low`'s MoE output, its nudged rows' gradients into `push` on the way.
        fn backward(t: *Self, keep: *const batch.Keep, rows: usize, low: usize) !void {
            const o = t.ops();
            const e = t.e;
            // the head (its bf16 rows' gradient in place) back to the final mixer's output, then its read-out
            try o.headBack(t.logits, t.head, t.dmix, rows, t.parts);
            try o.zero(t.g, rows * wide);
            try o.hcReadoutBack(t.mixer, hcKept(keep.mixer), t.dmix, 0, t.g, t.hs, rows);
            var l = layers_n;
            while (l > low) {
                l -= 1;
                const kp = keep.layers[l];
                const w = t.w[l];
                // the MoE write-back h' = h + moe inject: its output's gradient, and its inject gates'
                try o.hcBranchBack(w.mlp_hc, t.g, kp.mlp_hc.inject, kp.moe_branch, t.dbranch, t.dinj, rows);
                try t.copyPush(l);
                if (l == low) return;
                try o.zero(t.dmix, rows * dims);
                var pk = e.pocketOf(l);
                if (pk) |*x| {
                    x.scale = t.host.dev + 4 * pack_rows * 4;
                    x.from = t.from[l];
                }
                const lw = &e.layers[l];
                try o.moeBackPacked(.{ .up = lw.experts_up.ptr, .down = lw.experts_down.ptr, .routed = experts_n, .width = moe_w, .dim = dims }, lw.router.ptr, .{ .x = kp.moe_x, .ids = kp.moe_ids, .wts = kp.moe_wts }, pk, t.dbranch, t.dmix, t.ms, rows, top_k);
                try o.hcReadoutBack(w.mlp_hc, hcKept(kp.mlp_hc), t.dmix, t.dinj, t.g, t.hs, rows);
                // the mixer's write-back, the mixer one sequence at a time, its read-out
                try o.hcBranchBack(w.attn_hc, t.g, kp.attn_hc.inject, kp.mix_branch, t.dbranch, t.dinj, rows);
                try o.zero(t.dmix, rows * dims);
                for (t.spans[0..t.nseq]) |sp| {
                    const gy = t.dbranch + sp.at * dims * 4;
                    const dx = t.dmix + sp.at * dims * 4;
                    if (w.gdn) |g| {
                        try t.gops().backward(g, t.gs, kp.proj + sp.at * gdn.proj_width * 2, null, null, gy, dx, sp.rows, false);
                    } else {
                        var at = w.attn.?;
                        at.inv = e.batch_s.?.inv; // RoPE's frequencies in the pass's scratch, wherever it now sits
                        const kept: fb.Ops.AttnKept = .{ .pq = kp.pq + sp.at * q_heads * 2 * head_dim * 2, .pk = kp.pk + sp.at * kv_heads * head_dim * 2, .pv = kp.pv + sp.at * kv_heads * head_dim * 2, .gated = kp.gated + sp.at * out_k * 2, .pos0 = 0 };
                        try o.attnBack(at, kept, gy, dx, t.as, sp.rows, false);
                    }
                }
                try o.hcReadoutBack(w.attn_hc, hcKept(kp.attn_hc), t.dmix, t.dinj, t.g, t.hs, rows);
            }
        }

        fn hcKept(k: batch.HcKeep) fb.Ops.HcKept {
            return .{ .h = k.h, .normed = k.normed, .v = k.v, .up = k.up, .inject = k.inject };
        }
    };
}
