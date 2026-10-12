//! Flash Next's MTP drafts on CUDA (Python qwen4_exp/cuda/mtp.py, decode.py draft and mtp_decode, forward.commit).
//!
//! The head reads the next token's embedding (RMSNorm, fc_embedding) and the main model's streams after the last MoE
//! write-back (one RMSNorm over all four streams, fc_hidden per stream), adds the embedding branch to each stream, runs
//! one full-attention layer with its own hyper-connections, 512 experts and KV cache, then its own mixer and the
//! lm_head rows of the 79,591-id draft vocabulary. A chain keeps its first draft and stops before a later one whose
//! temperature-1 probability falls under the confidence.
//!
//! A verify window (the pending token and the drafts) runs as one batched pass (cuda_batch.zig), whose rows keep the
//! bits they have alone, and each row is drawn at its own keyed position with the sampler serial decoding uses; so the
//! kept rows are the tokens serial decoding emits. A partly kept window restores the DeltaNet states, conv windows and
//! the n-gram tail as they stood before it and replays the kept rows' recurrence from the projection rows the pass
//! kept (the same launch over fewer rows: the same bits); the attention caches past the kept rows are simply
//! overwritten later.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const vocab = wt.vocab;
const eps = wt.eps;
const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const state_n = wt.state_n;
const out_sk = wt.out_sk;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;
const ple_tail_n = wt.ple_tail_n;
const Q4 = wt.Q4;
const HcW = wt.HcW;

pub const Sampling = lanes.sampling.Sampling;

/// Drafts a round at most (Python DEPTH).
pub const max_chain: usize = 6;
/// The live recipe's --mtp-confidence (Python's CONFIDENCE default is 0.7).
pub const default_confidence: f64 = 0.6;
/// A verify window's rows at most: the pending token and a chain, or a copied continuation.
pub const max_rows: usize = 16;
/// qmm.split_for(2560, 2560): fc_embedding and fc_hidden sum K in four slices, in order.
const fc_sk: usize = 4;
/// Head rows one launch takes at most (a prompt's rows go through in pieces).
const piece_rows: usize = 256;
/// Normed-stream partial sums a row (cuda_batch.zig's pss_n).
const pss_n = (dims / 256) * streams;
const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const proj_bytes = gdn.proj_width * 2;

const vocab_text = @embedFile("draft_vocab_default.txt");

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// The draft vocabulary's ids, sorted (Python draft_token_ids("default")): column j of the draft head is ids[j].
pub fn draftIds(gpa: std.mem.Allocator) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, vocab_text, " \t\r\n,");
    while (it.next()) |part| {
        const id = try std.fmt.parseInt(u32, part, 10);
        if (id >= vocab) continue;
        try out.append(gpa, id);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (out.items, 0..) |id, i| {
        if (i > 0 and id == out.items[n - 1]) continue;
        out.items[n] = id;
        n += 1;
    }
    out.shrinkRetainingCapacity(n);
    return out.toOwnedSlice(gpa);
}

// ---- sampling ------------------------------------------------------------------------------------------------------

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A row's first maximum (argmax's choice): its column.
pub fn argmaxRow(row: []const u16) usize {
    var best = promote(row[0]);
    var at: usize = 0;
    for (row[1..], 1..) |b, i| {
        const v = promote(b);
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return at;
}

/// The keyed draw over one row of bf16 scores, column j being token ids[j] (or j): the server's sampler
/// (lanes.sampling.choose) over the row's top_k columns by value then id, which is the token it draws over every
/// column (it keeps those top_k first); top_k off reads every column. Greedy: the first maximum.
pub fn drawRow(gpa: std.mem.Allocator, row: []const u16, ids: ?[]const u32, position: u64, sm: ?Sampling) !u32 {
    const id = struct {
        fn of(map: ?[]const u32, col: usize) u64 {
            return if (map) |m| m[col] else col;
        }
    }.of;
    const s = sm orelse return @intCast(id(ids, argmaxRow(row)));
    if (s.temperature <= 0) return @intCast(id(ids, argmaxRow(row)));
    const n = row.len;
    const k = if (s.top_k == 0 or s.top_k >= n) n else @as(usize, s.top_k);
    const vals = try gpa.alloc(f64, k);
    defer gpa.free(vals);
    const keys = try gpa.alloc(u64, k);
    defer gpa.free(keys);
    if (k == n) {
        for (vals, keys, row, 0..) |*v, *key, b, col| {
            v.* = promote(b);
            key.* = id(ids, col);
        }
    } else {
        // the k best columns, best first; a later column with an equal value ranks below (the id order)
        const top_v = try gpa.alloc(f32, k);
        defer gpa.free(top_v);
        const top_c = try gpa.alloc(usize, k);
        defer gpa.free(top_c);
        var held: usize = 0;
        for (row, 0..) |b, col| {
            const v = promote(b);
            if (held == k and !(v > top_v[k - 1])) continue;
            var at: usize = if (held < k) held else k - 1;
            if (held < k) held += 1;
            while (at > 0 and v > top_v[at - 1]) : (at -= 1) {
                top_v[at] = top_v[at - 1];
                top_c[at] = top_c[at - 1];
            }
            top_v[at] = v;
            top_c[at] = col;
        }
        for (vals, keys, top_v, top_c) |*v, *key, tv, c| {
            v.* = tv;
            key.* = id(ids, c);
        }
    }
    return @intCast(try lanes.sampling.choose(gpa, vals, keys, position, s));
}

// ---- the head's weights --------------------------------------------------------------------------------------------

pub const Head = struct {
    gpa: std.mem.Allocator,
    norm_e: cuda.DeviceBuffer, // [D] fp32
    norm_h: cuda.DeviceBuffer, // [S D] fp32
    fc_e: Q4,
    fc_h: Q4,
    layer: load.Layer,
    mixer: HcW,
    /// lm_head's rows at `ids`, in their order.
    draft: Q4,
    ids: []u32,
    module: cuda.Module,
    best_fn: cuda.Function,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, dir: []const u8) !*Head {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.gpa = gpa;
        var store = try wt.Store.open(gpa, io, dir);
        defer store.deinit();
        var buf: [120]u8 = undefined;
        const ne = try wt.asF32(gpa, try store.tensor("language_model.mtp.pre_fc_norm_embedding.weight"));
        defer gpa.free(ne);
        const nh = try wt.asF32(gpa, try store.tensor("language_model.mtp.pre_fc_norm_hidden.weight"));
        defer gpa.free(nh);
        if (ne.len != dims or nh.len != wide) return error.UnexpectedTensor;
        h.norm_e = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ne));
        errdefer h.norm_e.free();
        h.norm_h = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(nh));
        errdefer h.norm_h.free();
        h.fc_e = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_embedding");
        errdefer h.fc_e.free();
        h.fc_h = try fc(gpa, driver, &store, &buf, "language_model.mtp.fc_hidden");
        errdefer h.fc_h.free();
        h.mixer = try wt.loadHc(gpa, driver, &store, "language_model.mtp.hyper_connection_mixer", false);
        errdefer h.mixer.free();
        {
            var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
            defer pack_mod.unload();
            const pack_fn = try pack_mod.function(wt.pack_symbol);
            h.layer = try load.loadLayerAt(gpa, driver, stream, &store, pack_fn, "language_model.mtp.layers.0", false, false);
        }
        errdefer h.layer.free(gpa);
        h.ids = try draftIds(gpa);
        errdefer gpa.free(h.ids);
        {
            const head_w = try wt.join3(gpa, &store, &buf, "language_model.lm_head", "");
            defer gpa.free(head_w.w);
            defer gpa.free(head_w.s);
            defer gpa.free(head_w.b);
            const wrow = (dims / 8) * 4;
            const grow = (dims / 32) * 2;
            if (head_w.w.len != vocab * wrow or head_w.s.len != vocab * grow) return error.UnexpectedTensor;
            const n = h.ids.len;
            const w = try gpa.alloc(u8, n * wrow);
            defer gpa.free(w);
            const s = try gpa.alloc(u8, n * grow);
            defer gpa.free(s);
            const b = try gpa.alloc(u8, n * grow);
            defer gpa.free(b);
            for (h.ids, 0..) |id, j| {
                @memcpy(w[j * wrow ..][0..wrow], head_w.w[id * wrow ..][0..wrow]);
                @memcpy(s[j * grow ..][0..grow], head_w.s[id * grow ..][0..grow]);
                @memcpy(b[j * grow ..][0..grow], head_w.b[id * grow ..][0..grow]);
            }
            h.draft = try wt.uploadQ4(gpa, driver, w, s, b, n, dims / 8);
        }
        errdefer h.draft.free();
        h.module = try cuda.Module.load(driver, cuda.kernels.flashnext_mtp);
        errdefer h.module.unload();
        h.best_fn = try h.module.function("fn_row_best");
        return h;
    }

    fn fc(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *wt.Store, buf: []u8, name: []const u8) !Q4 {
        const t = try wt.join3(gpa, store, buf, name, "");
        defer gpa.free(t.w);
        defer gpa.free(t.s);
        defer gpa.free(t.b);
        if (t.w.len != dims * (dims / 8) * 4) return error.UnexpectedTensor;
        return wt.uploadQ4(gpa, driver, t.w, t.s, t.b, dims, dims / 8);
    }

    pub fn deinit(h: *Head) void {
        h.module.unload();
        h.draft.free();
        h.gpa.free(h.ids);
        h.layer.free(h.gpa);
        h.mixer.free();
        h.fc_h.free();
        h.fc_e.free();
        h.norm_h.free();
        h.norm_e.free();
        h.gpa.destroy(h);
    }

    /// fn_row_best over `rows` rows of `v` scores `ld` apart: [rows, 4] fp32 at `out` (best, log-sum-exp, id bits).
    fn best(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.best_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }
};

/// Device bytes the head's weights take beyond the main model's (for the server's admission estimate).
pub fn headBytes() usize {
    const experts = (wt.experts_n + 1) * 3 * wt.moe_w * dims / 2 * 5 / 4;
    const attn = (wt.proj_n + dims) * dims / 2 * 5 / 4;
    const draft = 79_591 * dims / 2 * 5 / 4;
    return experts + attn + draft + 2 * dims * dims / 2 * 5 / 4;
}

// ---- a sequence's drafting state -----------------------------------------------------------------------------------

/// One draft and its temperature-1 probability under the head.
pub const Pick = struct { token: u32, prob: f64 };

/// A chain of drafts the head holds for the next window.
pub const Chain = struct {
    tokens: [max_chain]u32 = undefined,
    probs: [max_chain]f64 = undefined,
    n: usize = 0,

    pub fn slice(c: *const Chain) []const u32 {
        return c.tokens[0..c.n];
    }
};

/// A sequence's head cache and the buffers a window needs: its rows' scores and streams, and what it rolls back to.
pub const Draft = struct {
    gpa: std.mem.Allocator,
    kc: cuda.DeviceBuffer,
    vc: cuda.DeviceBuffer,
    ikc: cuda.DeviceBuffer,
    /// Entries the head's cache holds (Python mtp_len), the last `drafted` of them a chain's.
    len: usize = 0,
    drafted: usize = 0,
    cap: usize,
    /// The last pass's rows of main-model streams [cap rows, wide]: the prompt's, then each window's.
    streams: cuda.DeviceBuffer,
    /// The head's last output row (its streams after the MoE write-back): the next chained draft's input.
    out: cuda.DeviceBuffer,
    /// A window's scores [max_rows, vocab], the draft head's [ids], best values [max_rows, 4].
    logits: cuda.DeviceBuffer,
    dlogits: cuda.DeviceBuffer,
    best: cuda.DeviceBuffer,
    /// Before a window: each DeltaNet layer's state, conv window and (from the pass) projection rows; the n-gram tail.
    saved: cuda.DeviceBuffer,
    keep: batch.Keep = .{},
    history: [2]i64 = .{ 0, 0 },
    unused_history: [2]i64 = .{ 0, 0 },
    fed0: usize = 0,
    rows: usize = 0,
    ids: [max_rows]u32 = undefined,
    /// The scores a sampled draw reads on the host (a window's rows, or the draft head's row).
    host: []u16,
    best_host: [max_rows * 4]f32 = undefined,

    const state_at: usize = 0;
    const hist_at: usize = linear_n * state_n * 4;
    const proj_at: usize = hist_at + linear_n * conv_bytes;
    const tail_at: usize = proj_at + linear_n * max_rows * proj_bytes;
    const saved_bytes: usize = tail_at + ple_tail_n * wide * 2;

    /// Device bytes `init(cap, draft_n)` allocates.
    pub fn bytes(cap: usize, draft_n: usize) usize {
        return cap * (2 * kv_heads * head_dim * 2 + index_dim * 2) + cap * wide * 2 + wide * 2 + max_rows * vocab * 2 + draft_n * 2 + max_rows * 16 + saved_bytes;
    }

    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, cap: usize, draft_n: usize) !Draft {
        var d: Draft = .{ .gpa = gpa, .cap = cap, .kc = undefined, .vc = undefined, .ikc = undefined, .streams = undefined, .out = undefined, .logits = undefined, .dlogits = undefined, .best = undefined, .saved = undefined, .host = undefined };
        d.kc = try cuda.DeviceBuffer.alloc(driver, cap * kv_heads * head_dim * 2);
        errdefer d.kc.free();
        d.vc = try cuda.DeviceBuffer.alloc(driver, cap * kv_heads * head_dim * 2);
        errdefer d.vc.free();
        d.ikc = try cuda.DeviceBuffer.alloc(driver, cap * index_dim * 2);
        errdefer d.ikc.free();
        d.streams = try cuda.DeviceBuffer.alloc(driver, @max(cap, max_rows) * wide * 2);
        errdefer d.streams.free();
        d.out = try cuda.DeviceBuffer.alloc(driver, wide * 2);
        errdefer d.out.free();
        d.logits = try cuda.DeviceBuffer.alloc(driver, max_rows * vocab * 2);
        errdefer d.logits.free();
        d.dlogits = try cuda.DeviceBuffer.alloc(driver, draft_n * 2);
        errdefer d.dlogits.free();
        d.best = try cuda.DeviceBuffer.alloc(driver, max_rows * 16);
        errdefer d.best.free();
        d.saved = try cuda.DeviceBuffer.alloc(driver, saved_bytes);
        errdefer d.saved.free();
        d.host = try gpa.alloc(u16, @max(max_rows * vocab, draft_n));
        errdefer gpa.free(d.host);
        try d.kc.fill8(0, null);
        try d.vc.fill8(0, null);
        try d.ikc.fill8(0, null);
        var li: usize = 0;
        for (&d.keep.layers, 0..) |*lk, layer| {
            if (!wt.linearLayer(layer)) continue;
            lk.state = d.saved.ptr + state_at + li * state_n * 4;
            lk.hist = d.saved.ptr + hist_at + li * conv_bytes;
            lk.proj = d.saved.ptr + proj_at + li * max_rows * proj_bytes;
            li += 1;
        }
        return d;
    }

    pub fn deinit(d: *Draft) void {
        d.gpa.free(d.host);
        d.saved.free();
        d.best.free();
        d.dlogits.free();
        d.logits.free();
        d.out.free();
        d.streams.free();
        d.ikc.free();
        d.vc.free();
        d.kc.free();
    }

    /// Drop the head's cache (a new prompt).
    pub fn reset(d: *Draft) void {
        d.len = 0;
        d.drafted = 0;
        d.rows = 0;
    }
};

fn copyAsync(stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
    if (n == 0) return;
    const d = stream.d;
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
}

fn download(stream: cuda.Stream, dst: []u8, src: u64) !void {
    const d = stream.d;
    try stream.synchronize();
    try d.check(d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
}

/// The head's work over a sequence: the prompt pass with its streams kept, verify windows and their rollback, the
/// head's forward and its chains. `Tri` is the captured-kernel launcher.
pub fn Ops(comptime Tri: type) type {
    return struct {
        e: *prompt.Engine,
        tri: Tri,
        h: *Head,

        const Self = @This();

        fn stream(o: Self) cuda.Stream {
            return o.e.stream.*;
        }

        /// The prompt's pass with every row's streams kept, and the head's cache over every row but the last (the
        /// first token, still undrawn, is the last row's next): Python prefill_chunk's mtp_forward.
        pub fn prefill(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32) !void {
            if (ids.len > d.cap) return error.PromptTooLong;
            d.reset();
            try o.e.forward(Tri, o.tri, seq, ids, .{ .streams = d.streams.ptr });
            if (ids.len > 1) try o.forwardHead(d, ids[1..], d.streams.ptr, false);
        }

        /// The head over `next.len` rows from main-model streams `rows` [n, wide]: its cache takes them at entries
        /// d.len.. (advancing d.len), its last output row lands in d.out and, with `scores`, the draft head's scores
        /// of that row in d.dlogits.
        pub fn forwardHead(o: Self, d: *Draft, next: []const u32, rows: u64, scores: bool) !void {
            const e = o.e;
            const tri = o.tri;
            const h = o.h;
            if (next.len == 0) return;
            if (d.len + next.len > d.cap or d.len + next.len > wt.nch * 512) return error.PromptTooLong;
            const want = @max(@as(usize, 32), 2 * @min(next.len, piece_rows));
            const s = try e.reserve(want);
            const piece = @max(@as(usize, 1), @min(piece_rows, s.cap / 2));
            var p: batch.Pass(Tri) = .{
                .gpa = e.gpa,
                .io = e.io,
                .stream = e.stream.*,
                .tri = tri,
                .k = &e.batch_k,
                .s = s,
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
                .seq = .{ .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .tail = d.kc, .history = &d.unused_history },
                .hooks = .{ .logits = .none },
            };
            var at: usize = 0;
            while (at < next.len) {
                const m = @min(piece, next.len - at);
                p.rows = m;
                p.pos0 = d.len;
                p.pending = null;
                try p.stage(next[at..][0..m]);
                // the embedding branch: the token's row (stream 0 of the four copies), normed, fc_embedding
                try tri.embed(s.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, s.ple_keys, dims, streams, m);
                try tri.rmsnorm(s.ple_keys, h.norm_e.ptr, s.mixed, s.xsm, eps, wide, m, dims, dims);
                try p.lane(s.mixed, s.xsm, h.fc_e, s.dummy, s.part, m, fc_sk);
                try p.sumSlices(s.part, s.branch, true, m * dims, fc_sk);
                // the hidden branch: the four streams normed as one row, fc_hidden on each stream
                try tri.rmsnorm(rows + at * wide * 2, h.norm_h.ptr, s.norm, s.gq, eps, wide, m, wide, wide);
                try p.lane(s.norm, s.gq, h.fc_h, s.dummy, s.part, streams * m, fc_sk);
                try p.sumSlices(s.part, s.ple_gated, true, streams * m * dims, fc_sk);
                try tri.addStreams(s.branch, s.ple_gated, s.h, m, dims, streams);
                // the layer: attention over the head's own cache, its experts
                try p.writeback(0, m);
                try p.readout(h.layer.attn_hc, 0, m, .{});
                try p.attnBlock(&h.layer, 0, .{});
                try tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
                try p.readout(h.layer.mlp_hc, 0, m, .{});
                try p.moeBlock(&h.layer, .{});
                p.pending = 0;
                try p.writeback(0, m);
                try p.copy(d.out.ptr, s.h + (m - 1) * wide * 2, wide * 2);
                d.len += m;
                at += m;
                if (at == next.len and scores) {
                    // the last row's streams and sums at row 0, where the captured kernels' pointers stay aligned
                    if (m > 1) {
                        try p.copy(s.h, s.h + (m - 1) * wide * 2, wide * 2);
                        try p.copy(s.pss, s.pss + (m - 1) * pss_n * 4, pss_n * 4);
                    }
                    try p.readout(h.mixer, 0, 1, .{});
                    try p.lane(s.mixed, s.xsm, h.draft, d.dlogits.ptr, s.dummy, 1, 1);
                }
            }
        }

        /// Python draft.absorb: the chain's entries trimmed, then the kept rows (main streams `rows`, next tokens).
        pub fn absorb(o: Self, d: *Draft, next: []const u32, rows: u64, scores: bool) !void {
            if (d.drafted > 0) {
                d.len -= d.drafted;
                d.drafted = 0;
            }
            try o.forwardHead(d, next, rows, scores);
        }

        /// The draft at `position` from the draft head's scores and its temperature-1 probability (Python
        /// Engine.sample_draft): the first maximum when greedy, else the keyed draw over the draft vocabulary.
        pub fn pick(o: Self, d: *Draft, position: u64, sm: ?Sampling) !Pick {
            const h = o.h;
            const n = h.ids.len;
            try h.best(o.stream(), d.dlogits.ptr, n, n, 1, d.best.ptr);
            try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0..4]), d.best.ptr);
            const top = d.best_host[0];
            const lse = d.best_host[1];
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy) {
                const col: usize = @intCast(@as(i32, @bitCast(d.best_host[2])));
                return .{ .token = h.ids[col], .prob = @exp(@as(f64, top) - @as(f64, lse)) };
            }
            const row = d.host[0..n];
            try download(o.stream(), std.mem.sliceAsBytes(row), d.dlogits.ptr);
            const tok = try drawRow(o.e.gpa, row, h.ids, position, sm);
            const col = std.sort.binarySearch(u32, h.ids, tok, struct {
                fn order(t: u32, x: u32) std.math.Order {
                    return std.math.order(t, x);
                }
            }.order) orelse return .{ .token = tok, .prob = 0 };
            return .{ .token = tok, .prob = @exp(@as(f64, promote(row[col])) - @as(f64, lse)) };
        }

        /// Python decode.draft: absorb the kept rows, then chain up to `count` drafts from `position`, always keeping
        /// the first, stopping before a later one under `confidence` (or after a first one under it).
        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            var j: usize = 0;
            while (j < @min(count, max_chain)) : (j += 1) {
                const got = try o.pick(d, position + j, sm);
                const low = confidence > 0 and got.prob < confidence;
                if (low and j > 0) break;
                out.tokens[out.n] = got.token;
                out.probs[out.n] = got.prob;
                out.n += 1;
                if (low) break;
                if (j + 1 < count) {
                    try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
                    d.drafted += 1;
                }
            }
        }

        /// One forward over the window `ids` from seq.fed (the pending token, then drafts): every row's scores in
        /// d.logits, its streams in d.streams, and (for a rollback) the states it started from.
        pub fn verify(o: Self, seq: *prompt.Seq, d: *Draft, ids: []const u32) !void {
            const n = ids.len;
            if (n == 0 or n > max_rows) return error.WindowTooWide;
            if (seq.fed + n > seq.cap) return error.PromptTooLong;
            d.fed0 = seq.fed;
            d.rows = n;
            d.history = seq.history;
            @memcpy(d.ids[0..n], ids);
            const keeps = n > 1;
            if (keeps) try copyAsync(o.stream(), d.saved.ptr + Draft.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
            try o.e.forward(Tri, o.tri, seq, ids, .{ .logits = .all, .logits_out = d.logits.ptr, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr });
        }

        /// The window's rows drawn at their keyed positions (seq position fed0 + 1 + r), as serial decoding draws.
        pub fn draws(o: Self, d: *Draft, sm: ?Sampling, out: []u32) !void {
            const n = d.rows;
            const greedy = if (sm) |s| s.temperature <= 0 else true;
            if (greedy) {
                try o.h.best(o.stream(), d.logits.ptr, vocab, vocab, n, d.best.ptr);
                try download(o.stream(), std.mem.sliceAsBytes(d.best_host[0 .. 4 * n]), d.best.ptr);
                for (out[0..n], 0..) |*t, r| t.* = @intCast(@as(i32, @bitCast(d.best_host[4 * r + 2])));
                return;
            }
            const rows = d.host[0 .. n * vocab];
            try download(o.stream(), std.mem.sliceAsBytes(rows), d.logits.ptr);
            for (out[0..n], 0..) |*t, r| t.* = try drawRow(o.e.gpa, rows[r * vocab ..][0..vocab], null, d.fed0 + 1 + r, sm);
        }

        /// Keep the window's first `keep` rows (Python forward.commit): the DeltaNet states replayed over them from
        /// the window's start, the conv windows and n-gram tail shifted by them alone, the n-gram history after them.
        pub fn rollback(o: Self, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
            if (keep == 0 or keep > d.rows) return error.CannotRewind;
            if (keep == d.rows) return;
            const e = o.e;
            const st = o.stream();
            const s = if (e.batch_s) |*bs| bs else return error.CannotRewind;
            const k = &e.batch_k;
            seq.fed = d.fed0 + keep;
            var hist = d.history;
            for (d.ids[0..keep]) |t| hist = .{ hist[1], t };
            seq.history = hist;
            try copyAsync(st, seq.tail.ptr, d.saved.ptr + Draft.tail_at, ple_tail_n * wide * 2);
            try shift(k, st, seq.tail.ptr, s.ple_nrow, wide, keep, ple_tail_n, wide);
            var li: usize = 0;
            for (e.layers[0..e.n_layers], 0..) |*lw, layer| {
                if (!lw.linear) continue;
                defer li += 1;
                const lk = d.keep.layers[layer];
                try copyAsync(st, seq.state[li].ptr, lk.state, state_n * 4);
                try copyAsync(st, seq.conv[li].ptr, lk.hist, conv_bytes);
                var f: cuda.Args = .{};
                f.add(lk.proj);
                f.add(seq.conv[li].ptr);
                f.add(lw.cw.ptr);
                f.add(lw.a_log.ptr);
                f.add(lw.dt.ptr);
                f.add(s.gq);
                f.add(s.gk);
                f.add(s.gv);
                f.add(s.ggb);
                try cuda.launch.launch(k.front, .{ .grid = .{ .x = @intCast(keep), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, st, &f);
                var a: cuda.Args = .{};
                a.add(s.gq);
                a.add(s.gk);
                a.add(s.gv);
                a.add(s.ggb);
                a.add(seq.state[li].ptr);
                a.add(cint(keep));
                a.add(s.ys);
                try cuda.launch.launch(k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, st, &a);
                try shift(k, st, seq.conv[li].ptr, lk.proj, gdn.proj_width, keep, gdn.taps - 1, gdn.conv_dim);
            }
            d.rows = keep;
        }

        fn shift(k: *const batch.Kernels, st: cuda.Stream, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
            var a: cuda.Args = .{};
            a.add(win);
            a.add(rows_ptr);
            a.add(@as(c_longlong, @intCast(stride)));
            a.add(cint(n));
            a.add(cint(taps));
            a.add(cint(c));
            try cuda.launch.launch(k.shift, .{ .grid = .{ .x = @intCast((c + 255) / 256) }, .block = .{ .x = 256 } }, st, &a);
        }
    };
}

/// What a drafted decode did: rounds, drafts offered and kept, and its wall time.
pub const Stats = struct {
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    ms: f64 = 0,
    /// Wall time in the windows (forward and draws) and in the head (rollback, absorb, chain).
    verify_ms: f64 = 0,
    draft_ms: f64 = 0,
};

/// Python mtp_decode over a prefilled sequence: `out` holds the first token; decode until `count` tokens.
pub fn decode(comptime Tri: type, o: Ops(Tri), seq: *prompt.Seq, d: *Draft, prompt_len: usize, out: *std.ArrayList(u32), count: usize, depth: usize, confidence: f64, sm: ?Sampling, stop: []const u32) !Stats {
    const gpa = o.e.gpa;
    var st: Stats = .{};
    var ch: Chain = .{};
    var sampled: [max_rows]u32 = undefined;
    var ids: [max_rows]u32 = undefined;
    const t0 = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
    const isStop = struct {
        fn f(list: []const u32, t: u32) bool {
            return std.mem.indexOfScalar(u32, list, t) != null;
        }
    }.f;
    try o.draft(d, out.items[out.items.len - 1 ..], d.streams.ptr + (prompt_len - 1) * wide * 2, seq.fed + 1, @min(depth, count - out.items.len), sm, confidence, &ch);
    while (out.items.len < count and !isStop(stop, out.items[out.items.len - 1])) {
        ids[0] = out.items[out.items.len - 1];
        @memcpy(ids[1..][0..ch.n], ch.slice());
        const rows = 1 + ch.n;
        const tv = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
        try o.verify(seq, d, ids[0..rows]);
        try o.draws(d, sm, sampled[0..rows]);
        const td = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
        st.verify_ms += @as(f64, @floatFromInt(td - tv)) / 1e6;
        defer st.draft_ms += @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - td)) / 1e6;
        var keep: usize = 1;
        for (ch.slice(), 0..) |t, i| {
            if (sampled[i] != t or isStop(stop, sampled[i])) break;
            keep += 1;
        }
        try o.rollback(seq, d, keep);
        st.rounds += 1;
        st.drafted += ch.n;
        st.accepted += keep - 1;
        const take = @min(keep, count - out.items.len);
        try out.appendSlice(gpa, sampled[0..take]);
        if (out.items.len >= count or isStop(stop, out.items[out.items.len - 1])) break;
        try o.draft(d, sampled[0..keep], d.streams.ptr, seq.fed + 1, @min(depth, count - out.items.len), sm, confidence, &ch);
    }
    st.ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - t0)) / 1e6;
    return st;
}

/// Serial decoding through the same pass and draws: one row a forward (the reference drafted decoding must equal).
pub fn serial(comptime Tri: type, o: Ops(Tri), seq: *prompt.Seq, out: *std.ArrayList(u32), count: usize, sm: ?Sampling, stop: []const u32) !f64 {
    const gpa = o.e.gpa;
    const t0 = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    while (out.items.len < count and std.mem.indexOfScalar(u32, stop, out.items[out.items.len - 1]) == null) {
        var one = [_]u32{out.items[out.items.len - 1]};
        try o.e.predict(Tri, o.tri, seq, &one);
        for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
        try out.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
    }
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - t0)) / 1e6;
}

fn envUsize(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn envF64(name: [:0]const u8, default: f64) f64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch default;
}

/// The drafted check (flashnext-prompt with TF_FLASHNEXT_DRAFTS): one load, then per prompt serial decoding and
/// drafted decoding of `count` tokens, compared token for token, with tokens a second, rounds and drafts kept.
/// TF_FLASHNEXT_SEED (with TF_FLASHNEXT_TEMPERATURE, _TOP_P, _TOP_K) draws keyed samples instead of greedy ones.
pub fn check(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, prompts: []const []const u32, count: usize) !bool {
    var e = try prompt.Engine.init(gpa, io, driver, stream, model_dir);
    defer e.deinit();
    const h = try Head.init(gpa, io, driver, stream, model_dir);
    defer h.deinit();
    const o: Ops(Tri) = .{ .e = e, .tri = tri, .h = h };
    const depth = envUsize("TF_FLASHNEXT_DRAFTS", max_chain);
    const confidence = envF64("TF_FLASHNEXT_CONFIDENCE", default_confidence);
    const sm: ?Sampling = if (std.c.getenv("TF_FLASHNEXT_SEED")) |seed| .{
        .seed = std.fmt.parseInt(u64, std.mem.span(seed), 10) catch 0,
        .temperature = envF64("TF_FLASHNEXT_TEMPERATURE", 1.0),
        .top_p = envF64("TF_FLASHNEXT_TOP_P", 0.95),
        .top_k = @intCast(envUsize("TF_FLASHNEXT_TOP_K", 20)),
    } else null;
    const repeat = envUsize("TF_FLASHNEXT_REPEAT", 1);
    const stop: []const u32 = &.{};
    var same = true;
    for (prompts, 0..) |ids, pi| {
        const cap = ids.len + count + max_rows;
        var seq = try e.newSeq(cap);
        defer seq.deinit(gpa);
        var d = try Draft.init(gpa, driver, cap, h.ids.len);
        defer d.deinit();
        for (0..repeat) |rep| {
            // serial: the plain prompt pass, then one row a forward
            seq.fed = 0;
            seq.history = .{ e.eos, e.eos };
            try resetSeq(&seq);
            var ref: std.ArrayList(u32) = .empty;
            defer ref.deinit(gpa);
            try e.predict(Tri, tri, &seq, ids);
            const row = try gpa.alloc(u16, vocab);
            defer gpa.free(row);
            for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
            try ref.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
            const serial_ms = try serial(Tri, o, &seq, &ref, count, sm, stop);
            // drafted: the prompt pass with its streams and the head's cache, then windows
            seq.fed = 0;
            seq.history = .{ e.eos, e.eos };
            try resetSeq(&seq);
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(gpa);
            try o.prefill(&seq, &d, ids);
            for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
            try got.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
            const st = try decode(Tri, o, &seq, &d, ids.len, &got, count, depth, confidence, sm, stop);
            const equal = std.mem.eql(u32, ref.items, got.items);
            same = same and equal;
            const n_ref: f64 = @floatFromInt(ref.items.len - 1);
            const n_got: f64 = @floatFromInt(got.items.len - 1);
            std.debug.print("prompt {d} ({d} ids) run {d}: serial {d:.2} tok/s, drafted {d:.2} tok/s ({d:.2}x), {d} rounds, {d} drafts, {d} kept ({d:.1}%), {d:.2} tokens a round: {s}\n", .{
                pi,                                      ids.len,                            rep,
                n_ref / (serial_ms / 1e3),               n_got / (st.ms / 1e3),              (n_got / (st.ms / 1e3)) / (n_ref / (serial_ms / 1e3)),
                st.rounds,                               st.drafted,                         st.accepted,
                if (st.drafted > 0) 100.0 * @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0.0,
                if (st.rounds > 0) n_got / @as(f64, @floatFromInt(st.rounds)) else 0.0,
                if (equal) "same tokens" else "DIFFER",
            });
            std.debug.print("  drafted rounds: {d:.2} ms a window ({d:.2} rows), {d:.2} ms in the head a round\n", .{ st.verify_ms / @as(f64, @floatFromInt(@max(st.rounds, 1))), @as(f64, @floatFromInt(st.rounds + st.drafted)) / @as(f64, @floatFromInt(@max(st.rounds, 1))), st.draft_ms / @as(f64, @floatFromInt(@max(st.rounds, 1))) });
            std.debug.print("serial ", .{});
            for (ref.items) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\ndrafted", .{});
            for (got.items) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
    return same;
}

/// A sequence's caches back to empty (the check reruns a prompt on one allocation).
fn resetSeq(seq: *prompt.Seq) !void {
    for (seq.conv) |*b| try b.fill8(0, null);
    for (seq.state) |*b| try b.fill8(0, null);
    try seq.tail.fill8(0, null);
}

test "the draft vocabulary is 79,591 sorted ids under the vocabulary" {
    const ids = try draftIds(std.testing.allocator);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 79_591), ids.len);
    for (ids[1..], ids[0 .. ids.len - 1]) |b, a| try std.testing.expect(b > a);
}

test "a keyed draw over the top-k columns is the draw over every column" {
    const gpa = std.testing.allocator;
    var row: [300]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&row) |*v| v.* = @as(u16, 0x3f00) + prng.random().int(u8); // values near 0.5..1, many ties
    const all = struct {
        fn draw(a: std.mem.Allocator, r: []const u16, position: u64, s: Sampling) !u64 {
            const vals = try a.alloc(f64, r.len);
            defer a.free(vals);
            const ids = try a.alloc(u64, r.len);
            defer a.free(ids);
            for (vals, ids, r, 0..) |*v, *id, b, i| {
                v.* = promote(b);
                id.* = i;
            }
            return lanes.sampling.choose(a, vals, ids, position, s);
        }
    }.draw;
    for (0..64) |position| {
        const s: Sampling = .{ .seed = 11, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
        try std.testing.expectEqual(try all(gpa, &row, position, s), try drawRow(gpa, &row, null, position, s));
    }
    try std.testing.expectEqual(@as(u32, @intCast(argmaxRow(&row))), try drawRow(gpa, &row, null, 0, null));
}
