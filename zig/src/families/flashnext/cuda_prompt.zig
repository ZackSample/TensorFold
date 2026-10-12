//! One prompt, drafts off: the batched pass (cuda_batch.zig) runs the prompt's rows through each layer at once, on
//! the decode kernels at M rows, so its logits and caches are those of feeding one fused decode row at a time (the
//! row-by-row path below, kept for checks: TF_FLASHNEXT_PROMPT=rows). Then the argmax.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const qmm = @import("cuda_qmm.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const weights = @import("cuda_prompt_wt.zig");
const step = @import("cuda_prompt_step.zig");
const load = @import("cuda_prompt_load.zig");
pub const batch = @import("cuda_batch.zig");
const fb = @import("cuda_fn_back.zig");

const layers_n = weights.layers_n;
const linear_n = weights.linear_n;
const attn_n = weights.attn_n;
const dims = weights.dims;
const streams = weights.streams;
const wide = weights.wide;
const vocab = weights.vocab;
const experts_n = weights.experts_n;
const slots = weights.slots;
const moe_w = weights.moe_w;
const head_dim = weights.head_dim;
const index_dim = weights.index_dim;
const proj_n = weights.proj_n;
const out_k = weights.out_k;
const ple_layer = weights.ple_layer;
const inj_stride = weights.inj_stride;
const wts_stride = weights.wts_stride;
const ple_tail_n = weights.ple_tail_n;
const state_n = weights.state_n;
const kv_heads = weights.kv_heads;
const Store = weights.Store;
const Face = weights.Face;
const Q4 = weights.Q4;
const HcW = weights.HcW;
const linearLayer = weights.linearLayer;
const linearIndex = weights.linearIndex;
const attnIndex = weights.attnIndex;
const promote = weights.promote;
const asF32 = weights.asF32;
const uploadQ4 = weights.uploadQ4;
const join3 = weights.join3;
const loadHc = weights.loadHc;
const stitch = weights.stitch;
const Scratch = step.Scratch;
const writeback = step.writeback;
const readout = step.readout;
const gdnStep = step.gdnStep;
const attnStep = step.attnStep;
const moeStep = step.moeStep;
const pleStep = step.pleStep;
const Layer = load.Layer;
const loadLayer = load.loadLayer;

pub fn argmax(raw: []const u8) u32 {
    var best = promote(std.mem.readInt(u16, raw[0..2], .little));
    var at: usize = 0;
    var i: usize = 1;
    while (i < vocab) : (i += 1) {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return @intCast(at);
}

/// The five best scores of one row, best first, as "id:score" pairs on stderr (the capture's top5 field).
fn printTop(raw: []const u8) void {
    var ids: [5]usize = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (0..vocab) |i| {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v <= vals[4]) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = i;
    }
    std.debug.print("top5", .{});
    for (ids, vals) |id, v| std.debug.print(" {d}:{d:.4}", .{ id, v });
    std.debug.print("\n", .{});
}

const Pass = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: *Store,
    model_dir: []const u8,
    s: *Scratch,
    table: embed.Table,
    rows: usize,
    mode: []u8,
    history: *[2]i64,
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    mixer: HcW,
    head_q: Q4,
    logits: []u8,
    plan_fn: cuda.Function,
    up_fn: cuda.Function,
    down_fn: cuda.Function,
};

fn embedRow(comptime Tri: type, p: *Pass, tri: Tri, token: u32, h: u64) !void {
    const id: i32 = @intCast(token);
    // stream-ordered: a synchronous copy does not wait for the non-blocking stream, so a later row's id
    // would overwrite this one before its queued _embed reads it
    try p.s.ids.uploadAsync(0, std.mem.asBytes(&id), p.stream.handle);
    try tri.embed(p.s.ids.ptr, p.table.w.ptr, p.table.s.ptr, p.table.b.ptr, h, dims, streams, 1);
}

fn runLayer(comptime Tri: type, p: *Pass, tri: Tri, lw: *Layer, layer: usize, ids: []const u32, pos0: usize) !void {
    for (ids, 0..) |token, t| {
        const h = p.s.h.ptr + @as(u64, t) * wide * 2;
        const inj = p.s.inj.ptr + @as(u64, t) * inj_stride;
        const y = p.s.y.ptr + @as(u64, t) * slots * dims * 4;
        const wt = p.s.wts.ptr + @as(u64, t) * wts_stride;
        if (layer == ple_layer and p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        }
        if (layer == ple_layer) try pleStep(Tri, p.gpa, p.io, p.driver, p.stream, tri, p.s, p.model_dir, p.history, token, h, lw.key, lw.val, lw.nk.ptr, lw.nq.ptr, lw.nc.ptr, lw.ple_cw.ptr, &p.tail);
        if (p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        } else try writeback(tri, p.s, h, 0, inj, y, wt);
        try readout(tri, p.s, lw.attn_hc, h, inj);
        if (lw.linear) {
            const li = linearIndex(layer);
            try gdnStep(p.gpa, p.driver, p.stream, p.s, lw.proj, lw.out, p.conv[li], lw.cw.ptr, lw.a_log.ptr, lw.dt.ptr, lw.norm.ptr, &p.state[li]);
        } else {
            const ai = attnIndex(layer);
            try attnStep(Tri, p.driver, p.stream, tri, p.s, lw.proj, lw.out, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, p.kc[ai].ptr, p.vc[ai].ptr, p.ikc[ai].ptr, @intCast(pos0 + t));
        }
        try writeback(tri, p.s, h, 4, inj, y, wt);
        try readout(tri, p.s, lw.mlp_hc, h, inj);
        try moeStep(Tri, tri, p.s, p.stream, p.plan_fn, p.up_fn, p.down_fn, lw.router.ptr, lw.experts_up.ptr, lw.experts_down.ptr, y, wt);
        p.mode[t] = 2;
    }
}

fn zeros(gpa: std.mem.Allocator, driver: *cuda.Driver, n: usize, len: usize) ![]cuda.DeviceBuffer {
    const out = try gpa.alloc(cuda.DeviceBuffer, n);
    var i: usize = 0;
    errdefer {
        for (out[0..i]) |*b| b.free();
        gpa.free(out);
    }
    while (i < n) : (i += 1) {
        out[i] = try cuda.DeviceBuffer.alloc(driver, len);
        try out[i].fill8(0, null);
    }
    return out;
}

fn head(comptime Tri: type, p: *Pass, tri: Tri, row: usize) !void {
    if (p.mode[row] != 2) return error.UnexpectedTensor;
    const h = p.s.h.ptr + @as(u64, row) * wide * 2;
    const inj = p.s.inj.ptr + @as(u64, row) * inj_stride;
    const y = p.s.y.ptr + @as(u64, row) * slots * dims * 4;
    const wt = p.s.wts.ptr + @as(u64, row) * wts_stride;
    try writeback(tri, p.s, h, 2, inj, y, wt);
    p.mode[row] = 0;
    try readout(tri, p.s, p.mixer, h, inj);
    try qmm.matmul(p.driver, p.stream.*, p.s.mixed.ptr, p.s.xsm.ptr, p.head_q.w.ptr, p.head_q.s.ptr, p.head_q.b.ptr, p.s.logits.ptr, 1, vocab, dims);
    try p.stream.synchronize();
    try p.s.logits.download(0, p.logits);
}

fn forwardRows(comptime Tri: type, p: *Pass, tri: Tri, layers: []Layer, ids: []const u32, pos0: usize) !void {
    @memset(p.mode, 0);
    for (ids, 0..) |token, t| try embedRow(Tri, p, tri, token, p.s.h.ptr + @as(u64, t) * wide * 2);
    for (layers, 0..) |*lw, layer| try runLayer(Tri, p, tri, lw, layer, ids, pos0);
    try head(Tri, p, tri, ids.len - 1);
}

pub const vocabulary: usize = vocab;
/// Prompt plus reply tokens a sequence can hold: the model's window (the block indexer keeps a row's attention at
/// 2,048 selected keys plus its tail however long the sequence).
pub const max_tokens: usize = weights.window;

/// Device bytes `Engine.newSeq(cap)` allocates: the convolution tails, the recurrent states, the key and value
/// caches with the indexer keys and their pooled blocks, and the n-gram tail.
pub fn seqBytes(cap: usize) usize {
    return seqBytesAt(cap, 0);
}

/// seqBytes for a KV cache of `kv_bits` (0 bf16, 8 int8 codes with fp16 scales a group of 32).
pub fn seqBytesAt(cap: usize, kv_bits: usize) usize {
    const linear = linear_n * ((gdn.taps - 1) * gdn.conv_dim * 2 + state_n * 4);
    const attn = attn_n * (cap * (2 * kvRowBytes(kv_bits) + index_dim * 2) + pooledBytes(cap));
    return linear + attn + ple_tail_n * wide * 2;
}

/// One position's keys (or values) in one layer: bf16, or int8 codes plus their fp16 scales.
pub fn kvRowBytes(kv_bits: usize) usize {
    return if (kv_bits == 8) kv_heads * head_dim + kv_heads * (head_dim / 32) * 2 else kv_heads * head_dim * 2;
}

/// TF_FLASHNEXT_KV=int8: int8 KV (kvcache.py's -cq 8, the Python engine's default); bf16 otherwise.
pub fn kvBitsFromEnv() usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV") orelse return 0;
    return if (std.mem.eql(u8, std.mem.span(v), "int8")) 8 else 0;
}

/// One attention layer's pooled indexer keys for a cache of `cap` tokens: a bf16 row of 128 a block of 4.
pub fn pooledBytes(cap: usize) usize {
    return (cap + weights.index_ratio - 1) / weights.index_ratio * index_dim * 2;
}

/// One loaded checkpoint: the embedding table, the mixer, the head, and a scratch row the forwards share.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: Store,
    model_dir: []u8,
    table: embed.Table,
    mixer: HcW,
    head_q: Q4,
    scratch: Scratch,
    scratch_rows: usize,
    mode: []u8,
    eos: i64,
    experts: cuda.Module,
    plan_fn: cuda.Function,
    up_fn: cuda.Function,
    down_fn: cuda.Function,
    layers: [layers_n]Layer,
    n_layers: usize,
    /// The batched pass's kernels and scratch (grown to the widest chunk run), and whether prompts run row by row.
    batch_k: batch.Kernels,
    /// The n-gram table, open for the engine's life (the batched pass gathers its rows from it).
    ngram: ple.Ngram,
    batch_s: ?batch.Scratch = null,
    /// Block scores for the sparse rows of an attention block, grown to the longest sequence's need.
    scores: ?cuda.DeviceBuffer = null,
    /// New sequences' KV cache: 0 bf16, 8 int8 (TF_FLASHNEXT_KV).
    kv_bits: usize = 0,
    rowwise: bool,
    /// Rows a batched chunk takes at most (TF_FLASHNEXT_CHUNK; batch.default_chunk by default).
    chunk: usize,
    /// Each layer's Living Weights pocket as the forward adds it (null: none): loaded from a taught folder's widened
    /// shared experts, or uploaded by the learner. Every pass with one runs it through the batched pass's moe_add.
    pockets: [layers_n]?PocketRows = @splat(null),
    /// The pocket's kernels (flashnext_train_mix.cu), loaded with the first pocket.
    fb_k: ?fb.Kernels = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, model_dir: []const u8) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.driver = driver;
        e.stream = stream;
        e.n_layers = 0;
        errdefer for (e.layers[0..e.n_layers]) |*l| l.free(gpa);
        e.model_dir = try gpa.dupe(u8, model_dir);
        errdefer gpa.free(e.model_dir);
        e.store = try Store.open(gpa, io, e.model_dir);
        errdefer e.store.deinit();
        const weight = try e.store.tensor(embed.weight_name);
        const scales = try e.store.tensor(embed.scales_name);
        const biases = try e.store.tensor(embed.biases_name);
        e.table = try embed.Table.upload(driver, weight, scales, biases);
        errdefer e.table.deinit();
        e.eos = try ple.eosOf(gpa, io, e.model_dir);
        e.scratch = try Scratch.alloc(driver, 1);
        errdefer e.scratch.free();
        try e.scratch.dummy.fill8(0, null);
        e.scratch_rows = 1;
        e.mode = try gpa.alloc(u8, 1);
        errdefer gpa.free(e.mode);
        e.mixer = try loadHc(gpa, driver, &e.store, "language_model.model.hyper_connection_mixer", false);
        errdefer e.mixer.free();
        var buf: [80]u8 = undefined;
        const head_w = try join3(gpa, &e.store, &buf, "language_model.lm_head", "");
        defer gpa.free(head_w.w);
        defer gpa.free(head_w.s);
        defer gpa.free(head_w.b);
        if (head_w.w.len != vocab * (dims / 8) * 4) return error.UnexpectedTensor;
        e.head_q = try uploadQ4(gpa, driver, head_w.w, head_w.s, head_w.b, vocab, dims / 8);
        errdefer e.head_q.free();
        e.experts = try cuda.Module.load(driver, cuda.kernels.experts);
        errdefer e.experts.unload();
        e.plan_fn = try e.experts.function(weights.plan_symbol);
        e.up_fn = try e.experts.function(weights.up_symbol);
        e.down_fn = try e.experts.function(weights.down_symbol);
        var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
        defer pack_mod.unload();
        const pack_fn = try pack_mod.function(weights.pack_symbol);
        e.pockets = @splat(null);
        e.fb_k = null;
        errdefer e.freePockets();
        const rows = try gpa.alloc(f32, 3 * load.max_pocket * dims);
        defer gpa.free(rows);
        var learned: usize = 0;
        for (0..layers_n) |i| {
            var pk: load.Pocket = .{ .gate = rows[0 .. load.max_pocket * dims], .up = rows[load.max_pocket * dims ..][0 .. load.max_pocket * dims], .down = rows[2 * load.max_pocket * dims ..][0 .. load.max_pocket * dims] };
            e.layers[i] = try loadLayer(gpa, driver, stream, &e.store, pack_fn, i, &pk);
            e.n_layers += 1;
            if (pk.n > 0) {
                // the neurons in use: the first ones, up to the last with a nonzero input row
                var used: usize = 0;
                for (0..pk.n) |j| {
                    if (std.mem.indexOfNone(f32, pk.up[j * dims ..][0..dims], &.{0}) != null) used = j + 1;
                }
                try e.setPocket(i, pk.gate[0 .. pk.n * dims], pk.up[0 .. pk.n * dims], pk.down[0 .. pk.n * dims], used);
                learned += used;
            }
        }
        if (learned > 0) std.log.info("flashnext: the folder's Living Weights pocket: {d} learned neurons beside the shared experts", .{learned});
        e.store.release();
        e.store.closeFile();
        e.ngram = try ple.Ngram.init(gpa, io, e.model_dir);
        errdefer e.ngram.deinit();
        e.batch_k = try batch.Kernels.load(driver);
        e.batch_s = null;
        e.scores = null;
        e.kv_bits = kvBitsFromEnv();
        e.rowwise = if (std.c.getenv("TF_FLASHNEXT_PROMPT")) |v| std.mem.eql(u8, std.mem.span(v), "rows") else false;
        e.chunk = batch.default_chunk;
        if (std.c.getenv("TF_FLASHNEXT_CHUNK")) |v| e.chunk = std.fmt.parseInt(usize, std.mem.span(v), 10) catch batch.default_chunk;
        if (e.chunk == 0) e.chunk = batch.default_chunk;
        return e;
    }

    fn freePockets(e: *Engine) void {
        for (&e.pockets) |*slot| if (slot.*) |*pk| {
            pk.gate.free();
            pk.up.free();
            pk.down.free();
            slot.* = null;
        };
        if (e.fb_k) |*k| k.deinit();
        e.fb_k = null;
    }

    /// Layer `layer`'s pocket for the forward: gate rows, up rows and outputs fp32 [n, dims] (n neurons, the first
    /// `used` in use); made on first use.
    pub fn setPocket(e: *Engine, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        const n = gate.len / dims;
        if (n == 0 or n > load.max_pocket or up.len != gate.len or down.len != gate.len or used > n) return error.UnexpectedTensor;
        if (e.fb_k == null) e.fb_k = try fb.Kernels.load(e.driver);
        if (e.pockets[layer]) |pk| if (pk.cap != n) return error.PocketOfAnotherSize;
        if (e.pockets[layer] == null) {
            var g = try cuda.DeviceBuffer.alloc(e.driver, n * dims * 4);
            errdefer g.free();
            var u = try cuda.DeviceBuffer.alloc(e.driver, n * dims * 4);
            errdefer u.free();
            const d = try cuda.DeviceBuffer.alloc(e.driver, n * dims * 4);
            e.pockets[layer] = .{ .gate = g, .up = u, .down = d, .cap = n };
        }
        const pk = &e.pockets[layer].?;
        // stream-ordered after every pass already queued that reads the old rows
        try pk.gate.uploadAsync(0, std.mem.sliceAsBytes(gate), e.stream.handle);
        try pk.up.uploadAsync(0, std.mem.sliceAsBytes(up), e.stream.handle);
        try pk.down.uploadAsync(0, std.mem.sliceAsBytes(down), e.stream.handle);
        try e.stream.synchronize();
        pk.used = used;
    }

    /// Layer `layer`'s pocket into gate, up and down ([n, dims] each); the neurons in use (0: none).
    pub fn pocketRead(e: *Engine, layer: usize, gate: []f32, up: []f32, down: []f32) usize {
        const pk = e.pockets[layer] orelse return 0;
        if (gate.len != pk.cap * dims or up.len != gate.len or down.len != gate.len) return 0;
        e.stream.synchronize() catch return 0;
        pk.gate.download(0, std.mem.sliceAsBytes(gate)) catch return 0;
        pk.up.download(0, std.mem.sliceAsBytes(up)) catch return 0;
        pk.down.download(0, std.mem.sliceAsBytes(down)) catch return 0;
        return pk.used;
    }

    /// Layer `layer`'s pocket as the kernels take it, or null when it has no neuron in use.
    pub fn pocketOf(e: *const Engine, layer: usize) ?fb.Ops.Pocket {
        const pk = e.pockets[layer] orelse return null;
        if (pk.used == 0) return null;
        return .{ .gate = pk.gate.ptr, .up = pk.up.ptr, .down = pk.down.ptr, .n = pk.used };
    }

    fn pocketed(e: *const Engine) bool {
        for (e.pockets) |slot| if (slot) |pk| if (pk.used > 0) return true;
        return false;
    }

    /// The serving pass's moe_add: each layer's pocket behind its shared gate, added to its MoE output.
    fn servePocket(ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const pk = e.pocketOf(layer) orelse return;
        const o: fb.Ops = .{ .k = &e.fb_k.?, .s = e.stream.* };
        try o.pocketRows(pk, .{ .x = x, .ids = ids, .wts = wts }, out, true, rows, dims, slots - 1);
    }

    pub fn deinit(e: *Engine) void {
        e.freePockets();
        if (e.batch_s) |*bs| bs.free(e.gpa);
        if (e.scores) |*b| b.free();
        e.batch_k.deinit();
        e.ngram.deinit();
        e.experts.unload();
        for (e.layers[0..e.n_layers]) |*l| l.free(e.gpa);
        e.head_q.free();
        e.mixer.free();
        e.gpa.free(e.mode);
        e.scratch.free();
        e.table.deinit();
        e.store.deinit();
        e.gpa.free(e.model_dir);
        e.gpa.destroy(e);
    }

    fn grow(e: *Engine, rows: usize) !void {
        if (rows <= e.scratch_rows) return;
        var next = try Scratch.alloc(e.driver, rows);
        errdefer next.free();
        try next.dummy.fill8(0, null);
        const mode = try e.gpa.alloc(u8, rows);
        e.scratch.free();
        e.gpa.free(e.mode);
        e.scratch = next;
        e.mode = mode;
        e.scratch_rows = rows;
    }

    /// A fresh cache. `cap` is how many tokens it can hold, prompt plus generated.
    pub fn newSeq(e: *Engine, cap: usize) !Seq {
        if (cap == 0) return error.UnexpectedTensor;
        const logits = try e.gpa.alloc(u8, vocab * 2);
        errdefer e.gpa.free(logits);
        const conv = try zeros(e.gpa, e.driver, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
        errdefer freeOwned(e.gpa, conv);
        const state = try zeros(e.gpa, e.driver, linear_n, state_n * 4);
        errdefer freeOwned(e.gpa, state);
        const code_bytes: usize = if (e.kv_bits == 8) 1 else 2;
        const kc = try zeros(e.gpa, e.driver, attn_n, cap * kv_heads * head_dim * code_bytes);
        errdefer freeOwned(e.gpa, kc);
        const vc = try zeros(e.gpa, e.driver, attn_n, cap * kv_heads * head_dim * code_bytes);
        errdefer freeOwned(e.gpa, vc);
        const scales: usize = if (e.kv_bits == 8) attn_n else 0;
        const ks = try zeros(e.gpa, e.driver, scales, cap * kv_heads * (head_dim / 32) * 2);
        errdefer freeOwned(e.gpa, ks);
        const vs = try zeros(e.gpa, e.driver, scales, cap * kv_heads * (head_dim / 32) * 2);
        errdefer freeOwned(e.gpa, vs);
        const ikc = try zeros(e.gpa, e.driver, attn_n, cap * index_dim * 2);
        errdefer freeOwned(e.gpa, ikc);
        const pooled = try zeros(e.gpa, e.driver, attn_n, pooledBytes(cap));
        errdefer freeOwned(e.gpa, pooled);
        var tail = try cuda.DeviceBuffer.alloc(e.driver, ple_tail_n * wide * 2);
        errdefer tail.free();
        try tail.fill8(0, null);
        return .{
            .conv = conv,
            .state = state,
            .kc = kc,
            .vc = vc,
            .ikc = ikc,
            .ks = ks,
            .vs = vs,
            .kv_bits = e.kv_bits,
            .pooled = pooled,
            .tail = tail,
            .history = .{ e.eos, e.eos },
            .fed = 0,
            .logits = logits,
            .cap = cap,
        };
    }

    /// Run `ids` on from `seq.fed` and leave the next token's scores in `seq.logits`.
    pub fn predict(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32) !void {
        if (e.rowwise) return e.predictRows(Tri, tri, seq, ids);
        return e.forward(Tri, tri, seq, ids, .{});
    }

    /// Block scores for a `cap`-token sequence's chunks of up to `rows` rows, reallocated only to grow.
    pub fn reserveScores(e: *Engine, cap: usize, rows: usize) !u64 {
        const need = batch.scoreBytes(cap, rows);
        if (e.scores) |*b| {
            if (b.len >= need) return b.ptr;
            try e.stream.synchronize();
            b.free();
            e.scores = null;
        }
        e.scores = try cuda.DeviceBuffer.alloc(e.driver, need);
        return e.scores.?.ptr;
    }

    /// Scratch for chunks of `rows` rows, reallocated only to grow.
    pub fn reserve(e: *Engine, rows: usize) !*batch.Scratch {
        if (e.batch_s) |*bs| {
            if (bs.cap >= rows) return bs;
            try e.stream.synchronize();
            bs.free(e.gpa);
            e.batch_s = null;
        }
        e.batch_s = try batch.Scratch.alloc(e.gpa, e.driver, rows);
        return &e.batch_s.?;
    }

    /// The batched pass over `ids` from `seq.fed`, in chunks of at most `chunk` rows; the last chunk's head per
    /// `hooks.logits` (its last row lands in `seq.logits`). A training caller's hooks (keep, nudge, taps, every
    /// row's logits) describe one chunk: pass at most `chunk` rows with them.
    pub fn forward(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32, hooks: batch.Hooks) !void {
        if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
        const training = hooks.keep != null or hooks.nudge != null or hooks.moe_add != null or hooks.taps != null or hooks.logits == .all;
        if (training and ids.len > e.chunk) return error.UnexpectedTensor;
        var with = hooks;
        if (with.moe_add == null and e.pocketed()) with.moe_add = .{ .ctx = e, .add = servePocket };
        const s = try e.reserve(@min(ids.len, e.chunk));
        const scores = try e.reserveScores(seq.cap, s.cap);
        var p: batch.Pass(Tri) = .{
            .gpa = e.gpa,
            .io = e.io,
            .stream = e.stream.*,
            .tri = tri,
            .k = &e.batch_k,
            .s = s,
            .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram },
            .seq = seq.view(scores),
            .hooks = with,
        };
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(s.cap, ids.len - at);
            const last = at + n == ids.len;
            p.hooks.logits = if (last) hooks.logits else .none;
            p.hooks.streams = if (hooks.streams != 0) hooks.streams + at * wide * 2 else 0;
            try p.chunk(ids[at..][0..n], seq.fed + at, if (last and hooks.logits != .none) seq.logits else null);
            at += n;
        }
        if (hooks.logits == .none) try e.stream.synchronize();
        seq.fed += ids.len;
    }

    /// One decode round: each of `seqs` takes its token of `ids` at its own position, every row in one batched pass
    /// (the stateless kernels over all rows, each stream's conv windows, states and caches its own), and its next
    /// token's scores land in its `logits`, the bits a pass of that stream alone leaves. `out` is device scratch for
    /// the rows' scores [seqs.len, vocab] bf16.
    pub fn round(e: *Engine, comptime Tri: type, tri: Tri, seqs: []const *Seq, ids: []const u32, out: u64) !void {
        if (seqs.len == 0 or seqs.len != ids.len) return error.UnexpectedTensor;
        var cap: usize = 0;
        for (seqs) |q| {
            if (q.fed + 1 > q.cap) return error.UnexpectedTensor;
            cap = @max(cap, q.cap);
        }
        const rows = try e.gpa.alloc(batch.Lane, seqs.len);
        defer e.gpa.free(rows);
        const s = try e.reserve(seqs.len);
        // each row's attention runs alone on the block scores
        const scores = try e.reserveScores(cap, 1);
        for (rows, seqs) |*l, q| l.* = .{ .seq = q.view(scores), .pos = q.fed };
        var p: batch.Pass(Tri) = .{
            .gpa = e.gpa,
            .io = e.io,
            .stream = e.stream.*,
            .tri = tri,
            .k = &e.batch_k,
            .s = s,
            .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram },
            .seq = rows[0].seq,
            .lanes = rows,
            // a taught folder's pockets, row-wise like every other launch of the round
            .hooks = .{ .logits = .all, .logits_out = out, .moe_add = if (e.pocketed()) .{ .ctx = e, .add = servePocket } else null },
        };
        try p.chunk(ids, 0, null);
        try e.stream.synchronize();
        const d = e.stream.d;
        for (seqs, 0..) |q, r| {
            try d.check(d.api.cuMemcpyDtoH_v2(q.logits.ptr, out + r * vocab * 2, vocab * 2), "cuMemcpyDtoH");
            q.fed += 1;
        }
    }

    /// The row-by-row path: every row one fused decode step (the batched pass's reference).
    pub fn predictRows(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32) !void {
        if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
        // dense bf16 attention only: the reference path stops where the indexer would start choosing
        if (seq.fed + ids.len > weights.dense_nch * 512 or seq.kv_bits != 0) return error.UnexpectedTensor;
        try e.grow(ids.len);
        var pass = Pass{
            .gpa = e.gpa,
            .io = e.io,
            .driver = e.driver,
            .stream = e.stream,
            .store = &e.store,
            .model_dir = e.model_dir,
            .s = &e.scratch,
            .table = e.table,
            .rows = ids.len,
            .mode = e.mode,
            .history = &seq.history,
            .conv = seq.conv,
            .state = seq.state,
            .kc = seq.kc,
            .vc = seq.vc,
            .ikc = seq.ikc,
            .tail = seq.tail,
            .mixer = e.mixer,
            .head_q = e.head_q,
            .logits = seq.logits,
            .plan_fn = e.plan_fn,
            .up_fn = e.up_fn,
            .down_fn = e.down_fn,
        };
        try forwardRows(Tri, &pass, tri, &e.layers, ids, seq.fed);
        seq.fed += ids.len;
    }
};

/// A layer's pocket on the device: fp32 [cap, dims] gate rows, up rows and outputs, the first `used` neurons live.
pub const PocketRows = struct { gate: cuda.DeviceBuffer, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, cap: usize, used: usize = 0 };

/// One stream's caches. `fed` tokens have already gone through the model.
pub const Seq = struct {
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    ks: []cuda.DeviceBuffer,
    vs: []cuda.DeviceBuffer,
    kv_bits: usize,
    pooled: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    history: [2]i64,
    fed: usize,
    logits: []u8,
    cap: usize,

    /// The caches a batched pass reads and leaves; `scores` the engine's block scores (Engine.reserveScores).
    pub fn view(s: *Seq, scores: u64) batch.SeqState {
        return .{ .conv = s.conv, .state = s.state, .kc = s.kc, .vc = s.vc, .ikc = s.ikc, .ks = s.ks, .vs = s.vs, .kv_bits = s.kv_bits, .pooled = s.pooled, .tail = s.tail, .history = &s.history, .cap = s.cap, .scores = scores };
    }

    pub fn deinit(s: *Seq, gpa: std.mem.Allocator) void {
        freeBufs(s.conv);
        freeBufs(s.state);
        freeBufs(s.kc);
        freeBufs(s.vc);
        freeBufs(s.ikc);
        freeBufs(s.pooled);
        freeBufs(s.ks);
        freeBufs(s.vs);
        gpa.free(s.conv);
        gpa.free(s.state);
        gpa.free(s.kc);
        gpa.free(s.vc);
        gpa.free(s.ikc);
        gpa.free(s.pooled);
        gpa.free(s.ks);
        gpa.free(s.vs);
        if (s.tail.ptr != 0) s.tail.free();
        gpa.free(s.logits);
    }
};

fn freeBufs(bs: []cuda.DeviceBuffer) void {
    for (bs) |*b| b.free();
}

fn freeOwned(gpa: std.mem.Allocator, bs: []cuda.DeviceBuffer) void {
    freeBufs(bs);
    gpa.free(bs);
}

/// `out` receives `new_tokens` greedy ids. The first is the prompt's argmax. The rest are fed back one at a time.
pub fn run(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, tokens: []const u32, new_tokens: usize, out: []u32) !void {
    if (tokens.len == 0 or new_tokens == 0 or out.len < new_tokens or tokens.len + new_tokens > max_tokens) return error.UnexpectedTensor;
    var e = try Engine.init(gpa, io, driver, stream, model_dir);
    defer e.deinit();
    var seq = try e.newSeq(tokens.len + new_tokens);
    defer seq.deinit(gpa);
    const loaded = cuda.usage(true);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try e.predict(Tri, tri, &seq, tokens);
    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    out[0] = argmax(seq.logits);
    printTop(seq.logits);
    // TF_FLASHNEXT_LOGITS=path: the prompt's last row of scores (bf16), to compare the two paths byte for byte
    if (std.c.getenv("TF_FLASHNEXT_LOGITS")) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = seq.logits });
    std.debug.print("token {d}\n", .{out[0]});
    var tok = out[0];
    for (1..new_tokens) |i| {
        var one = [_]u32{tok};
        try e.predict(Tri, tri, &seq, &one);
        tok = argmax(seq.logits);
        if (i < 4) printTop(seq.logits);
        out[i] = tok;
        std.debug.print("token {d}\n", .{tok});
    }
    const t2 = std.Io.Clock.awake.now(io).toNanoseconds();
    const ms = struct {
        fn of(ns: i96) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    std.debug.print("timing prompt {d} rows {d:.1} ms, decode {d} tokens {d:.2} ms/token\n", .{ tokens.len, ms(t1 - t0), new_tokens - 1, if (new_tokens > 1) ms(t2 - t1) / @as(f64, @floatFromInt(new_tokens - 1)) else 0 });
    const gib = struct {
        fn of(b: u64) f64 {
            return @as(f64, @floatFromInt(b)) / (1 << 30);
        }
    }.of;
    const used = cuda.usage(false);
    std.debug.print("memory: {d:.2} GiB on the device after the load and the sequence, peak {d:.2} GiB during the prompt and decode (+{d:.2} GiB)\n", .{ gib(loaded.device), gib(used.peak), gib(used.peak - loaded.device) });
}

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(linearLayer(0));
    try std.testing.expect(!linearLayer(3));
    try std.testing.expect(!linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), attnIndex(47));
}

test "a sequence's device bytes grow by the twelve attention layers' keys, values, indexer keys and pooled blocks" {
    const per_token = attn_n * (2 * kv_heads * head_dim * 2 + index_dim * 2);
    try std.testing.expectEqual(per_token, seqBytes(6) - seqBytes(5));
    try std.testing.expectEqual(per_token + attn_n * index_dim * 2, seqBytes(5) - seqBytes(4));
    try std.testing.expect(seqBytes(max_tokens) > max_tokens * per_token);
}
