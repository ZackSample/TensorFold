//! The batched prompt pass: a chunk of rows through each layer at once, on the same kernels the row-by-row path
//! launches one row at a time, now with M rows. Every kernel keeps a row's bits whatever M is (the qmm lane tile,
//! the grouped experts, the captured Triton forms at M > 1 from capture_rows.py), the DeltaNet chain runs the chunk's
//! rows in one launch, and attention reads each row's own causal prefix; so a chunk leaves the logits, the KV cache,
//! the DeltaNet conv windows and states and the n-gram tail exactly as feeding its rows one at a time would.
//!
//! A training caller (Living Weights) passes `Hooks`: `keep` names per-layer buffers the pass copies the tensors a
//! backward reads into (cuda_fn_back.zig's HcKept, MoeKept, AttnKept and gdn_train.zig's projection rows, conv
//! history and state at the chunk start), `nudge` adds an fp32 row vector to a layer's MoE output, and `taps` reads
//! each layer's MoE input (the shared expert's normed input) and shared gate at labelled rows.

const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_prompt_load.zig");

const layers_n = wt.layers_n;
const linear_n = wt.linear_n;
const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const eps = wt.eps;
const vocab = wt.vocab;
const experts_n = wt.experts_n;
const slots = wt.slots;
const moe_w = wt.moe_w;
const tile = wt.tile;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_heads = wt.index_heads;
const index_dim = wt.index_dim;
const half = wt.half;
const proj_n = wt.proj_n;
const out_k = wt.out_k;
const nch = wt.nch;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const state_n = wt.state_n;
const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const Layer = load.Layer;

/// Normed-stream partial sums a row: D / 256 blocks of each stream.
const pss_n = (dims / 256) * streams;
/// The widest projection row: the DeltaNet's q|k|v|z|b|a (attention's q|k|v|indexer is 13952).
const proj_row = @max(gdn.proj_width, proj_n);
/// Attention rows a _chunks launch takes (its scratch: rows x nch x heads x dim fp32).
pub const att_rows: usize = 256;
/// Rows a pass holds at most unless asked for more (the dense cache tops out at nch x 512 keys anyway).
pub const default_chunk: usize = nch * 512;
/// Pick lists the one-block plan takes (experts.cu's PSMALL); wider calls rank in blocks of 1024.
const plan_small: usize = 1024;

const cubin_qmm: [:0]const u8 = "_ZN6tf_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";
const qmm_smem: u32 = 9472;
const plan_rank_symbol: [:0]const u8 = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
const plan_offsets_symbol: [:0]const u8 = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
const plan_scatter_symbol: [:0]const u8 = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
/// flashnext_experts.cu: the decode experts' arithmetic over `rows_tile` rows a warp (two tiles of 16).
const rows_up_symbol: [:0]const u8 = "_ZN13tf_fn_experts11expert_rowsILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
const rows_down_symbol: [:0]const u8 = "_ZN13tf_fn_experts11expert_rowsILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
/// Pairs an expert item takes: the plan's tile, the kernels' two row tiles of 16.
const rows_tile: usize = 32;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Which rows get the head: the last (a prompt), every row (a training pass's loss), or none.
pub const Logits = enum { last, all, none };

/// Rows whose MoE input and shared gate a pass reads, each layer: `normed` [layers, rows.len, dims] bf16 and
/// `gates` [layers, rows.len] fp32 (0: not read). `index` is a device copy of `rows` (i32).
pub const Taps = struct { rows: []const u32, index: u64, normed: u64, gates: u64 = 0 };

/// One hyper-connection read-out's kept tensors (cuda_fn_back.zig's HcKept): its input streams h [rows, wide],
/// normed [rows, wide] bf16, v the down projection fp32 [rows, low + 4], up [rows, wide] bf16, inject [rows, 4].
pub const HcKeep = struct { h: u64 = 0, normed: u64 = 0, v: u64 = 0, up: u64 = 0, inject: u64 = 0 };

/// A layer's kept tensors; 0 leaves one out. Row strides are the tensors' own widths.
pub const LayerKeep = struct {
    attn_hc: HcKeep = .{},
    mlp_hc: HcKeep = .{},
    /// The mixer branch the write-back adds after the attention or DeltaNet block [rows, dims] bf16.
    mix_branch: u64 = 0,
    /// The MoE output its write-back adds [rows, dims] bf16 (with the nudge, when one is given).
    moe_branch: u64 = 0,
    /// The MoE layer (MoeKept): its input x [rows, dims] bf16, ids [rows, 11] i32, weights [rows, 11] fp32.
    moe_x: u64 = 0,
    moe_ids: u64 = 0,
    moe_wts: u64 = 0,
    /// Attention (AttnKept): q|gate [rows, 12288], k and v [rows, 512] bf16, o_proj's input gated [rows, 6144] bf16.
    pq: u64 = 0,
    pk: u64 = 0,
    pv: u64 = 0,
    gated: u64 = 0,
    /// DeltaNet (gdn_train.zig's forward): projection rows [rows, 16480] bf16, the conv history [3, 10240] bf16
    /// and the chain state [48, 128, 128] fp32 as they were before the chunk.
    proj: u64 = 0,
    hist: u64 = 0,
    state: u64 = 0,
    /// The chain's end state written here instead of in place: the sequence's state stays as it was before the
    /// chunk (a drafted window keeps it for a rollback, cuda_mtp.zig). 0: in place.
    state_out: u64 = 0,
};

/// What a pass keeps: each layer's tensors, the final mixer's read-out and the streams it read (after the last
/// MoE write-back). `pos0` is set to the chunk's first position.
pub const Keep = struct {
    layers: [layers_n]LayerKeep = @splat(.{}),
    mixer: HcKeep = .{},
    pos0: usize = 0,
};

/// Called on the pass's stream once a layer has routed: `out` [rows, dims] fp32 holds the layer's nudge (zeros
/// without one) and what the callback adds into it joins the layer's MoE output before the bf16 rounding, e.g.
/// Living Weights' pocket neurons: fn_back's pocketRows(pocket, .{ x, ids, wts }, out, true, rows, dims, 10).
/// x [rows, dims] bf16 is the MoE input, ids and wts [rows, 11] its experts and weights (the shared one last).
pub const MoeAdd = struct {
    ctx: *anyopaque,
    add: *const fn (ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void,
};

/// Every tensor `Keep` names for `rows` rows, in buffers this owns: a training pass's whole keep list (8.1 MiB a row
/// over the 48 layers, most of it the read-outs' streams, normed rows and up projections, plus 108 MiB of DeltaNet
/// conv histories and states at the chunk start).
pub const KeepStore = struct {
    keep: Keep = .{},
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    fn get(k: *KeepStore, gpa: std.mem.Allocator, d: *const cuda.Driver, len: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, len);
        errdefer b.free();
        try k.bufs.append(gpa, b);
        k.bytes += len;
        return b.ptr;
    }

    fn hc(k: *KeepStore, gpa: std.mem.Allocator, d: *const cuda.Driver, rows: usize, inject: bool) !HcKeep {
        return .{
            .h = try k.get(gpa, d, rows * wide * 2),
            .normed = try k.get(gpa, d, rows * wide * 2),
            .v = try k.get(gpa, d, rows * (low + streams) * 4),
            .up = try k.get(gpa, d, rows * wide * 2),
            .inject = if (inject) try k.get(gpa, d, rows * streams * 2) else 0,
        };
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, rows: usize) !KeepStore {
        var k: KeepStore = .{};
        errdefer k.deinit(gpa);
        for (&k.keep.layers, 0..) |*l, layer| {
            l.attn_hc = try k.hc(gpa, d, rows, true);
            l.mlp_hc = try k.hc(gpa, d, rows, true);
            l.mix_branch = try k.get(gpa, d, rows * dims * 2);
            l.moe_branch = try k.get(gpa, d, rows * dims * 2);
            l.moe_x = try k.get(gpa, d, rows * dims * 2);
            l.moe_ids = try k.get(gpa, d, rows * slots * 4);
            l.moe_wts = try k.get(gpa, d, rows * slots * 4);
            if (wt.linearLayer(layer)) {
                l.proj = try k.get(gpa, d, rows * gdn.proj_width * 2);
                l.hist = try k.get(gpa, d, (gdn.taps - 1) * gdn.conv_dim * 2);
                l.state = try k.get(gpa, d, state_n * 4);
            } else {
                l.pq = try k.get(gpa, d, rows * q_heads * 2 * head_dim * 2);
                l.pk = try k.get(gpa, d, rows * kv_heads * head_dim * 2);
                l.pv = try k.get(gpa, d, rows * kv_heads * head_dim * 2);
                l.gated = try k.get(gpa, d, rows * out_k * 2);
            }
        }
        k.keep.mixer = try k.hc(gpa, d, rows, false);
        return k;
    }

    pub fn deinit(k: *KeepStore, gpa: std.mem.Allocator) void {
        for (k.bufs.items) |*b| b.free();
        k.bufs.deinit(gpa);
        k.* = undefined;
    }
};

/// A training caller's view into the pass. Without any, the pass is the server's prompt pass.
pub const Hooks = struct {
    logits: Logits = .last,
    /// Device [rows, vocab] bf16 for `.all` (else the pass's own buffer for the last row).
    logits_out: u64 = 0,
    keep: ?*Keep = null,
    /// Each layer's [rows, dims] fp32 added to its MoE output before the bf16 rounding (0: none).
    nudge: ?*const [layers_n]u64 = null,
    /// Rows a callback adds to each layer's MoE output (after the nudge), on the pass's stream.
    moe_add: ?MoeAdd = null,
    taps: ?Taps = null,
    /// Device [rows, wide] bf16: every row's streams after the last MoE write-back, before the mixer (the MTP
    /// head's input, cuda_mtp.zig); 0: none.
    streams: u64 = 0,
};

/// The kernels the pass launches besides the captured Triton ones, loaded once.
pub const Kernels = struct {
    qmm: cuda.Module,
    experts: cuda.Module,
    rows: cuda.Module,
    batch: cuda.Module,
    lane: cuda.Function,
    front: cuda.Function,
    chain: cuda.Function,
    chain_to: cuda.Function,
    gnorm: cuda.Function,
    plan: cuda.Function,
    rank: cuda.Function,
    offsets: cuda.Function,
    scatter: cuda.Function,
    up: cuda.Function,
    down: cuda.Function,
    sum: cuda.Function,
    shift: cuda.Function,
    combine: cuda.Function,
    copy: cuda.Function,
    copy32: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.qmm = try cuda.Module.load(d, cuda.kernels.qmm);
        errdefer k.qmm.unload();
        k.experts = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.experts.unload();
        k.rows = try cuda.Module.load(d, cuda.kernels.flashnext_experts);
        errdefer k.rows.unload();
        k.batch = try cuda.Module.load(d, cuda.kernels.flashnext_batch);
        errdefer k.batch.unload();
        k.lane = try k.qmm.function(cubin_qmm);
        try k.lane.allowDynamicShared(qmm_smem);
        k.plan = try k.experts.function(wt.plan_symbol);
        k.rank = try k.experts.function(plan_rank_symbol);
        k.offsets = try k.experts.function(plan_offsets_symbol);
        k.scatter = try k.experts.function(plan_scatter_symbol);
        k.up = try k.rows.function(rows_up_symbol);
        k.down = try k.rows.function(rows_down_symbol);
        k.sum = try k.batch.function("fn_sum_slices");
        k.shift = try k.batch.function("fn_shift_window");
        k.combine = try k.batch.function("fn_combine");
        k.copy = try k.batch.function("fn_copy2d");
        k.copy32 = try k.batch.function("fn_copy2d32");
        k.front = try k.batch.function("fn_gdn_front");
        k.chain = try k.batch.function("fn_gdn_rows");
        k.chain_to = try k.batch.function("fn_gdn_rows_to");
        k.gnorm = try k.batch.function("fn_gdn_norm");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.batch.unload();
        k.rows.unload();
        k.experts.unload();
        k.qmm.unload();
    }
};

/// Device scratch for chunks of up to `cap` rows.
pub const Scratch = struct {
    cap: usize,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    ids: u64 = 0,
    pos: u64 = 0,
    h: u64 = 0,
    pss: u64 = 0,
    norm: u64 = 0,
    dn: u64 = 0,
    hc_part: u64 = 0,
    act: u64 = 0,
    xs_act: u64 = 0,
    inj: u64 = 0,
    mixed: u64 = 0,
    xsm: u64 = 0,
    proj: u64 = 0,
    part: u64 = 0,
    gout: u64 = 0,
    gxs: u64 = 0,
    ys: u64 = 0,
    gq: u64 = 0,
    gk: u64 = 0,
    gv: u64 = 0,
    ggb: u64 = 0,
    q: u64 = 0,
    iq: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    ao: u64 = 0,
    gated: u64 = 0,
    gxs_attn: u64 = 0,
    router_l: u64 = 0,
    picks: u64 = 0,
    wts: u64 = 0,
    members: u64 = 0,
    items: u64 = 0,
    counts: u64 = 0,
    rank: u64 = 0,
    hist: u64 = 0,
    moe_act: u64 = 0,
    y: u64 = 0,
    branch: u64 = 0,
    extra: u64 = 0,
    ple_emb: u64 = 0,
    ple_xs: u64 = 0,
    ple_keys: u64 = 0,
    ple_vals: u64 = 0,
    ple_part: u64 = 0,
    ple_gated: u64 = 0,
    ple_pss: u64 = 0,
    ple_nrow: u64 = 0,
    logits: u64 = 0,
    inv: u64 = 0,
    dummy: u64 = 0,

    fn add(s: *Scratch, gpa: std.mem.Allocator, d: *const cuda.Driver, len: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, @max(len, 256));
        errdefer b.free();
        try s.bufs.append(gpa, b);
        return b.ptr;
    }

    /// Bytes `alloc(cap)` takes on the device.
    pub fn bytes(cap: usize) usize {
        const pairs = cap * slots;
        const nblk = cdiv(pairs, plan_small);
        const items = cdiv(pairs, tile) + experts_n + 1;
        var n: usize = 0;
        n += cap * 4 + (cdiv(cap, att_rows) + 1) * 16;
        n += cap * wide * 2 * 2 + cap * pss_n * 4 + cap * (low + 4) * 2 + hc_sk * cap * (low + 4) * 4;
        n += cap * low * 2 + cap * (low / 32) * 4 + cap * streams * 2 + cap * dims * 2 + cap * (dims / 32) * 4;
        n += cap * proj_row * 2 + out_sk * cap * dims * 4 + cap * gdn.value_dim * (2 + 4 + 4) + cap * (gdn.value_dim / 32) * 4;
        n += cap * (2 * gdn.nk * gdn.dk + 2 * gdn.nv) * 4;
        n += cap * q_heads * head_dim * 2 + cap * index_heads * index_dim * 2;
        n += att_rows * nch * q_heads * (head_dim + 2) * 4 + cap * out_k * 2 * 2 + cap * (out_k / 32) * 4;
        n += cap * (experts_n + 1) * 4 + pairs * 4 * 2 + pairs * 4 + items * 3 * 4 + 8 + pairs * 4 + nblk * (experts_n + 1) * 4;
        n += pairs * moe_w * 2 + pairs * dims * 4 + cap * dims * 2 + cap * dims * 4;
        n += cap * dims * 2 + cap * (dims / 32) * 4 + cap * wide * 2 + cap * dims * 2 + 4 * cap * dims * 4;
        n += cap * wide * 2 * 2 + cap * streams * 4 + vocab * 2 + half * 4 + 4096;
        return n;
    }

    pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize) !Scratch {
        var s: Scratch = .{ .cap = cap };
        errdefer s.free(gpa);
        const pairs = cap * slots;
        const nblk = cdiv(pairs, plan_small);
        const items = cdiv(pairs, tile) + experts_n + 1;
        s.ids = try s.add(gpa, d, cap * 4);
        s.pos = try s.add(gpa, d, (cdiv(cap, att_rows) + 1) * 16);
        s.h = try s.add(gpa, d, cap * wide * 2);
        s.pss = try s.add(gpa, d, cap * pss_n * 4);
        s.norm = try s.add(gpa, d, cap * wide * 2);
        s.dn = try s.add(gpa, d, cap * (low + 4) * 2);
        s.hc_part = try s.add(gpa, d, hc_sk * cap * (low + 4) * 4);
        s.act = try s.add(gpa, d, cap * low * 2);
        s.xs_act = try s.add(gpa, d, cap * (low / 32) * 4);
        s.inj = try s.add(gpa, d, cap * streams * 2);
        s.mixed = try s.add(gpa, d, cap * dims * 2);
        s.xsm = try s.add(gpa, d, cap * (dims / 32) * 4);
        s.proj = try s.add(gpa, d, cap * proj_row * 2);
        s.part = try s.add(gpa, d, out_sk * cap * dims * 4);
        s.gout = try s.add(gpa, d, cap * gdn.value_dim * 2);
        s.gxs = try s.add(gpa, d, cap * (gdn.value_dim / 32) * 4);
        s.ys = try s.add(gpa, d, cap * gdn.value_dim * 4);
        s.gq = try s.add(gpa, d, cap * gdn.nk * gdn.dk * 4);
        s.gk = try s.add(gpa, d, cap * gdn.nk * gdn.dk * 4);
        s.gv = try s.add(gpa, d, cap * gdn.value_dim * 4);
        s.ggb = try s.add(gpa, d, cap * gdn.nv * 2 * 4);
        s.q = try s.add(gpa, d, cap * q_heads * head_dim * 2);
        s.iq = try s.add(gpa, d, cap * index_heads * index_dim * 2);
        s.po = try s.add(gpa, d, att_rows * nch * q_heads * head_dim * 4);
        s.pm = try s.add(gpa, d, att_rows * nch * q_heads * 4);
        s.pl = try s.add(gpa, d, att_rows * nch * q_heads * 4);
        s.ao = try s.add(gpa, d, cap * out_k * 2);
        s.gated = try s.add(gpa, d, cap * out_k * 2);
        s.gxs_attn = try s.add(gpa, d, cap * (out_k / 32) * 4);
        s.router_l = try s.add(gpa, d, cap * (experts_n + 1) * 4);
        s.picks = try s.add(gpa, d, pairs * 4);
        s.wts = try s.add(gpa, d, pairs * 4);
        s.members = try s.add(gpa, d, pairs * 4);
        s.items = try s.add(gpa, d, items * 3 * 4);
        s.counts = try s.add(gpa, d, 8);
        s.rank = try s.add(gpa, d, pairs * 4);
        s.hist = try s.add(gpa, d, nblk * (experts_n + 1) * 4);
        s.moe_act = try s.add(gpa, d, pairs * moe_w * 2);
        s.y = try s.add(gpa, d, pairs * dims * 4);
        s.branch = try s.add(gpa, d, cap * dims * 2);
        s.extra = try s.add(gpa, d, cap * dims * 4);
        s.ple_emb = try s.add(gpa, d, cap * dims * 2);
        s.ple_xs = try s.add(gpa, d, cap * (dims / 32) * 4);
        s.ple_keys = try s.add(gpa, d, cap * wide * 2);
        s.ple_vals = try s.add(gpa, d, cap * dims * 2);
        s.ple_part = try s.add(gpa, d, 4 * cap * dims * 4);
        s.ple_gated = try s.add(gpa, d, cap * wide * 2);
        s.ple_pss = try s.add(gpa, d, cap * streams * 4);
        s.ple_nrow = try s.add(gpa, d, cap * wide * 2);
        s.logits = try s.add(gpa, d, vocab * 2);
        var inv: [half]f32 = undefined;
        for (&inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / @as(f64, half)));
        s.inv = try s.add(gpa, d, half * 4);
        try s.bufs.items[s.bufs.items.len - 1].upload(0, std.mem.sliceAsBytes(&inv));
        s.dummy = try s.add(gpa, d, 4096);
        try s.bufs.items[s.bufs.items.len - 1].fill8(0, null);
        return s;
    }

    pub fn free(s: *Scratch, gpa: std.mem.Allocator) void {
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(gpa);
        s.* = undefined;
    }
};

/// A sequence's state the pass reads and leaves as the row-by-row path would.
pub const SeqState = struct {
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    history: *[2]i64,
};

/// The engine's resident weights the pass reads.
pub const Model = struct {
    table_w: u64,
    table_s: u64,
    table_b: u64,
    layers: []const Layer,
    mixer: HcW,
    head: Q4,
    ngram: *const ple.Ngram,
};

/// TF_FLASHNEXT_PROFILE=1: each chunk's wall time (synchronized) and its host n-gram gather on stderr.
fn profile() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROFILE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn Pass(comptime Tri: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        io: std.Io,
        stream: cuda.Stream,
        tri: Tri,
        k: *const Kernels,
        s: *const Scratch,
        m: Model,
        seq: SeqState,
        hooks: Hooks,
        rows: usize = 0,
        pos0: usize = 0,
        /// The MoE write-back the next layer (or the head) applies first, and the layer it came from.
        pending: ?usize = null,

        // ---- launches ----------------------------------------------------------------------------------------

        /// qmm.cu's lane tile over m rows: out bf16 [m, n] (sk 1) or the unreduced fp32 slices [sk, m, n] in part.
        pub fn lane(self: *Self, x: u64, xs: u64, w: Q4, out: u64, part: u64, m: usize, sk: usize) !void {
            const bm: usize = 16;
            const bn: usize = 64;
            const k = w.k;
            const rows_t = cdiv(m, bm);
            const cols_t = cdiv(w.n, bn);
            const sweep = @max(@as(usize, 1), @min(rows_t, (12 << 20) / (bm * k * 2)));
            const npad = (w.n + 127) / 128 * 128;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(xs);
            a.add(w.w.ptr);
            a.add(w.s.ptr);
            a.add(w.b.ptr);
            a.add(out);
            a.add(part);
            for ([_]usize{ m, w.n, k, sk, npad, k, sweep }) |v| a.add(cint(v));
            try cuda.launch.launch(self.k.lane, .{ .grid = .{ .x = @intCast(rows_t * cols_t), .z = @intCast(sk) }, .block = .{ .x = 128 }, .shared = qmm_smem }, self.stream, &a);
        }

        pub fn sumSlices(self: *Self, part: u64, out: u64, bf16: bool, n: usize, sk: usize) !void {
            var a: cuda.Args = .{};
            a.add(part);
            a.add(out);
            a.add(@as(c_int, @intFromBool(bf16)));
            a.add(@as(c_longlong, @intCast(n)));
            a.add(cint(sk));
            try cuda.launch.launch(self.k.sum, .{ .grid = .{ .x = @intCast(cdiv(n, 256)) }, .block = .{ .x = 256 } }, self.stream, &a);
        }

        pub fn shiftWindow(self: *Self, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
            var a: cuda.Args = .{};
            a.add(win);
            a.add(rows_ptr);
            a.add(@as(c_longlong, @intCast(stride)));
            a.add(cint(n));
            a.add(cint(taps));
            a.add(cint(c));
            try cuda.launch.launch(self.k.shift, .{ .grid = .{ .x = @intCast(cdiv(c, 256)) }, .block = .{ .x = 256 } }, self.stream, &a);
        }

        /// rows x width 16-bit (or 32-bit) values between pitched buffers, or the `index` rows of src.
        fn copy2d(self: *Self, wide32: bool, src: u64, src_pitch: usize, dst: u64, dst_pitch: usize, width: usize, rows: usize, index: u64) !void {
            if (rows == 0 or width == 0) return;
            var a: cuda.Args = .{};
            a.add(src);
            a.add(@as(c_longlong, @intCast(src_pitch)));
            a.add(dst);
            a.add(@as(c_longlong, @intCast(dst_pitch)));
            a.add(cint(width));
            a.add(cint(rows));
            a.add(index);
            try cuda.launch.launch(if (wide32) self.k.copy32 else self.k.copy, .{ .grid = .{ .x = @intCast(cdiv(width, 256)), .y = @intCast(rows) }, .block = .{ .x = 256 } }, self.stream, &a);
        }

        pub fn copy(self: *Self, dst: u64, src: u64, n: usize) !void {
            if (dst == 0 or n == 0) return;
            const d = self.stream.d;
            try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, self.stream.handle), "cuMemcpyDtoDAsync");
        }

        pub fn upload(self: *Self, dst: u64, bytes: []const u8) !void {
            const d = self.stream.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, self.stream.handle), "cuMemcpyHtoDAsync");
        }

        // ---- the blocks ----------------------------------------------------------------------------------------

        /// The pending MoE output (or none: mode 0, the streams' sums alone) written back into rows [r0, r0 + n).
        pub fn writeback(self: *Self, r0: usize, n: usize) !void {
            const s = self.s;
            const dummy = s.dummy;
            const h = s.h + r0 * wide * 2;
            const pss = s.pss + r0 * pss_n * 4;
            const inj = s.inj + r0 * streams * 2;
            const rs = self.rows * dims;
            const from = self.pending orelse {
                try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", dummy, dummy, "*bf16", dummy, "*bf16", rs, n, dims, streams, 0, 1, 1, 1);
                return;
            };
            self.pending = null;
            const y = s.y + r0 * slots * dims * 4;
            const wts = s.wts + r0 * slots * 4;
            const nudge: u64 = if (self.hooks.moe_add != null) s.extra else if (self.hooks.nudge) |nd| nd[from] else 0;
            const keep: u64 = if (self.hooks.keep) |kp| kp.layers[from].moe_branch else 0;
            if (nudge == 0 and keep == 0) {
                try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", inj, y, "*fp32", wts, "*fp32", rs, n, dims, streams, 2, 10, 11, 1);
                return;
            }
            // the slots summed as MODE 2 sums them (fp32 fma in slot order), the nudge added before the rounding,
            // then written back as a plain bf16 branch (MODE 1): the same bits as MODE 2 when the nudge is 0
            const branch = if (keep != 0 and r0 == 0) keep else s.branch;
            var a: cuda.Args = .{};
            a.add(y);
            a.add(wts);
            a.add(if (nudge != 0) nudge + r0 * dims * 4 else @as(u64, 0));
            a.add(branch);
            a.add(cint(dims));
            a.add(cint(slots));
            try cuda.launch.launch(self.k.combine, .{ .grid = .{ .x = @intCast(n), .y = @intCast(dims / 256) }, .block = .{ .x = 256 } }, self.stream, &a);
            if (keep != 0 and branch != keep) try self.copy(keep + r0 * dims * 2, branch, n * dims * 2);
            try self.tri.hcWriteback(h, h, pss, branch, "*bf16", inj, dummy, "*bf16", dummy, "*fp32", rs, n, dims, streams, 1, 8, 9, 1);
        }

        /// A read-out over rows [r0, r0 + n) of the streams: mixed and its group sums, and the inject gates.
        pub fn readout(self: *Self, hc_w: HcW, r0: usize, n: usize, keep: HcKeep) !void {
            const s = self.s;
            const h = s.h + r0 * wide * 2;
            const pss = s.pss + r0 * pss_n * 4;
            const inj = s.inj + r0 * streams * 2;
            const norm = s.norm + r0 * wide * 2;
            const act = s.act + r0 * low * 2;
            const xs_act = s.xs_act + r0 * (low / 32) * 4;
            try self.copy(keep.h, h, n * wide * 2);
            try self.tri.hcDown(h, pss, hc_w.scale.ptr, norm, hc_w.down.w.ptr, hc_w.down.s.ptr, hc_w.down.b.ptr, s.dn, s.hc_part, eps, n, hc_w.n_down, wide, dims, dims / 256, streams, hc_sk);
            const inj_ptr = if (hc_w.has_inj == 1) inj else act;
            try self.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, hc_sk, n, streams, low, hc_w.n_down, hc_w.has_inj);
            try self.tri.hcUpmix(act, xs_act, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, wide, low, dims, streams);
            try self.copy(keep.normed, norm, n * wide * 2);
            if (keep.v != 0) try self.sumSlices(s.hc_part, keep.v, false, n * hc_w.n_down, hc_sk);
            if (keep.up != 0) try self.tri.upRows(act, xs_act, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, keep.up, n, wide, low);
            if (keep.inject != 0 and hc_w.has_inj == 1) try self.copy(keep.inject, inj, n * streams * 2);
        }

        fn pleBlock(self: *Self, lw: *const Layer, ids: []const u32) !void {
            const s = self.s;
            const n = self.rows;
            // the rows' n-gram embeddings (host gathers from the mapped table) are already uploaded by `stage`
            _ = ids;
            try self.lane(s.ple_emb, s.ple_xs, lw.key, s.ple_keys, s.dummy, n, 1);
            try self.lane(s.ple_emb, s.ple_xs, lw.val, s.dummy, s.ple_part, n, 4);
            try self.sumSlices(s.ple_part, s.ple_vals, true, n * dims, 4);
            try self.tri.pleGate(s.ple_keys, s.ple_vals, s.h, lw.nk.ptr, lw.nq.ptr, s.ple_gated, s.ple_pss, eps, n, dims, streams);
            try self.tri.pleConv(s.ple_gated, s.ple_pss, lw.nc.ptr, self.seq.tail.ptr, lw.ple_cw.ptr, s.h, s.h, s.ple_nrow, eps, n, dims, streams, 4, 3);
            try self.shiftWindow(self.seq.tail.ptr, s.ple_nrow, wide, n, ple_tail_n, wide);
        }

        fn gdnBlock(self: *Self, lw: *const Layer, li: usize, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            try self.lane(s.mixed, s.xsm, lw.proj, s.proj, s.dummy, n, 1);
            try self.copy(keep.proj, s.proj, n * gdn.proj_width * 2);
            try self.copy(keep.hist, self.seq.conv[li].ptr, (gdn.taps - 1) * gdn.conv_dim * 2);
            try self.copy(keep.state, self.seq.state[li].ptr, state_n * 4);
            // every row's conv, norms and gates at once, then the recurrence (a warp's 4 state rows each, 16 a block),
            // then the gated norm per row: the one-block chain's bits (flashnext_batch.cu)
            var f: cuda.Args = .{};
            f.add(s.proj);
            f.add(self.seq.conv[li].ptr);
            f.add(lw.cw.ptr);
            f.add(lw.a_log.ptr);
            f.add(lw.dt.ptr);
            f.add(s.gq);
            f.add(s.gk);
            f.add(s.gv);
            f.add(s.ggb);
            try cuda.launch.launch(self.k.front, .{ .grid = .{ .x = @intCast(n), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, self.stream, &f);
            var a: cuda.Args = .{};
            a.add(s.gq);
            a.add(s.gk);
            a.add(s.gv);
            a.add(s.ggb);
            a.add(self.seq.state[li].ptr);
            if (keep.state_out != 0) a.add(keep.state_out);
            a.add(cint(n));
            a.add(s.ys);
            try cuda.launch.launch(if (keep.state_out != 0) self.k.chain_to else self.k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, self.stream, &a);
            var b: cuda.Args = .{};
            b.add(s.ys);
            b.add(s.proj);
            b.add(lw.norm.ptr);
            b.add(eps);
            b.add(s.gout);
            b.add(s.gxs);
            try cuda.launch.launch(self.k.gnorm, .{ .grid = .{ .x = @intCast(n), .y = gdn.nv }, .block = .{ .x = gdn.dv } }, self.stream, &b);
            try self.shiftWindow(self.seq.conv[li].ptr, s.proj, gdn.proj_width, n, gdn.taps - 1, gdn.conv_dim);
            try self.lane(s.gout, s.gxs, lw.out, s.dummy, s.part, n, out_sk);
        }

        pub fn attnBlock(self: *Self, lw: *const Layer, ai: usize, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            const pa = s.proj;
            try self.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
            if (keep.pq != 0) try self.copy2d(false, pa, proj_n, keep.pq, q_heads * 2 * head_dim, q_heads * 2 * head_dim, n, 0);
            if (keep.pk != 0) try self.copy2d(false, pa + q_heads * 2 * head_dim * 2, proj_n, keep.pk, kv_heads * head_dim, kv_heads * head_dim, n, 0);
            if (keep.pv != 0) try self.copy2d(false, pa + (q_heads * 2 + kv_heads) * head_dim * 2, proj_n, keep.pv, kv_heads * head_dim, kv_heads * head_dim, n, 0);
            const kc = self.seq.kc[ai].ptr;
            const vc = self.seq.vc[ai].ptr;
            try self.tri.attnPrep(pa, s.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, s.q, kc, vc, s.dummy, s.dummy, s.iq, self.seq.ikc[ai].ptr, s.dummy, s.dummy, eps, n, 0);
            var r0: usize = 0;
            var b: usize = 0;
            while (r0 < n) : ({
                r0 += att_rows;
                b += 1;
            }) {
                const m = @min(att_rows, n - r0);
                const keys = self.pos0 + r0 + m;
                if (keys > nch * 512) return error.UnexpectedTensor;
                const pos = s.pos + b * 16;
                try self.tri.attnChunks(s.q + r0 * q_heads * head_dim * 2, kc, vc, s.dummy, s.dummy, pos, s.po, s.pm, s.pl, s.dummy, s.dummy, s.dummy, m, kv_heads, cdiv(keys, 512));
                try self.tri.attnMerge(s.po, s.pm, s.pl, pos, s.ao + r0 * out_k * 2, s.dummy, s.dummy, m, kv_heads);
            }
            const gated = if (keep.gated != 0) keep.gated else s.gated;
            try self.tri.attnGate(s.ao, pa, gated, s.gxs_attn, n, proj_n, q_heads, head_dim);
            try self.lane(gated, s.gxs_attn, lw.out, s.dummy, s.part, n, out_sk);
        }

        /// Router, each row's ten experts and the shared one, the grouped expert launches: y [rows, 11, dims] fp32.
        pub fn moeBlock(self: *Self, lw: *const Layer, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            const pairs = n * slots;
            try self.copy(keep.moe_x, s.mixed, n * dims * 2);
            try self.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
            try self.tri.topkMany(s.router_l, s.picks, s.wts, n);
            try self.copy(keep.moe_ids, s.picks, pairs * 4);
            try self.copy(keep.moe_wts, s.wts, pairs * 4);
            const e = experts_n + 1;
            if (pairs <= plan_small) {
                var a: cuda.Args = .{};
                a.add(s.picks);
                a.add(cint(pairs));
                a.add(cint(e));
                a.add(cint(rows_tile));
                a.add(s.members);
                a.add(s.items);
                a.add(s.counts);
                try cuda.launch.launch(self.k.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, self.stream, &a);
            } else {
                const nblk = cdiv(pairs, plan_small);
                var a: cuda.Args = .{};
                a.add(s.picks);
                a.add(cint(pairs));
                a.add(cint(e));
                a.add(s.rank);
                a.add(s.hist);
                try cuda.launch.launch(self.k.rank, .{ .grid = .{ .x = @intCast(nblk) }, .block = .{ .x = 1024 } }, self.stream, &a);
                var o: cuda.Args = .{};
                o.add(cint(nblk));
                o.add(cint(e));
                o.add(cint(rows_tile));
                o.add(s.hist);
                o.add(s.items);
                o.add(s.counts);
                try cuda.launch.launch(self.k.offsets, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, self.stream, &o);
                var c: cuda.Args = .{};
                c.add(s.picks);
                c.add(cint(pairs));
                c.add(cint(e));
                c.add(s.rank);
                c.add(s.hist);
                c.add(s.members);
                try cuda.launch.launch(self.k.scatter, .{ .grid = .{ .x = @intCast(cdiv(pairs, 256)) }, .block = .{ .x = 256 } }, self.stream, &c);
            }
            const items = cdiv(pairs, rows_tile) + e;
            try self.expert(self.k.up, s.mixed, dims, slots, lw.experts_up.ptr, dims / 32, moe_w / 32, s.moe_act, moe_w, items);
            try self.expert(self.k.down, s.moe_act, moe_w, 0, lw.experts_down.ptr, moe_w / 32, dims / 32, s.y, dims, items);
        }

        fn expert(self: *Self, f: cuda.Function, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, out: u64, n: usize, items: usize) !void {
            const s = self.s;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(cint(x_stride));
            a.add(cint(slot_n));
            a.add(w);
            a.add(cint(kg));
            a.add(cint(nb));
            a.add(s.items);
            a.add(s.counts);
            a.add(s.members);
            a.add(out);
            a.add(cint(n));
            a.add(@as(f32, 0));
            const blocks = @min(cdiv(items * nb, 4), 4096);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 128 } }, self.stream, &a);
        }

        fn tapRows(self: *Self, layer: usize) !void {
            const t = self.hooks.taps orelse return;
            if (t.rows.len == 0) return;
            for (t.rows) |r| if (r >= self.rows) return error.UnexpectedTensor;
            try self.copy2d(false, self.s.mixed, dims, t.normed + layer * t.rows.len * dims * 2, dims, dims, t.rows.len, t.index);
        }

        /// The shared expert's gate (slot 11's weight) at the tapped rows, once the layer has routed.
        fn tapGates(self: *Self, layer: usize) !void {
            const t = self.hooks.taps orelse return;
            if (t.rows.len == 0 or t.gates == 0) return;
            try self.copy2d(true, self.s.wts + (slots - 1) * 4, slots, t.gates + layer * t.rows.len * 4, 1, 1, t.rows.len, t.index);
        }

        // ---- the pass ------------------------------------------------------------------------------------------

        /// Token ids, positions and (when the n-gram layer is there) each row's n-gram embedding onto the device.
        pub fn stage(self: *Self, ids: []const u32) !void {
            const s = self.s;
            const n = ids.len;
            const id32 = try self.gpa.alloc(i32, n);
            defer self.gpa.free(id32);
            for (id32, ids) |*o, v| o.* = @intCast(v);
            try self.upload(s.ids, std.mem.sliceAsBytes(id32));
            const blocks = cdiv(n, att_rows);
            // one int a block, 16 bytes apart: _chunks and _merge take POS0 16-byte aligned
            const pos = try self.gpa.alloc(i32, blocks * 4);
            defer self.gpa.free(pos);
            @memset(pos, 0);
            for (0..blocks) |b| pos[4 * b] = @intCast(self.pos0 + b * att_rows);
            try self.upload(s.pos, std.mem.sliceAsBytes(pos));
        }

        /// The rows' n-gram embeddings, gathered on a host thread while the GPU embeds and runs layer 0.
        const Gather = struct {
            ngram: *const ple.Ngram,
            history: []i64,
            ids: []const u32,
            emb: []u16,
            xs: []f32,
            failed: ?anyerror = null,
            ns: i96 = 0,
            io: std.Io,

            fn main(g: *Gather) void {
                const t0 = std.Io.Clock.awake.now(g.io).toNanoseconds();
                g.ngram.rows(g.history, g.ids, g.emb, g.xs) catch |err| {
                    g.failed = err;
                };
                g.ns = std.Io.Clock.awake.now(g.io).toNanoseconds() - t0;
            }
        };

        /// One chunk of rows at positions pos0.. through every layer and the head.
        pub fn chunk(self: *Self, ids: []const u32, pos0: usize, logits_host: ?[]u8) !void {
            const s = self.s;
            const n = ids.len;
            const t0 = std.Io.Clock.awake.now(self.io).toNanoseconds();
            defer if (profile()) {
                self.stream.synchronize() catch {};
                std.debug.print("batch: {d} rows at {d} in {d:.1} ms\n", .{ n, pos0, @as(f64, @floatFromInt(std.Io.Clock.awake.now(self.io).toNanoseconds() - t0)) / 1e6 });
            };
            if (n == 0 or n > s.cap or pos0 + n > nch * 512) return error.UnexpectedTensor;
            self.rows = n;
            self.pos0 = pos0;
            self.pending = null;
            if (self.hooks.keep) |kp| kp.pos0 = pos0;
            const emb = try self.gpa.alloc(u16, n * dims);
            defer self.gpa.free(emb);
            const xs = try self.gpa.alloc(f32, n * (dims / 32));
            defer self.gpa.free(xs);
            var gather: Gather = .{ .ngram = self.m.ngram, .history = self.seq.history[0..], .ids = ids, .emb = emb, .xs = xs, .io = self.io };
            var thread: ?std.Thread = try std.Thread.spawn(.{}, Gather.main, .{&gather});
            defer if (thread) |th| th.join();
            try self.stage(ids);
            try self.tri.embed(s.ids, self.m.table_w, self.m.table_s, self.m.table_b, s.h, dims, streams, n);
            for (self.m.layers, 0..) |*lw, layer| {
                const keep: LayerKeep = if (self.hooks.keep) |kp| kp.layers[layer] else .{};
                if (layer == ple_layer) {
                    thread.?.join();
                    thread = null;
                    if (gather.failed) |err| return err;
                    if (profile()) std.debug.print("batch: {d} rows' n-gram rows gathered on the host in {d:.1} ms\n", .{ n, @as(f64, @floatFromInt(gather.ns)) / 1e6 });
                    try self.upload(s.ple_emb, std.mem.sliceAsBytes(emb));
                    try self.upload(s.ple_xs, std.mem.sliceAsBytes(xs));
                    if (self.pending != null) try self.writeback(0, n);
                    try self.pleBlock(lw, ids);
                }
                try self.writeback(0, n);
                try self.readout(lw.attn_hc, 0, n, keep.attn_hc);
                if (lw.linear) {
                    try self.gdnBlock(lw, wt.linearIndex(layer), keep);
                } else {
                    try self.attnBlock(lw, wt.attnIndex(layer), keep);
                }
                if (keep.mix_branch != 0) try self.sumSlices(s.part, keep.mix_branch, true, n * dims, out_sk);
                try self.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", n * dims, n, dims, streams, 4, 1, 1, out_sk);
                try self.readout(lw.mlp_hc, 0, n, keep.mlp_hc);
                try self.tapRows(layer);
                try self.moeBlock(lw, keep);
                try self.tapGates(layer);
                if (self.hooks.moe_add) |cb| {
                    // the layer's nudge (or zeros) in the pass's own buffer, then what the callback adds to it
                    const given: u64 = if (self.hooks.nudge) |nd| nd[layer] else 0;
                    if (given != 0) {
                        try self.copy(s.extra, given, n * dims * 4);
                    } else {
                        const d = self.stream.d;
                        try d.check(d.api.cuMemsetD32Async(s.extra, 0, n * dims, self.stream.handle), "cuMemsetD32Async");
                    }
                    try cb.add(cb.ctx, layer, s.mixed, s.picks, s.wts, s.extra, n);
                }
                self.pending = layer;
            }
            try self.head(logits_host);
        }

        fn head(self: *Self, logits_host: ?[]u8) !void {
            const s = self.s;
            const n = self.rows;
            switch (self.hooks.logits) {
                .none => {
                    const mixer_keep: ?HcKeep = if (self.hooks.keep) |kp| (if (kp.mixer.h != 0) kp.mixer else null) else null;
                    if (mixer_keep != null or self.hooks.streams != 0) try self.writeback(0, n);
                    try self.copy(self.hooks.streams, s.h, n * wide * 2);
                    if (mixer_keep) |mk| try self.readout(self.m.mixer, 0, n, mk);
                    return;
                },
                .all => {
                    try self.writeback(0, n);
                    try self.copy(self.hooks.streams, s.h, n * wide * 2);
                    try self.readout(self.m.mixer, 0, n, if (self.hooks.keep) |kp| kp.mixer else .{});
                    const out = if (self.hooks.logits_out != 0) self.hooks.logits_out else return error.UnexpectedTensor;
                    try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
                    if (logits_host) |lh| {
                        try self.stream.synchronize();
                        const d = self.stream.d;
                        try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
                    }
                },
                .last => {
                    // the last row's streams, sums, gates and MoE slots moved to row 0, where every pointer the
                    // captured kernels take stays 16-byte aligned (an odd row's inject gates and weights are not)
                    const r = n - 1;
                    if (self.hooks.streams != 0) {
                        // every row written back (each row's bits are its own) and kept, then the last row's
                        // streams and sums at row 0 for the mixer, as below
                        try self.writeback(0, n);
                        try self.copy(self.hooks.streams, s.h, n * wide * 2);
                        if (r > 0) {
                            try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                            try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                        }
                    } else if (r > 0) {
                        try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                        try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                        try self.copy(s.inj, s.inj + r * streams * 2, streams * 2);
                        try self.copy(s.y, s.y + r * slots * dims * 4, slots * dims * 4);
                        try self.copy(s.wts, s.wts + r * slots * 4, slots * 4);
                        if (self.hooks.moe_add != null) try self.copy(s.extra, s.extra + r * dims * 4, dims * 4);
                    }
                    if (self.hooks.streams == 0) try self.writeback(0, 1);
                    try self.readout(self.m.mixer, 0, 1, .{});
                    try self.lane(s.mixed, s.xsm, self.m.head, s.logits, s.dummy, 1, 1);
                    if (logits_host) |lh| {
                        try self.stream.synchronize();
                        const d = self.stream.d;
                        try d.check(d.api.cuMemcpyDtoH_v2(lh.ptr, s.logits, vocab * 2), "cuMemcpyDtoH");
                    }
                },
            }
        }
    };
}

test "a pass's scratch at the dense cap stays near a gigabyte and a half" {
    try std.testing.expect(Scratch.bytes(default_chunk) < 2 << 30);
    try std.testing.expect(Scratch.bytes(1) < Scratch.bytes(2));
}
