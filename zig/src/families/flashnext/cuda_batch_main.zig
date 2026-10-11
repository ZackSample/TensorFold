//! The batched pass's training hooks on one prompt (`zig build flashnext-batch`): a pass that keeps every layer's
//! tensors, taps three rows, adds zero nudges and heads every row must give the prompt pass's last-row scores byte
//! for byte; the taps must be the kept MoE inputs and gates; each kept write-back must rebuild the streams it made;
//! and a nudge at one row and layer must leave every earlier row's scores alone and move its own.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const batch = prompt.batch;
const dims = 2560;
const wide = 4 * dims;
const vocab = prompt.vocabulary;
const layers_n = 48;

fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

const Bufs = struct {
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,

    fn get(b: *Bufs, len: usize) !u64 {
        var x = try cuda.DeviceBuffer.alloc(b.d, len);
        errdefer x.free();
        try x.fill8(0, null);
        try b.list.append(b.gpa, x);
        return x.ptr;
    }

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.gpa);
    }
};

fn down(d: *const cuda.Driver, gpa: std.mem.Allocator, ptr: u64, len: usize) ![]u8 {
    const out = try gpa.alloc(u8, len);
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, ptr, len), "cuMemcpyDtoH");
    return out;
}

fn hcKeep(b: *Bufs, rows: usize, inject: bool) !batch.HcKeep {
    return .{
        .h = try b.get(rows * wide * 2),
        .normed = try b.get(rows * wide * 2),
        .v = try b.get(rows * (320 + 4) * 4),
        .up = try b.get(rows * wide * 2),
        .inject = if (inject) try b.get(rows * 4 * 2) else 0,
    };
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-batch <model-dir> <kernel-dir> <ids-file>\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    const n = ids.items.len;
    if (n < 12) return 2;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .set = &set, .s = stream };
    var e = try prompt.Engine.init(gpa, io, &driver, &stream, args[1]);
    defer e.deinit();

    // the prompt pass: the last row's scores
    var base = try e.newSeq(n + 1);
    defer base.deinit(gpa);
    try e.forward(flash.Tri, tri, &base, ids.items, .{});
    const want = try gpa.dupe(u8, base.logits);
    defer gpa.free(want);

    var bufs: Bufs = .{ .gpa = gpa, .d = &driver };
    defer bufs.deinit();
    var keep: batch.Keep = .{};
    for (&keep.layers, 0..) |*k, l| {
        k.attn_hc = try hcKeep(&bufs, n, true);
        k.mlp_hc = try hcKeep(&bufs, n, true);
        k.mix_branch = try bufs.get(n * dims * 2);
        k.moe_branch = try bufs.get(n * dims * 2);
        k.moe_x = try bufs.get(n * dims * 2);
        k.moe_ids = try bufs.get(n * 11 * 4);
        k.moe_wts = try bufs.get(n * 11 * 4);
        if (l % 4 == 3) {
            k.pq = try bufs.get(n * 12288 * 2);
            k.pk = try bufs.get(n * 512 * 2);
            k.pv = try bufs.get(n * 512 * 2);
            k.gated = try bufs.get(n * 6144 * 2);
        } else {
            k.proj = try bufs.get(n * 16480 * 2);
            k.hist = try bufs.get(3 * 10240 * 2);
            k.state = try bufs.get(48 * 128 * 128 * 4);
        }
    }
    keep.mixer = try hcKeep(&bufs, n, false);
    var nudges: [layers_n]u64 = undefined;
    const zero = try bufs.get(n * dims * 4);
    for (&nudges) |*p| p.* = zero;
    const tap_rows = [_]u32{ 0, 10, @intCast(n - 1) };
    var index_b = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&tap_rows));
    defer index_b.free();
    const taps: batch.Taps = .{ .rows = &tap_rows, .index = index_b.ptr, .normed = try bufs.get(layers_n * tap_rows.len * dims * 2), .gates = try bufs.get(layers_n * tap_rows.len * 4) };
    const all = try bufs.get(n * vocab * 2);

    var seq = try e.newSeq(n + 1);
    defer seq.deinit(gpa);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try e.forward(flash.Tri, tri, &seq, ids.items, .{ .logits = .all, .logits_out = all, .keep = &keep, .nudge = &nudges, .taps = taps });
    try stream.synchronize();
    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    var kept: usize = 0;
    for (bufs.list.items) |x| kept += x.len;
    std.debug.print("keeping pass: {d} rows in {d:.1} ms, {d} MiB of keeps, taps, nudges and scores\n", .{ n, @as(f64, @floatFromInt(t1 - t0)) / 1e6, kept >> 20 });
    var failures: usize = 0;

    // (1) zero nudges, every row's head: the last row's scores are the prompt pass's
    const got = try down(&driver, gpa, all + (n - 1) * vocab * 2, vocab * 2);
    defer gpa.free(got);
    const same = std.mem.eql(u8, got, want);
    std.debug.print("{s} all-row head with zero nudges and every keep: last row's scores {s} the prompt pass's\n", .{ if (same) "PASS" else "FAIL", if (same) "byte-equal to" else "differ from" });
    if (!same) failures += 1;

    // (2) the taps are the kept MoE inputs and shared gates
    var tap_ok = true;
    for (0..layers_n) |l| {
        const t_rows = try down(&driver, gpa, taps.normed + l * tap_rows.len * dims * 2, tap_rows.len * dims * 2);
        defer gpa.free(t_rows);
        const x = try down(&driver, gpa, keep.layers[l].moe_x, n * dims * 2);
        defer gpa.free(x);
        const g = try down(&driver, gpa, taps.gates + l * tap_rows.len * 4, tap_rows.len * 4);
        defer gpa.free(g);
        const w = try down(&driver, gpa, keep.layers[l].moe_wts, n * 11 * 4);
        defer gpa.free(w);
        for (tap_rows, 0..) |r, i| {
            if (!std.mem.eql(u8, t_rows[i * dims * 2 ..][0 .. dims * 2], x[r * dims * 2 ..][0 .. dims * 2])) tap_ok = false;
            if (!std.mem.eql(u8, g[i * 4 ..][0..4], w[(r * 11 + 10) * 4 ..][0..4])) tap_ok = false;
        }
    }
    std.debug.print("{s} taps: the three rows' MoE inputs and shared gates equal the kept ones at all 48 layers\n", .{if (tap_ok) "PASS" else "FAIL"});
    if (!tap_ok) failures += 1;

    // (3) the kept write-backs rebuild the streams: mlp_hc.h = bf16(attn_hc.h + mix_branch * attn inject) (the
    // captured _hc_writeback fuses the product into the add: glue.py's bf16 rounding of it is folded away) and
    // the next layer's attn_hc.h from mlp_hc.h, moe_branch and the mlp inject (layer 1's n-gram branch excepted)
    var wb_off: usize = 0;
    var wb_n: usize = 0;
    for (0..layers_n) |l| {
        const k = keep.layers[l];
        const pairs = [_]struct { before: u64, branch: u64, inject: u64, after: u64 }{
            .{ .before = k.attn_hc.h, .branch = k.mix_branch, .inject = k.attn_hc.inject, .after = k.mlp_hc.h },
            .{ .before = k.mlp_hc.h, .branch = k.moe_branch, .inject = k.mlp_hc.inject, .after = if (l + 1 < layers_n) keep.layers[l + 1].attn_hc.h else keep.mixer.h },
        };
        for (pairs, 0..) |pr, which| {
            if (which == 1 and l + 1 == 1) continue; // the n-gram branch lands between layer 0's MoE and layer 1
            const hb = try down(&driver, gpa, pr.before, n * wide * 2);
            defer gpa.free(hb);
            const br = try down(&driver, gpa, pr.branch, n * dims * 2);
            defer gpa.free(br);
            const ij = try down(&driver, gpa, pr.inject, n * 4 * 2);
            defer gpa.free(ij);
            const ha = try down(&driver, gpa, pr.after, n * wide * 2);
            defer gpa.free(ha);
            const off0 = wb_off;
            defer if (wb_off != off0 and std.c.getenv("TF_FLASHNEXT_VERBOSE") != null) std.debug.print("  layer {d} {s}: {d} differ\n", .{ l, if (which == 0) "mixer" else "moe", wb_off - off0 });
            for ([_]usize{ 0, 10, n - 1 }) |r| for (0..4) |s| for (0..dims) |c| {
                const h0 = bf(std.mem.readInt(u16, hb[2 * (r * wide + s * dims + c) ..][0..2], .little));
                const b = bf(std.mem.readInt(u16, br[2 * (r * dims + c) ..][0..2], .little));
                const g = bf(std.mem.readInt(u16, ij[2 * (r * 4 + s) ..][0..2], .little));
                const want_h = toBf16(@mulAdd(f32, b, g, h0));
                const got_h = std.mem.readInt(u16, ha[2 * (r * wide + s * dims + c) ..][0..2], .little);
                wb_n += 1;
                if (want_h != got_h) {
                    if (wb_off - off0 < 3 and l < 2 and std.c.getenv("TF_FLASHNEXT_VERBOSE") != null) std.debug.print("    r {d} s {d} c {d}: h {d} branch {d} inject {d} want {x:0>4} got {x:0>4}\n", .{ r, s, c, h0, b, g, want_h, got_h });
                    wb_off += 1;
                }
            };
        }
    }
    std.debug.print("{s} kept write-backs: {d} of {d} rebuilt stream values differ\n", .{ if (wb_off == 0) "PASS" else "FAIL", wb_off, wb_n });
    if (wb_off != 0) failures += 1;

    // (4) a nudge at row 10 after layer 20: rows 0..9 keep their scores, row 10's move
    const nud = try bufs.get(n * dims * 4);
    {
        const host = try gpa.alloc(f32, dims);
        defer gpa.free(host);
        for (host, 0..) |*v, i| v.* = if (i % 2 == 0) 0.25 else -0.25;
        try driver.check(driver.api.cuMemcpyHtoD_v2(nud + 10 * dims * 4, host.ptr, dims * 4), "cuMemcpyHtoD");
    }
    var one: [layers_n]u64 = @splat(0);
    one[20] = nud;
    const all2 = try bufs.get(n * vocab * 2);
    var seq2 = try e.newSeq(n + 1);
    defer seq2.deinit(gpa);
    try e.forward(flash.Tri, tri, &seq2, ids.items, .{ .logits = .all, .logits_out = all2, .nudge = &one });
    try stream.synchronize();
    const a = try down(&driver, gpa, all, 11 * vocab * 2);
    defer gpa.free(a);
    const b = try down(&driver, gpa, all2, 11 * vocab * 2);
    defer gpa.free(b);
    const before_same = std.mem.eql(u8, a[0 .. 10 * vocab * 2], b[0 .. 10 * vocab * 2]);
    var moved: f32 = 0;
    for (0..vocab) |i| {
        const x = bf(std.mem.readInt(u16, a[2 * (10 * vocab + i) ..][0..2], .little));
        const y = bf(std.mem.readInt(u16, b[2 * (10 * vocab + i) ..][0..2], .little));
        moved = @max(moved, @abs(x - y));
    }
    const nudge_ok = before_same and moved > 0;
    std.debug.print("{s} nudge at row 10, layer 20: rows 0-9 {s}, row 10's scores move by up to {d:.3}\n", .{ if (nudge_ok) "PASS" else "FAIL", if (before_same) "unchanged" else "CHANGED", moved });
    if (!nudge_ok) failures += 1;
    return if (failures == 0) 0 else 1;
}
