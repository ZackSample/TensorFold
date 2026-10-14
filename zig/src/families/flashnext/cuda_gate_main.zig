//! The pocket neurons' gates on given prompts (`flashnext-gate`): for each line of a list ("tag<TAB>rows<TAB>ids", rows
//! comma-separated), the prompt runs through a taught folder (its pockets loaded as serving adds them) and every
//! learned neuron's gate row times the MoE input at each listed row is printed ("tag row layer neuron value"): the
//! input fit's u x times the gate's sharpening. A prompt longer than a training chunk runs its head plainly first,
//! then its tail (holding every listed row) with the taps.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const batch = prompt.batch;
const dims = 2560;
const layers_n = 48;

fn bf(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-gate <model-dir> <kernel-dir> <list-file>\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(64 << 20));
    defer gpa.free(text);

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

    // every layer's pocket gate rows, read once
    var gates: [layers_n][]f32 = undefined;
    var used: [layers_n]usize = @splat(0);
    for (0..layers_n) |l| {
        const cap = if (e.pockets[l]) |pk| pk.cap else 0;
        gates[l] = try gpa.alloc(f32, cap * dims);
        if (cap == 0) continue;
        const up = try gpa.alloc(f32, cap * dims);
        defer gpa.free(up);
        const dn = try gpa.alloc(f32, cap * dims);
        defer gpa.free(dn);
        used[l] = e.pocketRead(l, gates[l], up, dn);
    }
    defer for (gates) |g| gpa.free(g);
    var total: usize = 0;
    for (used) |u| total += u;
    std.debug.print("pocket: {d} neurons in use\n", .{total});

    const max_rows = 64;
    var tapped_b = try cuda.DeviceBuffer.alloc(&driver, layers_n * max_rows * dims * 2);
    defer tapped_b.free();
    var index_b = try cuda.DeviceBuffer.alloc(&driver, 4 * max_rows);
    defer index_b.free();
    const host = try gpa.alloc(u8, layers_n * max_rows * dims * 2);
    defer gpa.free(host);

    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const tag = cols.next() orelse continue;
        var rows: std.ArrayList(usize) = .empty;
        defer rows.deinit(gpa);
        var rt = std.mem.tokenizeAny(u8, cols.next() orelse continue, ", ");
        while (rt.next()) |part| try rows.append(gpa, try std.fmt.parseInt(usize, part, 10));
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, cols.next() orelse continue, ", ");
        while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
        const n = ids.items.len;
        if (rows.items.len == 0 or rows.items.len > max_rows) return error.TooManyRows;
        var lo: usize = n;
        for (rows.items) |r| {
            if (r >= n) return error.RowPastPrompt;
            lo = @min(lo, r);
        }
        // the tail holding every listed row, at most a chunk; the head before it runs plainly
        const start = if (n > e.chunk) n - e.chunk else 0;
        if (lo < start) return error.RowPastPrompt;
        var seq = try e.newSeq(n + 1);
        defer seq.deinit(gpa);
        if (start > 0) try e.forward(flash.Tri, tri, &seq, ids.items[0..start], .{});
        const rel = try gpa.alloc(u32, rows.items.len);
        defer gpa.free(rel);
        for (rows.items, rel) |r, *x| x.* = @intCast(r - start);
        try index_b.upload(0, std.mem.sliceAsBytes(rel));
        const taps: batch.Taps = .{ .rows = rel, .index = index_b.ptr, .normed = tapped_b.ptr };
        try e.forward(flash.Tri, tri, &seq, ids.items[start..], .{ .taps = taps });
        try stream.synchronize();
        const m = rel.len;
        try tapped_b.download(0, host[0 .. layers_n * m * dims * 2]);
        for (0..layers_n) |l| {
            for (0..m) |ri| {
                const x = host[(l * m + ri) * dims * 2 ..][0 .. dims * 2];
                for (0..used[l]) |i| {
                    var dot: f64 = 0;
                    const g = gates[l][i * dims ..][0..dims];
                    for (0..dims) |k| dot += @as(f64, g[k]) * @as(f64, bf(std.mem.readInt(u16, x[k * 2 ..][0..2], .little)));
                    std.debug.print("{s} {d} {d} {d} {d:.4}\n", .{ tag, rows.items[ri], l, i, dot });
                }
            }
        }
    }
    return 0;
}
