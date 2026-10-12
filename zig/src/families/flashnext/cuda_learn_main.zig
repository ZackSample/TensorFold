//! Living Weights' trainer on Flash Next, checked on the GPU (`zig build flashnext-learn`): sequences packed side by
//! side give each one's loss as it gives alone; a target pass's push (dL/d a layer's MoE output at a nudged row)
//! against central finite differences of the loss as that row's nudge moves along it and along a random direction;
//! a pocket neuron in the forward moves the loss as its output's gradient says; and each pass's time.
//!
//! usage: flashnext-learn <model-dir> <kernel-dir> <ids-file>
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const train = flash.train;
const Trainer = train.Trainer(flash.Tri);
const dims = 2560;

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

fn dot(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += @as(f64, x) * y;
    return s;
}

fn norm16(xs: []const u16) f64 {
    var s: f64 = 0;
    for (xs) |b| {
        const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
        s += v * v;
    }
    return @sqrt(s);
}

/// The loss's slope as sequence `sq`'s key-k nudge at layer l moves along `dir`: a least-squares line through twelve
/// points at +-h, +-2h ... +-6h (bf16's rounding and the routing's flips make any one difference noisy).
fn slope(t: *Trainer, sq: *const train.Seq, k: usize, l: usize, dir: []const f32, h: f64) !f64 {
    const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
    var num: f64 = 0;
    var den: f64 = 0;
    for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
        const x = sgn * @as(f64, @floatFromInt(m)) * h;
        for (t.deltaAt(0, k, l), dir) |*o, u| o.* = @floatCast(x * u);
        var r: [1]train.Result = undefined;
        try t.pass(&.{sq.*}, .loss, turn, &r);
        if (std.c.getenv("FN_TRACE") != null) std.debug.print("    h {d:.4} loss {d:.5}\n", .{ x, r[0].loss });
        num += x * r[0].loss;
        den += x * x;
    };
    @memset(t.deltaAt(0, k, l), 0);
    return num / den;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-learn <model-dir> <kernel-dir> <ids-file>\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    if (ids.items.len > 256) ids.shrinkRetainingCapacity(256);
    const n = ids.items.len;
    if (n < 16) return 2;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .set = &set, .s = stream };
    const l0 = now(io);
    var e = try prompt.Engine.init(gpa, io, &driver, &stream, args[1]);
    defer e.deinit();
    _ = try e.reserve(train.pack_rows);
    const l1 = now(io);
    const t = try Trainer.init(gpa, io, e, tri);
    defer t.deinit();
    const l2 = now(io);
    const use = cuda.usage(false);
    std.debug.print("engine loaded in {d:.1} s, trainer in {d:.1} s; device bytes held {d:.2} GiB\n", .{ l1 - l0, l2 - l1, @as(f64, @floatFromInt(use.device)) / (1 << 30) });
    var failures: usize = 0;
    const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };

    // the sequence: the prompt with its last tokens as the answer (FN_ANSWER of them, 8 by default)
    const seq_ids = ids.items;
    const answer: usize = if (std.c.getenv("FN_ANSWER")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8 else 8;
    const start = n - @min(answer, n / 2);

    // (1) packed sequences: each one's loss as alone (a pass's rows are each sequence's own)
    {
        var seqs: [6]train.Seq = undefined;
        var alone: [6]f32 = undefined;
        for (&seqs, 0..) |*s, i| {
            const len = @min(n, 120) - 3 * i; // odd and even lengths, so some sequences sit after a gap row
            s.* = .{ .ids = seq_ids[0..len], .start = len - 6, .decide = len - 7 };
            var one: [1]train.Result = undefined;
            try t.pass(seqs[i .. i + 1], .loss, turn, &one);
            alone[i] = one[0].loss;
        }
        var got: [6]train.Result = undefined;
        const p0 = now(io);
        try t.pass(&seqs, .loss, turn, &got);
        const p1 = now(io);
        var worst: f32 = 0;
        for (got, alone) |g, a| worst = @max(worst, @abs(g.loss - a));
        const ok = worst <= 1e-4;
        std.debug.print("{s} six sequences side by side ({d} rows in {d:.0} ms): each loss as alone within {e} (alone: {d:.4} {d:.4} {d:.4} ...)\n", .{ if (ok) "PASS" else "FAIL", t.rows, (p1 - p0) * 1e3, worst, alone[0], alone[1], alone[2] });
        if (!ok) failures += 1;
    }

    // (2) push against finite differences, at four layers: two rows of the sequence, keys 0 and 1
    const layers = [_]usize{ 12, 23, 31, 42 };
    const rows = [_]usize{ start - 1, start - 5 };
    var sq: train.Seq = .{ .ids = seq_ids, .start = start, .decide = start - 1 };
    var mask: u64 = 0;
    for (layers) |l| mask |= @as(u64, 1) << @intCast(l);
    sq.nudge = .{ .layers = .{ mask, mask, 0 }, .rows = .{ rows[0], rows[1], 0 } };
    for (layers) |l| for (0..2) |k| @memset(t.deltaAt(0, k, l), 0);
    var out: [1]train.Result = undefined;
    const q0 = now(io);
    try t.pass(&.{sq}, .target, turn, &out);
    const q1 = now(io);
    const base = out[0].loss;
    std.debug.print("target pass: {d} rows, loss {d:.5}, back to layer {d} in {d:.0} ms\n", .{ n - 1, base, layers[0], (q1 - q0) * 1e3 });
    const push = try gpa.alloc(f32, layers.len * 2 * dims);
    defer gpa.free(push);
    for (layers, 0..) |l, li| for (0..2) |k| @memcpy(push[(li * 2 + k) * dims ..][0..dims], t.pushAt(0, k, l));
    var prng = std.Random.DefaultPrng.init(7);
    const step_frac: f64 = if (std.c.getenv("FN_STEP")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.004 else 0.004;
    var worst_rel: f64 = 0;
    var lines: usize = 0;
    for (layers, 0..) |l, li| for (0..2) |k| {
        const p = push[(li * 2 + k) * dims ..][0..dims];
        const pn = @sqrt(dot(p, p));
        const rn = norm16(t.residual(l + 1, rows[k]));
        var dir: [dims]f32 = undefined;
        for (0..2) |which| {
            // along the push, then half along it and half along a random direction
            var r: [dims]f32 = undefined;
            var s: f64 = 0;
            for (&r) |*x| {
                x.* = prng.random().floatNorm(f32);
                s += @as(f64, x.*) * x.*;
            }
            for (&dir, p, r) |*x, v, q| x.* = @floatCast(if (which == 0) v / pn else (v / pn + q / @sqrt(s)) / std.math.sqrt2);
            const want = dot(p, &dir);
            const fd = try slope(t, &sq, k, l, &dir, step_frac * @max(rn, 1.0));
            const rel = @abs(fd - want) / @max(@abs(want), 1e-30);
            worst_rel = @max(worst_rel, rel);
            std.debug.print("  layer {d} row {d} ({s}, |residual| {d:.1}): push . u {e:.4}, finite differences' slope {e:.4}, relative error {d:.4}\n", .{ l, rows[k], if (which == 0) "along push" else "half random", rn, want, fd, rel });
            lines += 1;
        }
    };
    const grad_ok = worst_rel < 0.1;
    std.debug.print("{s} push against finite differences: worst relative error {d:.4} ({d} checks)\n", .{ if (grad_ok) "PASS" else "FAIL", worst_rel, lines });
    if (!grad_ok) failures += 1;

    // (3) a pocket neuron at layer 31: its output's gradient (gate act push) against the loss moving with it
    {
        const l: usize = 31;
        const r = rows[0];
        var keys: [1]train.Result = undefined;
        sq.nudge = .{};
        try t.pass(&.{sq}, .keys, turn, &keys);
        // input weights that read 2 on this row and little elsewhere: the row less the sequence's mean row, scaled
        var z: [dims]f64 = @splat(0);
        const rows_n = sq.ids.len - 1;
        for (0..rows_n) |j| {
            if (j == r) continue;
            for (t.normed(l, j), 0..) |b, i| z[i] -= @as(f64, @as(f32, @bitCast(@as(u32, b) << 16))) / @as(f64, @floatFromInt(rows_n - 1));
        }
        const x = t.normed(l, r);
        var zx: f64 = 0;
        for (x, 0..) |b, i| {
            z[i] += @as(f32, @bitCast(@as(u32, b) << 16));
            zx += z[i] * @as(f32, @bitCast(@as(u32, b) << 16));
        }
        var u: [dims]f32 = undefined;
        for (&u, z) |*o, v| o.* = @floatCast(2 * v / zx);
        var most: f64 = -1e30;
        for (0..rows_n) |j| {
            if (j == r) continue;
            var d: f64 = 0;
            for (t.normed(l, j), u) |b, w| d += @as(f64, @as(f32, @bitCast(@as(u32, b) << 16))) * w;
            most = @max(most, d);
        }
        std.debug.print("  pocket neuron: u x = 2 on row {d}, at most {d:.3} on the sequence's other rows\n", .{ r, most });
        const neurons = 64;
        const gate = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(gate);
        const upr = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(upr);
        const down = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(down);
        @memset(gate, 0);
        @memset(upr, 0);
        @memset(down, 0);
        for (0..dims) |i| {
            gate[i] = 16 * u[i];
            upr[i] = u[i] / 16;
        }
        try e.setPocket(l, gate, upr, down, 1);
        sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(l), 0, 0 }, .rows = .{ r, 0, 0 } };
        @memset(t.deltaAt(0, 0, l), 0);
        var got: [1]train.Result = undefined;
        try t.pass(&.{sq}, .target, turn, &got);
        const xr = t.normed(l, r);
        var zg: f64 = 0;
        var zu: f64 = 0;
        for (xr, 0..) |b, i| {
            const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
            zg += gate[i] * v;
            zu += upr[i] * v;
        }
        const act = zg / (1 + @exp(-zg)) * zu;
        const g = t.gateAt(l, r);
        const pg = t.pushAt(0, 0, l);
        // the neuron's output along the push, by 1e-2 of the residual's size, and back
        var pn: f64 = 0;
        for (pg) |v| pn += @as(f64, v) * v;
        pn = @sqrt(pn);
        const resid = t.residual(l + 1, r);
        var rn: f64 = 0;
        for (resid) |b| {
            const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
            rn += v * v;
        }
        rn = @sqrt(rn);
        const step = step_frac * rn / @max(g * act, 1e-6);
        sq.nudge = .{};
        var num: f64 = 0;
        var den: f64 = 0;
        for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
            const xstep = sgn * @as(f64, @floatFromInt(m)) * step;
            for (down[0..dims], pg) |*o, v| o.* = @floatCast(xstep * v / pn);
            try e.setPocket(l, gate, upr, down, 1);
            var rr: [1]train.Result = undefined;
            try t.pass(&.{sq}, .loss, turn, &rr);
            num += xstep * rr[0].loss;
            den += xstep * xstep;
        };
        const fd = num / den;
        {
            // the forward's change at that row after layer l, against g act v (as the learner's verifyForward reads it)
            for (down[0..dims], pg) |*o, v| o.* = @floatCast(rn * v / pn);
            try e.setPocket(l, gate, upr, down, 1);
            var rr: [1]train.Result = undefined;
            try t.pass(&.{sq}, .target, turn, &rr);
            var with: [dims]f32 = undefined;
            for (&with, t.residual(l + 1, r)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
            @memset(down, 0);
            try e.setPocket(l, gate, upr, down, 1);
            try t.pass(&.{sq}, .target, turn, &rr);
            var dd: f64 = 0;
            var dw: f64 = 0;
            for (&with, t.residual(l + 1, r), pg) |w, b, v| {
                const delta = w - @as(f32, @bitCast(@as(u32, b) << 16));
                dd += delta * delta;
                dw += delta * v / pn;
            }
            std.debug.print("  pocket neuron in the forward: the residual moves {d:.3} (along v {d:.3}); g act |v| = {d:.3}\n", .{ @sqrt(dd), dw, g * act * rn });
        }
        const want = g * act * pn;
        const rel = @abs(fd - want) / @max(@abs(want), 1e-30);
        const ok = rel < 0.1;
        std.debug.print("{s} a pocket neuron at layer {d} (gate {d:.3}, activation {d:.3}): gate act |push| {e:.4}, finite difference of its output {e:.4}, relative error {d:.4}\n", .{ if (ok) "PASS" else "FAIL", l, g, act, want, fd, rel });
        if (!ok) failures += 1;
        @memset(down, 0);
        @memset(gate, 0);
        @memset(upr, 0);
        try e.setPocket(l, gate, upr, down, 0);
    }

    // (4) a full pass's time: sixteen sequences, keys then loss then target down to layer 12
    {
        var seqs: [16]train.Seq = undefined;
        var rowsum: usize = 0;
        for (&seqs, 0..) |*s, i| {
            const len = @min(n, 40 + 3 * i);
            rowsum += len - 1;
            s.* = .{ .ids = seq_ids[0..len], .start = len - 6, .decide = len - 7, .nudge = .{ .layers = .{ @as(u64, 1) << 12 | @as(u64, 1) << 30, 0, 0 }, .rows = .{ len - 7, 0, 0 } } };
            @memset(t.deltaAt(i, 0, 12), 0);
            @memset(t.deltaAt(i, 0, 30), 0);
        }
        var got: [16]train.Result = undefined;
        for ([_]train.Mode{ .keys, .loss, .target }) |m| {
            const a = now(io);
            try t.pass(&seqs, m, turn, &got);
            const b = now(io);
            std.debug.print("pass {s}: 16 sequences, {d} rows, {d:.0} ms\n", .{ @tagName(m), rowsum, (b - a) * 1e3 });
        }
    }
    const peak = cuda.usage(false);
    std.debug.print("device bytes held at the end {d:.2} GiB, peak {d:.2} GiB\n", .{ @as(f64, @floatFromInt(peak.device)) / (1 << 30), @as(f64, @floatFromInt(peak.peak)) / (1 << 30) });
    std.debug.print("{d} failures\n", .{failures});
    return if (failures == 0) 0 else 1;
}
