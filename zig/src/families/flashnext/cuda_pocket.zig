//! Living Weights' pockets in Flash Next's CUDA forward: each layer's SwiGLU neurons beside its shared expert.
const std = @import("std");
const cuda = @import("cuda");
const fb = @import("cuda_fn_back.zig");
const wt = @import("cuda_prompt_wt.zig");

const layers_n = wt.layers_n;
const dims = wt.dims;

/// A layer's pocket on the device: fp32 [cap, dims] gate rows, up rows and outputs, the first `used` neurons live.
pub const Rows = struct { gate: cuda.DeviceBuffer, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, cap: usize, used: usize = 0 };

/// Every layer's pocket (null: none) and the kernels that run them, loaded with the first.
pub const Pockets = struct {
    layers: [layers_n]?Rows = @splat(null),
    k: ?fb.Kernels = null,

    pub fn deinit(p: *Pockets) void {
        for (&p.layers) |*slot| if (slot.*) |*pk| {
            pk.gate.free();
            pk.up.free();
            pk.down.free();
            slot.* = null;
        };
        if (p.k) |*k| k.deinit();
        p.k = null;
    }

    /// Layer `layer`'s neurons: gate, up and output rows fp32 [n, dims], the first `used` live; made on first use.
    pub fn set(p: *Pockets, d: *cuda.Driver, s: *cuda.Stream, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        const n = gate.len / dims;
        if (n == 0 or n > fb.max_pocket or up.len != gate.len or down.len != gate.len or used > n) return error.UnexpectedTensor;
        if (p.k == null) p.k = try fb.Kernels.load(d);
        if (p.layers[layer]) |pk| if (pk.cap != n) return error.PocketOfAnotherSize;
        if (p.layers[layer] == null) {
            var g = try cuda.DeviceBuffer.alloc(d, n * dims * 4);
            errdefer g.free();
            var u = try cuda.DeviceBuffer.alloc(d, n * dims * 4);
            errdefer u.free();
            const o = try cuda.DeviceBuffer.alloc(d, n * dims * 4);
            p.layers[layer] = .{ .gate = g, .up = u, .down = o, .cap = n };
        }
        const pk = &p.layers[layer].?;
        // stream-ordered after every queued pass that reads the old rows
        try pk.gate.uploadAsync(0, std.mem.sliceAsBytes(gate), s.handle);
        try pk.up.uploadAsync(0, std.mem.sliceAsBytes(up), s.handle);
        try pk.down.uploadAsync(0, std.mem.sliceAsBytes(down), s.handle);
        try s.synchronize();
        pk.used = used;
    }

    /// Layer `layer`'s pocket into gate, up and down ([n, dims] each); the neurons in use (0: none).
    pub fn read(p: *const Pockets, s: *cuda.Stream, layer: usize, gate: []f32, up: []f32, down: []f32) usize {
        const pk = p.layers[layer] orelse return 0;
        if (gate.len != pk.cap * dims or up.len != gate.len or down.len != gate.len) return 0;
        s.synchronize() catch return 0;
        pk.gate.download(0, std.mem.sliceAsBytes(gate)) catch return 0;
        pk.up.download(0, std.mem.sliceAsBytes(up)) catch return 0;
        pk.down.download(0, std.mem.sliceAsBytes(down)) catch return 0;
        return pk.used;
    }

    /// Layer `layer`'s pocket as the kernels take it (activations in `act`), or null with no neuron in use.
    pub fn of(p: *const Pockets, layer: usize, act: u64) ?fb.Ops.Pocket {
        const pk = p.layers[layer] orelse return null;
        if (pk.used == 0) return null;
        return .{ .gate = pk.gate.ptr, .up = pk.up.ptr, .down = pk.down.ptr, .n = pk.used, .act = act };
    }

    /// The layers with a neuron in use, a bit each: a pass adds a pocket there alone.
    pub fn mask(p: *const Pockets) u64 {
        var m: u64 = 0;
        for (p.layers, 0..) |slot, l| if (slot) |pk| if (pk.used > 0) {
            m |= @as(u64, 1) << @intCast(l);
        };
        return m;
    }
};

/// The neurons in use of a pocket read from the shards: the first ones, up to the last with a nonzero input row.
pub fn usedOf(up: []const f32, n: usize) usize {
    var used: usize = 0;
    for (0..n) |j| {
        if (std.mem.indexOfNone(f32, up[j * dims ..][0..dims], &.{0}) != null) used = j + 1;
    }
    return used;
}

test "a pocket's neurons in use end at its last nonzero input row" {
    var up: [3 * dims]f32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 0), usedOf(&up, 3));
    up[dims + 7] = 0.5;
    try std.testing.expectEqual(@as(usize, 2), usedOf(&up, 3));
    var p: Pockets = .{};
    try std.testing.expectEqual(@as(u64, 0), p.mask());
    try std.testing.expect(p.of(5, 0) == null);
}
