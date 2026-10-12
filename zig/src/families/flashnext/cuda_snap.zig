//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` prompt tokens, copied on the
//! device into one buffer (the 36 DeltaNet conv windows and recurrent states, the 12 attention layers' first `at` key,
//! value and indexer-key rows with int8 KV's scales, the pooled indexer blocks those rows complete, the n-gram tail)
//! with the n-gram history and the position beside it. A stream restored
//! from it runs the rest of its prompt as if it had run the first `at` tokens itself: the batched pass leaves the same
//! bits however a prompt is cut into chunks.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");

const conv_bytes = (gdn.taps - 1) * gdn.conv_dim * 2;
const state_bytes = wt.state_n * 4;
const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;

/// One position's keys (or values) in one layer: bf16 (kv_bits 0) or int8 codes; and int8's fp16 scales.
fn kvCodes(kv_bits: usize) usize {
    return wt.kv_heads * wt.head_dim * @as(usize, if (kv_bits == 8) 1 else 2);
}
fn kvScales(kv_bits: usize) usize {
    return if (kv_bits == 8) wt.kv_heads * (wt.head_dim / 32) * 2 else 0;
}

/// Device bytes a state after `at` tokens takes: the DeltaNet's are fixed, the attention caches grow by the token
/// (and the pooled indexer keys by the complete block of 4).
pub fn bytes(at: usize, kv_bits: usize) usize {
    return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * (at * (2 * (kvCodes(kv_bits) + kvScales(kv_bits)) + index_row) + at / wt.index_ratio * index_row) + tail_bytes;
}

/// One kept state.
pub const Snap = struct {
    buf: cuda.DeviceBuffer,
    at: usize,
    history: [2]i64,
    kv_bits: usize,
};

const Way = enum { save, restore };

/// Each piece of the sequence to (save) or from (restore) its place in the buffer at `base`, on `stream`.
fn move(stream: cuda.Stream, seq: *prompt.Seq, base: u64, at: usize, way: Way) !void {
    const d = stream.d;
    var off: u64 = 0;
    const Piece = struct { bufs: []cuda.DeviceBuffer, n: usize };
    const bits = seq.kv_bits;
    // a block's pooled keys are made when its last row runs: the complete blocks' are the state's, the rest are
    // pooled again as the rows after `at` complete them
    const pieces = [_]Piece{
        .{ .bufs = seq.conv, .n = conv_bytes },
        .{ .bufs = seq.state, .n = state_bytes },
        .{ .bufs = seq.kc, .n = at * kvCodes(bits) },
        .{ .bufs = seq.vc, .n = at * kvCodes(bits) },
        .{ .bufs = seq.ks, .n = at * kvScales(bits) },
        .{ .bufs = seq.vs, .n = at * kvScales(bits) },
        .{ .bufs = seq.ikc, .n = at * index_row },
        .{ .bufs = seq.pooled, .n = at / wt.index_ratio * index_row },
        .{ .bufs = (&seq.tail)[0..1], .n = tail_bytes },
    };
    for (pieces) |p| for (p.bufs) |b| {
        if (p.n == 0) continue;
        const dst, const src = if (way == .save) .{ base + off, b.ptr } else .{ b.ptr, base + off };
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
        off += p.n;
    };
    std.debug.assert(off == bytes(at, bits));
}

/// A copy of `seq` as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq) !*Snap {
    const at = seq.fed;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytes(at, seq.kv_bits)), .at = at, .history = seq.history, .kv_bits = seq.kv_bits };
    errdefer snap.buf.free();
    try move(stream, seq, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

/// `seq` becomes `snap`'s state: its next token is the one after `snap.at`. Refused before any copy when the
/// sequence cannot hold that many tokens.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (snap.kv_bits != seq.kv_bits) return error.OtherKvForm;
    try move(stream, seq, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    stream.synchronize() catch {};
    snap.buf.free();
    gpa.destroy(snap);
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, 0) - bytes(0, 0));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, 0) - bytes(0, 0));
    // int8: codes and a group of 32's fp16 scale
    try std.testing.expectEqual(@as(usize, 12 * (2 * (2 * 256 + 2 * 8 * 2) + 128 * 2)), bytes(1, 8) - bytes(0, 8));
    try std.testing.expect(bytes(0, 0) > 100 << 20 and bytes(0, 0) < 120 << 20);
}
