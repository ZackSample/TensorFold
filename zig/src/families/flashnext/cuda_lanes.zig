//! Flash Next on the native CUDA server: each stream's prompt on its own caches, then shared decode rounds, every
//! live stream's next token in one batched pass (cuda_batch.zig's lanes), and the prompt cache's kept states
//! (cuda_snap.zig) so a later request with the same prefix resumes there. No draft head.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const snap = @import("cuda_snap.zig");

const be = lanes.backend;
const ring = 1024;
/// Streams a round takes at most (a round with more live streams takes them in turns).
const round_streams = 32;

/// `Tri` is the captured-kernel launcher. `open` loads the checkpoint and serves greedy or keyed draws.
pub fn Namespace(comptime Tri: type) type {
    return struct {
        pub const model_type = "qwen4_exp";
        pub const formats: []const []const u8 = &.{"mlx-q4g32"};
        // dense attention over the captured scratch (NCH chunks of 512 keys, QSA off): no sparse indexer yet
        pub const default_context: i64 = @intCast(prompt.max_tokens);
        // a prompt runs as one batched pass (cuda_batch.zig), the whole prompt at once: no staggered segments
        pub const max_segments: u32 = 1;
        pub const prompt_rows: u32 = @intCast(prompt.max_tokens);

        /// The prompt cache's copies of a stream's state (cuda_snap.zig), called with `Loaded.target`.
        pub const snap_bytes = snapBytes;
        pub const snap_save = snapSave;
        pub const snap_restore = snapRestore;
        pub const snap_drop = snapDrop;
        /// The batched pass leaves the same bits however a prompt is cut, so states keep and resume at any token
        /// (not planned on a grid), and nothing reads past a state's position (no draft head: no lookahead). The
        /// window is short (dense attention), so prompts from 512 tokens keep states, 256 apart.
        pub const cache_rules = struct {
            pub const lookahead: u32 = 0;
            pub const planned = false;
            pub const min_gap: u32 = 256;
            pub const min_prompt: u32 = 512;
        };

        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

        /// One stream's prompt-and-reply driver, as native/cuda.zig types it; this family has none.
        pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

        pub const Loaded = struct {
            backend: be.Backend,
            facts: lanes.Model,
            rows: u32,
            /// Device bytes each admitted stream allocates: its own caches at the server's window. The passes'
            /// scratch is shared and allocated at load (counted with the model).
            stream_bytes: usize,
            /// Every stream allocates its own sequence; none rides on the engine's.
            free_streams: u32 = 0,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            lone: ?LoneRun = null,
            learn_bytes: usize = 0,
            /// What the prompt cache's copies are taken from: the backend itself (snap_* look up the owner stream).
            target: *anyopaque,
        };

        /// What the device holds of the checkpoint, for admission: the n-gram tables stay in the mapped shards.
        /// 0 when the headers do not read, and the load then names what is wrong.
        pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
            return weights.deviceBytes(gpa, io, dir) catch 0;
        }

        /// A request or load this engine refuses, in words; null: none of its own.
        pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return switch (err) {
                error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
                error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
                else => null,
            };
        }

        const Lane = struct { seq: prompt.Seq };

        const Cuda = struct {
            gpa: std.mem.Allocator,
            context: usize,
            engine: *prompt.Engine,
            set: cuda.aot.Set,
            stream: cuda.Stream,
            lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
            /// A round's scores, every row [round_streams, vocab] bf16.
            logits: cuda.DeviceBuffer,
            drawn: [ring]u32 = undefined,
            next: u64 = 0,

            fn tri(self: *Cuda) Tri {
                return .{ .set = &self.set, .s = self.stream };
            }

            fn backend(self: *Cuda) be.Backend {
                return .{ .ptr = self, .vtable = &.{
                    .prefill = prefillFn,
                    .first = firstFn,
                    .queue = queueFn,
                    .read = readFn,
                    .verify = verifyFn,
                    .keep = keepFn,
                    .draft = draftFn,
                    .release = releaseFn,
                } };
            }

            fn facts() lanes.Model {
                return .{
                    // two rows reproduce one-row steps (any chunk does), and the round loop shares rounds only for
                    // a width of two or more; a window stays one row, since prefill turns each stream's drafts off
                    .exact_width = 2,
                    .gpu_tokens = false,
                    .mtp = false,
                    .speculate = false,
                    .drafts = 0,
                    .streams_exact = true,
                    .hidden_rows = true,
                    .batch_rows = round_streams,
                    .max_streams = round_streams,
                };
            }

            fn take(self: *Cuda, token: u32) u64 {
                const h = self.next;
                self.drawn[h % ring] = token;
                self.next += 1;
                return h;
            }

            fn tokenAt(self: *Cuda, seq: *prompt.Seq, s: *lanes.Stream, position: u64) !u32 {
                if (s.sampling) |sm| {
                    const n = prompt.vocabulary;
                    const vals = try self.gpa.alloc(f64, n);
                    defer self.gpa.free(vals);
                    const ids = try self.gpa.alloc(u64, n);
                    defer self.gpa.free(ids);
                    for (vals, ids, 0..) |*v, *id, i| {
                        const bits = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
                        v.* = @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
                        id.* = @intCast(i);
                    }
                    return @intCast(try lanes.sampling.choose(self.gpa, vals, ids, position, sm));
                }
                return prompt.argmax(seq.logits);
            }

            fn lane(self: *Cuda, s: *lanes.Stream) !*Lane {
                return self.lanes.getPtr(s) orelse return error.UnknownStream;
            }

            /// The prompt from the kept state the cache named (if it restores), cut at each mark where the cache
            /// keeps the state; the last chunk leaves the first token's scores.
            fn fill(self: *Cuda, s: *lanes.Stream, seq: *prompt.Seq, ids: []const u32) !void {
                s.cached = 0;
                var at: usize = 0;
                if (s.reuse.saved) |saved| {
                    const kept: *snap.Snap = @ptrCast(@alignCast(saved));
                    if (kept.at != s.reuse.at or kept.at == 0 or kept.at >= ids.len) {
                        s.reuse_failed = true;
                    } else if (snap.restore(self.stream, seq, kept)) |_| {
                        at = kept.at;
                        s.cached = @intCast(at);
                    } else |err| switch (err) {
                        error.SequenceTooShort => s.reuse_failed = true,
                        else => return err,
                    }
                }
                for (s.reuse.marks) |m| {
                    if (m <= at or m >= ids.len) continue;
                    if (s.isCancelled()) return error.Cancelled;
                    try self.engine.forward(Tri, self.tri(), seq, ids[at..m], .{ .logits = .none });
                    at = m;
                    if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(m));
                }
                if (s.isCancelled()) return error.Cancelled;
                try self.engine.predict(Tri, self.tri(), seq, ids[at..]);
            }

            /// Windows wider than one row (none while drafts are off): each stream's rows one at a time.
            fn verifyRows(self: *Cuda, windows: []const be.Window, out: []be.Verified) !void {
                for (windows, out) |w, *o| {
                    const l = try self.lane(w.stream);
                    const rows = w.rows();
                    var input = w.pending;
                    for (0..rows) |r| {
                        if (w.positions[r] != l.seq.fed + 1) return error.PositionMismatch;
                        var one = [_]u32{input};
                        try self.engine.predict(Tri, self.tri(), &l.seq, &one);
                        o.sampled[r] = try self.tokenAt(&l.seq, w.stream, w.positions[r]);
                        if (r + 1 < rows) input = w.tokens[r];
                    }
                    if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
                }
            }
        };

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
            _ = o.drafts; // this model is served one token at a time; there is no draft head
            if (o.context > prompt.max_tokens) return error.ContextOverDenseAttention;
            if (o.segments != 1 or o.slide) return error.UnsupportedOption;
            // every launch is a captured Triton cubin: TENSORFOLD_CUDA_KERNELS names the set
            const kernels = captured orelse return error.CapturedKernelsRequired;
            const driver: *cuda.Driver = @constCast(ctx.d);
            const own = try gpa.create(Cuda);
            errdefer gpa.destroy(own);
            own.gpa = gpa;
            own.context = o.context;
            own.stream = try cuda.Stream.init(driver, true);
            errdefer own.stream.deinit();
            own.set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, kernels);
            errdefer own.set.deinit();
            own.engine = try prompt.Engine.init(gpa, io, driver, &own.stream, dir);
            errdefer own.engine.deinit();
            // the passes' scratch at the widest prompt chunk the window allows, shared by every stream's prompt and
            // every round: allocated now, so admission counts it once with the model
            _ = try own.engine.batchScratch(@min(o.context, own.engine.chunk));
            own.logits = try cuda.DeviceBuffer.alloc(driver, round_streams * prompt.vocabulary * 2);
            errdefer own.logits.free();
            own.lanes = .empty;
            own.next = 0;
            return .{
                .backend = own.backend(),
                .facts = Cuda.facts(),
                .rows = 2,
                .stream_bytes = prompt.seqBytes(o.context),
                .ctx = own,
                .deinit = release,
                .target = own,
            };
        }

        fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            var it = own.lanes.valueIterator();
            while (it.next()) |l| l.seq.deinit(own.gpa);
            own.lanes.deinit(own.gpa);
            own.logits.free();
            own.engine.deinit();
            own.set.deinit();
            own.stream.deinit();
            own.gpa.destroy(own);
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const ids = s.prompt();
            if (ids.len == 0 or ids.len + s.max_new > self.context) return error.PromptTooLong;
            // one row a stream a round: a copied draft the stream rejected would need its DeltaNet states rewound
            s.drafts = false;
            s.proposer = null;
            const gop = try self.lanes.getOrPut(self.gpa, s);
            if (gop.found_existing) gop.value_ptr.seq.deinit(self.gpa);
            gop.value_ptr.* = .{ .seq = self.engine.newSeq(ids.len + s.max_new) catch |err| {
                self.lanes.removeByPtr(gop.key_ptr);
                return err;
            } };
            self.fill(s, &gop.value_ptr.seq, ids) catch |err| {
                releaseFn(ptr, s);
                return err;
            };
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            const l = try self.lane(s);
            if (position != l.seq.fed) return error.PositionMismatch;
            return self.take(try self.tokenAt(&l.seq, s, position));
        }

        fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            _ = ptr;
            _ = s;
            _ = feed;
            _ = position;
            return error.NotPipelined;
        }

        fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const self = of(ptr);
            if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
            return self.drawn[handle % ring];
        }

        /// Every window's pending token in shared rounds of up to round_streams rows, each stream's scores the
        /// bits it would get alone; then each row drawn with its stream's sampling at its position.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            const seqs = try self.gpa.alloc(*prompt.Seq, windows.len);
            defer self.gpa.free(seqs);
            const ids = try self.gpa.alloc(u32, windows.len);
            defer self.gpa.free(ids);
            for (windows, seqs, ids) |w, *q, *id| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                if (w.rows() != 1) return self.verifyRows(windows, out);
                const l = try self.lane(w.stream);
                if (w.positions[0] != l.seq.fed + 1) return error.PositionMismatch;
                q.* = &l.seq;
                id.* = w.pending;
            }
            var at: usize = 0;
            while (at < windows.len) : (at += round_streams) {
                const n = @min(round_streams, windows.len - at);
                try self.engine.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr);
            }
            for (windows, out, seqs) |w, *o, q| o.sampled[0] = try self.tokenAt(q, w.stream, w.positions[0]);
        }

        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            _ = ptr;
            for (windows, paths) |w, path| {
                if (path.len != w.rows()) return error.CannotRewind;
                for (path, 0..) |r, i| if (r != i) return error.CannotRewind;
            }
        }

        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            _ = ptr;
            _ = requests;
            return error.NoDraftHead;
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.lanes.fetchRemove(s)) |kv| {
                var l = kv.value;
                l.seq.deinit(self.gpa);
            }
        }

        // -- the prompt cache's copies ------------------------------------------------------------------------

        fn snapBytes(ptr: *anyopaque, at: u32) u64 {
            _ = ptr;
            return snap.bytes(at);
        }

        /// The owner stream's state, its prompt pass standing at `at`.
        fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            if (l.seq.fed != at) return error.NotAtMark;
            return try snap.save(self.gpa, self.engine.driver, self.stream, &l.seq);
        }

        fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
            const self = of(ptr);
            const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
            const l = try self.lane(s);
            try snap.restore(self.stream, &l.seq, @ptrCast(@alignCast(saved)));
        }

        fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            snap.drop(self.gpa, self.stream, @ptrCast(@alignCast(saved)));
        }
    };
}
