//! Flash Next on the native CUDA server: one stream's prompt, then verify windows of the MTP head's drafts (or one
//! token at a time with --no-drafts), every row drawn greedily or keyed at its position.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");

const be = lanes.backend;
const ring = 1024;

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

        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

        /// One stream's prompt-and-reply driver, as native/cuda.zig types it; this family has none: a lone drafted
        /// stream runs in the lane core, whose copied continuations beside the head's chains outpace mtp_decode.
        pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

        pub const Loaded = struct {
            backend: be.Backend,
            facts: lanes.Model,
            rows: u32,
            /// Device bytes each admitted stream allocates: its own caches at the server's window.
            stream_bytes: usize,
            /// Every stream allocates its own sequence; none rides on the engine's.
            free_streams: u32 = 0,
            ctx: *anyopaque,
            deinit: *const fn (*anyopaque) void,
            lone: ?LoneRun = null,
            learn_bytes: usize = 0,
        };

        /// What the device holds of the checkpoint, for admission: the n-gram tables stay in the mapped shards, the
        /// MTP head (counted here, loaded unless --no-drafts) on the device.
        /// 0 when the headers do not read, and the load then names what is wrong.
        pub fn deviceWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
            const main = weights.deviceBytes(gpa, io, dir) catch return 0;
            return main + mtp.headBytes();
        }

        /// A request or load this engine refuses, in words; null: none of its own.
        pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return switch (err) {
                error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
                error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
                else => null,
            };
        }

        /// A stream's caches; with the head, its cache, a window's buffers and the drafts it holds for the next round.
        const Lane = struct {
            seq: prompt.Seq,
            draft: ?mtp.Draft = null,
            chain: mtp.Chain = .{},
            prompt_len: usize = 0,
        };

        const Ops = mtp.Ops(Tri);

        const Cuda = struct {
            gpa: std.mem.Allocator,
            context: usize,
            engine: *prompt.Engine,
            head: ?*mtp.Head = null,
            depth: usize = mtp.max_chain,
            confidence: f64 = mtp.default_confidence,
            set: cuda.aot.Set,
            stream: cuda.Stream,
            lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
            drawn: [ring]u32 = undefined,
            next: u64 = 0,

            fn tri(self: *Cuda) Tri {
                return .{ .set = &self.set, .s = self.stream };
            }

            fn ops(self: *Cuda) Ops {
                return .{ .e = self.engine, .tri = self.tri(), .h = self.head.? };
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
                    .tree = treeFn,
                    .probabilities = probabilitiesFn,
                } };
            }

            /// With the head: windows up to 16 rows (a chain of 6, or a copied continuation), drafts asked to the
            /// most and cut by the head's confidence (draft_probabilities: the round loop asks the widest chain).
            fn facts(self: *const Cuda) lanes.Model {
                const drafting = self.head != null;
                return .{
                    .exact_width = if (drafting) mtp.max_rows else 1,
                    .gpu_tokens = false,
                    .mtp = drafting,
                    .speculate = drafting,
                    .speculate_early = false,
                    .drafts = if (drafting) @intCast(self.depth) else 0,
                    .draft_probabilities = drafting,
                    .batch_rows = if (drafting) mtp.max_rows else 1,
                    .max_streams = 1,
                };
            }

            fn take(self: *Cuda, token: u32) u64 {
                const h = self.next;
                self.drawn[h % ring] = token;
                self.next += 1;
                return h;
            }

            fn value(self: *const Cuda, feed: be.Feed) u32 {
                return switch (feed) {
                    .handle => |h| self.drawn[h % ring],
                    .value => |v| v,
                };
            }

            /// The token after the sequence's last row (seq.logits) at `position`: the draws serial decoding makes.
            fn tokenAt(self: *Cuda, seq: *prompt.Seq, s: *lanes.Stream, position: u64) !u32 {
                if (s.sampling == null) return prompt.argmax(seq.logits);
                const row = try self.gpa.alloc(u16, prompt.vocabulary);
                defer self.gpa.free(row);
                for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
                return mtp.drawRow(self.gpa, row, null, position, s.sampling);
            }

            fn lane(self: *Cuda, s: *lanes.Stream) !*Lane {
                return self.lanes.getPtr(s) orelse return error.UnknownStream;
            }
        };

        fn envUsize(name: [:0]const u8, default: usize) usize {
            const v = std.c.getenv(name) orelse return default;
            return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
        }

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
            if (o.context > prompt.max_tokens) return error.ContextOverDenseAttention;
            if (o.segments != 1 or o.slide) return error.UnsupportedOption;
            // every launch is a captured Triton cubin: TENSORFOLD_CUDA_KERNELS names the set
            const kernels = captured orelse return error.CapturedKernelsRequired;
            const driver: *cuda.Driver = @constCast(ctx.d);
            const own = try gpa.create(Cuda);
            errdefer gpa.destroy(own);
            own.* = .{ .gpa = gpa, .context = o.context, .engine = undefined, .set = undefined, .stream = undefined };
            own.stream = try cuda.Stream.init(driver, true);
            errdefer own.stream.deinit();
            own.set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, kernels);
            errdefer own.set.deinit();
            own.engine = try prompt.Engine.init(gpa, io, driver, &own.stream, dir);
            errdefer own.engine.deinit();
            // the batched pass's scratch for a prompt filling the window, once for the engine (every stream's
            // prompt, window and head pass runs on it): allocated now, inside the load's reserve
            _ = try own.engine.reserve(@min(o.context, prompt.batch.default_chunk));
            // the MTP head unless --no-drafts (TF_FLASHNEXT_DRAFTS and _CONFIDENCE: the chain's rule, 6 and 0.60)
            if (o.drafts) {
                own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);
                own.depth = @max(1, @min(mtp.max_chain, envUsize("TF_FLASHNEXT_DRAFTS", mtp.max_chain)));
                if (std.c.getenv("TF_FLASHNEXT_CONFIDENCE")) |v| own.confidence = std.fmt.parseFloat(f64, std.mem.span(v)) catch mtp.default_confidence;
            }
            errdefer if (own.head) |h| h.deinit();
            const window = @min(o.context + mtp.max_rows, prompt.max_tokens);
            var stream_bytes = prompt.seqBytes(window);
            if (own.head) |h| stream_bytes += mtp.Draft.bytes(window, h.ids.len);
            return .{
                .backend = own.backend(),
                .facts = own.facts(),
                .rows = if (own.head != null) mtp.max_rows else 1,
                // the stream's caches (and with the head, its cache and a window's buffers)
                .stream_bytes = stream_bytes,
                .ctx = own,
                .deinit = release,
            };
        }

        fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            var it = own.lanes.valueIterator();
            while (it.next()) |l| dropLane(own, l);
            own.lanes.deinit(own.gpa);
            if (own.head) |h| h.deinit();
            own.engine.deinit();
            own.set.deinit();
            own.stream.deinit();
            own.gpa.destroy(own);
        }

        fn dropLane(self: *Cuda, l: *Lane) void {
            if (l.draft) |*d| d.deinit();
            l.seq.deinit(self.gpa);
        }

        fn of(ptr: *anyopaque) *Cuda {
            return @ptrCast(@alignCast(ptr));
        }

        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const ids = s.prompt();
            if (ids.len == 0 or ids.len + s.max_new > self.context) return error.PromptTooLong;
            const gop = try self.lanes.getOrPut(self.gpa, s);
            if (gop.found_existing) dropLane(self, gop.value_ptr);
            const drafting = self.head != null and s.drafts;
            // a window's rows past the reply's last token (its drafts) still need cache slots
            const cap = @min(ids.len + s.max_new + (if (drafting) mtp.max_rows else 0), prompt.max_tokens);
            gop.value_ptr.* = .{ .seq = self.engine.newSeq(cap) catch |err| {
                self.lanes.removeByPtr(gop.key_ptr);
                return err;
            }, .prompt_len = ids.len };
            const l = gop.value_ptr;
            errdefer {
                dropLane(self, l);
                self.lanes.removeByPtr(gop.key_ptr);
            }
            if (drafting) {
                l.draft = try mtp.Draft.init(self.gpa, self.engine.driver, cap, self.head.?.ids.len);
                try self.ops().prefill(&l.seq, &l.draft.?, ids);
            } else try self.engine.predict(Tri, self.tri(), &l.seq, ids);
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

        /// A drafting stream's window in one batched pass, each row drawn at its keyed position; a stream without the
        /// head feeds its one row.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            for (windows, out) |w, *o| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                const l = try self.lane(w.stream);
                const rows = w.rows();
                if (l.draft) |*d| {
                    if (rows > mtp.max_rows) return error.WindowTooWide;
                    for (w.positions, 0..) |p, r| if (p != l.seq.fed + 1 + r) return error.PositionMismatch;
                    var ids: [mtp.max_rows]u32 = undefined;
                    ids[0] = w.pending;
                    @memcpy(ids[1..rows], w.tokens);
                    try self.ops().verify(&l.seq, d, ids[0..rows]);
                    try self.ops().draws(d, w.stream.sampling, o.sampled[0..rows]);
                } else {
                    if (rows != 1) return error.CannotRewind;
                    if (w.positions[0] != l.seq.fed + 1) return error.PositionMismatch;
                    var one = [_]u32{w.pending};
                    try self.engine.predict(Tri, self.tri(), &l.seq, &one);
                    o.sampled[0] = try self.tokenAt(&l.seq, w.stream, w.positions[0]);
                }
                if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            }
        }

        /// Each window keeps its first rows (a chain's prefix): the DeltaNet states replayed over them alone.
        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            for (windows, paths) |w, path| {
                if (path.len == 0 or path.len > w.rows()) return error.CannotRewind;
                for (path, 0..) |r, i| if (r != i) return error.CannotRewind;
                const l = try self.lane(w.stream);
                if (l.draft) |*d| try self.ops().rollback(&l.seq, d, path.len) else if (path.len != w.rows()) return error.CannotRewind;
            }
        }

        /// The head absorbs the kept rows (or the prompt's last row with the first token) and chains the next
        /// window's drafts: up to the asked depth, cut by its confidence (Python decode.draft).
        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            if (self.head == null) return error.NoDraftHead;
            for (requests) |r| {
                if (r.lanes != null) return error.TreesNotBuilt;
                const l = try self.lane(r.stream);
                const d = if (l.draft) |*x| x else return error.NoDraftHead;
                d.settle(&l.seq); // a window the round kept whole (keep is called only for a cut one)
                if (r.position != l.seq.fed + 1) return error.PositionMismatch;
                // the window's rows past the cache's room cannot run: fewer drafts there
                const room = l.seq.cap - l.seq.fed - 1;
                const count = @min(@min(r.depth, self.depth), room);
                if (r.rows) |path| {
                    for (path, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                    if (r.follow.len < path.len) return error.PositionMismatch;
                    try self.ops().draft(d, r.follow[0..path.len], d.streams.ptr, r.position, count, r.stream.sampling, self.confidence, &l.chain);
                } else {
                    const first = self.value(r.first orelse return error.NoFirstToken);
                    try self.ops().draft(d, &.{first}, d.streams.ptr + (l.prompt_len - 1) * weights.wide * 2, r.position, count, r.stream.sampling, self.confidence, &l.chain);
                }
            }
        }

        /// The held chain as host tokens: the round loop verifies them as it would copied ones.
        fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.draft == null) return null;
            return .{ .count = @intCast(l.chain.n), .tokens = try gpa.dupe(u32, l.chain.slice()) };
        }

        /// Each held draft's temperature-1 probability under the head.
        fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
            const self = of(ptr);
            const l = try self.lane(s);
            if (l.draft == null or out.len > l.chain.n) return false;
            @memcpy(out, l.chain.probs[0..out.len]);
            return true;
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            if (self.lanes.fetchRemove(s)) |kv| {
                var l = kv.value;
                self.stream.synchronize() catch {};
                dropLane(self, &l);
            }
        }
    };
}
