//! Flash Next on the native CUDA server: each stream's prompt on its own caches, then verify windows of the MTP
//! head's drafts (a drafting stream: one batched pass a window) or shared decode rounds (every live stream without
//! drafts takes its next token in one batched pass, cuda_batch.zig's lanes), every row drawn greedily or keyed at
//! its position; and the prompt cache's kept states (cuda_snap.zig), so a later request without drafts and with the
//! same prefix resumes there.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const weights = @import("cuda_prompt_wt.zig");
const slide_mod = @import("cuda_slide.zig");
const train = @import("cuda_train.zig");
const living = @import("living");
const mtp = @import("cuda_mtp.zig");
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
        // the model's window: past 2,048 keys the block indexer picks each row's 512 best blocks of 4 (cuda_batch.zig)
        pub const default_context: i64 = @intCast(prompt.max_tokens);
        // a prompt runs as one batched pass (cuda_batch.zig) in chunks of this many rows: no staggered segments
        pub const max_segments: u32 = 1;
        pub const prompt_rows: u32 = @intCast(prompt.batch.default_chunk);

        /// The prompt cache's copies of a stream's state (cuda_snap.zig), called with `Loaded.target`.
        pub const snap_bytes = snapBytes;
        pub const snap_save = snapSave;
        pub const snap_restore = snapRestore;
        pub const snap_drop = snapDrop;
        /// The batched pass leaves the same bits however a prompt is cut, so states keep and resume at any token
        /// (not planned on a grid), and nothing reads past a state's position (no lookahead: a drafting stream, whose
        /// head reads every prompt row's streams, runs its whole prompt and keeps nothing). Prompts from 512 tokens
        /// keep states, 256 apart.
        pub const cache_rules = struct {
            pub const lookahead: u32 = 0;
            pub const planned = false;
            pub const min_gap: u32 = 256;
            pub const min_prompt: u32 = 512;
        };

        pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

        /// Living Weights' learner (--slide): lessons put neurons in each layer's pocket beside its shared expert,
        /// live in this engine's forward and, once kept, in the folder's widened shards.
        pub const slide = slide_mod.Slide(Tri);

        /// The first file a lesson would rewrite through a link (config.json, the index, a shard holding a shared
        /// expert), or null: a teaching copy may hard-link every other shard.
        pub const linked = slide_mod.linked;

        /// What a learner over the loaded engine `p` learns into; `dir` must outlive it.
        pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
            const own: *Cuda = @ptrCast(@alignCast(p));
            return .{ .e = own.engine, .tri = own.tri(), .dir = dir, .lay = own.lay };
        }

        /// One stream's prompt-and-reply driver, as native/cuda.zig types it; this family has none: a lone drafted
        /// stream runs in the lane core, whose copied continuations beside the head's chains outpace mtp_decode.
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
            /// --slide: the engine its learner trains (the family's own context), and what its trainer allocates at
            /// the first lesson, kept out of the streams' room.
            learns: ?*anyopaque = null,
            learn_bytes: usize = 0,
            /// What the prompt cache's copies are taken from: the backend itself (snap_* look up the owner stream).
            target: *anyopaque,
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
            /// A round's scores, every row [round_streams, vocab] bf16.
            logits: cuda.DeviceBuffer,
            drawn: [ring]u32 = undefined,
            next: u64 = 0,
            lay: living.Layout = undefined, // --slide: the pocket's layout, from config.json at load

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
            /// Without it two rows reproduce one-row steps (any chunk does) and the round loop shares rounds only for
            /// a width of two or more; a window stays one row, since prefill turns such a stream's copied drafts off.
            /// Several streams' windows come in one verify: a drafted window runs as its own batched pass, the
            /// streams without drafts share rounds.
            fn facts(self: *const Cuda) lanes.Model {
                const drafting = self.head != null;
                return .{
                    .exact_width = if (drafting) mtp.max_rows else 2,
                    .gpu_tokens = false,
                    .mtp = drafting,
                    .speculate = drafting,
                    .speculate_early = false,
                    .drafts = if (drafting) @intCast(self.depth) else 0,
                    .draft_probabilities = drafting,
                    .streams_exact = true,
                    .hidden_rows = true,
                    .batch_rows = if (drafting) 4 * mtp.max_rows else round_streams,
                    .max_streams = round_streams,
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
                        error.SequenceTooShort, error.OtherKvForm => s.reuse_failed = true,
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
        };

        fn envUsize(name: [:0]const u8, default: usize) usize {
            const v = std.c.getenv(name) orelse return default;
            return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
        }

        pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, captured: ?[]const u8, o: Options) !Loaded {
            if (o.context > prompt.max_tokens) return error.ContextOverWindow;
            if (o.segments != 1) return error.UnsupportedOption;
            // every launch is a captured Triton cubin: TENSORFOLD_CUDA_KERNELS names the set
            const kernels = captured orelse return error.CapturedKernelsRequired;
            const driver: *cuda.Driver = @constCast(ctx.d);
            const own = try gpa.create(Cuda);
            errdefer gpa.destroy(own);
            own.* = .{ .gpa = gpa, .context = o.context, .engine = undefined, .set = undefined, .stream = undefined, .logits = undefined };
            own.stream = try cuda.Stream.init(driver, true);
            errdefer own.stream.deinit();
            own.set = try cuda.aot.Set.load(gpa, io, driver, ctx.device, kernels);
            errdefer own.set.deinit();
            own.engine = try prompt.Engine.init(gpa, io, driver, &own.stream, dir);
            errdefer own.engine.deinit();
            // the batched pass's scratch for a prompt filling the window, once for the engine (every stream's
            // prompt, window and head pass runs on it): allocated now, inside the load's reserve, not at the first long
            // prompt; --slide: the trainer's passes run on the same scratch, up to its pack of rows
            _ = try own.engine.reserve(@max(@min(o.context, prompt.batch.default_chunk), if (o.slide) @as(usize, train.pack_rows) else 0));
            // the MTP head unless --no-drafts (TF_FLASHNEXT_DRAFTS and _CONFIDENCE: the chain's rule, 6 and 0.60)
            if (o.drafts) {
                own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);
                own.depth = @max(1, @min(mtp.max_chain, envUsize("TF_FLASHNEXT_DRAFTS", mtp.max_chain)));
                if (std.c.getenv("TF_FLASHNEXT_CONFIDENCE")) |v| own.confidence = std.fmt.parseFloat(f64, std.mem.span(v)) catch mtp.default_confidence;
            }
            errdefer if (own.head) |h| h.deinit();
            const window = @min(o.context + mtp.max_rows, prompt.max_tokens);
            // and an attention block's block scores at the window (64 MiB at 262,144 tokens)
            _ = try own.engine.reserveScores(window, @min(o.context, prompt.batch.default_chunk));
            own.logits = try cuda.DeviceBuffer.alloc(driver, round_streams * prompt.vocabulary * 2);
            errdefer own.logits.free();
            own.lanes = .empty;
            own.next = 0;
            if (o.slide) own.lay = try slide_mod.layoutOf(gpa, io, dir);
            var stream_bytes = prompt.seqBytesAt(window, own.engine.kv_bits);
            if (own.head) |h| stream_bytes += mtp.Draft.bytes(window, h.ids.len, own.engine.kv_bits);
            return .{
                .backend = own.backend(),
                .facts = own.facts(),
                .rows = if (own.head != null) mtp.max_rows else 2,
                // the stream's caches (and with the head, its cache and a window's buffers)
                .stream_bytes = stream_bytes,
                .ctx = own,
                .deinit = release,
                .learns = if (o.slide) own else null,
                .learn_bytes = if (o.slide) train.memory() else 0,
                .target = own,
            };
        }

        fn release(p: *anyopaque) void {
            const own: *Cuda = @ptrCast(@alignCast(p));
            var it = own.lanes.valueIterator();
            while (it.next()) |l| dropLane(own, l);
            own.lanes.deinit(own.gpa);
            own.logits.free();
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
            // without the head's drafts, one row a stream a round: a copied draft the stream rejected would need its
            // DeltaNet states rewound, which only a drafting stream's window keeps
            if (!drafting) {
                s.drafts = false;
                s.proposer = null;
            }
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
                // the head reads every prompt row's streams: the whole prompt runs (a kept state stays untouched)
                s.cached = 0;
                l.draft = try mtp.Draft.init(self.gpa, self.engine.driver, cap, self.head.?.ids.len, self.engine.kv_bits);
                try self.ops().prefill(&l.seq, &l.draft.?, ids);
            } else try self.fill(s, &l.seq, ids);
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

        /// A drafting stream's window in one batched pass, each row drawn at its keyed position; the streams without
        /// drafts feed their one row each in shared rounds of up to round_streams rows, each stream's scores the bits
        /// it would get alone, then each row drawn with its stream's sampling at its position.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            const seqs = try self.gpa.alloc(*prompt.Seq, windows.len);
            defer self.gpa.free(seqs);
            const ids = try self.gpa.alloc(u32, windows.len);
            defer self.gpa.free(ids);
            const at_window = try self.gpa.alloc(usize, windows.len);
            defer self.gpa.free(at_window);
            var plain: usize = 0;
            for (windows, out, 0..) |w, *o, wi| {
                if (w.held != 0 or w.parents != null) return error.HeldDraftsNotBuilt;
                const l = try self.lane(w.stream);
                const rows = w.rows();
                if (l.draft) |*d| {
                    if (rows > mtp.max_rows) return error.WindowTooWide;
                    for (w.positions, 0..) |p, r| if (p != l.seq.fed + 1 + r) return error.PositionMismatch;
                    var row_ids: [mtp.max_rows]u32 = undefined;
                    row_ids[0] = w.pending;
                    @memcpy(row_ids[1..rows], w.tokens);
                    try self.ops().verify(&l.seq, d, row_ids[0..rows]);
                    try self.ops().draws(d, w.stream.sampling, o.sampled[0..rows]);
                } else {
                    if (rows != 1) return error.CannotRewind;
                    if (w.positions[0] != l.seq.fed + 1) return error.PositionMismatch;
                    seqs[plain] = &l.seq;
                    ids[plain] = w.pending;
                    at_window[plain] = wi;
                    plain += 1;
                }
                if (w.tokens.len > 0) @memcpy(o.drafts[0..w.tokens.len], w.tokens);
            }
            var at: usize = 0;
            while (at < plain) : (at += round_streams) {
                const n = @min(round_streams, plain - at);
                try self.engine.round(Tri, self.tri(), seqs[at..][0..n], ids[at..][0..n], self.logits.ptr);
            }
            for (seqs[0..plain], at_window[0..plain]) |q, wi| out[wi].sampled[0] = try self.tokenAt(q, windows[wi].stream, windows[wi].positions[0]);
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
                // the window's kept rows: a lone stream's round calls keep first only for a cut window, a shared round
                // calls it after the drafts (rollback is idempotent, so keep then finds it done)
                if (r.rows) |path| try self.ops().rollback(&l.seq, d, path.len) else d.settle(&l.seq);
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

        // -- the prompt cache's copies ------------------------------------------------------------------------

        fn snapBytes(ptr: *anyopaque, at: u32) u64 {
            return snap.bytes(at, of(ptr).engine.kv_bits);
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
