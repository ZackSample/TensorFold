//! Flash Next (qwen4_exp) as Living Weights reads it: its family (living/families.zig), the layout its config gives,
//! and the pocket's writer into its shards. Its host's pocketSaveSwiglu writes through `Shards`.
const std = @import("std");
const living = @import("living");

pub const family = living.families.qwen4_exp;

/// Every layer's shared expert carries the pocket; all but the last learn (as Nemotron's: a nudge after the last
/// layer has no gradient for the trainer to read). `width` is the config's shared expert width, a pocket written
/// earlier (`pocket`, text_config's tensorfold_learned_neurons) still in it.
pub fn layout(hidden: usize, layers: usize, width: usize, pocket: usize) !living.Layout {
    if (layers > living.family.max_layers or pocket >= width) return error.BadConfig;
    var every: living.Layers = .empty;
    for (0..layers) |l| every.set(l);
    return .fromExperts(hidden, width - pocket, layers, every);
}

/// The layout of a config.json's text: hidden_size, num_hidden_layers, the shared expert's width and the pocket's.
pub fn layoutOf(text: std.json.ObjectMap) !living.Layout {
    const k = family.keys;
    const int = living.families.intField;
    return layout(int(text, k.hidden) orelse return error.BadConfig, int(text, k.layers) orelse return error.BadConfig, int(text, k.width) orelse return error.BadConfig, int(text, k.pocket) orelse 0);
}

/// The writer over `core` (its shard_edit and safetensors).
pub fn Shards(comptime core: type) type {
    const W = living.shards.Io(core);
    return struct {
        /// Layers `layers`' neurons into `dir`'s shards, widened (and text_config told) on the first write: `gate` and
        /// `up` [layers, neurons, D] the rows the forward multiplies, `down` the outputs; handed back as written.
        pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, lay: living.Layout, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
            return W.write(gpa, io, dir, family, lay, layers, neurons, &.{ gate, up }, down);
        }
    };
}
