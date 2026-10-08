//! P2-3: decode math — port of knot's `runtime.rs` calibration/softmax and
//! `engine.rs` answer assembly inputs. f32 semantics copied verbatim
//! (sequential sums, `(v*10000).round()/10000`, clamp ranges).
const std = @import("std");

pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

pub fn clampTemperature(t: f64) f64 {
    if (std.math.isNan(t)) return 1.0;
    return std.math.clamp(t, TEMP_MIN, TEMP_MAX);
}

/// `{qtype}:{size}` — the per-bucket override key (knot `temp_bucket`).
pub fn tempBucket(alloc: std.mem.Allocator, qtype: []const u8, k: usize) ![]const u8 {
    const size: []const u8 = switch (k) {
        0...2 => "2",
        3...5 => "3-5",
        6...10 => "6-10",
        else => "11+",
    };
    return std.fmt.allocPrint(alloc, "{s}:{s}", .{ qtype, size });
}

/// Per-checkpoint calibration: 3 per-type temperatures + optional per-bucket
/// map, straight from `rl_agent_config.json` (knot `Calibration::from_config`).
pub const Calibration = struct {
    temperature: [3]f64 = @splat(1.0),
    by_options: std.StringHashMapUnmanaged(f64) = .{},

    pub fn fromConfig(alloc: std.mem.Allocator, cfg: std.json.Value) !Calibration {
        var self = Calibration{};
        if (cfg != .object) return self;
        if (cfg.object.get("temperature")) |tv| {
            if (tv == .array) {
                for (tv.array.items[0..@min(tv.array.items.len, 3)], 0..) |v, i| {
                    if (v == .float) self.temperature[i] = clampTemperature(v.float);
                    if (v == .integer) self.temperature[i] = clampTemperature(@floatFromInt(v.integer));
                }
            }
        }
        if (cfg.object.get("temperature_by_options")) |bv| {
            if (bv == .object) {
                var it = bv.object.iterator();
                while (it.next()) |e| {
                    const val: f64 = switch (e.value_ptr.*) {
                        .float => |f| f,
                        .integer => |i| @floatFromInt(i),
                        else => continue,
                    };
                    try self.by_options.put(alloc, e.key_ptr.*, val);
                }
            }
        }
        return self;
    }

    /// Effective temperature for a question (bucket override, else per-type).
    pub fn forQuestion(self: *const Calibration, qtype: []const u8, idx: usize, k: usize, buf: *[64]u8) f64 {
        const size: []const u8 = switch (k) {
            0...2 => "2",
            3...5 => "3-5",
            6...10 => "6-10",
            else => "11+",
        };
        const key = std.fmt.bufPrint(buf, "{s}:{s}", .{ qtype, size }) catch unreachable;
        const t = self.by_options.get(key) orelse self.temperature[idx];
        return clampTemperature(t);
    }
};

/// Softmax over `logits` scaled by `temperature` (f32, knot order).
pub fn scaledSoftmax(alloc: std.mem.Allocator, logits: []const f32, temperature: f32) ![]f32 {
    var max: f32 = -std.math.inf(f32);
    for (logits) |z| max = @max(max, z);
    const t = temperature;
    var out = try alloc.alloc(f32, logits.len);
    var sum: f32 = 0;
    for (logits, 0..) |z, i| {
        out[i] = @exp((z / t) - (max / t));
        sum += out[i];
    }
    for (out) |*e| e.* = e.* / sum;
    return out;
}

/// 1 − normalized entropy confidence (Laya `confidence_from_probs`).
pub fn confidenceFromProbs(p: []const f32) f32 {
    const k = p.len;
    if (k < 2) return 1.0;
    var h: f32 = 0;
    for (p) |x| {
        if (x > 0.0) h += -x * @log(x);
    }
    const c = 1.0 - h / @log(@as(f32, @floatFromInt(k)));
    return @max(c, 0.0);
}

pub fn answerConfidence(p: []const f32) f32 {
    var m: f32 = -std.math.inf(f32);
    for (p) |x| m = @max(m, x);
    return m;
}

/// Round to 4 decimals, f32, away-from-zero (Rust f32 `round`).
pub fn r4(v: f32) f32 {
    return @round(v * 10_000.0) / 10_000.0;
}
