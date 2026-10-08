//! Phase-2 canary: wire question → Laya internal form + option rendering.
//! Port of knot `crates/knot/src/prompt.rs` (`de_instructions`,
//! `InternalQuestion::from_wire`, `normalize_choice_labels`,
//! `render_options`, `render_criterion`, `serialize_state`) and
//! `pyjson::py_str`, message strings verbatim (they are 422 `detail`s).
const std = @import("std");
const pyjson = @import("pyjson.zig");

pub const Reject = error{Rejected};

/// Message-carrying rejection (knot maps these to HTTP 422 with this text).
pub const Bag = struct {
    alloc: std.mem.Allocator,
    msg: ?[]const u8 = null,

    pub fn fail(self: *Bag, comptime fmt: []const u8, args: anytype) Reject {
        self.msg = std.fmt.allocPrint(self.alloc, fmt, args) catch @panic("OOM");
        return error.Rejected;
    }
};

/// Rust `{:?}` for a string — the quoting style knot's messages use
/// (`question {qid:?}: ...`).
pub fn rustDebug(alloc: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try out.appendSlice(alloc, "\"");
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            '\n' => try out.appendSlice(alloc, "\\n"),
            '\r' => try out.appendSlice(alloc, "\\r"),
            '\t' => try out.appendSlice(alloc, "\\t"),
            else => {
                if (c < 0x20 or c == 0x7f) {
                    const hex = "0123456789abcdef";
                    const esc = [_]u8{ '\\', 'x', hex[c >> 4], hex[c & 15] };
                    try out.appendSlice(alloc, &esc);
                } else {
                    try out.append(alloc, c);
                }
            },
        }
    }
    try out.appendSlice(alloc, "\"");
    return out.toOwnedSlice(alloc);
}

pub const QType = enum { choice, score, noul };

pub fn typeName(t: QType) []const u8 {
    return switch (t) {
        .choice => "choice",
        .score => "score",
        .noul => "noul",
    };
}

/// Laya's internal dict: `t`, `ins`, `crit`, `labels`.
pub const Internal = struct {
    t: QType,
    ins: []const u8,
    crit: ?std.json.Value,
    labels: ?std.json.Value,
};

/// `de_instructions`: string passes through, null / empty array / empty
/// object become empty text, anything else is Python `json.dumps`.
fn deInstructions(alloc: std.mem.Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        .null => "",
        .array => |a| if (a.items.len == 0) "" else pyjson.dumps(alloc, v),
        .object => |o| if (o.count() == 0) "" else pyjson.dumps(alloc, v),
        else => pyjson.dumps(alloc, v),
    };
}

/// Python equality over scalars (`[1, 1.0]` and `[True, 1]` collapse).
/// Rust matches on (String,String)/(Bool,Bool)/(Number,Number) plus the
/// Python `True == 1` cross — coercing every scalar to f64 reproduces all
/// of those outcomes (serde numbers are never NaN).
fn pyEq(a: std.json.Value, b: std.json.Value) bool {
    if (a == .string or b == .string) {
        // Only String==String is true in Rust; mixed string/number is false.
        return a == .string and b == .string and std.mem.eql(u8, a.string, b.string);
    }
    const fa: f64 = switch (a) {
        .float => |v| v,
        .integer => |v| @floatFromInt(v),
        .bool => |v| if (v) @as(f64, 1.0) else 0.0,
        else => return false,
    };
    const fb: f64 = switch (b) {
        .float => |v| v,
        .integer => |v| @floatFromInt(v),
        .bool => |v| if (v) @as(f64, 1.0) else 0.0,
        else => return false,
    };
    return fa == fb;
}

/// `pyjson::py_str`: Python `str(scalar)` used for choice answer keys.
pub fn pyStr(alloc: std.mem.Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        .bool => |b| if (b) "True" else "False",
        .integer => |i| std.fmt.allocPrint(alloc, "{d}", .{i}),
        .float => |f| pyjson.floatRepr(alloc, f),
        else => pyjson.dumps(alloc, v),
    };
}

/// `normalize_choice_labels`: a list of labels becomes `{label: null}`,
/// rejecting null / structured / duplicate labels with 422s.
fn normalizeChoiceLabels(bag: *Bag, qid: []const u8, crit: *?std.json.Value) !void {
    const arr = switch (crit.* orelse return) {
        .array => |a| a,
        else => return,
    };
    const alloc = bag.alloc;
    const qid_dbg = try rustDebug(alloc, qid);
    for (arr.items, 0..) |label, i| {
        switch (label) {
            .null => return bag.fail(
                "question {s}: choice label {d} is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool",
                .{ qid_dbg, i },
            ),
            .array, .object => return bag.fail(
                "question {s}: choice label {d} is not a scalar; a label is rendered as option text and used as the answer key, so it must be a string, number or bool",
                .{ qid_dbg, i },
            ),
            else => {},
        }
    }
    var obj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
    for (arr.items, 0..) |label, i| {
        for (0..i) |j| {
            if (pyEq(arr.items[j], label)) {
                return bag.fail(
                    "question {s}: choice label {d} repeats label {d}; the labels are the answer keys, so every option needs its own (1, 1.0 and True are one key)",
                    .{ qid_dbg, i, j },
                );
            }
        }
        const key = try pyStr(alloc, label);
        try obj.put(alloc, key, .null);
    }
    crit.* = .{ .object = obj };
}

/// `render_criterion`: strings pass through, structured values become
/// Python `json.dumps` text.
pub fn renderCriterion(alloc: std.mem.Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        else => pyjson.dumps(alloc, v),
    };
}

/// `serialize_state`: strings pass through, containers become Python
/// `json.dumps(state, ensure_ascii=False)`.
pub fn serializeState(alloc: std.mem.Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        else => pyjson.dumps(alloc, v),
    };
}

pub const NoulLabels = struct { f: []const u8, t: []const u8 };

/// noul `resolve_noul_labels`: exactly `false`+`true` keys → trimmed strings.
fn resolveNoulLabels(bag: *Bag, labels: ?std.json.Value) !NoulLabels {
    const bad_msg = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
    const no_labels = NoulLabels{ .f = "false", .t = "true" };
    const m = labels orelse return no_labels;
    if (m != .object) return bag.fail("{s}", .{bad_msg});
    const obj = m.object;
    if (obj.count() != 2 or !obj.contains("false") or !obj.contains("true"))
        return bag.fail("{s}", .{bad_msg});
    // Values must be strings (Rust `.as_str().unwrap_or("")` → non-string
    // becomes empty → rejected by the distinct/empty checks below).
    const fv = obj.get("false").?;
    const tv = obj.get("true").?;
    const f2: []const u8 = if (fv == .string) std.mem.trim(u8, fv.string, " ") else "";
    const t2: []const u8 = if (tv == .string) std.mem.trim(u8, tv.string, " ") else "";
    if (f2.len == 0 or t2.len == 0 or std.mem.eql(u8, f2, t2))
        return bag.fail("{s}", .{bad_msg});
    return .{ .f = f2, .t = t2 };
}

/// `render_options`: one rendered option per criterion, in label order;
/// noul semantic order is [false, true].
pub fn renderOptions(
    alloc: std.mem.Allocator,
    bag: *Bag,
    q: Internal,
) ![][]const u8 {
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    switch (q.t) {
        .choice => {
            const c = q.crit orelse return bag.fail("choice needs criteria object", .{});
            if (c != .object) return bag.fail("choice needs criteria object", .{});
            const obj = c.object;
            if (obj.count() == 0) return bag.fail("choice needs at least one criterion", .{});
            var it = obj.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                const v = e.value_ptr.*;
                const opt: []const u8 = switch (v) {
                    .null => k,
                    .string => |s| if (s.len == 0) k else try std.fmt.allocPrint(alloc, "{s}: {s}", .{ k, try renderCriterion(alloc, v) }),
                    else => try std.fmt.allocPrint(alloc, "{s}: {s}", .{ k, try renderCriterion(alloc, v) }),
                };
                try out.append(alloc, opt);
            }
        },
        .score => {
            const c = q.crit orelse return bag.fail("score needs criteria list", .{});
            if (c != .array) return bag.fail("score needs criteria list", .{});
            const arr = c.array;
            if (arr.items.len == 0) return bag.fail("score needs at least one level", .{});
            for (arr.items, 0..) |lvl, i| {
                if (lvl == .null) {
                    return bag.fail(
                        "score level {d} is null; give every level a description, index 0 first",
                        .{i},
                    );
                }
                try out.append(alloc, try std.fmt.allocPrint(
                    alloc,
                    "level {d}: {s}",
                    .{ i, try renderCriterion(alloc, lvl) },
                ));
            }
        },
        .noul => {
            const labs = try resolveNoulLabels(bag, q.labels);
            if (q.crit) |c| {
                if (c != .null) {
                    if (c != .object) {
                        return bag.fail(
                            "noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it",
                            .{},
                        );
                    }
                    var bad: std.ArrayListUnmanaged([]const u8) = .empty;
                    var it = c.object.iterator();
                    while (it.next()) |e| {
                        const lower = try std.ascii.allocLowerString(alloc, e.key_ptr.*);
                        if (!std.mem.eql(u8, lower, "true") and !std.mem.eql(u8, lower, "false"))
                            try bad.append(alloc, e.key_ptr.*);
                    }
                    if (bad.items.len > 0) {
                        var arr: std.ArrayListUnmanaged(u8) = .empty;
                        try arr.appendSlice(alloc, "[");
                        for (bad.items, 0..) |b, i| {
                            if (i > 0) try arr.appendSlice(alloc, ", ");
                            try arr.appendSlice(alloc, try rustDebug(alloc, b));
                        }
                        try arr.appendSlice(alloc, "]");
                        return bag.fail(
                            "noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got {s}",
                            .{arr.items},
                        );
                    }
                }
            }
            const crit = if (q.crit != null and q.crit.? != .null and q.crit.? == .object) q.crit.? else null;
            const render = struct {
                fn r(a: std.mem.Allocator, c: ?std.json.Value, key: []const u8, dflt: []const u8) ![]const u8 {
                    if (c) |obj| {
                        if (obj.object.get(key)) |v| {
                            if (v == .null) return dflt;
                            if (v == .string and v.string.len == 0) return dflt;
                            return renderCriterion(a, v);
                        }
                    }
                    return dflt;
                }
            }.r;
            try out.append(alloc, try std.fmt.allocPrint(
                alloc,
                "{s}: {s}",
                .{ labs.f, try renderCriterion(alloc, .{ .string = try render(alloc, crit, "false", "no, the statement does not hold") }) },
            ));
            try out.append(alloc, try std.fmt.allocPrint(
                alloc,
                "{s}: {s}",
                .{ labs.t, try renderCriterion(alloc, .{ .string = try render(alloc, crit, "true", "yes, the statement holds") }) },
            ));
        },
    }
    return out.toOwnedSlice(alloc);
}

/// `InternalQuestion::from_wire`: reject what cannot be answered, normalize
/// to the internal `t`/`ins`/`crit`/`labels` form.
pub fn fromWire(
    alloc: std.mem.Allocator,
    bag: *Bag,
    qid: []const u8,
    q: std.json.Value,
) !Internal {
    if (q != .object) return bag.fail("question {s} must be an object", .{try rustDebug(alloc, qid)});
    const obj = q.object;
    const t: QType = blk: {
        const tv = obj.get("type") orelse return bag.fail("question {s}: 'type' is required", .{try rustDebug(alloc, qid)});
        if (tv != .string) return bag.fail("question {s}: 'type' must be a string", .{try rustDebug(alloc, qid)});
        if (std.mem.eql(u8, tv.string, "choice")) break :blk .choice;
        if (std.mem.eql(u8, tv.string, "score")) break :blk .score;
        if (std.mem.eql(u8, tv.string, "noul")) break :blk .noul;
        return bag.fail("unknown question type {s}", .{tv.string});
    };
    const ins = try deInstructions(alloc, obj.get("instructions") orelse .null);
    if (std.mem.trim(u8, ins, " \t\r\n").len == 0) {
        return bag.fail("question {s}: 'instructions' must not be empty", .{try rustDebug(alloc, qid)});
    }
    var crit: ?std.json.Value = obj.get("criteria") orelse null;
    const labels: ?std.json.Value = obj.get("labels");
    if (t == .choice) {
        try normalizeChoiceLabels(bag, qid, &crit);
    } else if (t == .noul) {
        if (crit) |c| {
            if (c == .object) {
                // `_to_internal`: `{str(k).lower(): v}` — order preserved.
                var lowered = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                var it = c.object.iterator();
                while (it.next()) |e| {
                    const lower = try std.ascii.allocLowerString(alloc, e.key_ptr.*);
                    try lowered.put(alloc, lower, e.value_ptr.*);
                }
                crit = .{ .object = lowered };
            }
        }
    }
    return .{ .t = t, .ins = ins, .crit = crit, .labels = labels };
}
