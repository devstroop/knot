//! Python-compatible JSON rendering — port of knot's
//! `crates/knot/src/pyjson.rs` (Laya common.py/agent.py semantics):
//! `json.dumps(value, ensure_ascii=False)` with default (", ", ": ")
//! separators, Python's escaping rules, and Python's shortest float repr
//! (fixed notation for decimal exponents in [-4, 16), else `d[.ddd]e±NN`
//! with a sign and at least two exponent digits — `1e-05`, `1e+16`).
//!
//! Parity target: 55-case corpus vs Python and knot's own dumps
//! (pyjson_cases.json / pyjson_py.json / pyjson_rust.json).
const std = @import("std");

pub fn dumps(alloc: std.mem.Allocator, value: std.json.Value) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try writeValue(&out, alloc, value, false);
    return out.toOwnedSlice(alloc);
}

/// serde_json wire style: compact `","`/`":"` separators, same escaping
/// and float repr as `dumps` (knot serves HTTP bodies this way).
pub fn dumpsCompact(alloc: std.mem.Allocator, value: std.json.Value) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try writeValue(&out, alloc, value, true);
    return out.toOwnedSlice(alloc);
}

fn writeValue(out: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, v: std.json.Value, compact: bool) !void {
    const sep: []const u8 = if (compact) "," else ", ";
    const kv: []const u8 = if (compact) ":" else ": ";
    switch (v) {
        .null => try out.appendSlice(alloc, "null"),
        .bool => |b| try out.appendSlice(alloc, if (b) "true" else "false"),
        .integer => |i| {
            const s = try std.fmt.allocPrint(alloc, "{d}", .{i});
            defer alloc.free(s);
            try out.appendSlice(alloc, s);
        },
        .float => |f| {
            const s = try floatRepr(alloc, f);
            defer alloc.free(s);
            try out.appendSlice(alloc, s);
        },
        // Raw numeric text (0.17 keeps out-of-range/specified numbers as
        // source strings): already-valid JSON number text — pass through.
        .number_string => |raw| try out.appendSlice(alloc, raw),
        .string => |s| try writeString(out, alloc, s),
        .array => |arr| {
            try out.append(alloc, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(alloc, sep);
                try writeValue(out, alloc, item, compact);
            }
            try out.append(alloc, ']');
        },
        .object => |obj| {
            try out.append(alloc, '{');
            var first = true;
            var it = obj.iterator();
            while (it.next()) |e| {
                if (!first) try out.appendSlice(alloc, sep);
                first = false;
                try writeString(out, alloc, e.key_ptr.*);
                try out.appendSlice(alloc, kv);
                try writeValue(out, alloc, e.value_ptr.*, compact);
            }
            try out.append(alloc, '}');
        },
    }
}

/// Python `repr` of a finite float, mirroring knot's float_repr exactly.
pub fn floatRepr(alloc: std.mem.Allocator, v: f64) ![]u8 {
    if (v == 0.0) {
        // -0.0 check via the sign bit (0.17 has no f64.is_sign_negative).
        const neg_zero = (@as(u64, @bitCast(v)) >> 63) == 1;
        return try alloc.dupe(u8, if (neg_zero) "-0.0" else "0.0");
    }
    const sci = try std.fmt.allocPrint(alloc, "{e}", .{v});
    defer alloc.free(sci);
    const neg = sci[0] == '-';
    const s = if (neg) sci[1..] else sci;
    const eidx = std.mem.indexOfScalar(u8, s, 'e') orelse return error.NoExponent;
    const exp_part = if (s[eidx + 1] == '+') s[eidx + 2 ..] else s[eidx + 1 ..];
    const exp: i32 = try std.fmt.parseInt(i32, exp_part, 10);
    var digits_buf: [64]u8 = undefined;
    var dn: usize = 0;
    for (s[0..eidx]) |c| {
        if (c != '.') {
            digits_buf[dn] = c;
            dn += 1;
        }
    }
    const digits = digits_buf[0..dn];
    const sign: []const u8 = if (neg) "-" else "";
    var out: std.ArrayListUnmanaged(u8) = .empty;
    if (exp >= -4 and exp < 16) {
        try out.appendSlice(alloc, sign);
        if (exp >= 0) {
            const int_len: usize = @intCast(exp + 1);
            if (digits.len >= int_len) {
                try out.appendSlice(alloc, digits[0..int_len]);
                const frac = digits[int_len..];
                if (frac.len == 0) {
                    try out.appendSlice(alloc, ".0");
                } else {
                    try out.append(alloc, '.');
                    try out.appendSlice(alloc, frac);
                }
            } else {
                try out.appendSlice(alloc, digits);
                var z: usize = 0;
                while (z < int_len - digits.len) : (z += 1) try out.append(alloc, '0');
                try out.appendSlice(alloc, ".0");
            }
        } else {
            try out.appendSlice(alloc, "0.");
            var z: usize = 0;
            const zeros: usize = @intCast(-exp - 1);
            while (z < zeros) : (z += 1) try out.append(alloc, '0');
            try out.appendSlice(alloc, digits);
        }
    } else {
        try out.appendSlice(alloc, sign);
        try out.append(alloc, digits[0]);
        if (digits.len > 1) {
            try out.append(alloc, '.');
            try out.appendSlice(alloc, digits[1..]);
        }
        try out.append(alloc, 'e');
        try out.append(alloc, if (exp < 0) '-' else '+');
        const es = try std.fmt.allocPrint(alloc, "{d}", .{@abs(exp)});
        defer alloc.free(es);
        if (es.len == 1) try out.append(alloc, '0');
        try out.appendSlice(alloc, es);
    }
    return out.toOwnedSlice(alloc);
}

/// Python's json encoder escapes only `"`, `\`, and controls < 0x20
/// (`\b\t\n\f\r`, else `\u00xx` lowercase); everything else — DEL, non-ASCII
/// — passes through raw (ensure_ascii=False). Byte-wise: escapes are ASCII
/// and UTF-8 continuation bytes are >= 0x80, so a byte scan is exact.
fn writeString(out: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, s: []const u8) !void {
    try out.append(alloc, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            0x08 => try out.appendSlice(alloc, "\\b"),
            0x09 => try out.appendSlice(alloc, "\\t"),
            0x0A => try out.appendSlice(alloc, "\\n"),
            0x0C => try out.appendSlice(alloc, "\\f"),
            0x0D => try out.appendSlice(alloc, "\\r"),
            else => if (c < 0x20) {
                const esc = try std.fmt.allocPrint(alloc, "\\u{x:0>4}", .{c});
                defer alloc.free(esc);
                try out.appendSlice(alloc, esc);
            } else {
                try out.append(alloc, c);
            },
        }
    }
    try out.append(alloc, '"');
}

test "float repr boundaries" {
    const alloc = std.testing.allocator;
    const cases = [_]struct { f: f64, want: []const u8 }{
        .{ .f = 1.0, .want = "1.0" },
        .{ .f = 100.0, .want = "100.0" },
        .{ .f = 0.0001, .want = "0.0001" },
        .{ .f = 1e-5, .want = "1e-05" },
        .{ .f = -1.5e-7, .want = "-1.5e-07" },
        .{ .f = 1e15, .want = "1000000000000000.0" },
        .{ .f = 1e16, .want = "1e+16" },
        .{ .f = -2.5e20, .want = "-2.5e+20" },
        .{ .f = 1234.5, .want = "1234.5" },
    };
    for (cases) |c| {
        const got = try floatRepr(alloc, c.f);
        defer alloc.free(got);
        try std.testing.expectEqualStrings(c.want, got);
    }
    const neg0 = try floatRepr(alloc, -0.0);
    defer alloc.free(neg0);
    try std.testing.expectEqualStrings("-0.0", neg0);
}
