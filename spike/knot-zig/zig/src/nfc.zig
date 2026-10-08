//! NFC — canonical normalization for the tokenizer port.
//!
//! Pipeline: canonical (recursive) decomposition + algorithmic Hangul
//! decomposition -> canonical ordering of combining marks -> canonical
//! composition with the behavior-derived pair table (nfc_tables.zig) and the
//! Hangul jamo rules. Equivalent to Python's `unicodedata.normalize("NFC", s)`
//! for this Unicode version — gated by the nfc_corpus.json cross-check.
//!
//! Tables are generated (make_nfc_tables.py); lookup is a linear scan, which
//! is fine at spike scale (short strings); binary search is the perf note for
//! a real port.

const std = @import("std");
const t = @import("nfc_tables.zig");

const SBASE: u21 = 0xAC00;
const SCOUNT: u21 = 11172;
const LBASE: u21 = 0x1100;
const VBASE: u21 = 0x1161;
const TBASE: u21 = 0x11A7;

pub fn cccOf(cp: u21) u8 {
    for (t.ccc_cp, 0..) |c, i| {
        if (c == cp) return t.ccc_val[i];
        if (c > cp) return 0;
    }
    return 0;
}

fn decompOf(cp: u21) ?[]const u21 {
    for (t.decomp_cp, 0..) |c, i| {
        if (c == cp) return t.decomp_flat[t.decomp_off[i]..t.decomp_off[i + 1]];
        if (c > cp) return null;
    }
    return null;
}

fn expand(cp: u21, out: *std.ArrayListUnmanaged(u21), alloc: std.mem.Allocator, depth: u8) !void {
    if (depth > 8) {
        try out.append(alloc, cp);
        return;
    }
    // Hangul syllable -> L V (T), algorithmic (tables exclude Hangul).
    if (cp >= SBASE and cp < SBASE + SCOUNT) {
        const s = cp - SBASE;
        try out.append(alloc, LBASE + @divFloor(s, 588));
        try out.append(alloc, VBASE + @divFloor(@mod(s, 588), 28));
        const tt = @mod(s, 28);
        if (tt != 0) try out.append(alloc, TBASE + tt);
        return;
    }
    if (decompOf(cp)) |parts| {
        for (parts) |p| try expand(p, out, alloc, depth + 1);
        return;
    }
    try out.append(alloc, cp);
}

/// Canonical ordering: stable insertion sort of combining marks (strict `>` so
/// equal ccc keeps input order; starters ccc=0 stop the movement naturally).
fn reorder(cps: []u21) void {
    var i: usize = 1;
    while (i < cps.len) : (i += 1) {
        const v = cps[i];
        const c = cccOf(v);
        if (c == 0) continue;
        var j: usize = i;
        while (j > 0 and cccOf(cps[j - 1]) > c) {
            cps[j] = cps[j - 1];
            j -= 1;
        }
        cps[j] = v;
    }
}

fn compPair(a: u21, b: u21) ?u21 {
    // Hangul jamo: L + V -> LV
    if (a >= LBASE and a <= LBASE + 18 and b >= VBASE and b <= VBASE + 20) {
        const s = (a - LBASE) * 588 + (b - VBASE) * 28;
        return SBASE + s;
    }
    // Hangul jamo: LV + T -> LVT
    if (b > TBASE and b <= TBASE + 27) {
        if (a >= SBASE and a < SBASE + SCOUNT and @mod(a - SBASE, 28) == 0) {
            return a + (b - TBASE);
        }
    }
    for (t.comp_a, 0..) |ca, i| {
        if (ca > a or (ca == a and t.comp_b[i] > b)) return null;
        if (ca == a and t.comp_b[i] == b) return t.comp_x[i];
    }
    return null;
}

/// Canonical composition in place, with the Unicode blocking rule.
fn compose(cps: *std.ArrayListUnmanaged(u21)) void {
    if (cps.items.len < 2) return;
    var starter: ?usize = null;
    var max_ccc: i16 = -1; // max ccc strictly between starter and next char
    if (cccOf(cps.items[0]) == 0) starter = 0;
    var i: usize = 1;
    while (i < cps.items.len) {
        const b = cps.items[i];
        const cb: i16 = cccOf(b);
        if (starter) |si| {
            if (max_ccc < cb) {
                if (compPair(cps.items[si], b)) |x| {
                    cps.items[si] = x;
                    _ = cps.orderedRemove(i); // next char shifts into i
                    // Re-derive the blocking state: chars now strictly
                    // between the starter and the next unprocessed char.
                    max_ccc = -1;
                    var k = si + 1;
                    while (k < i) : (k += 1) {
                        const cc: i16 = cccOf(cps.items[k]);
                        if (cc > max_ccc) max_ccc = cc;
                    }
                    continue; // reprocess index i (shifted char)
                }
            }
        }
        if (cb == 0) {
            starter = i;
            max_ccc = -1;
        } else if (cb > max_ccc) {
            max_ccc = cb;
        }
        i += 1;
    }
}

fn utf8LeadLen(b: u8) ?usize {
    if (b < 0x80) return 1;
    if (b & 0xE0 == 0xC0) return 2;
    if (b & 0xF0 == 0xE0) return 3;
    if (b & 0xF8 == 0xF0) return 4;
    return null;
}

fn utf8EncodeOne(cp: u21, out: *[4]u8) usize {
    if (cp < 0x80) {
        out[0] = @intCast(cp);
        return 1;
    } else if (cp < 0x800) {
        out[0] = @intCast(0xC0 | (cp >> 6));
        out[1] = @intCast(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = @intCast(0xE0 | (cp >> 12));
        out[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        out[2] = @intCast(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = @intCast(0xF0 | (cp >> 18));
        out[1] = @intCast(0x80 | ((cp >> 12) & 0x3F));
        out[2] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        out[3] = @intCast(0x80 | (cp & 0x3F));
        return 4;
    }
}

/// `unicodedata.normalize("NFC", text)` equivalent for this Unicode version.
pub fn normalize(alloc: std.mem.Allocator, text: []const u8) ![]u8 {
    var cps: std.ArrayListUnmanaged(u21) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const bl = utf8LeadLen(text[i]) orelse return error.BadUtf8;
        if (i + bl > text.len) return error.BadUtf8;
        const cp = try std.unicode.utf8Decode(text[i .. i + bl]);
        try expand(cp, &cps, alloc, 0);
        i += bl;
    }
    reorder(cps.items);
    compose(&cps);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    for (cps.items) |cp| {
        var buf: [4]u8 = undefined;
        const n = utf8EncodeOne(cp, &buf);
        try out.appendSlice(alloc, buf[0..n]);
    }
    return out.toOwnedSlice(alloc);
}

test "nfc: stability (idempotence)" {
    const alloc = std.testing.allocator;
    // reorder-then-compose sequence: normalize twice must be a fixed point
    const r1 = try normalize(alloc, "a\xCC\x81\xCC\x96");
    defer alloc.free(r1);
    const r2 = try normalize(alloc, r1);
    defer alloc.free(r2);
    try std.testing.expectEqualStrings(r1, r2);
}

test "nfc: hangul jamo algorithm" {
    const alloc = std.testing.allocator;
    // ᄀ + ᅡ -> 가
    const a = try normalize(alloc, "\xE1\x84\x80\xE1\x85\xA1");
    defer alloc.free(a);
    try std.testing.expectEqualStrings("\xEA\xB0\x80", a);
    // 각 = ᄀ + ᅡ + ᆨ -> 각
    const b = try normalize(alloc, "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8");
    defer alloc.free(b);
    try std.testing.expectEqualStrings("\xEA\xB0\x81", b);
}
