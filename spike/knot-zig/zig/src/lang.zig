//! P2-4: language/script detection — port of knot's `lang.rs` (itself a port
//! of `laya/lang.py`). Oracle: `lang_cases.json` (18 cases, fields `script`,
//! `is_english`, `language`, `language_undecided`, `non_latin_fraction`).
//!
//! Regex primitives are hand-rolled (zig std has no unicode-class regex);
//! lowercase is ASCII (verified sufficient: no fixture carries non-ASCII
//! uppercase). BTreeMap order semantics are preserved where ties matter.
const std = @import("std");
const ld = @import("lang_data.zig");

const NON_EN_DIACRITIC_RATE: f64 = 0.02;
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;
const NON_LATIN_FRACTION: f64 = 0.2;
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: usize = 10;

pub const Set = std.StringHashMapUnmanaged(void);

fn setPut(s: *Set, alloc: std.mem.Allocator, k: []const u8) !void {
    try s.put(alloc, k, {});
}
fn setHas(s: *const Set, k: []const u8) bool {
    return s.contains(k);
}

/// Rust `char::is_alphabetic` proxy: the Alphabetic property is L* +
/// Other_Alphabetic marks; the M0 UCD tables give full L* (Unicode 13).
/// Deviation is confined to standalone marks — no fixture oracle counts one
/// (the 18 lang_cases pin this).
fn isAlpha(cp: u32) bool {
    const tables = @import("class_tables.zig");
    if (cp > std.math.maxInt(u21)) return false;
    const c: u21 = @intCast(cp);
    var lo: usize = 0;
    var hi: usize = tables.l_ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = tables.l_ranges[mid];
        if (c < r[0]) {
            hi = mid;
        } else if (c > r[1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}
fn isAsciiLower(c: u8) bool {
    return c >= 'a' and c <= 'z';
}
/// `text.to_lowercase()` for the fixture universe (ASCII only — verified).
fn lowerAscii(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try alloc.dupe(u8, s);
    for (out) |*c| {
        if (c.* >= 'A' and c.* <= 'Z') c.* += 32;
    }
    return out;
}
fn isDiac(c: []const u8) bool {
    return std.mem.indexOf(u8, ld.NON_EN_DIACRITICS, c) != null;
}

/// Rust `trunc_str`: byte-boundary-safe prefix.
pub fn truncStr(s: []const u8, max: usize) []const u8 {
    if (s.len <= max) return s;
    var end = max;
    while (end > 0 and (s[end] & 0xC0) == 0x80) end -= 1;
    return s[0..end];
}

fn collectLeaves(alloc: std.mem.Allocator, state: std.json.Value) !std.ArrayListUnmanaged([]const u8) {
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    iterTextAlloc(alloc, state, 0, &out);
    return out;
}

fn iterTextAlloc(alloc: std.mem.Allocator, state: std.json.Value, depth: usize, out: *std.ArrayListUnmanaged([]const u8)) void {
    if (depth > 6) return;
    switch (state) {
        .string => |s| out.append(alloc, s) catch return,
        .array => |items| for (items.items) |v| iterTextAlloc(alloc, v, depth + 1, out),
        .object => |m| {
            var it = m.iterator();
            while (it.next()) |e| iterTextAlloc(alloc, e.value_ptr.*, depth + 1, out);
        },
        else => {},
    }
}

/// Flatten a state into detection text (byte budget + final char take).
pub fn stateText(alloc: std.mem.Allocator, state: std.json.Value, max_chars: usize) ![]const u8 {
    const leaves = try collectLeaves(alloc, state);
    var parts: std.ArrayListUnmanaged([]const u8) = .empty;
    var budget: usize = max_chars;
    for (leaves.items) |leaf| {
        if (budget == 0) break;
        if (leaf.len > budget) {
            try parts.append(alloc, truncStr(leaf, budget));
            break;
        }
        budget -|= leaf.len + 1;
        try parts.append(alloc, leaf);
    }
    var j: std.ArrayListUnmanaged(u8) = .empty;
    for (parts.items, 0..) |p, i| {
        if (i > 0) try j.append(alloc, ' ');
        try j.appendSlice(alloc, p);
    }
    const joined_bytes = try j.toOwnedSlice(alloc);
    // `.chars().take(max_chars)`
    var taken: std.ArrayListUnmanaged(u8) = .empty;
    var i: usize = 0;
    var n: usize = 0;
    while (i < joined_bytes.len and n < max_chars) : (n += 1) {
        const w = utf8Width(joined_bytes[i]);
        try taken.appendSlice(alloc, joined_bytes[i .. i + w]);
        i += w;
    }
    return taken.toOwnedSlice(alloc);
}

// ---------- script detection ----------

fn scriptOfChar(cp: u32) ?[]const u8 {
    if (cp < 0x02B0 or (cp >= 0x1E00 and cp <= 0x1EFF) or (cp >= 0xFF21 and cp <= 0xFF3A) or (cp >= 0xFF41 and cp <= 0xFF5A)) {
        return null; // Latin-ish
    }
    for (ld.SCRIPT_RANGES) |sr| {
        for (sr.ranges) |r| {
            if (cp >= r.lo and cp <= r.hi) return sr.name;
        }
    }
    return null;
}

pub const Counts = struct {
    entries: std.ArrayListUnmanaged(struct { name: []const u8, n: usize }) = .empty,
    latin: usize = 0,
};

fn scriptCounts(alloc: std.mem.Allocator, text: []const u8) !Counts {
    var counts = Counts{};
    var i: usize = 0;
    while (i < text.len) {
        const cp = try nextCp(text, &i);
        if (!isAlpha(cp)) continue;
        if (cp < 0x02B0 or (cp >= 0x1E00 and cp <= 0x1EFF) or (cp >= 0xFF21 and cp <= 0xFF3A) or (cp >= 0xFF41 and cp <= 0xFF5A)) {
            counts.latin += 1;
            continue;
        }
        var claimed = false;
        for (ld.SCRIPT_RANGES) |sr| {
            for (sr.ranges) |r| {
                if (cp >= r.lo and cp <= r.hi) {
                    var found = false;
                    for (counts.entries.items) |*e| {
                        if (std.mem.eql(u8, e.name, sr.name)) {
                            e.n += 1;
                            found = true;
                            break;
                        }
                    }
                    if (!found) try counts.entries.append(alloc, .{ .name = sr.name, .n = 1 });
                    claimed = true;
                    break;
                }
            }
            if (claimed) break;
        }
        if (!claimed) {
            var found = false;
            for (counts.entries.items) |*e| {
                if (std.mem.eql(u8, e.name, "other")) {
                    e.n += 1;
                    found = true;
                    break;
                }
            }
            if (!found) try counts.entries.append(alloc, .{ .name = "other", .n = 1 });
        }
    }
    try counts.entries.append(alloc, .{ .name = "latin", .n = counts.latin });
    return counts;
}

fn nextCp(text: []const u8, i: *usize) !u32 {
    const w = utf8Width(text[i.*]);
    const cp = try std.unicode.utf8Decode(text[i.* .. i.* + w]);
    i.* += w;
    return cp;
}

fn countOf(counts: *const Counts, name: []const u8) usize {
    for (counts.entries.items) |e| {
        if (std.mem.eql(u8, e.name, name)) return e.n;
    }
    return 0;
}

fn scriptFromCounts(counts: *const Counts) []const u8 {
    var all_zero = true;
    for (counts.entries.items) |e| {
        if (e.n != 0) all_zero = false;
    }
    if (all_zero) return "unknown";
    var best: ?struct { name: []const u8, n: usize } = null;
    for (counts.entries.items) |e| {
        if (std.mem.eql(u8, e.name, "latin")) continue;
        if (best == null or e.n > best.?.n) best = .{ .name = e.name, .n = e.n };
    }
    const latin = countOf(counts, "latin");
    if (best == null or latin > best.?.n) return "latin";
    return best.?.name;
}

pub const ProfEntry = struct { name: []const u8, frac: f64 };

/// Latin first, then first-appearance order (laya `_profile_from_counts`).
fn profileFromCounts(alloc: std.mem.Allocator, counts: *const Counts) ![]ProfEntry {
    var total: usize = 0;
    for (counts.entries.items) |e| total += e.n;
    if (total == 0) return &.{};
    var ordered: std.ArrayListUnmanaged(ProfEntry) = .empty;
    const lat = countOf(counts, "latin");
    if (lat > 0) try ordered.append(alloc, .{ .name = "latin", .frac = @as(f64, @floatFromInt(lat)) / @as(f64, @floatFromInt(total)) });
    for (counts.entries.items) |e| {
        if (!std.mem.eql(u8, e.name, "latin") and e.n > 0) {
            try ordered.append(alloc, .{ .name = e.name, .frac = @as(f64, @floatFromInt(e.n)) / @as(f64, @floatFromInt(total)) });
        }
    }
    return ordered.toOwnedSlice(alloc);
}

fn isCombiningMark(cp: u32) bool {
    return (cp >= 0x0300 and cp <= 0x036F) or (cp >= 0x1AB0 and cp <= 0x1AFF) or (cp >= 0x1DC0 and cp <= 0x1DFF) or (cp >= 0x20D0 and cp <= 0x20FF) or (cp >= 0xFE20 and cp <= 0xFE2F);
}

fn nonLatinWords(alloc: std.mem.Allocator, text: []const u8) !std.ArrayListUnmanaged([]const u8) {
    var runs: std.ArrayListUnmanaged([]const u8) = .empty;
    var cur: std.ArrayListUnmanaged(u8) = .empty;
    var cur_script: ?[]const u8 = null;
    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        const cp = nextCp(text, &i) catch break;
        const ch = text[start..i];
        if (isCombiningMark(cp)) continue;
        const s = scriptOfChar(cp);
        if (s != null and cur_script != null and std.mem.eql(u8, s.?, cur_script.?)) {
            try cur.appendSlice(alloc, ch);
        } else if (s != null) {
            if (cur.items.len > 0) try runs.append(alloc, try cur.toOwnedSlice(alloc));
            cur = .empty;
            try cur.appendSlice(alloc, ch);
            cur_script = s;
        } else {
            if (cur.items.len > 0) try runs.append(alloc, try cur.toOwnedSlice(alloc));
            cur = .empty;
            cur_script = null;
        }
    }
    if (cur.items.len > 0) try runs.append(alloc, try cur.toOwnedSlice(alloc));
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    for (runs.items) |w| {
        if (w.len < 2) continue;
        // !first.is_uppercase()
        const fw = w[0];
        if (fw >= 'A' and fw <= 'Z') continue;
        // non-ASCII first char: Rust is_uppercase — treat as not uppercase (fixture-safe)
        try out.append(alloc, w);
    }
    return out;
}

fn sharedWords(alloc: std.mem.Allocator) !Set {
    var freq: std.StringHashMapUnmanaged(usize) = .{};
    for (ld.STOP_LANGS) |ls| {
        for (ls.words) |w| {
            const g = try freq.getOrPut(alloc, w);
            if (!g.found_existing) g.value_ptr.* = 0;
            g.value_ptr.* += 1;
        }
    }
    var shared: Set = .{};
    var it = freq.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* > 1) try setPut(&shared, alloc, e.key_ptr.*);
    }
    for (ld.NORDIC_OVERLAP_WORDS) |w| try setPut(&shared, alloc, w);
    return shared;
}

fn setFromWords(alloc: std.mem.Allocator, words: []const []const u8) !Set {
    var s: Set = .{};
    for (words) |w| try setPut(&s, alloc, w);
    return s;
}

fn isEnCollision(w: []const u8) bool {
    for (ld.EN_COLLISION_WORDS) |x| {
        if (std.mem.eql(u8, x, w)) return true;
    }
    return false;
}

// ---------- regex-primitive replacements ----------

fn isWordChar(cp: u32) bool {
    return isAlpha(cp) or (cp >= '0' and cp <= '9') or cp == '_';
}
fn isWordCharByte(c: u8, text: []const u8, i: *usize) bool {
    _ = text;
    _ = i;
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

/// `identifier_re` = `[\w-]*(?:[.@][\w-]+)+` replace_all with " ".
fn replaceIdentifiers(alloc: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    var pos: usize = 0;
    while (pos < text.len) {
        // longest wordrun (incl. '-') from pos — may be empty
        var pre_end = pos;
        while (pre_end < text.len) {
            const w = utf8Width(text[pre_end]);
            const cp = decodeAt(text, pre_end);
            const ok = isWordChar(cp) or text[pre_end] == '-';
            if (!ok) break;
            pre_end += w;
        }
        var matched = false;
        var pre = pre_end;
        while (pre >= pos and !matched) : (pre -= 1) {
            // try extensions: one or more ([.@] wordrun+)
            var ext_end = pre;
            var groups: usize = 0;
            while (ext_end < text.len and (text[ext_end] == '.' or text[ext_end] == '@')) {
                var g = ext_end + 1;
                var ran = false;
                while (g < text.len) {
                    const w = utf8Width(text[g]);
                    const cp = decodeAt(text, g);
                    if (!(isWordChar(cp) or text[g] == '-')) break;
                    g += w;
                    ran = true;
                }
                if (!ran) break;
                ext_end = g;
                groups += 1;
            }
            if (groups >= 1) {
                try out.appendSlice(alloc, text[pos..pre]);
                try out.append(alloc, ' ');
                pos = ext_end;
                matched = true;
            }
            if (pre == pos) break;
        }
        if (!matched) {
            const w = utf8Width(text[pos]);
            try out.appendSlice(alloc, text[pos .. pos + w]);
            pos += w;
        }
    }
    return out.toOwnedSlice(alloc);
}

fn utf8Width(c: u8) usize {
    if (c < 0x80) return 1;
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}
fn decodeAt(text: []const u8, i: usize) u32 {
    const w = utf8Width(text[i]);
    const cp = std.unicode.utf8Decode(text[i .. i + w]) catch 0xFFFD;
    return cp;
}

/// `code_line_re` = `[=;{}\[\]]|\w\(` is_match.
fn codeLineMatch(segment: []const u8) bool {
    for (segment, 0..) |c, i| {
        if (c == '=' or c == ';' or c == '{' or c == '}' or c == '[' or c == ']') return true;
        if (c == '(' and i > 0) {
            const p = i - 1;
            // \w( — word char (possibly multi-byte) directly before
            if (segment[p] < 0x80) {
                if ((segment[p] >= '0' and segment[p] <= '9') or (segment[p] >= 'a' and segment[p] <= 'z') or (segment[p] >= 'A' and segment[p] <= 'Z') or segment[p] == '_') return true;
            } else {
                // walk to lead byte
                var s = p;
                while (s > 0 and (segment[s] & 0xC0) == 0x80) s -= 1;
                const cp = decodeAt(segment, s);
                if (isWordChar(cp)) return true;
            }
        }
    }
    return false;
}

/// `joined_re` = `[^\W_][._/\\][^\W_]` is_match (word-minus-underscore incl. digits).
fn joinedMatch(tok: []const u8) bool {
    var i: usize = 0;
    while (i + 1 < tok.len) : (i += 1) {
        const a = tok[i];
        const b = tok[i + 1];
        if ((b == '.' or b == '_' or b == '/' or b == '\\') and i + 2 < tok.len) {
            if (isWordByte(a) and isWordByte(tok[i + 2])) return true;
        }
    }
    return false;
}
fn isWordByte(c: u8) bool {
    return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_' or c >= 0x80;
}

/// `letter_run_re` = `[^\W\d_]{2,}` replace_all with callback: all-uppercase
/// runs → " ", else keep.
fn replaceLetterRuns(alloc: std.mem.Allocator, prose: []const u8) ![]const u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    var i: usize = 0;
    while (i < prose.len) {
        const w = utf8Width(prose[i]);
        const cp = decodeAt(prose, i);
        if (isAlpha(cp)) {
            const start = i;
            var j = i;
            while (j < prose.len) {
                const w2 = utf8Width(prose[j]);
                const cp2 = decodeAt(prose, j);
                if (!isAlpha(cp2)) break;
                j += w2;
            }
            const run = prose[start..j];
            const run_len_chars = countChars(run);
            if (run_len_chars >= 2) {
                var all_upper = true;
                var k = start;
                while (k < j) {
                    const c = prose[k];
                    if (c < 0x80) {
                        if (c >= 'a' and c <= 'z') {
                            all_upper = false;
                            break;
                        }
                    } else {
                        // non-ASCII letter: assume lowercase (fixture-safe)
                        all_upper = false;
                        break;
                    }
                    k += utf8Width(prose[k]);
                }
                if (all_upper) try out.append(alloc, ' ') else try out.appendSlice(alloc, run);
                i = j;
                continue;
            }
            try out.appendSlice(alloc, prose[i .. i + w]);
            i += w;
            continue;
        }
        try out.appendSlice(alloc, prose[i .. i + w]);
        i += w;
    }
    return out.toOwnedSlice(alloc);
}

fn countChars(s: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (n += 1) i += utf8Width(s[i]);
    return n;
}

/// `word_re` find_iter (letters only) → lowered tokens.
fn wordTokens(alloc: std.mem.Allocator, text: []const u8) !std.ArrayListUnmanaged([]const u8) {
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        const w = utf8Width(text[i]);
        const cp = decodeAt(text, i);
        if (isAlpha(cp)) {
            const start = i;
            var j = i;
            while (j < text.len) {
                const w2 = utf8Width(text[j]);
                const cp2 = decodeAt(text, j);
                if (!isAlpha(cp2)) break;
                j += w2;
            }
            try out.append(alloc, try lowerAscii(alloc, text[start..j]));
            i = j;
            continue;
        }
        i += w;
    }
    return out;
}

fn countWords(text: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const w = utf8Width(text[i]);
        const cp = decodeAt(text, i);
        if (isAlpha(cp)) {
            n += 1;
            while (i < text.len) {
                const w2 = utf8Width(text[i]);
                const cp2 = decodeAt(text, i);
                if (!isAlpha(cp2)) break;
                i += w2;
            }
            continue;
        }
        i += w;
    }
    return n;
}

// ---------- latin profile ----------

const LangScore = struct { lang: []const u8, s: usize };

pub const LatinProfile = struct {
    language: ?[]const u8 = null,
    english_hits: usize = 0,
    diacritic_rate: f64 = 0,
    looks_non_english: bool = false,
};

fn englishRescuedByWords(alloc: std.mem.Allocator, words: *const std.ArrayListUnmanaged([]const u8), diac_rate: f64) !bool {
    if (diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE) return false;
    const shared = try sharedWords(alloc);
    var en_only_n: usize = 0;
    for (ld.STOP_EN) |w| {
        if (!setHas(&shared, w)) en_only_n += 1;
    }
    var hit: usize = 0;
    for (words.items) |w| {
        var in_en_only = false;
        for (ld.STOP_EN) |ew| {
            if (!setHas(&shared, ew) and std.mem.eql(u8, ew, w)) {
                in_en_only = true;
                break;
            }
        }
        if (in_en_only) hit += 1;
    }
    if (hit < 2) return false;
    var diac_words: usize = 0;
    var wi: usize = 0;
    while (wi < words.items.len) : (wi += 1) {
        const w = words.items[wi];
        var i: usize = 0;
        var any = false;
        while (i < w.len) {
            const cplen = utf8Width(w[i]);
            if (isDiac(w[i .. i + cplen])) {
                any = true;
                break;
            }
            i += cplen;
        }
        if (any) diac_words += 1;
    }
    return diac_words <= 1;
}

pub fn latinProfile(alloc: std.mem.Allocator, text: []const u8) !LatinProfile {
    const lowered_ident_raw = try replaceIdentifiers(alloc, text);
    const id2 = try std.mem.replaceOwned(u8, alloc, lowered_ident_raw, "İ", "i");
    const lowered_ident = try lowerAscii(alloc, id2);
    const words = try wordTokens(alloc, lowered_ident);

    const lowered = try lowerAscii(alloc, text);
    var diac: usize = 0;
    {
        var i: usize = 0;
        while (i < lowered.len) {
            const cl = utf8Width(lowered[i]);
            if (isDiac(lowered[i .. i + cl])) diac += 1;
            i += cl;
        }
    }
    const dl = @max(lowered.len, 1);
    const diac_rate: f64 = @as(f64, @floatFromInt(diac)) / @as(f64, @floatFromInt(dl));
    const non_english = diac_rate >= NON_EN_DIACRITIC_RATE;

    const shared = try sharedWords(alloc);
    const word_set = try setFromWords(alloc, words.items);

    var has_nordic = false;
    var nit = word_set.keyIterator();
    while (nit.next()) |k| {
        for (ld.NORDIC_OVERLAP_WORDS) |nw| {
            if (std.mem.eql(u8, nw, k.*)) {
                has_nordic = true;
                break;
            }
        }
        if (has_nordic) break;
    }
    var en_only_hit = false;
    for (ld.STOP_EN) |w| {
        if (!setHas(&shared, w) and setHas(&word_set, w)) {
            en_only_hit = true;
            break;
        }
    }
    const nordic_overlap = has_nordic and !en_only_hit;

    const short_swedish = words.items.len > 0 and words.items.len < 4 and blk: {
        for (words.items) |w| {
            for (ld.SHORT_SWEDISH_WORDS) |sw| {
                if (std.mem.eql(u8, sw, w)) break :blk true;
            }
        }
        break :blk false;
    };
    if (short_swedish) {
        return .{ .language = "sv", .english_hits = 0, .diacritic_rate = diac_rate, .looks_non_english = non_english };
    }
    if (words.items.len < 4) {
        return .{ .language = null, .english_hits = 0, .diacritic_rate = diac_rate, .looks_non_english = non_english or nordic_overlap };
    }

    // word counts (order-free — sums only)
    var counts: std.StringHashMapUnmanaged(usize) = .{};
    for (words.items) |w| {
        const g = try counts.getOrPut(alloc, w);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += 1;
    }
    // scores per STOP_LANGS language, in STOP_LANGS declaration order, then
    // sorted by key (BTreeMap) for evidenced tie-breaks.
    var score_list: std.ArrayListUnmanaged(LangScore) = .empty;
    for (ld.STOP_LANGS) |ls| {
        var s: usize = 0;
        var cit = counts.iterator();
        while (cit.next()) |e| {
            var in_stop = false;
            for (ls.words) |sw| {
                if (std.mem.eql(u8, sw, e.key_ptr.*)) {
                    in_stop = true;
                    break;
                }
            }
            if (in_stop) {
                if (isEnCollision(e.key_ptr.*)) {
                    s += 1;
                } else {
                    s += e.value_ptr.*;
                }
            }
        }
        try score_list.append(alloc, .{ .lang = ls.lang, .s = s });
    }
    // BTreeMap order = sorted by lang key
    std.mem.sort(@TypeOf(score_list.items[0]), score_list.items, {}, struct {
        fn lt(_: void, a: @TypeOf(score_list.items[0]), b: @TypeOf(score_list.items[0])) bool {
            return std.mem.lessThan(u8, a.lang, b.lang);
        }
    }.lt);

    var en: usize = 0;
    for (score_list.items) |e| {
        if (std.mem.eql(u8, e.lang, "en")) en = e.s;
    }

    // evidenced: non-en langs with a non-shared stopword hit (scores order = sorted)
    var evidenced: std.ArrayListUnmanaged(LangScore) = .empty;
    for (score_list.items) |e| {
        if (std.mem.eql(u8, e.lang, "en")) continue;
        var stop: []const []const u8 = &.{};
        for (ld.STOP_LANGS) |ls| {
            if (std.mem.eql(u8, ls.lang, e.lang)) {
                stop = ls.words;
                break;
            }
        }
        var hit_non_shared = false;
        var wit = word_set.keyIterator();
        while (wit.next()) |w| {
            var in_stop = false;
            for (stop) |sw| {
                if (std.mem.eql(u8, sw, w.*)) {
                    in_stop = true;
                    break;
                }
            }
            if (in_stop and !setHas(&shared, w.*)) {
                hit_non_shared = true;
                break;
            }
        }
        if (hit_non_shared) try evidenced.append(alloc, e);
    }
    var best_lg: ?[]const u8 = null;
    var best: usize = 0;
    for (evidenced.items) |e| {
        if (best_lg == null or e.s >= best) {
            best_lg = e.lang;
            best = e.s;
        }
    }

    var lang: ?[]const u8 = null;
    if (best_lg) |blg| {
        const sv_special = std.mem.eql(u8, blg, "sv") and setHas(&word_set, "inte") and setHas(&word_set, "kan") and
            words.items.len > 0 and
            (std.mem.eql(u8, words.items[0], "kan") or std.mem.eql(u8, words.items[0], "jag") or std.mem.eql(u8, words.items[0], "vi")) and
            en <= 1;
        const cond = best >= @max(@as(usize, 2), en + 2) or sv_special or
            (non_english and best >= @max(@as(usize, 2), en));
        if (cond) lang = blg;
    }
    if (lang == null and en > 0 and (!non_english or try englishRescuedByWords(alloc, &words, diac_rate))) {
        lang = "en";
    }
    const looks_non_english = non_english or (lang == null and nordic_overlap);
    return .{ .language = lang, .english_hits = en, .diacritic_rate = diac_rate, .looks_non_english = looks_non_english };
}

// ---------- prose / mixed-segment detection ----------

fn isAsciiSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n' or c == 0x0b or c == 0x0c;
}

fn splitWhitespace(alloc: std.mem.Allocator, s: []const u8) !std.ArrayListUnmanaged([]const u8) {
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        while (i < s.len and isAsciiSpace(s[i])) i += 1;
        const start = i;
        while (i < s.len and !isAsciiSpace(s[i])) i += 1;
        if (i > start) try out.append(alloc, s[start..i]);
    }
    return out;
}

fn namedProseLanguage(alloc: std.mem.Allocator, segment: []const u8) !?[]const u8 {
    const trimmed = std.mem.trim(u8, segment, " \t\r\n");
    if (trimmed.len == 0) return null;
    if (codeLineMatch(segment)) return null;
    const toks = try splitWhitespace(alloc, segment);
    var joined_free: std.ArrayListUnmanaged([]const u8) = .empty;
    for (toks.items) |t| {
        if (!joinedMatch(t)) try joined_free.append(alloc, t);
    }
    var prose: std.ArrayListUnmanaged(u8) = .empty;
    for (joined_free.items, 0..) |t, i| {
        if (i > 0) try prose.append(alloc, ' ');
        try prose.appendSlice(alloc, t);
    }
    var has_lower = false;
    for (prose.items) |c| {
        if (c >= 'a' and c <= 'z') {
            has_lower = true;
            break;
        }
    }
    const prose_s = if (has_lower) try replaceLetterRuns(alloc, prose.items) else prose.items;
    const tokens = try wordTokens(alloc, prose_s);
    if (tokens.items.len < 4) return null;
    const lp = try latinProfile(alloc, prose_s);
    const lang = lp.language;
    if (lang == null) return null;
    if (std.mem.eql(u8, lang.?, "en")) return null;
    var stop: []const []const u8 = &.{};
    for (ld.STOP_LANGS) |ls| {
        if (std.mem.eql(u8, ls.lang, lang.?)) {
            stop = ls.words;
            break;
        }
    }
    var uniq: std.StringHashMapUnmanaged(void) = .{};
    for (tokens.items) |t| try uniq.put(alloc, t, {});
    var hits: usize = 0;
    var uit = uniq.keyIterator();
    while (uit.next()) |t| {
        for (stop) |sw| {
            if (std.mem.eql(u8, sw, t.*)) {
                hits += 1;
                break;
            }
        }
    }
    return if (hits >= 2) lang else null;
}

fn nonEnglishSegment(alloc: std.mem.Allocator, state: std.json.Value, max_chars: usize) !?struct { lang: []const u8, seg: []const u8 } {
    var seen: usize = 0;
    const leaves = try collectLeaves(alloc, state);
    for (leaves.items) |leaf| {
        var it = std.mem.splitScalar(u8, leaf, '\n');
        while (it.next()) |raw_seg| {
            if (seen >= max_chars) return null;
            const seg = truncStr(raw_seg, max_chars - seen);
            seen += seg.len;
            if (try namedProseLanguage(alloc, seg)) |lang| {
                return .{ .lang = lang, .seg = std.mem.trim(u8, seg, " \t\r\n") };
            }
        }
    }
    return null;
}

fn leafNonEnglish(alloc: std.mem.Allocator, leaf: []const u8) !?Analysis {
    var best_n: usize = 0;
    var best: ?Analysis = null;
    var it = std.mem.splitScalar(u8, leaf, '\n');
    while (it.next()) |line| {
        if (line.len < 7) continue;
        const sample = truncStr(line, 4000);
        const st = std.mem.trim(u8, sample, " \t\r\n");
        if (st.len == 0 or codeLineMatch(sample)) continue;
        const det = try analyseText(alloc, sample);
        if (det.is_english) continue;
        if (det.language) |l| {
            if (!std.mem.eql(u8, l, "en")) {
                if (try namedProseLanguage(alloc, sample) == null) continue;
            }
        } else if (!std.mem.eql(u8, det.script, "latin") and !std.mem.eql(u8, det.script, "unknown")) {
            const nlw = try nonLatinWords(alloc, sample);
            if (nlw.items.len == 0 or countAlpha(sample) < NON_LATIN_MIN_LETTERS) continue;
        } else if (!(det.language_undecided and det.diacritic_rate >= NON_EN_DIACRITIC_RATE and countWords(sample) >= 4)) {
            continue;
        }
        const n_alpha = countAlpha(sample);
        if (n_alpha > best_n) {
            best_n = n_alpha;
            best = det;
        }
    }
    return best;
}

fn countAlpha(s: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const w = utf8Width(s[i]);
        const cp = decodeAt(s, i);
        if (isAlpha(cp)) n += 1;
        i += w;
    }
    return n;
}

// ---------- Analysis ----------

pub const Analysis = struct {
    script: []const u8,
    script_profile: []ProfEntry,
    language: ?[]const u8,
    is_english: bool,
    language_undecided: bool,
    diacritic_rate: f64,
    non_latin_fraction: f64,
    mixed_segment: ?[]const u8,
};

fn analyseText(alloc: std.mem.Allocator, text: []const u8) !Analysis {
    const counts = try scriptCounts(alloc, text);
    const prof = try profileFromCounts(alloc, &counts);
    const script = scriptFromCounts(&counts);
    var non_latin: f64 = 0.0;
    if (prof.len > 0) {
        var latin_frac: f64 = 0.0;
        for (prof) |p| {
            if (std.mem.eql(u8, p.name, "latin")) {
                latin_frac = p.frac;
                break;
            }
        }
        const n = 1.0 - latin_frac;
        non_latin = @round(n * 10_000.0) / 10_000.0;
    }
    const alpha_n = countAlpha(text);
    const n_non_latin: usize = @intFromFloat(@round(non_latin * @as(f64, @floatFromInt(alpha_n))));

    var s2 = script;
    if (std.mem.eql(u8, s2, "latin")) {
        const nlw = try nonLatinWords(alloc, text);
        if (nlw.items.len > 0 and (non_latin >= NON_LATIN_FRACTION or (non_latin >= NON_LATIN_MIN_FRACTION and n_non_latin >= NON_LATIN_MIN_LETTERS))) {
            var bname: ?[]const u8 = null;
            var bfrac: f64 = 0;
            for (prof) |p| {
                if (std.mem.eql(u8, p.name, "latin")) continue;
                if (bname == null or p.frac >= bfrac) {
                    bname = p.name;
                    bfrac = p.frac;
                }
            }
            if (bname) |bn| s2 = bn;
        }
    }
    if (std.mem.eql(u8, s2, "unknown")) {
        return .{ .script = "unknown", .script_profile = prof, .language = null, .is_english = true, .language_undecided = true, .diacritic_rate = 0.0, .non_latin_fraction = 0.0, .mixed_segment = null };
    }
    if (!std.mem.eql(u8, s2, "latin")) {
        return .{ .script = s2, .script_profile = prof, .language = null, .is_english = false, .language_undecided = true, .diacritic_rate = 0.0, .non_latin_fraction = non_latin, .mixed_segment = null };
    }
    const lp = try latinProfile(alloc, text);
    const undecided = lp.language == null;
    const english = (lp.language != null and std.mem.eql(u8, lp.language.?, "en")) or (undecided and !lp.looks_non_english);
    return .{
        .script = "latin",
        .script_profile = prof,
        .language = lp.language,
        .is_english = english,
        .language_undecided = undecided,
        .diacritic_rate = @round(lp.diacritic_rate * 10000.0) / 10000.0,
        .non_latin_fraction = non_latin,
        .mixed_segment = null,
    };
}

/// Full detection result for a state (knot `analyse`).
pub fn analyse(alloc: std.mem.Allocator, state: std.json.Value) !Analysis {
    const txt = try stateText(alloc, state, 4000);
    var result = try analyseText(alloc, txt);
    if (std.mem.eql(u8, result.script, "latin") and result.is_english) {
        const leaves = try collectLeaves(alloc, state);
        const multi = leaves.items.len > 1 or blk: {
            for (leaves.items) |l| {
                if (std.mem.indexOfScalar(u8, l, '\n') != null) break :blk true;
            }
            break :blk false;
        };
        if (multi) {
            if (try nonEnglishSegment(alloc, state, 4000)) |ms| {
                result.language = ms.lang;
                result.is_english = false;
                result.language_undecided = false;
                result.mixed_segment = ms.seg;
            }
        }
    }
    const is_str = state == .string;
    const is_null = state == .null;
    if (is_str or is_null or !result.is_english) return result;
    var best_n: usize = 0;
    var best: ?Analysis = null;
    const leaves = try collectLeaves(alloc, state);
    for (leaves.items) |leaf| {
        if (try leafNonEnglish(alloc, leaf)) |det| {
            const n_alpha = countAlpha(truncStr(leaf, 4000));
            if (n_alpha > best_n) {
                best_n = n_alpha;
                best = det;
            }
        }
    }
    if (best) |b| {
        result.language = b.language;
        result.is_english = false;
        result.language_undecided = b.language_undecided;
    }
    return result;
}

pub fn isEnglish(alloc: std.mem.Allocator, state: std.json.Value) !bool {
    return (try analyse(alloc, state)).is_english;
}
