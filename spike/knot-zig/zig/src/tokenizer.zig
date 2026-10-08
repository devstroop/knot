//! Byte-level BPE tokenizer — port of the HF `tokenizers` 0.21 encode path
//! for knot's exact call: `Tokenizer::encode(text, add_special_tokens=false)`
//! (knot/crates/knot/src/prompt.rs `encode_ids`).
//!
//! Faithful pieces: added-token split (longest match first), ByteLevel
//! pre-tokenizer with the fixed GPT-2 regex pattern (`use_regex=true`),
//! bytes→unicode mapping, min-rank BPE (all occurrences merged per rank
//! step), vocab lookup. TemplateProcessing is skipped because knot encodes
//! with `add_special_tokens=false`.
//!
//! Known gaps the spike measures instead of hides:
//! - NFC normalizer: implemented (nfc.zig + generated nfc_tables.zig,
//!   gated by the nfc_corpus.json cross-check against Python's NFC).
//! - `\p{L}` / `\p{N}` / `\s` = FULL UCD range tables generated from Python's
//!   unicodedata (class_tables.zig: L*, N*, White_Space) — no curated subsets.
const std = @import("std");
const nfc = @import("nfc.zig");
const tables = @import("class_tables.zig");

const Added = struct { id: u32, content: []const u8 };

pub const Tokenizer = struct {
    arena: std.mem.Allocator,
    value: ?std.json.Value = null,
    vocab: std.StringHashMapUnmanaged(u32) = .{},
    merges: std.StringHashMapUnmanaged(u32) = .{},
    added: std.ArrayListUnmanaged(Added) = .empty,
    byte_map: [256]u21 = undefined,
    trace: bool = false,

    pub fn init(arena_alloc: std.mem.Allocator) Tokenizer {
        var self: Tokenizer = .{ .arena = arena_alloc };
        buildByteMap(&self.byte_map);
        return self;
    }

    pub fn load(self: *Tokenizer, text: []const u8) !void {
        const alloc = self.arena;
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, text, .{});
        self.value = parsed.value; // strings live in `alloc`; process-lifetime
        const root = self.value.?;
        const model = root.object.get("model").?;

        const vocab_v = model.object.get("vocab").?;
        var vit = vocab_v.object.iterator();
        while (vit.next()) |e| {
            try self.vocab.put(alloc, e.key_ptr.*, @intCast(e.value_ptr.*.integer));
        }

        const merges_v = model.object.get("merges").?;
        for (merges_v.array.items, 0..) |pair, idx| {
            const a = pair.array.items[0].string;
            const b = pair.array.items[1].string;
            const key = try std.fmt.allocPrint(alloc, "{s} {s}", .{ a, b });
            try self.merges.put(alloc, key, @intCast(idx));
        }

        if (root.object.get("added_tokens")) |ats| {
            for (ats.array.items) |a| {
                const id: u32 = @intCast(a.object.get("id").?.integer);
                const content = a.object.get("content").?.string;
                try self.added.append(alloc, .{ .id = id, .content = content });
            }
            // Longest-match-first, as HF's added-token trie behaves.
            std.mem.sort(Added, self.added.items, {}, struct {
                fn less(_: void, x: Added, y: Added) bool {
                    return x.content.len > y.content.len;
                }
            }.less);
        }
    }

    /// HF `token_to_id`: added tokens and the BPE vocab both consulted
    /// (the specials `[MASK]`/`<s>`/... may live in either).
    pub fn tokenToId(self: *Tokenizer, text: []const u8) ?u32 {
        for (self.added.items) |a| {
            if (std.mem.eql(u8, a.content, text)) return a.id;
        }
        return self.vocab.get(text);
    }

    /// Reverse lookup (`id_to_token`) for the mask-token string.
    pub fn idToToken(self: *Tokenizer, id: u32) ?[]const u8 {
        for (self.added.items) |a| {
            if (a.id == id) return a.content;
        }
        var it = self.vocab.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* == id) return e.key_ptr.*;
        }
        return null;
    }

    /// `encode(text, false)` — ids only, no offsets, no special tokens.
    pub fn encode(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8, out: *std.ArrayListUnmanaged(u32)) !void {
        // HF pipeline order: normalizer FIRST, then added-token split, then
        // pre-tokenizer + model.
        const norm = try nfc.normalize(alloc, text);
        var pos: usize = 0;
        while (pos < norm.len) {
            // Longest added-token match at pos (added list is sorted desc).
            var matched: ?Added = null;
            for (self.added.items) |a| {
                if (a.content.len > 0 and norm.len - pos >= a.content.len and
                    std.mem.eql(u8, norm[pos .. pos + a.content.len], a.content))
                {
                    matched = a;
                    break;
                }
            }
            if (matched) |a| {
                try out.append(alloc, a.id);
                pos += a.content.len;
                continue;
            }
            // Next added-token start (or end of text): the slice runs to the
            // next match so the pre-tokenizer sees a contiguous chunk.
            const next = nextAddedStart(self.added.items, norm, pos);
            try self.encodeText(alloc, norm[pos..next], out);
            pos = next;
        }
    }

    fn nextAddedStart(added: []const Added, text: []const u8, from: usize) usize {
        var best: ?usize = null;
        var i = from;
        while (i < text.len) : (i += 1) {
            for (added) |a| {
                if (a.content.len > 0 and text.len - i >= a.content.len and
                    std.mem.eql(u8, text[i .. i + a.content.len], a.content))
                {
                    if (best == null or i < best.?) best = i;
                    break;
                }
            }
        }
        return best orelse text.len;
    }

    fn encodeText(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8, out: *std.ArrayListUnmanaged(u32)) !void {
        if (text.len == 0) return;
        // UTF-8 -> codepoints (HF regexes operate on unicode scalars).
        var cps: std.ArrayListUnmanaged(u21) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            const bl = utf8LeadLen(text[i]) orelse return error.BadUtf8;
            if (i + bl > text.len) return error.BadUtf8;
            const cp = try std.unicode.utf8Decode(text[i .. i + bl]);
            try cps.append(alloc, cp);
            i += bl;
        }
        // GPT-2 pattern, leftmost-first, greedy per alternative.
        var start: usize = 0;
        var j: usize = 0;
        while (j < cps.items.len) {
            const end = matchAt(cps.items, j) orelse return error.NoPatternMatch;
            if (end <= j) return error.EmptyMatch;
            if (end > start) {
                try self.encodePiece(alloc, cps.items[start..end], out);
            }
            start = end;
            j = end;
        }
    }

    fn encodePiece(self: *Tokenizer, alloc: std.mem.Allocator, piece: []const u21, out: *std.ArrayListUnmanaged(u32)) !void {
        if (piece.len == 0) return;
        // ByteLevel: every UTF-8 byte of the piece -> bytes_to_unicode char,
        // as its own one-char symbol STRING (BPE symbols are strings —
        // merging concatenates, it never drops bytes).
        var seq: std.ArrayListUnmanaged([]const u8) = .empty;
        for (piece) |cp| {
            var b: [4]u8 = undefined;
            const bn = utf8Encode(cp, &b);
            for (b[0..bn]) |byte| {
                const mc = self.byte_map[byte];
                var mb: [4]u8 = undefined;
                const mn = utf8Encode(mc, &mb);
                const sym = try alloc.alloc(u8, mn);
                @memcpy(sym, mb[0..mn]);
                try seq.append(alloc, sym);
            }
        }
        if (self.trace) {
            var rendered: [96]u8 = undefined;
            var rn: usize = 0;
            for (piece) |cp| {
                var pb: [4]u8 = undefined;
                const n = utf8Encode(cp, &pb);
                if (rn + n <= rendered.len) {
                    @memcpy(rendered[rn .. rn + n], pb[0..n]);
                    rn += n;
                }
            }
            std.debug.print("  piece \"{s}\" symbols={d}\n", .{ rendered[0..rn], seq.items.len });
        }
        try self.bpe(alloc, seq.items, out);
    }

    fn bpe(self: *Tokenizer, alloc: std.mem.Allocator, symbols: []const []const u8, out: *std.ArrayListUnmanaged(u32)) !void {
        if (symbols.len == 0) return;
        var seq: std.ArrayListUnmanaged([]const u8) = .empty;
        try seq.appendSlice(alloc, symbols);
        while (seq.items.len > 1) {
            // Min-rank adjacent pair across the whole sequence.
            var best_rank: u32 = std.math.maxInt(u32);
            var best_i: usize = 0;
            var found = false;
            for (0..seq.items.len - 1) |k| {
                const key = try pairKey(alloc, seq.items[k], seq.items[k + 1]);
                if (self.merges.get(key)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_i = k;
                        found = true;
                    }
                }
            }
            if (!found) break;
            // Merge EVERY occurrence of that pair in one left-to-right pass
            // (HF BPE semantics: one rank step, all occurrences); the merged
            // symbol is the CONCATENATION, not either side.
            const pa = seq.items[best_i];
            const pb = seq.items[best_i + 1];
            var merged: std.ArrayListUnmanaged([]const u8) = .empty;
            var k: usize = 0;
            while (k < seq.items.len) {
                if (k + 1 < seq.items.len and std.mem.eql(u8, seq.items[k], pa) and
                    std.mem.eql(u8, seq.items[k + 1], pb))
                {
                    const joined = try std.fmt.allocPrint(alloc, "{s}{s}", .{ pa, pb });
                    try merged.append(alloc, joined);
                    k += 2;
                } else {
                    try merged.append(alloc, seq.items[k]);
                    k += 1;
                }
            }
            seq = merged; // superseded buffer leaks into the scratch arena
        }
        for (seq.items) |sym| {
            if (self.vocab.get(sym)) |id| {
                try out.append(alloc, id);
            } else {
                return error.UnknownToken;
            }
        }
    }

    fn pairKey(alloc: std.mem.Allocator, a: []const u8, b: []const u8) ![]const u8 {
        return std.fmt.allocPrint(alloc, "{s} {s}", .{ a, b });
    }
};

fn utf8LeadLen(b: u8) ?usize {
    if (b < 0x80) return 1;
    if (b & 0xE0 == 0xC0) return 2;
    if (b & 0xF0 == 0xE0) return 3;
    if (b & 0xF8 == 0xF0) return 4;
    return null;
}

fn utf8Encode(cp: u21, out: *[4]u8) usize {
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

fn buildByteMap(map: *[256]u21) void {
    // GPT-2 bytes_to_unicode: printable ranges map to themselves, the rest
    // to 256, 257, … in ascending byte order.
    var present: [256]bool = @splat(false);
    var cp: u32 = 33;
    while (cp <= 126) : (cp += 1) present[cp] = true; // '!'..'~'
    cp = 161;
    while (cp <= 172) : (cp += 1) present[cp] = true; // '¡'..'¬'
    cp = 174;
    while (cp <= 255) : (cp += 1) present[cp] = true; // '®'..'ÿ'
    var n: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        if (present[b]) {
            map[b] = @intCast(b);
        } else {
            map[b] = @intCast(256 + n);
            n += 1;
        }
    }
}

// --- character classes (curated ranges; UCD-complete = documented gap) ---

fn inRanges(ranges: []const [2]u21, cp: u21) bool {
    for (ranges) |r| {
        if (cp < r[0]) return false; // sorted: early exit
        if (cp <= r[1]) return true;
    }
    return false;
}

fn isWs(cp: u21) bool {
    return inRanges(tables.ws_ranges, cp);
}

fn isCrLf(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

fn isL(cp: u21) bool {
    return inRanges(tables.l_ranges, cp);
}

fn isN(cp: u21) bool {
    return inRanges(tables.n_ranges, cp);
}

fn isNonCrLfLN(cp: u21) bool {
    return !isCrLf(cp) and !isL(cp) and !isN(cp);
}

fn isPunctClass(cp: u21) bool {
    return !isWs(cp) and !isL(cp) and !isN(cp); // [^\s\p{L}\p{N}]
}

// --- GPT-2 pattern matching (leftmost-first, greedy alternatives) ---

fn contractionsAt(cps: []const u21, i: usize) ?usize {
    if (cps[i] != '\'') return null;
    const pats = [_][]const u21{ &[_]u21{'s'}, &[_]u21{'t'}, &[_]u21{ 'r', 'e' }, &[_]u21{ 'v', 'e' }, &[_]u21{'m'}, &[_]u21{ 'l', 'l' }, &[_]u21{'d'} };
    for (pats) |p| {
        if (i + 1 + p.len > cps.len) continue;
        var ok = true;
        for (p, 0..) |c, k| {
            const got = cps[i + 1 + k];
            const want_lower: u21 = if (c >= 'A' and c <= 'Z') c + 32 else c;
            const got_lower: u21 = if (got >= 'A' and got <= 'Z') got + 32 else got;
            if (got_lower != want_lower) {
                ok = false;
                break;
            }
        }
        if (ok) return i + 1 + p.len;
    }
    return null;
}

fn punctRunAt(cps: []const u21, j: usize) ?usize {
    const n = cps.len;
    if (j >= n or !isPunctClass(cps[j])) return null;
    var e = j + 1;
    while (e < n and isPunctClass(cps[e])) e += 1;
    while (e < n and isCrLf(cps[e])) e += 1; // trailing [\r\n]*
    return e;
}

fn matchAt(cps: []const u21, i: usize) ?usize {
    const n = cps.len;
    // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (contractionsAt(cps, i)) |e| return e;
    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    {
        if (isL(cps[i])) {
            var j = i;
            while (j < n and isL(cps[j])) j += 1;
            return j;
        } else if (isNonCrLfLN(cps[i]) and i + 1 < n and isL(cps[i + 1])) {
            // greedy optional consumes the non-letter, then L+
            var j = i + 1;
            while (j < n and isL(cps[j])) j += 1;
            return j;
        }
        // alternative fails: fall through to 3)
    }
    // 3) ` ?\p{N}+` — derived from HF's own offsets (NOT the classic
    // \p{N}{1,3}): tokens show ' 0'|'12345'|'6789' and ' 42'|' 100', i.e. an
    // optional space glued to a maximal digit run.
    {
        var j: usize = i;
        if (cps[i] == ' ' and i + 1 < n and isN(cps[i + 1])) j = i + 1;
        if (isN(cps[j])) {
            while (j < n and isN(cps[j])) j += 1;
            if (j > i) return j;
        }
    }
    // 4) ?[^\s\p{L}\p{N}]+[\r\n]*   (greedy optional space first)
    if (cps[i] == ' ') {
        if (punctRunAt(cps, i + 1)) |e| return e;
    }
    if (punctRunAt(cps, i)) |e| return e;
    // 5) \s+(?!\S) — tokenizers 0.21 evaluates this BEFORE the crlf-run
    //    alternative: greedy to end-of-input ('\n\n' at EOS merges), else
    //    backtracks to all-but-last ('\n\n[' splits into '\n' + '\n').
    //    Proven by the oracle probes: '\n\n'->ĊĊ but '\n\n['->Ċ,Ċ,'['.
    {
        var j = i;
        while (j < n and isWs(cps[j])) j += 1;
        if (j > i) {
            if (j == n) return j;
            const e = j - 1;
            if (e > i) return e;
        }
    }
    // 6) \s*[\r\n]+  (longest end over all crlf block starts)
    {
        var j = i;
        while (j < n and isWs(cps[j])) j += 1;
        if (j > i) {
            var best: ?usize = null;
            var k = i;
            while (k < j) : (k += 1) {
                if (isCrLf(cps[k])) {
                    var e = k;
                    while (e < j and isCrLf(cps[e])) e += 1;
                    if (best == null or e > best.?) best = e;
                }
            }
            if (best) |e| return e;
        }
    }
    // 7) \s+
    {
        var j = i;
        while (j < n and isWs(cps[j])) j += 1;
        if (j > i) return j;
    }
    return null;
}

test "byte map anchors" {
    var m: [256]u21 = undefined;
    buildByteMap(&m);
    try std.testing.expectEqual(@as(u21, 288), m[32]); // ' ' -> Ġ
    try std.testing.expectEqual(@as(u21, 256), m[0]); // NUL -> U+0100
    try std.testing.expectEqual(@as(u21, 33), m[33]); // '!' -> itself
    try std.testing.expectEqual(@as(u21, 172), m[172]); // '¬' -> itself
}

test "pattern: contractions and words" {
    const s = [_]u21{ 'd', 'o', 'n', '\'', 't' }; // "don't", apostrophe at 3
    try std.testing.expectEqual(@as(?usize, 5), contractionsAt(&s, 3)); // 't -> end 5
    try std.testing.expectEqual(@as(?usize, null), contractionsAt(&s, 1)); // 'o' not a quote
}
