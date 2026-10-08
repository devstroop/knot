const std = @import("std");
const Io = std.Io;
const knot_tok_spike = @import("knot_tok_spike");
const tokenizer_mod = knot_tok_spike.tokenizer;
const nfc_mod = knot_tok_spike.nfc;

fn readAll(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try alloc.alloc(u8, st.size);
    const got = try file.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

fn containsNfdCombiningAcute(s: []const u8) bool {
    // U+0301 in UTF-8 = CC 81
    return std.mem.indexOf(u8, s, "\xCC\x81") != null;
}

/// Tokenizer parity driver: strings.json + goldens.json (from the Rust
/// oracle) vs this port. Exit 0 only if every id list matches.
pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var paths: [4][]const u8 = undefined;
    var np: usize = 0;
    for (args) |a| {
        if (std.mem.endsWith(u8, a, ".json") and np < 4) {
            paths[np] = a;
            np += 1;
        }
    }
    if (np != 3 and np != 4) {
        std.debug.print(
            "usage: knot_tok_spike <tokenizer.json> <strings.json> <goldens.json> [nfc_corpus.json]\n",
            .{},
        );
        std.process.exit(2);
    }

    const tok_text = try readAll(io, arena, paths[0]);
    const strings_text = try readAll(io, arena, paths[1]);
    const goldens_text = try readAll(io, arena, paths[2]);
    const corpus_text: ?[]const u8 = if (np == 4) try readAll(io, arena, paths[3]) else null;

    var tk = tokenizer_mod.Tokenizer.init(arena);
    try tk.load(tok_text);

    const strings_v = (try std.json.parseFromSlice(std.json.Value, arena, strings_text, .{})).value;
    const goldens_v = (try std.json.parseFromSlice(std.json.Value, arena, goldens_text, .{})).value;
    // Stage-level diff vs HF: trace pieces for the first corpus string only.
    if (goldens_v.array.items.len > 0) {
        const s0 = goldens_v.array.items[0].object.get("s").?.string;
        tk.trace = std.mem.startsWith(u8, s0, "Does the context");
    }

    var total: usize = 0;
    var passed: usize = 0;
    var nfc_gaps: usize = 0;
    for (goldens_v.array.items) |g| {
        const s = g.object.get("s").?.string;
        const want = g.object.get("ids").?.array.items;
        total += 1;

        var got: std.ArrayListUnmanaged(u32) = .empty;
        tk.encode(arena, s, &got) catch |e| {
            std.debug.print("FAIL(encode:{s}) {s}\n", .{ @errorName(e), s });
            continue;
        };

        var first_diff: ?usize = null;
        const n = @min(want.len, got.items.len);
        for (0..n) |k| {
            const w: u32 = @intCast(want[k].integer);
            if (w != got.items[k]) {
                first_diff = k;
                break;
            }
        }
        if (first_diff == null and want.len == got.items.len) {
            passed += 1;
            continue;
        }
        const want_tokens = g.object.get("tokens").?.array.items;
        const is_nfd = containsNfdCombiningAcute(s);
        if (is_nfd) nfc_gaps += 1;
        std.debug.print(
            "FAIL({s}) {s}\n  want n={d}: {any}\n  hf   : {any}\n  got  n={d}: {any}\n",
            .{
                if (is_nfd) "NFC-GAP:normalizer-not-ported" else "MISMATCH",
                s[0..@min(s.len, 70)],
                want.len,
                want[0..@min(want.len, 14)],
                want_tokens[0..@min(want_tokens.len, 14)],
                got.items.len,
                got.items[0..@min(got.items.len, 14)],
            },
        );
    }
    // Optional NFC cross-check corpus: [ {raw, nfc}, … ] generated from
    // Python's unicodedata — every entry must normalize identically.
    var nfc_ok: usize = 0;
    var nfc_bad: usize = 0;
    if (corpus_text) |ct| {
        const carr = (try std.json.parseFromSlice(std.json.Value, arena, ct, .{})).value;
        for (carr.array.items) |row| {
            const raw = row.object.get("raw").?.string;
            const want = row.object.get("nfc").?.string;
            const got = try nfc_mod.normalize(arena, raw);
            if (std.mem.eql(u8, got, want)) {
                nfc_ok += 1;
            } else {
                nfc_bad += 1;
                if (nfc_bad <= 5) {
                    std.debug.print(
                        "NFC-MISMATCH {s}\n  want {s}\n  got  {s}\n",
                        .{
                            raw[0..@min(raw.len, 40)],
                            want[0..@min(want.len, 40)],
                            got[0..@min(got.len, 40)],
                        },
                    );
                }
            }
        }
        std.debug.print("nfc_corpus: {d}/{d} match\n", .{ nfc_ok, nfc_ok + nfc_bad });
    }
    std.debug.print(
        "summary: {d}/{d} exact, {d} NFC-gap, strings={d} goldens={d}\n",
        .{ passed, total, nfc_gaps, strings_v.array.items.len, total },
    );
    _ = knot_tok_spike;
    // NFC is implemented now: any non-exact golden row (NFD probe included)
    // or corpus mismatch is a REAL failure. 0 = all exact + corpus clean.
    if (passed == total and nfc_bad == 0) {
        std.process.exit(0);
    } else {
        std.process.exit(1);
    }
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    try std.testing.fuzz({}, testOne, .{});
}

fn testOne(context: void, smith: *std.testing.Smith) !void {
    _ = context;
    // Try command `zig build test --fuzz -Doptimize=ReleaseFast` to see if it manages to fail this test case!

    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    while (!smith.eos()) switch (smith.value(enum { add_data, dup_data })) {
        .add_data => {
            const slice = try list.addManyAsSlice(gpa, smith.value(u4));
            smith.bytes(slice);
        },
        .dup_data => {
            if (list.items.len == 0) continue;
            if (list.items.len > std.math.maxInt(u32)) return error.SkipZigTest;
            const len = smith.valueRangeAtMost(u32, 1, @min(32, list.items.len));
            const off = smith.valueRangeAtMost(u32, 0, @intCast(list.items.len - len));
            try list.appendSlice(gpa, list.items[off..][0..len]);
            try std.testing.expectEqualSlices(
                u8,
                list.items[off..][0..len],
                list.items[list.items.len - len ..],
            );
        },
    };
}
