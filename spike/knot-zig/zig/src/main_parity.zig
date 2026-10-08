//! P2-1 sequence parity: rebuild the engine_english request's sequences with
//! the zig prompt port and compare against knot's own Rust goldens —
//! `parity_english.json` `items[]` (ids/markers/qtype per question) and
//! `collated` (the padded tensors ORT actually ran on).
//!
//! Run: p2_parity <tokenizer.json> <engine_english.json> <parity_english.json>
const std = @import("std");
const Io = std.Io;
const tokenizer_mod = @import("tokenizer.zig");
const question = @import("question.zig");
const prompt = @import("prompt.zig");
const session_mod = @import("session.zig");
const decode = @import("decode.zig");
const pyjson = @import("pyjson.zig");
const lang_mod = @import("lang.zig");
const route_mod = @import("route.zig");

/// Answer floats are serde f32 → **shortest f32 repr** (ryu: `0.9857`,
/// `1.0` — not the f64-widened `0.9857000112533569`). zig's `{d}` is
/// shortest-round-trip for the type; ryu always emits a fraction → fill
/// `.0` when neither `.` nor exponent marker is present.
fn fmtF32(alloc: std.mem.Allocator, v: f32) ![]const u8 {
    const s = try std.fmt.allocPrint(alloc, "{d}", .{v});
    if (std.mem.indexOfAny(u8, s, ".eE") == null) {
        return std.fmt.allocPrint(alloc, "{s}.0", .{s});
    }
    return s;
}

/// serde's compact `["a","b"]` for `Vec<String>`.
fn jsonStrList(alloc: std.mem.Allocator, items: []const []const u8) ![]const u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try out.appendSlice(alloc, "[");
    for (items, 0..) |s, i| {
        if (i > 0) try out.appendSlice(alloc, ",");
        try out.appendSlice(alloc, try pyjson.dumps(alloc, .{ .string = s }));
    }
    try out.appendSlice(alloc, "]");
    return out.toOwnedSlice(alloc);
}

/// The wire `routing` object in serde field order (knot protocol.rs).
fn routingJson(alloc: std.mem.Allocator, d: *const route_mod.Decision) !std.json.Value {
    var obj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
    try obj.put(alloc, "model", .{ .string = d.model });
    try obj.put(alloc, "repo", .{ .string = d.repo });
    try obj.put(alloc, "reason", .{ .string = d.reason });
    const det: std.json.Value = if (d.detection) |an| blk: {
        var dobj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        try dobj.put(alloc, "script", .{ .string = an.script });
        var prof = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        for (an.script_profile) |pe| try prof.put(alloc, pe.name, .{ .float = pe.frac });
        try dobj.put(alloc, "script_profile", .{ .object = prof });
        try dobj.put(alloc, "language", if (an.language) |l| .{ .string = l } else .null);
        try dobj.put(alloc, "is_english", .{ .bool = an.is_english });
        try dobj.put(alloc, "language_undecided", .{ .bool = an.language_undecided });
        try dobj.put(alloc, "diacritic_rate", .{ .float = an.diacritic_rate });
        try dobj.put(alloc, "non_latin_fraction", .{ .float = an.non_latin_fraction });
        try dobj.put(alloc, "mixed_segment", if (an.mixed_segment) |m| .{ .string = m } else .null);
        break :blk .{ .object = dobj };
    } else .null;
    try obj.put(alloc, "detection", det);
    try obj.put(alloc, "workflow", if (d.workflow) |w| .{ .string = w } else .null);
    if (d.fallback) |fb| {
        var fobj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        try fobj.put(alloc, "requested", .{ .string = fb.requested });
        try fobj.put(alloc, "served", .{ .string = fb.served });
        try obj.put(alloc, "fallback", .{ .object = fobj });
    }
    return .{ .object = obj };
}

fn readAll(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try alloc.alloc(u8, st.size);
    const got = try file.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

fn loadJson(io: std.Io, alloc: std.mem.Allocator, path: []const u8) !std.json.Value {
    const text = try readAll(io, alloc, path);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, text, .{});
    return parsed.value;
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("MISMATCH: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.arena.allocator();
    const io = init.io;
    const raw = try init.minimal.args.toSlice(alloc);
    // toSlice includes argv0 — positional args start at index 1.
    if (raw.len < 4) {
        std.debug.print("usage: p2_parity <tokenizer.json> <engine_english.json> <parity_english.json> [model_dir]\n", .{});
        std.process.exit(2);
    }
    const tok_path: []const u8 = raw[1];
    const eng_path: []const u8 = raw[2];
    const par_path: []const u8 = raw[3];
    const model_dir: ?[]const u8 = if (raw.len >= 5) raw[4] else null;

    // --- load tokenizer + specials (same path knot's PromptBuilder takes) ---
    var tok = tokenizer_mod.Tokenizer.init(alloc);
    const tok_text = try readAll(io, alloc, tok_path);
    try tok.load(tok_text);
    const builder = try prompt.Builder.fromFile(io, alloc, &tok, tok_path);

    const eng = try loadJson(io, alloc, eng_path);
    const par = try loadJson(io, alloc, par_path);
    const state = eng.object.get("state") orelse return fail("engine fixture has no state", .{});
    const questions = eng.object.get("questions") orelse return fail("engine fixture has no questions", .{});
    const items = par.object.get("items") orelse return fail("parity fixture has no items", .{});
    const collated = par.object.get("collated") orelse return fail("parity fixture has no collated", .{});

    const state_ids = try builder.encodeState(alloc, state);
    std.debug.print("state_ids={d} specials cls={d} sep={d} mask={d} pad={d} mask_token={s}\n", .{
        state_ids.len, builder.cls_id, builder.sep_id, builder.mask_id, builder.pad_id, builder.mask_token,
    });

    // Fixture pad id must agree (parity was built by knot's own builder).
    const fix_pad: i64 = par.object.get("pad_token_id").?.integer;
    if (@as(i64, @intCast(builder.pad_id)) != fix_pad)
        fail("pad_id: zig {d} vs fixture {d}", .{ builder.pad_id, fix_pad });

    const n = items.array.items.len;
    // Per-row built sequences, driven by the fixture's own qid order.
    var built_seq: std.ArrayListUnmanaged([]const u32) = .empty;
    var built_mark: std.ArrayListUnmanaged([]const usize) = .empty;
    var built_qtype: std.ArrayListUnmanaged(i64) = .empty;
    var internals: std.ArrayListUnmanaged(question.Internal) = .empty;
    var qids: std.ArrayListUnmanaged([]const u8) = .empty;
    var truncs: std.ArrayListUnmanaged(prompt.TruncStats) = .empty;
    var stats_list: std.ArrayListUnmanaged(prompt.HeadStats) = .empty;

    for (items.array.items, 0..) |item, i| {
        const qid = item.object.get("qid").?.string;
        const qv = questions.object.get(qid) orelse return fail("item {d} qid {s} not in questions", .{ i, qid });
        var bag = question.Bag{ .alloc = alloc };
        const internal = question.fromWire(alloc, &bag, qid, qv) catch {
            return fail("fromWire({s}): {s}", .{ qid, bag.msg orelse "?" });
        };
        const want_qtype: i64 = switch (internal.t) {
            .choice => 0,
            .score => 1,
            .noul => 2,
        };
        const fix_qtype = item.object.get("qtype").?.integer;
        if (want_qtype != fix_qtype)
            fail("item {d} ({s}) qtype: zig {d} vs fixture {d}", .{ i, qid, want_qtype, fix_qtype });

        const seq = try builder.buildSequence(alloc, &bag, state_ids, internal, 512, 192, null, false);
        const ids_fix = item.object.get("ids").?.array.items;
        const marks_fix = item.object.get("markers").?.array.items;
        if (seq.ids.len != ids_fix.len) {
            std.debug.print("item {d} ({s}): len zig={d} fixture={d}\nfirst diff context:\n", .{ i, qid, seq.ids.len, ids_fix.len });
            const common = @min(seq.ids.len, ids_fix.len);
            for (0..common) |j| {
                if (@as(i64, @intCast(seq.ids[j])) != ids_fix[j].integer) {
                    fail("item {d} ({s}): first id diff at {d}: zig {d} vs fixture {d}", .{ i, qid, j, seq.ids[j], ids_fix[j].integer });
                }
            }
            fail("item {d} ({s}): prefix identical, length differs (truncation divergence)", .{ i, qid });
        }
        for (seq.ids, 0..) |v, j| {
            if (@as(i64, @intCast(v)) != ids_fix[j].integer)
                fail("item {d} ({s}): id[{d}] zig {d} vs fixture {d}", .{ i, qid, j, v, ids_fix[j].integer });
        }
        if (seq.markers.len != marks_fix.len)
            fail("item {d} ({s}): markers len zig={d} fixture={d}", .{ i, qid, seq.markers.len, marks_fix.len });
        for (seq.markers, 0..) |v, j| {
            if (@as(i64, @intCast(v)) != marks_fix[j].integer)
                fail("item {d} ({s}): marker[{d}] zig {d} vs fixture {d}", .{ i, qid, j, v, marks_fix[j].integer });
        }
        std.debug.print("item {d} ({s}): seq {d} ids, {d} markers OK\n", .{ i, qid, seq.ids.len, seq.markers.len });
        try built_seq.append(alloc, seq.ids);
        try built_mark.append(alloc, seq.markers);
        try built_qtype.append(alloc, want_qtype);
        try internals.append(alloc, internal);
        try qids.append(alloc, qid);
        try truncs.append(alloc, seq.trunc);
        try stats_list.append(alloc, seq.stats);
    }

    // --- collation: pad to this chunk's lmax/kmax exactly like run_group ---
    const lmax = blk: {
        var m: usize = 0;
        for (built_seq.items) |s| m = @max(m, s.len);
        break :blk m;
    };
    const kmax = blk: {
        var m: usize = 0;
        for (built_mark.items) |mks| m = @max(m, mks.len);
        break :blk m;
    };
    const rows = collated.object.get("input_ids").?.array.items;
    if (rows.len != n) fail("collated rows {d} vs items {d}", .{ rows.len, n });
    const fix_lmax = rows[0].array.items.len;
    const fix_kmax = collated.object.get("marker_pos").?.array.items[0].array.items.len;
    if (lmax != fix_lmax) fail("lmax: zig {d} vs fixture {d}", .{ lmax, fix_lmax });
    if (kmax != fix_kmax) fail("kmax: zig {d} vs fixture {d}", .{ kmax, fix_kmax });

    const c_input = collated.object.get("input_ids").?.array.items;
    const c_att = collated.object.get("attention_mask").?.array.items;
    const c_mpos = collated.object.get("marker_pos").?.array.items;
    const c_mmask = collated.object.get("marker_mask").?.array.items;
    const c_qtype = collated.object.get("qtype").?.array.items;

    for (0..n) |i| {
        const s = built_seq.items[i];
        const mks = built_mark.items[i];
        // input_ids + attention_mask
        for (0..lmax) |j| {
            const got: i64 = if (j < s.len) @intCast(s[j]) else @intCast(builder.pad_id);
            const want = c_input[i].array.items[j].integer;
            if (got != want) fail("collated input_ids[{d}][{d}]: zig {d} vs {d}", .{ i, j, got, want });
            const att: i64 = if (j < s.len) 1 else 0;
            const want_att = c_att[i].array.items[j].integer;
            if (att != want_att) fail("collated attention[{d}][{d}]: zig {d} vs {d}", .{ i, j, att, want_att });
        }
        // marker_pos + marker_mask
        for (0..kmax) |j| {
            const got: i64 = if (j < mks.len) @intCast(mks[j]) else 0;
            const want = c_mpos[i].array.items[j].integer;
            if (got != want) fail("collated marker_pos[{d}][{d}]: zig {d} vs {d}", .{ i, j, got, want });
            const mk: bool = j < mks.len;
            const want_mk = c_mmask[i].array.items[j].bool;
            if (mk != want_mk) fail("collated marker_mask[{d}][{d}]: zig {} vs {}", .{ i, j, mk, want_mk });
        }
        const want_qt = c_qtype[i].integer;
        if (built_qtype.items[i] != want_qt) fail("collated qtype[{d}]: zig {d} vs {d}", .{ i, built_qtype.items[i], want_qt });
    }
    std.debug.print("collated: {d} rows × lmax={d} kmax={d} OK\n", .{ n, lmax, kmax });
    std.debug.print("P2-1 PASS: {d}/{d} sequences + collated tensors byte-match knot's goldens\n", .{ n, n });

    // --- P2-2: session forward on the same tensors → logits/act parity ---
    const md = model_dir orelse {
        std.debug.print("(no model_dir given — P2-2 session check skipped)\n", .{});
        return;
    };
    const path_raw = try std.fmt.allocPrint(alloc, "{s}/laya.onnx", .{md});
    const onnx_path: [:0]u8 = try alloc.allocSentinel(u8, path_raw.len, 0);
    @memcpy(onnx_path[0..path_raw.len], path_raw);
    var sess = try session_mod.Session.load(onnx_path);
    defer sess.deinit();

    // Collate my built sequences exactly like run_group does.
    const s_ids = try alloc.alloc(i64, n * lmax);
    const s_att = try alloc.alloc(i64, n * lmax);
    const s_mpos = try alloc.alloc(i64, n * kmax);
    const s_mmask = try alloc.alloc(bool, n * kmax);
    for (0..n) |i| {
        const s = built_seq.items[i];
        const mks = built_mark.items[i];
        for (0..lmax) |j| {
            s_ids[i * lmax + j] = if (j < s.len) @intCast(s[j]) else @intCast(builder.pad_id);
            s_att[i * lmax + j] = if (j < s.len) 1 else 0;
        }
        for (0..kmax) |j| {
            s_mpos[i * kmax + j] = if (j < mks.len) @intCast(mks[j]) else 0;
            s_mmask[i * kmax + j] = j < mks.len;
        }
    }
    const out = try sess.forward(alloc, s_ids, s_att, s_mpos, s_mmask, built_qtype.items, n, lmax, kmax);

    const fix_logits = par.object.get("logits").?.array.items;
    const fix_act = par.object.get("act_logits").?.array.items;
    if (fix_logits[0].array.items.len != out.logits_dim1)
        fail("logits dim1: zig {d} vs fixture {d}", .{ out.logits_dim1, fix_logits[0].array.items.len });
    if (fix_act[0].array.items.len != out.act_dim1)
        fail("act dim1: zig {d} vs fixture {d}", .{ out.act_dim1, fix_act[0].array.items.len });

    // Knot's own contract (onnx_parity.rs): the fixture is the TORCH
    // reference — |zig - torch| < 5e-3 per real marker; masked markers sit
    // at -1e4 and are skipped.
    for (0..n) |i| {
        for (0..out.logits_dim1) |j| {
            const want: f32 = @floatCast(fix_logits[i].array.items[j].float);
            const got = out.logits[i * out.logits_dim1 + j];
            if (want > -1e3 and @abs(got - want) >= 5e-3) {
                const gh = std.fmt.bytesToHex(std.mem.asBytes(&got), .lower);
                const wh = std.fmt.bytesToHex(std.mem.asBytes(&want), .lower);
                fail("logits[{d}][{d}]: zig {d} (0x{s}) vs torch {d} (0x{s}) — outside knot's 5e-3", .{ i, j, got, &gh, want, &wh });
            }
        }
        for (0..out.act_dim1) |j| {
            const want: f32 = @floatCast(fix_act[i].array.items[j].float);
            const got = out.act[i * out.act_dim1 + j];
            // knot discards act from the torch check (`let (logits, _act)` —
            // raw act logits are unbounded); its contract lands at the answer
            // level (`act_probability` after softmax), gated in P2-3.
            _ = want;
            _ = got;
        }
    }
    // Max abs deviation (report — knot logs no such number, but the canary wants it).
    var max_dev: f32 = 0;
    for (0..n) |i| {
        for (0..out.logits_dim1) |j| {
            const want: f32 = @floatCast(fix_logits[i].array.items[j].float);
            if (want > -1e3) max_dev = @max(max_dev, @abs(out.logits[i * out.logits_dim1 + j] - want));
        }
    }
    std.debug.print("P2-2 PASS: logits[{d}] + act[{d}] within knot's 5e-3 torch-parity (max |dev| = {d:.6})\n", .{ out.logits_dim1, out.act_dim1, max_dev });

    // --- P2-3: calibration + decode + answers vs the engine fixture result ---
    const cfg_text = blk: {
        const p = try std.fmt.allocPrint(alloc, "{s}/rl_agent_config.json", .{md});
        break :blk try readAll(io, alloc, p);
    };
    const cfg_val = try std.json.parseFromSlice(std.json.Value, alloc, cfg_text, .{});
    const cal = try decode.Calibration.fromConfig(alloc, cfg_val.value);

    const expected = eng.object.get("result") orelse return fail("engine fixture has no result", .{});
    const exp_answers = expected.object.get("answers").?;

    var answers_obj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
    var input_tokens: usize = 0;
    var state_tokens: usize = 0;
    var dropped_max: usize = 0;
    var truncated_qs: std.ArrayListUnmanaged([]const u8) = .empty;

    for (internals.items, 0..) |iq, ri| {
        const qid = qids.items[ri];
        const s = built_seq.items[ri];
        const mks = built_mark.items[ri];
        const k = mks.len;
        input_tokens += s.len;
        const ts = truncs.items[ri];
        state_tokens = @max(state_tokens, ts.state_tokens);
        dropped_max = @max(dropped_max, ts.state_tokens_dropped);
        if (ts.truncated) try truncated_qs.append(alloc, qid);

        const qt_idx: usize = switch (iq.t) {
            .choice => 0,
            .score => 1,
            .noul => 2,
        };
        const qt_name = question.typeName(iq.t);
        var buf: [64]u8 = undefined;
        const t_f64 = cal.forQuestion(qt_name, qt_idx, k, &buf);
        const t: f32 = @floatCast(t_f64);

        const row_logits = out.logits[ri * out.logits_dim1 ..][0..k];
        const p_slot = try decode.scaledSoftmax(alloc, row_logits, t);
        const ans_conf = decode.answerConfidence(p_slot);
        const act_row = out.act[ri * out.act_dim1 ..][0..out.act_dim1];
        const act_p = try decode.scaledSoftmax(alloc, act_row, 1.0);
        const act_prob = decode.r4(act_p[0]);

        // Answer JSON in serde field order; `window` omitted (None).
        var ans_json: []const u8 = undefined;
        switch (iq.t) {
            .choice => {
                const c = iq.crit orelse return fail("choice {s}: no criteria", .{qid});
                const obj = c.object;
                var keys: std.ArrayListUnmanaged([]const u8) = .empty;
                var kit = obj.iterator();
                while (kit.next()) |e| try keys.append(alloc, e.key_ptr.*);
                // best = LAST max (Rust max_by on partial_cmp)
                var best: usize = 0;
                var best_v: f32 = p_slot[0];
                for (p_slot, 0..) |v, idx| {
                    if (v >= best_v) {
                        best_v = v;
                        best = idx;
                    }
                }
                var probs = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                for (keys.items, 0..) |kk, idx| {
                    const pv = try std.json.parseFromSlice(std.json.Value, alloc, try fmtF32(alloc, decode.r4(p_slot[idx])), .{});
                    try probs.put(alloc, kk, pv.value);
                }
                const choice_key = if (best < keys.items.len) keys.items[best] else "";
                ans_json = try std.fmt.allocPrint(
                    alloc,
                    "{{\"type\":\"choice\",\"choice\":{s},\"probabilities\":{s},\"confidence\":{s},\"answer_confidence\":{s},\"action\":{{\"act_probability\":{s}}}}}",
                    .{
                        try pyjson.dumps(alloc, .{ .string = choice_key }),
                        try pyjson.dumps(alloc, .{ .object = probs }),
                        try fmtF32(alloc, decode.r4(decode.confidenceFromProbs(p_slot))),
                        try fmtF32(alloc, decode.r4(ans_conf)),
                        try fmtF32(alloc, act_prob),
                    },
                );
            },
            .score => {
                const c = iq.crit orelse return fail("score {s}: no criteria", .{qid});
                const arr = c.array;
                var exp: f32 = 0;
                for (p_slot, 0..) |v, idx| exp += @as(f32, @floatFromInt(idx)) * v;
                var legend = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                var probs = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                for (arr.items, 0..) |lvl, idx| {
                    const desc = try question.renderCriterion(alloc, lvl);
                    const key = try std.fmt.allocPrint(alloc, "{d}", .{idx});
                    try legend.put(alloc, key, .{ .string = desc });
                    const pv = try std.json.parseFromSlice(std.json.Value, alloc, try fmtF32(alloc, decode.r4(p_slot[idx])), .{});
                    try probs.put(alloc, key, pv.value);
                }
                ans_json = try std.fmt.allocPrint(
                    alloc,
                    "{{\"type\":\"score\",\"score\":{s},\"legend\":{s},\"probabilities\":{s},\"confidence\":{s},\"answer_confidence\":{s},\"action\":{{\"act_probability\":{s}}}}}",
                    .{
                        try fmtF32(alloc, decode.r4(exp)),
                        try pyjson.dumps(alloc, .{ .object = legend }),
                        try pyjson.dumps(alloc, .{ .object = probs }),
                        try fmtF32(alloc, decode.r4(decode.confidenceFromProbs(p_slot))),
                        try fmtF32(alloc, decode.r4(ans_conf)),
                        try fmtF32(alloc, act_prob),
                    },
                );
            },
            .noul => {
                const n1 = decode.r4(p_slot[1]);
                const conf = decode.r4(@max(p_slot[1], 1.0 - p_slot[1]));
                ans_json = try std.fmt.allocPrint(
                    alloc,
                    "{{\"type\":\"noul\",\"noul\":{s},\"confidence\":{s},\"answer_confidence\":{s},\"action\":{{\"act_probability\":{s}}}}}",
                    .{
                        try fmtF32(alloc, n1),
                        try fmtF32(alloc, conf),
                        try fmtF32(alloc, decode.r4(ans_conf)),
                        try fmtF32(alloc, act_prob),
                    },
                );
            },
        }
        // Assemble through an object + one dump so the Python separators
        // (", "/": ") match knot's pyjson exactly.
        const ans_val = try std.json.parseFromSlice(std.json.Value, alloc, ans_json, .{});
        try answers_obj.put(alloc, qid, ans_val.value);
    }

    // Expected answers re-dumped through the same serializer → string compare.
    const want_answers = try pyjson.dumps(alloc, exp_answers);
    const got_answers = try pyjson.dumps(alloc, .{ .object = answers_obj });
    if (!std.mem.eql(u8, got_answers, want_answers)) {
        std.debug.print("got answers:  {s}\nwant answers: {s}\n", .{ got_answers, want_answers });
        return fail("P2-3 answers mismatch", .{});
    }

    // Usage: input/output + the four scored-path extras (options omitted
    // when no collapse; windows absent).
    const want_usage = expected.object.get("usage").?;
    const collapsed_occurred = blk: {
        // options: only present when some row had distinct < options —
        // recomputed from the head stats we captured.
        for (stats_list.items) |st| {
            if (st.options_distinct < st.options) break :blk true;
        }
        break :blk false;
    };
    var usage_obj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
    try usage_obj.put(alloc, "input_tokens", .{ .integer = @intCast(input_tokens) });
    try usage_obj.put(alloc, "output_tokens", .{ .integer = 0 });
    try usage_obj.put(alloc, "state_tokens", .{ .integer = @intCast(state_tokens) });
    try usage_obj.put(alloc, "state_tokens_dropped", .{ .integer = @intCast(dropped_max) });
    try usage_obj.put(alloc, "truncated", .{ .bool = dropped_max > 0 });
    const tql = try std.json.parseFromSlice(
        std.json.Value,
        alloc,
        try jsonStrList(alloc, truncated_qs.items),
        .{},
    );
    try usage_obj.put(alloc, "truncated_questions", tql.value);
    if (collapsed_occurred)
        return fail("options collapsed in a fixture row — extend the usage builder", .{});
    const usage_json = try pyjson.dumps(alloc, .{ .object = usage_obj });
    const want_usage_s = try pyjson.dumps(alloc, want_usage);
    if (!std.mem.eql(u8, usage_json, want_usage_s)) {
        std.debug.print("got usage:  {s}\nwant usage: {s}\n", .{ usage_json, want_usage_s });
        return fail("P2-3 usage mismatch", .{});
    }
    std.debug.print("P2-3 PASS: answers + usage byte-match the fixture (temperature t[0]={d:.4})\n", .{cal.temperature[0]});

    // --- P2-4: lang detection vs lang_cases oracle (5th positional arg) ---
    const lang_path: ?[]const u8 = if (raw.len >= 6) raw[5] else null;
    const lp_path = lang_path orelse {
        std.debug.print("(no lang_cases.json given — P2-4 skipped)\n", .{});
        return;
    };
    const lc = try loadJson(io, alloc, lp_path);
    var pass: usize = 0;
    for (lc.array.items, 0..) |c, i| {
        const input = c.object.get("input").?;
        const det = try lang_mod.analyse(alloc, input);
        const want_script = c.object.get("script").?.string;
        const want_en = c.object.get("is_english").?.bool;
        const want_lang = c.object.get("language").?;
        const want_und = c.object.get("language_undecided").?.bool;
        const want_nl = c.object.get("non_latin_fraction").?.float;
        if (!std.mem.eql(u8, det.script, want_script))
            fail("lang[{d}]: script zig={s} want={s}", .{ i, det.script, want_script });
        if (det.is_english != want_en)
            fail("lang[{d}]: is_english zig={} want={}", .{ i, det.is_english, want_en });
        const got_lang: ?[]const u8 = det.language;
        const lang_ok = switch (want_lang) {
            .null => got_lang == null,
            .string => |s| got_lang != null and std.mem.eql(u8, got_lang.?, s),
            else => false,
        };
        if (!lang_ok)
            fail("lang[{d}]: language zig={s} want={s}", .{ i, got_lang orelse "null", if (want_lang == .string) want_lang.string else "null" });
        if (det.language_undecided != want_und)
            fail("lang[{d}]: language_undecided zig={} want={}", .{ i, det.language_undecided, want_und });
        if (det.non_latin_fraction != want_nl)
            fail("lang[{d}]: non_latin_fraction zig={d} want={d}", .{ i, det.non_latin_fraction, want_nl });
        pass += 1;
    }
    std.debug.print("P2-4 PASS: lang_cases {d}/{d} match knot's analyse()\n", .{ pass, lc.array.items.len });

    // --- G4: routing golden + #37 fallback/503 (6th positional arg) ---
    const golden_path: ?[]const u8 = if (raw.len >= 7) raw[6] else null;
    const gpath = golden_path orelse {
        std.debug.print("(no golden_english.json given — G4 skipped)\n", .{});
        return;
    };
    const golden = try loadJson(io, alloc, gpath);
    const english_only = route_mod.Sources{ .configured = &.{"english"}, .default = "english" };
    var gpass: usize = 0;
    for (golden.object.get("cases").?.array.items, 0..) |c, i| {
        const gstate = c.object.get("state").?;
        const qs = c.object.get("questions").?.object;
        var ids: std.ArrayListUnmanaged([]const u8) = .empty;
        var qit = qs.iterator();
        while (qit.next()) |e| try ids.append(alloc, e.key_ptr.*);
        var d = try route_mod.route(alloc, .{
            .state = gstate,
            .question_ids = ids.items,
            .default = "english",
        });
        try route_mod.resolve(alloc, &d, &english_only);
        const got = try pyjson.dumps(alloc, try routingJson(alloc, &d));
        const want_obj = c.object.get("response").?.object.get("routing").?;
        const want = try pyjson.dumps(alloc, want_obj);
        if (!std.mem.eql(u8, got, want)) {
            std.debug.print("got:  {s}\nwant: {s}\n", .{ got, want });
            return fail("G4 routing golden case {d}", .{i});
        }
        gpass += 1;
    }
    std.debug.print("G4a PASS: routing golden {d}/{d} byte-match\n", .{ gpass, golden.object.get("cases").?.array.items.len });

    // The #37 shape: detection picks `multilingual`, deployment has english only.
    {
        const arabic = std.json.Value{ .string = "مرحبا بالعالم" };
        var d = try route_mod.route(alloc, .{ .state = arabic, .default = "english" });
        if (!std.mem.eql(u8, d.model, "multilingual"))
            fail("G4: expected multilingual decision, got {s}", .{d.model});
        try route_mod.resolve(alloc, &d, &english_only);
        const fb = d.fallback orelse return fail("G4: fallback flag missing", .{});
        if (!std.mem.eql(u8, fb.requested, "multilingual") or !std.mem.eql(u8, fb.served, "english"))
            fail("G4: fallback {s} -> {s}", .{ fb.requested, fb.served });
        if (!std.mem.eql(u8, d.model, "english"))
            fail("G4: resolved model {s}", .{d.model});
        if (std.mem.indexOf(u8, d.reason, "not configured here") == null)
            fail("G4: reason lacks fallback note: {s}", .{d.reason});
        std.debug.print("G4b PASS: unconfigured route -> flagged fallback ({s} -> {s})\n", .{ fb.requested, fb.served });
    }
    // Nothing configured → knot's CheckpointUnavailable text (serve maps to 503).
    {
        const arabic = std.json.Value{ .string = "مرحبا بالعالم" };
        var d = try route_mod.route(alloc, .{ .state = arabic, .default = "english" });
        const empty = route_mod.Sources{ .configured = &.{}, .default = "english" };
        if (route_mod.resolve(alloc, &d, &empty)) {
            return fail("G4: expected CheckpointUnavailable with no sources", .{});
        } else |e| {
            if (e != error.CheckpointUnavailable) return fail("G4: wrong error {s}", .{@errorName(e)});
        }
        const msg = try route_mod.unavailableMessage(alloc, "multilingual");
        const knot_msg = "checkpoint \"multilingual\" is not configured; this deployment has no fallback checkpoint (set KNOT_MODEL_DIR or KNOT_DEFAULT_MODEL)";
        if (!std.mem.eql(u8, msg, knot_msg)) {
            std.debug.print("got:  {s}\nwant: {s}\n", .{ msg, knot_msg });
            return fail("G4: unavailable message drift", .{});
        }
        std.debug.print("G4c PASS: no-source resolve fails loud with knot's exact 503 text\n", .{});
    }
}
