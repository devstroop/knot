//! P2-5: engine orchestration — request → route/resolve → build sequences →
//! collated forward → decode → compact response JSON. The components are the
//! P2-1..G4 ports; this module is the seam `main_http.zig` swaps in for its
//! `inference not wired` stub (ADR-008 gate G1/G2/G4 at the HTTP level).
const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");
const question = @import("question.zig");
const prompt = @import("prompt.zig");
const session_mod = @import("session.zig");
const decode = @import("decode.zig");
const route_mod = @import("route.zig");
const pyjson = @import("pyjson.zig");

pub const AGENT_MODEL = "laya-rl-agent";

/// f32 → wire float: serde's shortest-f32 repr with ryu's fraction fill.
fn fmtF32(alloc: std.mem.Allocator, v: f32) ![]const u8 {
    const s = try std.fmt.allocPrint(alloc, "{d}", .{v});
    if (std.mem.indexOfAny(u8, s, ".eE") == null) {
        return std.fmt.allocPrint(alloc, "{s}.0", .{s});
    }
    return s;
}

/// Wire float for HTTP: knot serializes through `serde_json::to_value`,
/// which WIDENS f32 → f64 first, so the body carries the f64-shortest repr
/// (`0.9287999868392944`), not the numpy short-f32 form the recorded
/// fixtures hold (`0.9288`). pyjson's floatRepr (55/55 vs Python/serde) is
/// the exact algorithm for the widened value.
fn num(alloc: std.mem.Allocator, v: f32) !std.json.Value {
    const s = try pyjson.floatRepr(alloc, @as(f64, @floatCast(v)));
    const p = try std.json.parseFromSlice(std.json.Value, alloc, s, .{});
    return p.value;
}

fn jsonStrList(alloc: std.mem.Allocator, items: []const []const u8) !std.json.Value {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try out.appendSlice(alloc, "[");
    for (items, 0..) |s, i| {
        if (i > 0) try out.appendSlice(alloc, ",");
        try out.appendSlice(alloc, try pyjson.dumps(alloc, .{ .string = s }));
    }
    try out.appendSlice(alloc, "]");
    const p = try std.json.parseFromSlice(std.json.Value, alloc, out.items, .{});
    return p.value;
}

fn readText(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.FileNotFound;
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try alloc.alloc(u8, st.size);
    const got = try file.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

pub const Engine = struct {
    tok: tokenizer_mod.Tokenizer,
    builder: prompt.Builder,
    sess: session_mod.Session,
    cal: decode.Calibration,
    name: []const u8,
    configured: [1][]const u8 = .{""},
    sources: route_mod.Sources = .{ .configured = &.{}, .default = "english" },
    max_len: usize = 512,
    head_max_len: usize = 192,

    /// In-place init: the builder and sources point INTO `self`, so the
    /// engine may not be moved afterwards.
    pub fn init(
        self: *Engine,
        io: std.Io,
        alloc: std.mem.Allocator,
        model_dir: []const u8,
        name: []const u8,
    ) !void {
        const tok_path = blk: {
            const nested = try std.fmt.allocPrint(alloc, "{s}/tokenizer/tokenizer.json", .{model_dir});
            break :blk if (std.Io.Dir.cwd().openFile(io, nested, .{})) |f| blk2: {
                f.close(io);
                break :blk2 nested;
            } else |_| try std.fmt.allocPrint(alloc, "{s}/tokenizer.json", .{model_dir});
        };
        self.tok = tokenizer_mod.Tokenizer.init(alloc);
        const tok_text = try readText(io, alloc, tok_path);
        try self.tok.load(tok_text);
        self.builder = try prompt.Builder.fromFile(io, alloc, &self.tok, tok_path);

        const onnx_path = try std.fmt.allocPrint(alloc, "{s}/laya.onnx", .{model_dir});
        const onnx_z: [:0]u8 = try alloc.allocSentinel(u8, onnx_path.len, 0);
        @memcpy(onnx_z[0..onnx_path.len], onnx_path);
        self.sess = try session_mod.Session.load(onnx_z);

        const cfg_path = try std.fmt.allocPrint(alloc, "{s}/rl_agent_config.json", .{model_dir});
        const cfg_text = try readText(io, alloc, cfg_path);
        const cfg = try std.json.parseFromSlice(std.json.Value, alloc, cfg_text, .{});
        self.cal = try decode.Calibration.fromConfig(alloc, cfg.value);
        if (cfg.value == .object) {
            if (cfg.value.object.get("max_len")) |v| {
                if (v == .integer and v.integer > 0) self.max_len = @intCast(v.integer);
            }
            if (cfg.value.object.get("head_max_len")) |v| {
                if (v == .integer and v.integer > 0) self.head_max_len = @intCast(v.integer);
            }
        }

        self.name = try alloc.dupe(u8, name);
        self.configured = .{self.name};
        self.sources = .{ .configured = &self.configured, .default = self.name };
    }

    pub fn deinit(self: *Engine) void {
        self.sess.deinit();
    }

    /// One request → compact response JSON.
    /// `err_out` carries the caller-facing detail for `error.Rejected`
    /// (knot 422) and `error.CheckpointUnavailable` (knot 503).
    pub fn predict(
        self: *Engine,
        alloc: std.mem.Allocator,
        req: std.json.Value,
        err_out: *?[]const u8,
    ) ![]u8 {
        if (req != .object) {
            err_out.* = "request body must be an object with a 'questions' field";
            return error.Rejected;
        }
        const questions_val = req.object.get("questions") orelse {
            err_out.* = "request body must be an object with a 'questions' field";
            return error.Rejected;
        };
        if (questions_val != .object) {
            err_out.* = "request body must be an object with a 'questions' field";
            return error.Rejected;
        }
        const qmap = questions_val.object;
        const state = req.object.get("state") orelse .null;

        var ids: std.ArrayListUnmanaged([]const u8) = .empty;
        var qit = qmap.iterator();
        while (qit.next()) |e| try ids.append(alloc, e.key_ptr.*);

        // Request-level params (jev gateway names are ignored, like knot).
        const model_param: ?[]const u8 = blk: {
            const m = req.object.get("model") orelse break :blk null;
            if (m != .string) break :blk null;
            if (std.mem.startsWith(u8, m.string, "jev")) break :blk null;
            break :blk m.string;
        };
        const task_param: ?[]const u8 = if (req.object.get("task")) |t| (if (t == .string) t.string else null) else null;
        const lang_param: ?[]const u8 = if (req.object.get("lang")) |t| (if (t == .string) t.string else null) else null;
        const lang_guess_param: ?[]const u8 = if (req.object.get("lang_guess")) |t| (if (t == .string) t.string else null) else null;

        var max_len = self.max_len;
        if (req.object.get("max_len")) |v| {
            if (v == .integer and v.integer > 0) max_len = @intCast(v.integer);
        }
        var head_max_len = self.head_max_len;
        if (req.object.get("head_max_len")) |v| {
            if (v == .integer and v.integer > 0) head_max_len = @intCast(v.integer);
        }

        // --- route + #37 resolve (gate G4 at request level) ---------------
        var decision = route_mod.route(alloc, .{
            .state = state,
            .question_ids = ids.items,
            .model = model_param,
            .task = task_param,
            .lang = lang_param,
            .lang_guess = lang_guess_param,
            .default = self.sources.default,
        }) catch |e| switch (e) {
            error.UnknownModel => {
                err_out.* = try std.fmt.allocPrint(
                    alloc,
                    "unknown model {s}",
                    .{try question.rustDebug(alloc, model_param orelse task_param orelse "?")},
                );
                return error.Rejected;
            },
            else => return e,
        };
        const requested = decision.model;
        route_mod.resolve(alloc, &decision, &self.sources) catch {
            err_out.* = try route_mod.unavailableMessage(alloc, requested);
            return error.CheckpointUnavailable;
        };

        const routing_val = try routingJson(alloc, &decision);

        // --- empty-questions short path (Laya `if not ids`) ---------------
        if (qmap.count() == 0) {
            var resp = try std.json.ObjectMap.init(alloc, &.{}, &.{});
            try resp.put(alloc, "model", .{ .string = AGENT_MODEL });
            try resp.put(alloc, "answers", .{ .object = try std.json.ObjectMap.init(alloc, &.{}, &.{}) });
            var usage = try std.json.ObjectMap.init(alloc, &.{}, &.{});
            try usage.put(alloc, "input_tokens", .{ .integer = 0 });
            try usage.put(alloc, "output_tokens", .{ .integer = 0 });
            try resp.put(alloc, "usage", .{ .object = usage });
            try resp.put(alloc, "routing", routing_val);
            return pyjson.dumpsCompact(alloc, .{ .object = resp });
        }

        // --- build rows ----------------------------------------------------
        const state_ids = try self.builder.encodeState(alloc, state);
        const Row = struct {
            qid: []const u8,
            ids: []u32,
            markers: []usize,
            qtype: usize,
            order: ?[]usize,
            options: usize,
            distinct: usize,
            trunc: prompt.TruncStats,
        };
        var rows: std.ArrayListUnmanaged(Row) = .empty;
        var qit2 = qmap.iterator();
        while (qit2.next()) |e| {
            const qid = e.key_ptr.*;
            if (qid.len == 0) {
                err_out.* = try std.fmt.allocPrint(alloc, "question id must be a non-empty string, got {s}", .{try question.rustDebug(alloc, qid)});
                return error.Rejected;
            }
            var bag = question.Bag{ .alloc = alloc };
            const iq = question.fromWire(alloc, &bag, qid, e.value_ptr.*) catch {
                err_out.* = bag.msg;
                return error.Rejected;
            };
            // wire option_order (validated like knot's run_group)
            var order: ?[]usize = null;
            if (e.value_ptr.* == .object) {
                if (e.value_ptr.*.object.get("option_order")) |oo| {
                    if (oo == .array) {
                        const arr = oo.array.items;
                        const n_opts = (try question.renderOptions(alloc, &bag, iq)).len;
                        if (arr.len != n_opts) {
                            err_out.* = try std.fmt.allocPrint(
                                alloc,
                                "question {s}: option_order must have one index per option ({d} expected, got {d})",
                                .{ try question.rustDebug(alloc, qid), n_opts, arr.len },
                            );
                            return error.Rejected;
                        }
                        const ord = try alloc.alloc(usize, arr.len);
                        for (arr, 0..) |v, i| {
                            if (v != .integer or v.integer < 0) {
                                err_out.* = try std.fmt.allocPrint(
                                    alloc,
                                    "question {s}: option_order must be a permutation of 0..{d}",
                                    .{ try question.rustDebug(alloc, qid), n_opts },
                                );
                                return error.Rejected;
                            }
                            ord[i] = @intCast(v.integer);
                        }
                        const sorted = try alloc.dupe(usize, ord);
                        std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
                        for (sorted, 0..) |v, i| {
                            if (v != i) {
                                err_out.* = try std.fmt.allocPrint(
                                    alloc,
                                    "question {s}: option_order must be a permutation of 0..{d}",
                                    .{ try question.rustDebug(alloc, qid), n_opts },
                                );
                                return error.Rejected;
                            }
                        }
                        order = ord;
                    }
                }
            }
            const seq = self.builder.buildSequence(alloc, &bag, state_ids, iq, max_len, head_max_len, order, false) catch |be| {
                if (be == error.Rejected) {
                    err_out.* = bag.msg;
                    return error.Rejected;
                }
                return be;
            };
            if (seq.markers.len != seq.stats.options) {
                err_out.* = try std.fmt.allocPrint(
                    alloc,
                    "question {s}: only {d} of its {d} option markers fit in max_len={d} with head_max_len={d}",
                    .{ try question.rustDebug(alloc, qid), seq.markers.len, seq.stats.options, max_len, head_max_len },
                );
                return error.Rejected;
            }
            try rows.append(alloc, .{
                .qid = qid,
                .ids = seq.ids,
                .markers = seq.markers,
                .qtype = switch (iq.t) {
                    .choice => 0,
                    .score => 1,
                    .noul => 2,
                },
                .order = order,
                .options = seq.stats.options,
                .distinct = seq.stats.options_distinct,
                .trunc = seq.trunc,
            });
        }

        // --- collate + forward ---------------------------------------------
        const n = rows.items.len;
        var lmax: usize = 0;
        var kmax: usize = 0;
        for (rows.items) |r| {
            lmax = @max(lmax, r.ids.len);
            kmax = @max(kmax, r.markers.len);
        }
        const pad: i64 = @intCast(self.builder.pad_id);
        const c_ids = try alloc.alloc(i64, n * lmax);
        const c_att = try alloc.alloc(i64, n * lmax);
        const c_mpos = try alloc.alloc(i64, n * kmax);
        const c_mmask = try alloc.alloc(bool, n * kmax);
        const c_qt = try alloc.alloc(i64, n);
        for (rows.items, 0..) |r, i| {
            for (0..lmax) |j| {
                c_ids[i * lmax + j] = if (j < r.ids.len) @intCast(r.ids[j]) else pad;
                c_att[i * lmax + j] = if (j < r.ids.len) 1 else 0;
            }
            for (0..kmax) |j| {
                c_mpos[i * kmax + j] = if (j < r.markers.len) @intCast(r.markers[j]) else 0;
                c_mmask[i * kmax + j] = j < r.markers.len;
            }
            c_qt[i] = @intCast(r.qtype);
        }
        const out = try self.sess.forward(alloc, c_ids, c_att, c_mpos, c_mmask, c_qt, n, lmax, kmax);

        // --- decode + usage -------------------------------------------------
        var answers = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        var input_tokens: usize = 0;
        var state_tokens: usize = 0;
        var dropped_max: usize = 0;
        var truncated_qs: std.ArrayListUnmanaged([]const u8) = .empty;
        var collapsed = try std.json.ObjectMap.init(alloc, &.{}, &.{});

        for (rows.items, 0..) |r, ri| {
            input_tokens += r.ids.len;
            state_tokens = @max(state_tokens, r.trunc.state_tokens);
            dropped_max = @max(dropped_max, r.trunc.state_tokens_dropped);
            if (r.trunc.truncated) try truncated_qs.append(alloc, r.qid);
            if (r.distinct < r.options) {
                var cobj = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                try cobj.put(alloc, "total", .{ .integer = @intCast(r.options) });
                try cobj.put(alloc, "distinct", .{ .integer = @intCast(r.distinct) });
                try cobj.put(alloc, "tokens_per_option", .null);
                try collapsed.put(alloc, r.qid, .{ .object = cobj });
            }

            const k = r.markers.len;
            const qt_names = [_][]const u8{ "choice", "score", "noul" };
            const qt_name = qt_names[r.qtype];
            var tbuf: [64]u8 = undefined;
            const t: f32 = @floatCast(self.cal.forQuestion(qt_name, r.qtype, k, &tbuf));

            const row_logits = out.logits[ri * out.logits_dim1 ..][0..k];
            const p_slot = try decode.scaledSoftmax(alloc, row_logits, t);
            // un-permute slot → option (wire option_order), Laya unpermute_probs
            const p: []f32 = if (r.order) |ord| blk: {
                if (ord.len != p_slot.len) break :blk p_slot;
                var popt = try alloc.alloc(f32, p_slot.len);
                for (ord, 0..) |opt, slot| {
                    if (opt < popt.len) popt[opt] = p_slot[slot];
                }
                break :blk popt;
            } else p_slot;
            const ans_conf = decode.answerConfidence(p);
            const act_row = out.act[ri * out.act_dim1 ..][0..out.act_dim1];
            const act_p = try decode.scaledSoftmax(alloc, act_row, 1.0);
            const act_prob = try num(alloc, decode.r4(act_p[0]));

            var ans = try std.json.ObjectMap.init(alloc, &.{}, &.{});
            try ans.put(alloc, "type", .{ .string = qt_name });
            const iq_qv = qmap.get(r.qid).?;
            switch (r.qtype) {
                0 => {
                    // keys from the WIRE criteria (post-normalize order = crit object)
                    var bag2 = question.Bag{ .alloc = alloc };
                    const iq = try question.fromWire(alloc, &bag2, r.qid, iq_qv);
                    const critv: std.json.Value = iq.crit orelse .null;
                    if (critv != .object) {
                        err_out.* = "choice needs criteria object";
                        return error.Rejected;
                    }
                    const obj = critv.object;
                    var keys: std.ArrayListUnmanaged([]const u8) = .empty;
                    var kit = obj.iterator();
                    while (kit.next()) |ke| try keys.append(alloc, ke.key_ptr.*);
                    var best: usize = 0;
                    var best_v: f32 = p[0];
                    for (p, 0..) |v, idx| {
                        if (v >= best_v) {
                            best_v = v;
                            best = idx;
                        }
                    }
                    var probs = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                    for (keys.items, 0..) |kk, idx| {
                        try probs.put(alloc, kk, try num(alloc, decode.r4(p[idx])));
                    }
                    try ans.put(alloc, "choice", .{ .string = if (best < keys.items.len) keys.items[best] else "" });
                    try ans.put(alloc, "probabilities", .{ .object = probs });
                    try ans.put(alloc, "confidence", try num(alloc, decode.r4(decode.confidenceFromProbs(p))));
                    try ans.put(alloc, "answer_confidence", try num(alloc, decode.r4(ans_conf)));
                },
                1 => {
                    var bag2 = question.Bag{ .alloc = alloc };
                    const iq = try question.fromWire(alloc, &bag2, r.qid, iq_qv);
                    const critv: std.json.Value = iq.crit orelse .null;
                    if (critv != .array) {
                        err_out.* = "score needs criteria list";
                        return error.Rejected;
                    }
                    const arr = critv.array;
                    var exp: f32 = 0;
                    for (p, 0..) |v, idx| exp += @as(f32, @floatFromInt(idx)) * v;
                    var legend = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                    var probs = try std.json.ObjectMap.init(alloc, &.{}, &.{});
                    for (arr.items, 0..) |lvl, idx| {
                        const key = try std.fmt.allocPrint(alloc, "{d}", .{idx});
                        try legend.put(alloc, key, .{ .string = try question.renderCriterion(alloc, lvl) });
                        try probs.put(alloc, key, try num(alloc, decode.r4(p[idx])));
                    }
                    try ans.put(alloc, "score", try num(alloc, decode.r4(exp)));
                    try ans.put(alloc, "legend", .{ .object = legend });
                    try ans.put(alloc, "probabilities", .{ .object = probs });
                    try ans.put(alloc, "confidence", try num(alloc, decode.r4(decode.confidenceFromProbs(p))));
                    try ans.put(alloc, "answer_confidence", try num(alloc, decode.r4(ans_conf)));
                },
                else => {
                    try ans.put(alloc, "noul", try num(alloc, decode.r4(p[1])));
                    try ans.put(alloc, "confidence", try num(alloc, decode.r4(@max(p[1], 1.0 - p[1]))));
                    try ans.put(alloc, "answer_confidence", try num(alloc, decode.r4(ans_conf)));
                },
            }
            var action = try std.json.ObjectMap.init(alloc, &.{}, &.{});
            try action.put(alloc, "act_probability", act_prob);
            try ans.put(alloc, "action", .{ .object = action });
            try answers.put(alloc, r.qid, .{ .object = ans });
        }

        // --- usage + response ------------------------------------------------
        var usage = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        try usage.put(alloc, "input_tokens", .{ .integer = @intCast(input_tokens) });
        try usage.put(alloc, "output_tokens", .{ .integer = 0 });
        try usage.put(alloc, "state_tokens", .{ .integer = @intCast(state_tokens) });
        try usage.put(alloc, "state_tokens_dropped", .{ .integer = @intCast(dropped_max) });
        try usage.put(alloc, "truncated", .{ .bool = dropped_max > 0 });
        try usage.put(alloc, "truncated_questions", try jsonStrList(alloc, truncated_qs.items));
        if (collapsed.count() > 0) {
            try usage.put(alloc, "options", .{ .object = collapsed });
        }

        var resp = try std.json.ObjectMap.init(alloc, &.{}, &.{});
        try resp.put(alloc, "model", .{ .string = AGENT_MODEL });
        try resp.put(alloc, "answers", .{ .object = answers });
        try resp.put(alloc, "usage", .{ .object = usage });
        try resp.put(alloc, "routing", routing_val);
        return pyjson.dumpsCompact(alloc, .{ .object = resp });
    }
};

/// The wire `routing` object in serde field order.
pub fn routingJson(alloc: std.mem.Allocator, d: *const route_mod.Decision) !std.json.Value {
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
