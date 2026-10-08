//! Phase-2 canary: sequence building — port of knot's `PromptBuilder`
//! (`from_file` special-token resolution, `encode_state`, `build_head`,
//! `build_sequence`) over the proven zig tokenizer port.
const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");
const question_mod = @import("question.zig");
const question = question_mod;

pub const HeadStats = struct {
    options: usize,
    options_distinct: usize,
    tokens_per_option: ?usize,
};

pub const TruncStats = struct {
    state_tokens: usize,
    state_tokens_used: usize,
    state_tokens_dropped: usize,
    truncated: bool,
};

pub const OPTION_MAX_TOKENS: usize = 48;

/// Resolve the special ids the way `PromptBuilder::from_file` does: the
/// declarations beside the tokenizer win, candidate strings only as fallback.
fn declaredSpecials(io: std.Io, alloc: std.mem.Allocator, tok_path: []const u8) !std.StringHashMapUnmanaged([]const u8) {
    var out: std.StringHashMapUnmanaged([]const u8) = .{};
    const dir = std.fs.path.dirname(tok_path) orelse return out;
    const keys = [_][]const u8{ "cls_token", "sep_token", "mask_token", "bos_token", "eos_token" };
    for ([_][]const u8{ "tokenizer_config.json", "special_tokens_map.json" }) |name| {
        const full = try std.fs.path.join(alloc, &.{ dir, name });
        const text = blk: {
            var file = std.Io.Dir.cwd().openFile(io, full, .{}) catch break :blk null;
            defer file.close(io);
            const st = try file.stat(io);
            const buf = try alloc.alloc(u8, st.size);
            const got = try file.readPositionalAll(io, buf, 0);
            break :blk buf[0..got];
        } orelse continue;
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, text, .{}) catch continue;
        const v = parsed.value;
        if (v != .object) continue;
        for (keys) |key| {
            if (out.contains(key)) continue;
            const val = v.object.get(key) orelse continue;
            const s: []const u8 = switch (val) {
                .string => |s| s,
                .object => |o| blk: {
                    const c = o.get("content") orelse break :blk "";
                    break :blk if (c == .string) c.string else "";
                },
                else => "",
            };
            if (s.len == 0) continue;
            try out.put(alloc, key, s);
        }
    }
    return out;
}

pub const Builder = struct {
    tok: *tokenizer_mod.Tokenizer,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    pad_id: u32,
    mask_token: []const u8,

    pub fn fromFile(io: std.Io, alloc: std.mem.Allocator, tok: *tokenizer_mod.Tokenizer, tok_path: []const u8) !Builder {
        const declared = try declaredSpecials(io, alloc, tok_path);
        const pick = struct {
            fn p(
                a: std.mem.Allocator,
                d: std.StringHashMapUnmanaged([]const u8),
                t: *tokenizer_mod.Tokenizer,
                key: []const u8,
                candidates: []const []const u8,
            ) !?u32 {
                if (d.get(key)) |content| {
                    if (try tokLookup(t, content)) |id| return id;
                }
                for (candidates) |cand| {
                    if (try tokLookup(t, cand)) |id| return id;
                }
                _ = a;
                return null;
            }
        }.p;
        const cls_id = (try pick(alloc, declared, tok, "cls_token", &.{ "[CLS]", "<s>", "<bos>" })) orelse
            return error.Model;
        const sep_id = (try pick(alloc, declared, tok, "sep_token", &.{ "[SEP]", "</s>", "<eos>" })) orelse
            return error.Model;
        const mask_id = (try pick(alloc, declared, tok, "mask_token", &.{ "[MASK]", "<mask>" })) orelse
            return error.Model;
        const pad_id = (try pick(alloc, declared, tok, "pad_token", &.{ "[PAD]", "<pad>" })) orelse 0;
        const mask_token = if (try tokIdToToken(tok, mask_id)) |t|
            std.mem.trimStart(u8, t, "▁")
        else
            "[MASK]";
        return .{
            .tok = tok,
            .cls_id = cls_id,
            .sep_id = sep_id,
            .mask_id = mask_id,
            .pad_id = pad_id,
            .mask_token = mask_token,
        };
    }

    fn encodeIds(self: *const Builder, alloc: std.mem.Allocator, text: []const u8) ![]u32 {
        var list: std.ArrayListUnmanaged(u32) = .empty;
        try self.tok.encode(alloc, text, &list);
        return list.toOwnedSlice(alloc);
    }

    /// `encode_state`: serialize (Python dumps for containers), neutralize
    /// mask-token leakage, tokenize.
    pub fn encodeState(self: *const Builder, alloc: std.mem.Allocator, state: std.json.Value) ![]u32 {
        const text = try question.serializeState(alloc, state);
        const cleaned = try std.mem.replaceOwned(u8, alloc, text, self.mask_token, " ");
        return self.encodeIds(alloc, cleaned);
    }

    /// `(ids, markers, stats)` for the question half of the sequence.
    pub fn buildHead(
        self: *const Builder,
        alloc: std.mem.Allocator,
        bag: *question.Bag,
        q: question.Internal,
        head_max_len: usize,
        order: ?[]const usize,
    ) !struct { ids: []u32, markers: []usize, stats: HeadStats } {
        const opts = try question.renderOptions(alloc, bag, q);
        const ord: []const usize = order orelse blk: {
            var all = try alloc.alloc(usize, opts.len);
            for (0..opts.len) |i| all[i] = i;
            break :blk all;
        };

        // question text: `{t} question: {ins}`, mask-token neutralized
        const ins_clean = try std.mem.replaceOwned(u8, alloc, q.ins, self.mask_token, " ");
        const head_text = try std.fmt.allocPrint(alloc, "{s} question: {s}", .{ question.typeName(q.t), ins_clean });
        var head_ids = try self.encodeIds(alloc, head_text);

        // options in caller order, each prefixed with the [MASK] marker
        var opt_ids_list: std.ArrayListUnmanaged([]u32) = .empty;
        for (ord) |i| {
            const raw = try std.fmt.allocPrint(alloc, " {s}", .{opts[i]});
            const opt_clean = try std.mem.replaceOwned(u8, alloc, raw, self.mask_token, " ");
            var token_ids = try self.encodeIds(alloc, opt_clean);
            if (token_ids.len > OPTION_MAX_TOKENS) token_ids = token_ids[0..OPTION_MAX_TOKENS];
            var with_mask = try alloc.alloc(u32, token_ids.len + 1);
            with_mask[0] = self.mask_id;
            @memcpy(with_mask[1..], token_ids);
            try opt_ids_list.append(alloc, with_mask);
        }
        const opt_ids = try opt_ids_list.toOwnedSlice(alloc);

        var opt_budget: isize = @as(isize, @intCast(head_max_len));
        for (opt_ids) |o| opt_budget -= @intCast(o.len);
        var per_option: ?usize = null;
        if (opt_budget < 16) {
            const per: usize = @max(4, (head_max_len -| 16) / @max(opt_ids.len, 1));
            per_option = per;
            for (opt_ids) |*o| {
                if (o.len > per) o.* = o.*[0..per];
                if (o.len == 0 or o.*[0] != self.mask_id) {
                    var fixed = try alloc.alloc(u32, o.len + 1);
                    fixed[0] = self.mask_id;
                    @memcpy(fixed[1..], o.*);
                    o.* = fixed;
                }
            }
            opt_budget = @as(isize, @intCast(head_max_len));
            for (opt_ids) |o| opt_budget -= @intCast(o.len);
        }
        const head_cap: usize = @max(8, @max(opt_budget, 0));
        if (head_ids.len > head_cap) head_ids = head_ids[0..head_cap];

        // CLS + head + SEP + marker-prefixed options + SEP
        var ids: std.ArrayListUnmanaged(u32) = .empty;
        try ids.append(alloc, self.cls_id);
        try ids.appendSlice(alloc, head_ids);
        try ids.append(alloc, self.sep_id);
        var markers: std.ArrayListUnmanaged(usize) = .empty;
        for (opt_ids) |o| {
            try markers.append(alloc, ids.items.len);
            try ids.appendSlice(alloc, o);
        }
        try ids.append(alloc, self.sep_id);

        // distinct option slices (Rust HashSet<&Vec<u32>>)
        var distinct: usize = 0;
        for (opt_ids, 0..) |o, i| {
            var seen = false;
            for (opt_ids[0..i]) |prev| {
                if (std.mem.eql(u32, o, prev)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) distinct += 1;
        }
        return .{
            .ids = try ids.toOwnedSlice(alloc),
            .markers = try markers.toOwnedSlice(alloc),
            .stats = .{
                .options = opt_ids.len,
                .options_distinct = distinct,
                .tokens_per_option = per_option,
            },
        };
    }

    /// Full sequence: head + clamped state + trailing SEP.
    pub fn buildSequence(
        self: *const Builder,
        alloc: std.mem.Allocator,
        bag: *question.Bag,
        state_ids: []const u32,
        q: question.Internal,
        max_len: usize,
        head_max_len: usize,
        option_order: ?[]const usize,
        truncate_left: bool,
    ) !struct { ids: []u32, markers: []usize, stats: HeadStats, trunc: TruncStats } {
        const head = try self.buildHead(alloc, bag, q, head_max_len, option_order);
        const room = max_len -| (head.ids.len + 1);
        const st: []const u32 = if (truncate_left)
            state_ids[state_ids.len -| room..]
        else
            state_ids[0..@min(state_ids.len, room)];

        var out = try alloc.alloc(u32, @min(head.ids.len + st.len + 1, max_len));
        var n: usize = 0;
        @memcpy(out[n .. n + head.ids.len], head.ids);
        n += head.ids.len;
        @memcpy(out[n .. n + st.len], st);
        n += st.len;
        out[n] = self.sep_id;
        n += 1;
        const ids = out[0..n];

        var markers: std.ArrayListUnmanaged(usize) = .empty;
        for (head.markers) |m| {
            if (m < max_len) try markers.append(alloc, m);
        }
        return .{
            .ids = ids,
            .markers = try markers.toOwnedSlice(alloc),
            .stats = head.stats,
            .trunc = .{
                .state_tokens = state_ids.len,
                .state_tokens_used = st.len,
                .state_tokens_dropped = state_ids.len - st.len,
                .truncated = st.len < state_ids.len,
            },
        };
    }
};

/// HF `token_to_id`: added tokens and the BPE vocab, both consulted.
pub fn tokLookup(tok: *tokenizer_mod.Tokenizer, text: []const u8) !?u32 {
    return tok.tokenToId(text);
}

pub fn tokIdToToken(tok: *tokenizer_mod.Tokenizer, id: u32) !?[]const u8 {
    return tok.idToToken(id);
}
