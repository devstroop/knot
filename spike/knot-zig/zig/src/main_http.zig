//! HTTP skeleton — phase 6 of the knot-zig canary: the TRANSPORT surface.
//!
//! Reproduces knot's status codes and error bodies byte-for-byte for every
//! NON-inference request (contract = ../knot_http_baseline.json, checked by
//! ../http_diff.py); valid inference requests answer 501 — the engine is not
//! wired (that is the M-ladder, not the canary).
//!
//! Honest gaps vs axum (documented, outside the baseline corpus): Content-
//! Length bodies only (chunked POST -> 411), serial accept loop, no auth,
//! no size guards, HTTP-parse errors close the connection instead of
//! answering.
//!
//! Run: http_skeleton <port> <model_dir>   (e.g. 8125 /root/.cache/knot/english)
const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const http = std.http;

const engine_mod = @import("engine.zig");

fn basename(path: []const u8) []const u8 {
    var end = path.len;
    while (end > 0 and path[end - 1] == '/') end -= 1;
    var start = end;
    while (start > 0 and path[start - 1] != '/') start -= 1;
    return path[start..end];
}

/// Mirror knot's snapshot_revision(): first line of the HF cache .metadata
/// file (40-hex etag) or null for a hand-copied directory.
fn snapshotEtag(io: std.Io, alloc: std.mem.Allocator, model_dir: []const u8) ?[]const u8 {
    for ([_][]const u8{
        "/.cache/huggingface/download/rl_agent_config.json.metadata",
        "/.cache/huggingface/download/model.safetensors.metadata",
    }) |rel| {
        const path = std.fmt.allocPrint(alloc, "{s}{s}", .{ model_dir, rel }) catch return null;
        const file = Io.Dir.cwd().openFile(io, path, .{}) catch continue;
        defer file.close(io);
        var buf: [256]u8 = undefined;
        const n = file.readPositionalAll(io, &buf, 0) catch continue;
        const text = buf[0..n];
        const nl = std.mem.indexOfScalar(u8, text, '\n') orelse text.len;
        const first = std.mem.trim(u8, text[0..nl], " \r");
        if (first.len == 40) {
            var hex_ok = true;
            for (first) |c| {
                const is_hex = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or
                    (c >= 'A' and c <= 'F');
                if (!is_hex) {
                    hex_ok = false;
                    break;
                }
            }
            if (hex_ok) return alloc.dupe(u8, first) catch null;
        }
    }
    return null;
}
fn jsonDetail(alloc: std.mem.Allocator, detail: []const u8) ![]u8 {
    // Detail literals never contain `"` or `\`, so a plain wrap is exact.
    return std.fmt.allocPrint(alloc, "{{\"detail\":\"{s}\"}}", .{detail});
}

fn respondJson(request: *http.Server.Request, body: []const u8, status: http.Status, keep_alive: bool) !void {
    try request.respond(body, .{
        .status = status,
        .keep_alive = keep_alive,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

fn respondEmpty(request: *http.Server.Request, status: http.Status, allow: []const u8) !void {
    // knot (axum) answers 404/405 with an empty body; we may not have read
    // the request body on these paths, so close rather than risk parsing it
    // as the next pipelined request head.
    var headers: []const http.Header = &.{};
    if (allow.len > 0) {
        headers = &.{.{ .name = "allow", .value = allow }};
    }
    try request.respond("", .{
        .status = status,
        .keep_alive = false,
        .extra_headers = headers,
    });
}

fn route(alloc: std.mem.Allocator, eng: *engine_mod.Engine, request: *http.Server.Request, name: []const u8, etag: ?[]const u8) !void {
    const method = request.head.method;
    // Dupe the target BEFORE the body reader invalidates Head's strings.
    const target = try alloc.dupe(u8, request.head.target);
    const path_end = std.mem.indexOfScalar(u8, target, '?') orelse target.len;
    const path = target[0..path_end];

    if (std.mem.eql(u8, path, "/health")) {
        if (method == .GET or method == .HEAD) {
            const etag_json = if (etag) |e|
                try std.fmt.allocPrint(alloc, "\"{s}\"", .{e})
            else
                "null";
            const body = try std.fmt.allocPrint(alloc, "{{\"status\":\"ok\",\"loaded\":[\"{s}\"],\"revisions\":{{\"{s}\":{s}}},\"device\":\"cpu\",\"device_is_preference\":false,\"checkpoint_devices\":{{\"{s}\":\"cpu\"}},\"cpu_fallbacks\":{{\"{s}\":{{\"count\":0,\"last_reason\":null}}}}}}", .{ name, name, etag_json, name, name });
            try respondJson(request, body, .ok, true);
        } else try respondEmpty(request, .method_not_allowed, "GET,HEAD");
        return;
    }
    if (std.mem.eql(u8, path, "/models")) {
        if (method == .GET or method == .HEAD) {
            const body = try std.fmt.allocPrint(alloc, "{{\"models\":[\"{s}\"]}}", .{name});
            try respondJson(request, body, .ok, true);
        } else try respondEmpty(request, .method_not_allowed, "GET,HEAD");
        return;
    }
    const is_systemone = std.mem.eql(u8, path, "/v1/systemone");
    const is_batch = std.mem.eql(u8, path, "/v1/systemone/batch");
    if (!is_systemone and !is_batch) {
        try respondEmpty(request, .not_found, "");
        return;
    }
    if (method != .POST) {
        try respondEmpty(request, .method_not_allowed, "POST");
        return;
    }
    const cl = request.head.content_length orelse {
        try respondJson(request, try jsonDetail(alloc, "length required"), .length_required, false);
        return;
    };
    var body_buf: [16384]u8 = undefined;
    const br = try request.readerExpectContinue(&body_buf);
    const body = try br.readAllocAll(alloc, cl);
    // --- knot-identical validation (exact strings from the baseline) ------
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch {
        try respondJson(request, try jsonDetail(alloc, "request body must be valid JSON"), .bad_request, true);
        return;
    };
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => {
            // No object: systemone ⇒ "no state"; batch ⇒ the object message.
            const m = if (is_batch)
                "request body must be an object with 'states' and 'questions' fields"
            else
                "'state' is required";
            try respondJson(request, try jsonDetail(alloc, m), .bad_request, true);
            return;
        },
    };
    if (is_batch) {
        // knot validates the batch shape BEFORE the single-shot fields
        // (`{}` on /batch ⇒ the batch message, never `'state' is required`).
        if (!obj.contains("states") or !obj.contains("questions")) {
            try respondJson(request, try jsonDetail(alloc, "request body must be an object with 'states' and 'questions' fields"), .bad_request, true);
            return;
        }
        try respondJson(request, try jsonDetail(alloc, "http skeleton: inference not wired"), .not_implemented, true);
        return;
    }
    const state_val = obj.get("state") orelse {
        try respondJson(request, try jsonDetail(alloc, "'state' is required"), .bad_request, true);
        return;
    };
    if (state_val == .null) {
        try respondJson(request, try jsonDetail(alloc, "'state' is required"), .bad_request, true);
        return;
    }

    const questions_msg =
        "request body must be an object with a 'questions' field";
    const batch_msg =
        "request body must be an object with 'states' and 'questions' fields";
    const missing_msg = if (is_batch) batch_msg else questions_msg;

    const questions = obj.get("questions") orelse {
        if (is_batch) {
            try respondJson(request, try jsonDetail(alloc, batch_msg), .bad_request, true);
        } else {
            try respondJson(request, try jsonDetail(alloc, questions_msg), .bad_request, true);
        }
        return;
    };
    if (is_batch and !obj.contains("states")) {
        try respondJson(request, try jsonDetail(alloc, batch_msg), .bad_request, true);
        return;
    }
    const qmap = switch (questions) {
        .object => |m| m,
        else => {
            try respondJson(request, try jsonDetail(alloc, missing_msg), .bad_request, true);
            return;
        },
    };
    var qit = qmap.iterator();
    while (qit.next()) |entry| {
        const qtype = switch (entry.value_ptr.*) {
            .object => |qm| qm.get("type") orelse {
                try respondJson(request, try jsonDetail(alloc, "'questions' must match the request schema"), .bad_request, true);
                return;
            },
            else => {
                try respondJson(request, try jsonDetail(alloc, "'questions' must match the request schema"), .bad_request, true);
                return;
            },
        };
        const ok_type = switch (qtype) {
            .string => |s| std.mem.eql(u8, s, "choice") or
                std.mem.eql(u8, s, "score") or std.mem.eql(u8, s, "noul"),
            else => false,
        };
        if (!ok_type) {
            try respondJson(request, try jsonDetail(alloc, "'questions' must match the request schema"), .bad_request, true);
            return;
        }
    }

    // --- valid request → engine (P2-5 replaces the 501 stub) -----------
    var eerr: ?[]const u8 = null;
    const resp = eng.predict(alloc, parsed.value, &eerr) catch |e| {
        switch (e) {
            error.Rejected => try respondJson(request, try jsonDetail(alloc, eerr orelse "invalid request"), .unprocessable_entity, true),
            error.CheckpointUnavailable => try respondJson(request, try jsonDetail(alloc, eerr orelse "unavailable"), .service_unavailable, true),
            else => try respondJson(request, try jsonDetail(alloc, "inference failed"), .internal_server_error, true),
        }
        return;
    };
    try respondJson(request, resp, .ok, true);
}
fn serveConn(io: std.Io, alloc: std.mem.Allocator, stream: net.Stream, eng: *engine_mod.Engine, name: []const u8, etag: ?[]const u8) void {
    var rbuf: [16384]u8 = undefined;
    var wbuf: [16384]u8 = undefined;
    var reader = stream.reader(io, &rbuf);
    var writer = stream.writer(io, &wbuf);
    var server = http.Server.init(&reader.interface, &writer.interface);
    while (true) {
        var request = server.receiveHead() catch break; // EOF or bad head
        route(alloc, eng, &request, name, etag) catch break;
    }
}

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var port: ?u16 = null;
    var model_dir: ?[]const u8 = null;
    // toSlice includes argv0 — start at index 1 (the exe path would otherwise
    // win the "first arg containing /" race for model_dir).
    var ai: usize = 1;
    while (ai < args.len) : (ai += 1) {
        const a = args[ai];
        if (port == null) {
            port = std.fmt.parseInt(u16, a, 10) catch null;
            if (port != null) continue;
        }
        if (model_dir == null and std.mem.indexOfScalar(u8, a, '/') != null) {
            model_dir = a;
        }
    }
    const p = port orelse {
        std.debug.print("usage: http_skeleton <port> <model_dir>\n", .{});
        std.process.exit(2);
    };
    const dir = model_dir orelse {
        std.debug.print("usage: http_skeleton <port> <model_dir>\n", .{});
        std.process.exit(2);
    };
    const name = basename(dir);
    const etag = snapshotEtag(io, arena, dir);

    // P2-5: the real engine (tokenizer + prompt + ORT + decode + routing).
    var eng: engine_mod.Engine = undefined;
    try eng.init(io, arena, dir, name);
    defer eng.deinit();

    var addr = try net.IpAddress.parseIp4("127.0.0.1", p);
    var listener = try addr.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    std.debug.print("http_skeleton listening 127.0.0.1:{d} model={s} etag={s}\n", .{ p, name, etag orelse "null" });

    while (true) {
        var stream = listener.accept(io) catch |e| {
            std.debug.print("accept: {s}\n", .{@errorName(e)});
            break;
        };
        serveConn(io, arena, stream, &eng, name, etag);
        stream.close(io);
    }
}
