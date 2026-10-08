//! pyjson three-way parity driver: render every corpus case through the zig
//! port and require byte-equality with BOTH Python's json.dumps and knot's
//! own pyjson::dumps.
//!
//!   pyjson_spike <pyjson_cases.json> <pyjson_py.json> <pyjson_rust.json>
//!
//! Exit 0 = all 55 match both oracles; 1 = any mismatch; 2 = usage.
const std = @import("std");
const Io = std.Io;
const pyjson = @import("pyjson.zig");

fn readAll(io: std.Io, alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try alloc.alloc(u8, st.size);
    const got = try file.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var paths: [3][]const u8 = undefined;
    var np: usize = 0;
    for (args) |a| {
        if (std.mem.endsWith(u8, a, ".json") and np < 3) {
            paths[np] = a;
            np += 1;
        }
    }
    if (np != 3) {
        std.debug.print(
            "usage: pyjson_spike <pyjson_cases.json> <pyjson_py.json> <pyjson_rust.json>\n",
            .{},
        );
        std.process.exit(2);
    }

    const cases_v = (try std.json.parseFromSlice(std.json.Value, arena, try readAll(io, arena, paths[0]), .{})).value;
    const py_v = (try std.json.parseFromSlice(std.json.Value, arena, try readAll(io, arena, paths[1]), .{})).value;
    const rust_v = (try std.json.parseFromSlice(std.json.Value, arena, try readAll(io, arena, paths[2]), .{})).value;

    const cases = cases_v.array.items;
    const py = py_v.array.items;
    const rust = rust_v.array.items;
    if (cases.len != py.len or cases.len != rust.len) {
        std.debug.print("case count mismatch: {d} vs {d} vs {d}\n", .{ cases.len, py.len, rust.len });
        std.process.exit(1);
    }

    var pass: usize = 0;
    var fail: usize = 0;
    for (cases, 0..) |case, i| {
        const got = try pyjson.dumps(arena, case);
        const want_py = py[i].string;
        const want_rust = rust[i].string;
        if (std.mem.eql(u8, got, want_py) and std.mem.eql(u8, got, want_rust)) {
            pass += 1;
        } else {
            fail += 1;
            std.debug.print(
                "MISMATCH case {d}\n  py  : {s}\n  rust: {s}\n  zig : {s}\n",
                .{ i, want_py, want_rust, got },
            );
        }
    }
    std.debug.print("pyjson: {d}/{d} match python+knot-rust\n", .{ pass, pass + fail });
    _ = Io;
    std.process.exit(if (fail == 0) 0 else 1);
}
