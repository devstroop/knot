const std = @import("std");
const Io = std.Io;

const ort_spike = @import("ort_spike");
const ort = @import("ort_c"); // translated from include/onnxruntime_c_api.h

const SEQ: usize = 16;
const NUM_MARKERS: usize = 2;
const MODEL: [*c]const u8 = "/root/.cache/knot/english/laya.onnx";

// Shared verbatim with ort_golden/src/main.rs.
const IDS = [_]i64{
    10795, 253,  3634, 3662, 2634, 18037, 65,   32,
    510,   5570, 9300, 253,  2953, 285,   5783, 253,
};
const ATT: [SEQ]i64 = @splat(1);
const MPOS = [_]i64{ 4, 12 };
const MMASK = [_]bool{ true, true };
const QTYPE = [_]i64{0};

fn fail(status: ?*ort.OrtStatus, comptime what: []const u8, api: *const ort.OrtApi) !void {
    if (status) |s| {
        const msg = api.GetErrorMessage.?(s);
        std.debug.print("ORT error in {s}: {s}\n", .{ what, std.mem.sliceTo(msg, 0) });
        api.ReleaseStatus.?(s);
        return error.OrtCallFailed;
    }
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    _ = Io;
    _ = ort_spike;

    const base_raw = ort.OrtGetApiBase();
    if (base_raw == null) return error.NoApiBase;
    const base: *const ort.OrtApiBase = base_raw;
    const api_raw = base.GetApi.?(@as(c_uint, ort.ORT_API_VERSION));
    if (api_raw == null) return error.NoApi;
    const api: *const ort.OrtApi = api_raw;
    const ver = base.GetVersionString.?();
    std.debug.print("ort_version={s} api_version={d}\n", .{ std.mem.sliceTo(ver, 0), ort.ORT_API_VERSION });

    var env: ?*ort.OrtEnv = null;
    try fail(api.CreateEnv.?(ort.ORT_LOGGING_LEVEL_WARNING, "ort-spike", &env), "CreateEnv", api);

    var so: ?*ort.OrtSessionOptions = null;
    try fail(api.CreateSessionOptions.?(&so), "CreateSessionOptions", api);

    var session: ?*ort.OrtSession = null;
    try fail(api.CreateSession.?(env, MODEL, so.?, &session), "CreateSession", api);

    var alloc: ?*ort.OrtMemoryInfo = null;
    try fail(
        api.CreateCpuMemoryInfo.?(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, &alloc),
        "CreateCpuMemoryInfo",
        api,
    );

    const seq_shape = [_]i64{ 1, @as(i64, SEQ) };
    const mk_shape = [_]i64{ 1, @as(i64, NUM_MARKERS) };
    const qt_shape = [_]i64{1};

    var t_ids: ?*ort.OrtValue = null;
    var t_att: ?*ort.OrtValue = null;
    var t_mpos: ?*ort.OrtValue = null;
    var t_mmask: ?*ort.OrtValue = null;
    var t_qt: ?*ort.OrtValue = null;

    try fail(api.CreateTensorWithDataAsOrtValue.?(alloc, @constCast(&IDS), IDS.len * 8, &seq_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids), "tensor input_ids", api);
    try fail(api.CreateTensorWithDataAsOrtValue.?(alloc, @constCast(&ATT), ATT.len * 8, &seq_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_att), "tensor attention_mask", api);
    try fail(api.CreateTensorWithDataAsOrtValue.?(alloc, @constCast(&MPOS), MPOS.len * 8, &mk_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_mpos), "tensor marker_pos", api);
    try fail(api.CreateTensorWithDataAsOrtValue.?(alloc, @constCast(&MMASK), MMASK.len, &mk_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &t_mmask), "tensor marker_mask", api);
    try fail(api.CreateTensorWithDataAsOrtValue.?(alloc, @constCast(&QTYPE), QTYPE.len * 8, &qt_shape, 1, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_qt), "tensor qtype", api);

    const in_names = [_][*c]const u8{ "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype" };
    const in_tensors = [5]?*ort.OrtValue{ t_ids, t_att, t_mpos, t_mmask, t_qt };
    const out_names = [_][*c]const u8{"logits"};
    var out_tensors: [1]?*ort.OrtValue = .{null};

    try fail(api.Run.?(
        session.?,
        null, // OrtRunOptions — defaults, like knot
        &in_names,
        @ptrCast(&in_tensors),
        in_tensors.len,
        &out_names,
        out_names.len,
        &out_tensors,
    ), "Run", api);

    var out_ptr: [*]f32 = undefined;
    try fail(api.GetTensorMutableData.?(out_tensors[0], @ptrCast(&out_ptr)), "GetTensorMutableData", api);

    const pair = [_]f32{ out_ptr[0], out_ptr[1] };
    const hex = std.fmt.bytesToHex(std.mem.asBytes(&pair), .lower);
    std.debug.print("logits_hex={s}\n", .{&hex});
    std.debug.print("logits_f32[0]={d} [1]={d}\n", .{ out_ptr[0], out_ptr[1] });

    _ = api.ReleaseValue.?(out_tensors[0]);
    _ = api.ReleaseMemoryInfo.?(alloc);
    _ = api.ReleaseSession.?(session);
    _ = api.ReleaseSessionOptions.?(so);
    _ = api.ReleaseEnv.?(env);
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
