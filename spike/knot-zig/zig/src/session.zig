//! P2-2: ONNX Runtime session over the raw C API — the same static ORT
//! 1.22.0 knot's ort-sys links (recipe proven in the M0 `ortzig` spike;
//! header translated via `addTranslateC` → `@import("ort_c")`).
//!
//! Shapes mirror knot's `OnnxRuntime::forward`: inputs
//! `[n,lmax] i64 ×2, [n,kmax] i64, [n,kmax] bool, [n] i64`; outputs
//! `logits [n,*] f32` and `act_logits [n,*] f32` (dims read back, not assumed).
const std = @import("std");
const ort = @import("ort_c");

fn chk(status: ?*ort.OrtStatus, what: []const u8, api: *const ort.OrtApi) !void {
    if (status) |s| {
        const msg = api.GetErrorMessage.?(s);
        std.debug.print("ORT error in {s}: {s}\n", .{ what, std.mem.sliceTo(msg, 0) });
        api.ReleaseStatus.?(s);
        return error.OrtCallFailed;
    }
}

pub const ForwardOut = struct {
    logits: []f32,
    act: []f32,
    logits_dim1: usize,
    act_dim1: usize,
};

pub const Session = struct {
    api: *const ort.OrtApi,
    env: ?*ort.OrtEnv = null,
    opts: ?*ort.OrtSessionOptions = null,
    sess: ?*ort.OrtSession = null,
    info: ?*ort.OrtMemoryInfo = null,

    /// `path_z` must be NUL-terminated (`allocPrintZ`).
    pub fn load(path_z: [:0]const u8) !Session {
        const base_raw = ort.OrtGetApiBase();
        if (base_raw == null) return error.NoApiBase;
        const base: *const ort.OrtApiBase = base_raw;
        const api_raw = base.GetApi.?(@as(c_uint, ort.ORT_API_VERSION));
        if (api_raw == null) return error.NoApi;
        const api: *const ort.OrtApi = api_raw;

        var self = Session{ .api = api };
        errdefer self.deinit();
        try chk(api.CreateEnv.?(ort.ORT_LOGGING_LEVEL_WARNING, "knot-zig", &self.env), "CreateEnv", api);
        try chk(api.CreateSessionOptions.?(&self.opts), "CreateSessionOptions", api);
        try chk(api.CreateSession.?(self.env, path_z.ptr, self.opts.?, &self.sess), "CreateSession", api);
        try chk(
            api.CreateCpuMemoryInfo.?(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, &self.info),
            "CreateCpuMemoryInfo",
            api,
        );
        return self;
    }

    pub fn deinit(self: *Session) void {
        const api = self.api;
        if (self.info) |v| _ = api.ReleaseMemoryInfo.?(v);
        if (self.sess) |v| _ = api.ReleaseSession.?(v);
        if (self.opts) |v| _ = api.ReleaseSessionOptions.?(v);
        if (self.env) |v| _ = api.ReleaseEnv.?(v);
        self.* = undefined;
    }

    fn shape2d(self: *Session, val: *ort.OrtValue, what: []const u8) !struct { d0: usize, d1: usize } {
        const api = self.api;
        var tinfo: ?*ort.OrtTensorTypeAndShapeInfo = null;
        try chk(api.GetTensorTypeAndShape.?(val, &tinfo), what, api);
        defer _ = api.ReleaseTensorTypeAndShapeInfo.?(tinfo);
        var elem: c_uint = 0;
        try chk(api.GetTensorElementType.?(tinfo.?, &elem), what, api);
        if (elem != ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return error.UnexpectedElemType;
        var count: usize = 0;
        try chk(api.GetDimensionsCount.?(tinfo.?, &count), what, api);
        if (count != 2) return error.UnexpectedRank;
        var dims: [2]i64 = undefined;
        try chk(api.GetDimensions.?(tinfo.?, &dims, 2), what, api);
        if (dims[0] < 0 or dims[1] < 0) return error.DynamicDim;
        return .{ .d0 = @intCast(dims[0]), .d1 = @intCast(dims[1]) };
    }

    /// One collated forward pass. Input buffers are borrowed (ORT reads them
    /// during `Run`); returned slices are arena-owned copies.
    pub fn forward(
        self: *Session,
        alloc: std.mem.Allocator,
        ids: []const i64,
        att: []const i64,
        mpos: []const i64,
        mmask: []const bool,
        qtype: []const i64,
        n: usize,
        lmax: usize,
        kmax: usize,
    ) !ForwardOut {
        const api = self.api;
        const ids_shape = [_]i64{ @intCast(n), @intCast(lmax) };
        const mk_shape = [_]i64{ @intCast(n), @intCast(kmax) };
        const qt_shape = [_]i64{@intCast(n)};

        var t_ids: ?*ort.OrtValue = null;
        var t_att: ?*ort.OrtValue = null;
        var t_mpos: ?*ort.OrtValue = null;
        var t_mmask: ?*ort.OrtValue = null;
        var t_qt: ?*ort.OrtValue = null;
        var out_t: [2]?*ort.OrtValue = .{ null, null };
        var logits: []f32 = &.{};
        var act: []f32 = &.{};
        errdefer {
            for (out_t) |ot| {
                if (ot) |v| {
                    _ = api.ReleaseValue.?(v);
                }
            }
            for ([_]?*ort.OrtValue{ t_ids, t_att, t_mpos, t_mmask, t_qt }) |t| {
                if (t) |v| {
                    _ = api.ReleaseValue.?(v);
                }
            }
        }

        try chk(api.CreateTensorWithDataAsOrtValue.?(self.info, @constCast(ids.ptr), ids.len * 8, &ids_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids), "tensor input_ids", api);
        try chk(api.CreateTensorWithDataAsOrtValue.?(self.info, @constCast(att.ptr), att.len * 8, &ids_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_att), "tensor attention_mask", api);
        try chk(api.CreateTensorWithDataAsOrtValue.?(self.info, @constCast(mpos.ptr), mpos.len * 8, &mk_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_mpos), "tensor marker_pos", api);
        try chk(api.CreateTensorWithDataAsOrtValue.?(self.info, @constCast(mmask.ptr), mmask.len, &mk_shape, 2, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &t_mmask), "tensor marker_mask", api);
        try chk(api.CreateTensorWithDataAsOrtValue.?(self.info, @constCast(qtype.ptr), qtype.len * 8, &qt_shape, 1, ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_qt), "tensor qtype", api);

        const in_names = [_][*c]const u8{ "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype" };
        const in_tensors = [5]?*ort.OrtValue{ t_ids, t_att, t_mpos, t_mmask, t_qt };
        const out_names = [_][*c]const u8{ "logits", "act_logits" };

        try chk(api.Run.?(
            self.sess.?,
            null,
            &in_names,
            @ptrCast(&in_tensors),
            in_tensors.len,
            &out_names,
            out_names.len,
            &out_t,
        ), "Run", api);

        // logits: [n, d1]
        {
            const sh = try self.shape2d(
                out_t[0].?,
                "logits shape",
            );
            if (sh.d0 != n) return error.UnexpectedBatch;
            var ptr: [*]f32 = undefined;
            try chk(api.GetTensorMutableData.?(out_t[0], @ptrCast(&ptr)), "logits data", api);
            logits = try alloc.dupe(f32, ptr[0 .. n * sh.d1]);
        }
        // act_logits: [n, d1]
        {
            const sh = try self.shape2d(out_t[1].?, "act shape");
            if (sh.d0 != n) return error.UnexpectedBatch;
            var ptr: [*]f32 = undefined;
            try chk(api.GetTensorMutableData.?(out_t[1], @ptrCast(&ptr)), "act data", api);
            act = try alloc.dupe(f32, ptr[0 .. n * sh.d1]);
        }

        for (out_t) |t| _ = api.ReleaseValue.?(t);
        for ([_]?*ort.OrtValue{ t_ids, t_att, t_mpos, t_mmask, t_qt }) |t| {
            _ = api.ReleaseValue.?(t);
        }
        return .{
            .logits = logits,
            .act = act,
            .logits_dim1 = logits.len / n,
            .act_dim1 = act.len / n,
        };
    }
};
