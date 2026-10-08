//! G4: checkpoint router — port of knot's `router.rs` decision logic plus
//! the #37 hybrid fallback (`with_fallback` / resolve → 503-class error).
//! Oracles: `golden_english.response.routing` (12 byte-exact blocks) and the
//! fallback/503 rules as merged in knot PR #39.
const std = @import("std");
const lang_mod = @import("lang.zig");
const qmod = @import("question.zig");

pub const BUNDLE_REPO = "convaiinnovations/laya";

const ModelSpec = struct { key: []const u8, repo: []const u8, sub: ?[]const u8 };
pub const DEFAULT_MODELS = [_]ModelSpec{
    .{ .key = "english", .repo = BUNDLE_REPO, .sub = null },
    .{ .key = "multilingual", .repo = BUNDLE_REPO, .sub = "multilingual" },
    .{ .key = "typed-decisions", .repo = BUNDLE_REPO, .sub = "typed-decisions" },
};

const Alias = struct { from: []const u8, to: []const u8 };
const ALIASES = [_]Alias{
    .{ .from = "en", .to = "english" },
    .{ .from = "laya", .to = "english" },
    .{ .from = "default", .to = "english" },
    .{ .from = "multi", .to = "multilingual" },
    .{ .from = "ml", .to = "multilingual" },
    .{ .from = "laya-multilingual", .to = "multilingual" },
    .{ .from = "typed", .to = "typed-decisions" },
    .{ .from = "typed_decisions", .to = "typed-decisions" },
    .{ .from = "laya-typed-decisions", .to = "typed-decisions" },
    .{ .from = "decisions", .to = "typed-decisions" },
};

const TypedWorkflow = struct { name: []const u8, ids: []const []const u8 };
const TYPED_DECISION_WORKFLOWS = [_]TypedWorkflow{
    .{ .name = "agent_trace_observability", .ids = &.{ "action", "needs_review", "outcome", "risk", "urgency" } },
    .{ .name = "customer_service", .ids = &.{ "action", "category", "churn_risk", "needs_human", "urgency" } },
    .{ .name = "invoice_processing", .ids = &.{ "discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency" } },
    .{ .name = "security_incidents", .ids = &.{ "credential_compromise", "disposition", "severity", "true_positive", "urgency" } },
};

const LANGUAGE_AGNOSTIC_CODES = [_][]const u8{ "c", "posix", "c.utf-8", "posix.utf-8", "und", "mul", "mis", "zxx", "art", "qaa" };
const ENGLISH_SUBTAGS = [_][]const u8{"en"};

fn asciiTrimLower(alloc: std.mem.Allocator, s: []const u8) ![]const u8 {
    const t = std.mem.trim(u8, s, " \t\r\n");
    const out = try alloc.alloc(u8, t.len);
    for (t, 0..) |c, i| out[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
    return out;
}

/// Name of the typed-decisions workflow whose question ids exactly match.
pub fn matchTypedWorkflow(alloc: std.mem.Allocator, question_ids: []const []const u8) !?[]const u8 {
    for (TYPED_DECISION_WORKFLOWS) |wf| {
        if (wf.ids.len != question_ids.len) continue;
        var all = true;
        outer: for (wf.ids) |want| {
            for (question_ids) |have| {
                if (std.mem.eql(u8, want, have)) {
                    continue :outer;
                }
            }
            all = false;
            break;
        }
        _ = alloc;
        if (all) return wf.name;
    }
    return null;
}

/// `normalise_name` — alias map + known-checkpoint gate (caller formats knot's
/// `unknown model {name:?}` InvalidRequest when this returns null).
pub fn normaliseName(alloc: std.mem.Allocator, name: []const u8) !?[]const u8 {
    const key0 = try asciiTrimLower(alloc, name);
    var key: []const u8 = key0;
    for (ALIASES) |a| {
        if (std.mem.eql(u8, a.from, key)) {
            key = a.to;
            break;
        }
    }
    for (DEFAULT_MODELS) |m| {
        if (std.mem.eql(u8, m.key, key)) return m.key;
    }
    return null;
}

/// True/False for a language code, or null when the code identifies nothing.
pub fn englishFromCode(alloc: std.mem.Allocator, value: []const u8) !?bool {
    const code0 = try asciiTrimLower(alloc, value);
    if (code0.len == 0) return null;
    const dot = std.mem.indexOfScalar(u8, code0, '.') orelse code0.len;
    const code = code0[0..dot];
    const primary0 = try std.mem.replaceOwned(u8, alloc, code, "_", "-");
    const dash = std.mem.indexOfScalar(u8, primary0, '-') orelse primary0.len;
    const primary = primary0[0..dash];
    if (primary.len == 0) return null;
    for (LANGUAGE_AGNOSTIC_CODES) |c| {
        if (std.mem.eql(u8, c, primary)) return null;
    }
    for (ENGLISH_SUBTAGS) |c| {
        if (std.mem.eql(u8, c, primary)) return true;
    }
    return false;
}

/// `repo_str`: `repo` or `repo/subfolder`.
pub fn repoStr(alloc: std.mem.Allocator, key: []const u8) ![]const u8 {
    for (DEFAULT_MODELS) |m| {
        if (std.mem.eql(u8, m.key, key)) {
            return if (m.sub) |s|
                try std.fmt.allocPrint(alloc, "{s}/{s}", .{ m.repo, s })
            else
                try alloc.dupe(u8, m.repo);
        }
    }
    return try alloc.dupe(u8, BUNDLE_REPO);
}

pub const Fallback = struct { requested: []const u8, served: []const u8 };

pub const Decision = struct {
    model: []const u8,
    repo: []const u8,
    reason: []const u8,
    detection: ?lang_mod.Analysis,
    workflow: ?[]const u8,
    fallback: ?Fallback = null,

    fn new(alloc: std.mem.Allocator, model: []const u8, reason: []const u8, detection: ?lang_mod.Analysis, workflow: ?[]const u8) !Decision {
        return .{
            .model = model,
            .repo = try repoStr(alloc, model),
            .reason = reason,
            .detection = detection,
            .workflow = workflow,
        };
    }

    /// Issue #37 (`with_fallback`): repoint at the checkpoint that will
    /// actually run and record the substitution.
    pub fn withFallback(self: *Decision, alloc: std.mem.Allocator, served: []const u8) !void {
        self.fallback = .{ .requested = self.model, .served = served };
        self.reason = try std.fmt.allocPrint(
            alloc,
            "{s} — checkpoint {s} is not configured here; served from {s} instead",
            .{ self.reason, try qmod.rustDebug(alloc, self.model), try qmod.rustDebug(alloc, served) },
        );
        self.model = served;
        self.repo = try repoStr(alloc, served);
    }
};

fn optDebug(alloc: std.mem.Allocator, v: ?[]const u8) ![]const u8 {
    if (v) |s| return std.fmt.allocPrint(alloc, "Some({s})", .{try qmod.rustDebug(alloc, s)});
    return "None";
}

pub const RouteParams = struct {
    state: std.json.Value,
    question_ids: []const []const u8 = &.{},
    model: ?[]const u8 = null,
    task: ?[]const u8 = null,
    lang: ?[]const u8 = null,
    lang_guess: ?[]const u8 = null,
    /// Router default (`KNOT_DEFAULT_MODEL` / "english").
    default: []const u8 = "english",
    auto_task_detection: bool = false,
    router_lang_guess: ?[]const u8 = null,
};

/// `router.route` — decision only; resolution against configured sources is
/// `resolve`.
pub fn route(alloc: std.mem.Allocator, p: RouteParams) !Decision {
    if (p.model) |model| {
        const key = (try normaliseName(alloc, model)) orelse {
            return error.UnknownModel;
        };
        return Decision.new(alloc, key, try std.fmt.allocPrint(alloc, "explicit model={s}", .{try qmod.rustDebug(alloc, model)}), null, null);
    }
    if (p.task) |task| {
        const lower = try asciiTrimLower(alloc, task);
        const norm = try std.mem.replaceOwned(u8, alloc, lower, "-", "_");
        const key = if (std.mem.eql(u8, norm, "typed_decisions"))
            "typed-decisions"
        else
            (try normaliseName(alloc, task)) orelse return error.UnknownModel;
        return Decision.new(alloc, key, try std.fmt.allocPrint(alloc, "explicit task={s}", .{try qmod.rustDebug(alloc, task)}), null, null);
    }

    const workflow = try matchTypedWorkflow(alloc, p.question_ids);
    if (workflow != null and p.auto_task_detection) {
        const wf = workflow.?;
        return Decision.new(
            alloc,
            "typed-decisions",
            try std.fmt.allocPrint(alloc, "question ids match the {s} typed-decisions workflow", .{try qmod.rustDebug(alloc, wf)}),
            null,
            wf,
        );
    }

    if (p.lang) |lang| {
        if (try englishFromCode(alloc, lang)) |is_en| {
            const key: []const u8 = if (is_en) "english" else "multilingual";
            return Decision.new(
                alloc,
                key,
                try std.fmt.allocPrint(alloc, "explicit lang={s}", .{try qmod.rustDebug(alloc, lang)}),
                null,
                workflow,
            );
        }
    }

    for ([_]?[]const u8{ p.lang_guess, p.router_lang_guess }) |hint| {
        if (hint) |h| {
            if (try englishFromCode(alloc, h)) |is_en| {
                const key: []const u8 = if (is_en) "english" else "multilingual";
                return Decision.new(
                    alloc,
                    key,
                    try std.fmt.allocPrint(
                        alloc,
                        "caller identified this as {s} text",
                        .{if (is_en) "English" else "non-English"},
                    ),
                    null,
                    workflow,
                );
            }
        }
    }

    const det = try lang_mod.analyse(alloc, p.state);
    const det_json_supported = true;
    _ = det_json_supported;
    if (std.mem.eql(u8, det.script, "unknown")) {
        const reason = try std.fmt.allocPrint(alloc, "no letters detected in state; using default ({s})", .{p.default});
        return Decision.new(alloc, p.default, reason, det, workflow);
    }
    if (!std.mem.eql(u8, det.script, "latin")) {
        const pct = 100.0 * det.non_latin_fraction;
        const reason = try std.fmt.allocPrint(
            alloc,
            "non-Latin script ({s}, {d:.0}% of letters); the English checkpoint cannot read it",
            .{ det.script, pct },
        );
        return Decision.new(alloc, "multilingual", reason, det, workflow);
    }
    if (!det.is_english) {
        const reason: []const u8 = if (det.mixed_segment) |mixed|
            try std.fmt.allocPrint(
                alloc,
                "Latin script, mostly English, but a line or field reads as {s} ({s}); the English checkpoint cannot read it",
                .{
                    try optDebug(alloc, det.language),
                    try qmod.rustDebug(alloc, lang_mod.truncStr(mixed, 60)),
                },
            )
        else if (det.language) |lg|
            try std.fmt.allocPrint(alloc, "Latin script but language looks like {s}, not English", .{try qmod.rustDebug(alloc, lg)})
        else
            try std.fmt.allocPrint(
                alloc,
                "Latin script, language not identified but {d:.0}% non-English letters; not safe for the English checkpoint",
                .{100.0 * det.diacritic_rate},
            );
        return Decision.new(alloc, "multilingual", reason, det, workflow);
    }
    if (det.language_undecided) {
        const reason = try std.fmt.allocPrint(alloc, "Latin script, language not identified and no non-English letters; using default ({s})", .{p.default});
        return Decision.new(alloc, p.default, reason, det, workflow);
    }
    return Decision.new(alloc, "english", "English Latin text", det, workflow);
}

/// #37 resolve: configured → unchanged; unconfigured → fallback to a
/// configured source (flagged); nothing configured → knot's
/// `CheckpointUnavailable` message (serve maps it to 503).
pub const ResolveError = error{CheckpointUnavailable};

pub const Sources = struct {
    configured: []const []const u8,
    default: []const u8,

    fn has(self: *const Sources, name: []const u8) bool {
        for (self.configured) |c| {
            if (std.mem.eql(u8, c, name)) return true;
        }
        return false;
    }

    fn fallbackSource(self: *const Sources) ?[]const u8 {
        if (self.has(self.default)) return self.default;
        if (self.configured.len > 0) return self.configured[0];
        return null;
    }
};

pub fn resolve(alloc: std.mem.Allocator, d: *Decision, sources: *const Sources) ResolveError!void {
    if (sources.has(d.model)) return;
    const served = sources.fallbackSource() orelse {
        // knot Error::CheckpointUnavailable display text
        return error.CheckpointUnavailable;
    };
    d.withFallback(alloc, served) catch return error.CheckpointUnavailable;
}

/// The exact `CheckpointUnavailable` display string (knot error.rs).
pub fn unavailableMessage(alloc: std.mem.Allocator, requested: []const u8) ![]const u8 {
    return std.fmt.allocPrint(
        alloc,
        "checkpoint {s} is not configured; this deployment has no fallback checkpoint (set KNOT_MODEL_DIR or KNOT_DEFAULT_MODEL)",
        .{try qmod.rustDebug(alloc, requested)},
    );
}
