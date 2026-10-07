# Changelog

All notable changes to this project are documented here, newest first. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
is not yet SemVer-stable (0.x — breaking changes land in minor series with a
migration guide).

## [Unreleased]

### Added

- `Predictor` moved to core as **`knot::Predictor`** (ADR-006 rule 6): the
  inference-side contract now lives beside `SystemOneRequest`/`Engine`, so a
  future nqlite adapter crate can depend on `knot` without pulling
  `knot-serve` — which would otherwise form a package cycle through the
  single binary (ADR-004). `knot_serve::Predictor` keeps working via a
  re-export; no behavior change, parity fixtures untouched. Gated with the
  engine (`onnx`/`candle`), so a no-runtime build stays clean.
- **`knot-nqlite` adapter crate** (ADR-006 seam B): an opt-in `Predictor`
  decorator that appends one nqlite transaction per decision *after* the
  response — `state -[:decided]-> decision -[:used_checkpoint]-> checkpoint`
  plus a hash-only decision row (length-limited excerpt; full state is never
  stored). Off by default: enable per deployment with the `audit` feature of
  `knot-serve` + `KNOT_AUDIT_DB=<store-file>` (capacity/excerpt via
  `KNOT_AUDIT_CAPACITY` / `KNOT_AUDIT_EXCERPT`). Bounded queue with counted
  overflow (never blocks the hot path, never silent), dedicated writer
  thread owning the single store handle, `PRUNE HISTORY` on boot.

### Changed (breaking)

- Renamed the decision engine `oio` → **knot**: crates `oio`/`oio-serve` are
  now `knot`/`knot-serve`, the installed binary is `knot`, config moved
  `OIO_*` → `KNOT_*` (hard cut, no shims), MCP tools are `knot_*`, and the
  default cache dir is `~/.cache/knot`. See [docs/MIGRATION.md](docs/MIGRATION.md)
  and [docs/ADR/007-knot-naming.md](docs/ADR/007-knot-naming.md).
- Repository moved `devstroop/oio` → `devstroop/knot` (redirect preserved);
  `oio` stays as the parent initiative name.

Behavior, wire protocol (`/v1/systemone`), checkpoint formats, and eval
semantics are unchanged — parity fixtures pass byte-identical across the
rename.

## 0.1.0 — history

Earlier work is recorded in git history and the [decision
records](docs/ADR/): Laya-compatible decision engine (ONNX + Candle
runtimes), frozen-head training pipeline with predeclared quality gates and a
passing held-out verdict, Gita retrieval/decision demos, and the
frozen-encoder feature cache. No changelog was kept before this file; see
`git log` for the full record.
