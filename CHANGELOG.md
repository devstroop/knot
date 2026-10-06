# Changelog

All notable changes to this project are documented here, newest first. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
is not yet SemVer-stable (0.x — breaking changes land in minor series with a
migration guide).

## [Unreleased]

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
