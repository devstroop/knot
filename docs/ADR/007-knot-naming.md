# ADR-007: knot naming (OIO parent, knot engine)

- Status: accepted
- Supersedes: 002 (naming clauses only — see below)

## Context

The `oio` name now denotes the parent initiative, Open Intelligence
Operations: the orchestrator layer plus everything under it. The decision
engine in this repository needs an identity of its own. Alternatives
considered: keeping `oio` on the engine (rejected — the name is needed at the
parent/orchestrator level), `judgement`/`deem` (rejected in favor of the
maintainer's choice), and scoped variants such as `oio-knot` (rejected —
the decision is the plain name `knot`, qualified as "OIO Knot" in prose).

## Decision

- Product: the decision engine is **Knot** ("OIO Knot" on first mention).
- Repository: `devstroop/knot` (via GitHub rename; redirects preserve history).
- Crates: `knot` (library) and `knot-serve`; the installed binary is `knot`.
- Config: `KNOT_*` env vars (hard cut, no `OIO_*` shims); mapping table in
  README/docs.
- MCP tools: `knot_status`, `knot_route`, `knot_predict`, `knot_predict_batch`.
- Default cache dir: `~/.cache/knot` (`KNOT_CACHE` override).
- Wire protocol (`/v1/systemone`), checkpoint formats, and eval semantics are
  unchanged by the rename; parity fixtures must pass byte-identical.

## Non-decisions (explicitly frozen)

- `crates.io`: the bare `knot` crate name is taken (unrelated RAG indexer),
  so `cargo publish` of a `knot` library is blocked. Local builds, path/git
  dependencies, and CI are unaffected, and nothing here publishes today.
  Revisit only if publishing matters.
- Data-plane identifiers were frozen here at rename time, to avoid
  invalidating prepared corpora, manifests, and feature caches for
  cosmetics. **Revoked 2026-10-07** (maintainer direction: no `oio` in live
  identifiers): the training dataset is now `knot-authored/score-smoke`
  (revision `knot-training-v1`) and internal `_knot_row_index` keys replace
  `_oio_row_index`; corpora and caches issued under the old IDs are
  regenerated or rebuilt as described in MIGRATION.md. Still frozen as
  recorded history: evidence artifacts (`demo/results.json`) keep their
  original names.
- Historical ADRs (001–006) keep their original names as written; history is
  not rewritten.

## Consequences

- ADR-002's `OIO_*` naming clause and `~/.cache/oio` default are superseded
  by this record. Its Apache-2.0 clause stands.
- Operators re-point one prefix (`OIO_*` → `KNOT_*`) and one cache path;
  MCP clients re-point four tool names. A Laya client repoints cleanly, as
  before — compat lives at the protocol level, not the branding level.
- Any behavioral difference detected by the parity fixtures after the rename
  is a bug in the rename, not a behavior change: back it out and redo it
  mechanically.
