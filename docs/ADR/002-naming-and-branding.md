# ADR-002: OIO_* naming, Apache-2.0, own cache dir

- Status: accepted
- Supersedes: —

## Context

Laya uses `LAYA_*` env vars and `~/laya_models`. oio is a separate product with
different defaults and a cleaner config surface.

## Decision

- Env config under `OIO_*` (see ARCHITECTURE.md).
- Cache under `~/.cache/oio` (overridable).
- Wire schema stays Laya/Jev-compatible — compat is at the protocol level, not
  the branding level.

## Consequences

- No silent cross-contamination with an existing Laya install.
- A Laya client repoints cleanly; an operator sets `OIO_*` once.
