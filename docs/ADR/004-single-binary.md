# ADR-004: one `oio-serve` binary, MCP behind `--mcp`

- Status: accepted
- Supersedes: —

## Context

PLAN carried the open decision "whether `oio-serve` and MCP ship in one
binary or two". MCP (PRD F12, "v2 of serve crate") needs the same engine,
config and auth surface as HTTP; only the transport differs.

## Decision

One binary. `oio-serve` serves the HTTP contract (`POST /v1/systemone`,
`/batch`, `/health`, `/models`) by default and switches to MCP stdio when
passed `--mcp` (`oio_serve::mcp::run_stdio`). Both paths construct the same
`Engine` from the same `OIO_*` configuration before branching
(`crates/oio-serve/src/main.rs`).

## Consequences

- One artifact to build, version and deploy; the HTTP and MCP surfaces cannot
  drift apart in engine behaviour or config parsing.
- An MCP-only embed still links the whole binary's dependency tree (axum
  included). If packaging needs ever diverge, split and supersede this ADR.
