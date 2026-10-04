# ADR-003: cargo workspace of oio + oio-serve

- Status: accepted
- Supersedes: —

## Context

One crate with lib+bin vs a small workspace.

## Decision

Workspace with `crates/oio` (library) and `crates/oio-serve` (binary). Heavy
dependencies (`ort`, `tokenizers`, `axum`) are feature-gated to keep
`cargo check -p oio` fast and CI-light.

## Consequences

- Library consumers don't compile axum/ONNX unless they opt in.
- Additional binaries (MCP, evals CLI) slot in as sibling crates later.
