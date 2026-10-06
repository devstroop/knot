# ADR

Append-only architecture decision records. Never edit a merged ADR —
supersede it with a new one (`Supersedes: 00x`).

| ID | Title | Status |
|---|---|---|
| [001](001-ort-first.md) | ort-first runtime, ONNX artifacts reused from Laya | accepted |
| [002](002-naming-and-branding.md) | OIO_* naming, Apache-2.0, own cache dir | accepted |
| [003](003-workspace-layout.md) | cargo workspace of oio + oio-serve | accepted |
| [004](004-single-binary.md) | one `oio-serve` binary, MCP behind `--mcp` | accepted |
| [005](005-artifacts-and-fixtures.md) | checkpoint artifact layout, parity-fixture format | accepted |
| [006](006-optional-nqlite-adapters.md) | optional persistence adapters (nqlite) outside the inference path | proposed |
| [007](007-knot-naming.md) | knot naming: OIO parent, knot engine (supersedes 002 naming) | accepted |
