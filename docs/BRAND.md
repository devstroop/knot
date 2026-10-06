# Brand: OIO, knot, nqlite

**OIO (Open Intelligence Operations)** is the parent initiative. Under it:

| Name | Role | In one line |
|---|---|---|
| **oio** | Orchestrator / operations layer (separate project, future home: `devstroop/oio`) | Wires pipelines: retrieve → check evidence → judge → record → escalate |
| **knot** (this repo, `devstroop/knot`) | Decision engine + model | Renders typed judgments (`choice` / `score` / `noul`) over supplied options; stateless, no memory, no retrieval |
| **nqlite** (`devstroop/nqlite`) | Deterministic context database | Remembers and retrieves: records, graph relations, bring-your-own vectors; never decides |

Glossary: *oio wires the pipeline, nqlite supplies the evidence, and wherever the wiring demands judgment, a knot resolves it.*

## Naming history

This engine was previously named `oio` (repository `devstroop/oio`, crates
`oio` / `oio-serve`, config `OIO_*`). It was renamed to **knot** in
[ADR-007](ADR/007-knot-naming.md); the `oio` name stays with the parent and
the future orchestrator. The GitHub repository moved with a redirect, so old
links keep working. When referring to the pre-rename era in prose, write
"knot (formerly oio)" on first mention.

## Usage in other projects' docs

- First mention: **knot (formerly oio)**.
- Crate references: `knot` (library), `knot-serve` (server package), binary `knot`.
- Config prefix: `KNOT_*` (see [MIGRATION.md](MIGRATION.md) for the `OIO_*` mapping).
- Never use bare "knot" for the crates.io RAG indexer, Knot DNS, or the
  weddings brand — in this ecosystem it always means OIO's decision engine.
