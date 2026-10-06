# knot docs

Reading order and map of the documentation tree. Top level first; deepen one
level per working round.

| File | Owns | Change when |
|---|---|---|
| [PRD.md](PRD.md) | What & why: goals, requirements F1–F12, metrics | Product direction shifts |
| [PLAN.md](PLAN.md) | Architecture of work: milestones M0–M8, gates | Milestone scope/exit criteria move |
| [SPEC.md](SPEC.md) | Invariants: prompt format, budgets, confidence, errors, routing | Behavior contracts change |
| [COMPAT.md](COMPAT.md) | Jev/Laya compatibility contract, porting notes | Wire semantics or diffs change |
| [JEV-WIRE.md](JEV-WIRE.md) | Reference notes: what Laya's wire code actually does (evidence for COMPAT) | Laya reference behaviour is re-read |
| [RESEARCH-COMPARE.md](RESEARCH-COMPARE.md) | Evidence: recorded Jev vs Laya vs knot wire, serve deltas, quality/latency numbers, scope ranking | New research evidence lands or gaps move |
| [DEMO.md](DEMO.md) | Generated three-way run report: Jev recorded/published vs Laya vs knot over the English/feishu/Hindi corpora | `scripts/demo.sh` re-runs (raw evidence in `demo/results.json`) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate layout, one `predict()` data flow, `Runtime` trait | Crates/layers reorganize |
| [BRAND.md](BRAND.md) | Brand map: OIO parent, knot engine, nqlite memory; naming history | The family or the names change |
| [MIGRATION.md](MIGRATION.md) | Upgrading from `oio`: env/binary/MCP/cache mapping | Anything user-facing is renamed again |
| [BENCHMARKING.md](BENCHMARKING.md) | Paired ONNX/Candle method, measured numbers with bindings | Method changes or numbers are re-measured |
| [TRAINING.md](TRAINING.md) | Staged pipeline, data policy, predeclared gates, recorded verdict | Pipeline, data, or gates change |
| [GITA-DEMO.md](GITA-DEMO.md) | Retrieval + decision demos, eval sets, nqlite comparison | Demo, sets, or retrievers change |
| [ADR/](ADR/) | Why decisions were made (append-only) | New decision taken |

Rules:

1. SPEC changes **before** code when behavior moves.
2. ADRs are never edited, only superseded (`Supersedes: 00x`).
3. Keep this tree project-facing: product, contracts, and decisions — not
   development logs. Completed work is recorded in git history, not here.
