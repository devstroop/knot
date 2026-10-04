# oio docs

Reading order and map of the documentation tree. Top level first; deepen one
level per working round.

| File | Owns | Change when |
|---|---|---|
| [PRD.md](PRD.md) | What & why: goals, requirements F1–F12, metrics | Product direction shifts |
| [PLAN.md](PLAN.md) | Architecture of work: milestones M0–M8, gates | Milestone scope/exit criteria move |
| [SPEC.md](SPEC.md) | Invariants: prompt format, budgets, confidence, errors, routing | Behavior contracts change |
| [COMPAT.md](COMPAT.md) | Jev/Laya compatibility contract, porting notes | Wire semantics or diffs change |
| [JEV-WIRE.md](JEV-WIRE.md) | Reference notes: what Laya's wire code actually does (evidence for COMPAT) | Laya reference behaviour is re-read |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate layout, one `predict()` data flow, `Runtime` trait | Crates/layers reorganize |
| [ADR/](ADR/) | Why decisions were made (append-only) | New decision taken |

Rules:

1. SPEC changes **before** code when behavior moves.
2. ADRs are never edited, only superseded (`Supersedes: 00x`).
3. Keep this tree project-facing: product, contracts, and decisions — not
   development logs. Completed work is recorded in git history, not here.
