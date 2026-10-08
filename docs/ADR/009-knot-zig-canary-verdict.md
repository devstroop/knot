# ADR-009: knot-in-zig canary verdict — record the green gates, keep Rust canonical, charter phase-3 entry

- Status: proposed
- Supersedes: — (discharges ADR-008's *revisit* clause; does not amend it)
- Related: [ADR-008](008-knot-zig-canary-phase2.md) (the charter this executes), [ADR-001](001-ort-first.md) (runtime pin), [ADR-005](005-artifacts-and-fixtures.md) (the oracle discipline), nqlite-zig ADR-001/002 (scoped knot out of its v1; *link Rust, transport Zig*), workspace `discussions/topic-12.md` (deliberation + final verdict table)

## Context

ADR-008 chartered a bounded phase-2 canary: wire the ORT session into the
zig HTTP path and gate it on G1 (transport), G2 (answer parity), G3 (one
benchmark row), G4 (the #37 single-checkpoint case). All four ran green on
2026-10-08, verified on ReleaseFast:

| Gate | Result |
|---|---|
| G1 transport | `http_diff` **15/15 with zero expected-differ** — the phase-1 stubs (`minimal_ok`, both content-type cases) now byte-match live knot; `golden_diff` **16/16** full-body (12 golden + engine + #37 fallback cases over the wire) |
| G2 answers | sequences/collated byte-match knot's Rust goldens; session within knot's own `< 5e-3` torch-parity (**max \|dev\| 1.1e-5**); answers + usage byte-match `engine_english`; `lang_cases` 18/18; routing golden 12/12 |
| G3 bench row (128 tok/req ×30, both release builds) | cold start 3.44 vs 3.24 s (×1.06) · binary **31.3 vs 36.2 MB (×0.86)** · peak RSS 1585 vs 1559 MB (×1.02) · latency 276.6 vs 300.2 ms/req (×0.92) · throughput 463 vs 426 tok/s (×1.09) |
| G4 #37 defined in zig | flagged `{multilingual → english}` fallback with reason suffix; `CheckpointUnavailable` message string-equal to knot's 503 detail; both confirmed over HTTP |

The technical risk the canary existed to test — *can zig reproduce knot's
observable contract byte-for-byte?* — is retired. What the canary did
**not** build: valid `/batch` (501 stub), the MCP surface, `predict_long`
windows, the `min_confidence` gate, shortlist, multi-checkpoint LRU,
integrity verification, `KNOT_DEVICE`/thread knobs, a CI matrix, and any
platform other than this Linux x86_64 box. The port reached
single-checkpoint parity in roughly four zig modules plus generators
(`question`/`prompt`/`session`/`decode`/`lang`/`route`/`engine`, ~3.5k
lines) — a real investment, and still one deployment shape.

## Decision

1. **Record the verdicts as knot evidence** (table above; gate scripts
   `p2_parity`, `http_diff.py`, `golden_diff.py`, `bench_p2.py` are the
   reproduction commands). The byte-parity risk is closed with oracle
   evidence on both sides of every gate.

2. **Scope stays unchanged.** Rust remains the canonical implementation
   (PRD §4 untouched); nqlite-zig ADR-002's *link Rust, transport Zig*
   stands. One green canary on one deployment shape is not a second
   implementation — the gap list above is the honest distance to parity,
   and G3's wins (8–14 %) are promising, not transformative.

3. **The spike is preserved as evidence**, not promoted: curated source +
   goldens + gate scripts land on branch `spike/knot-zig` of this repo
   (no CI impact, no `main` surface change), so every gate in the table
   stays reproducible after `/tmp` is gone.

4. **Phase-3 is chartered, not scheduled.** Re-entry requires a *driver*:
   (a) a deployment need for the zig host (footprint/latency/deploy
   shape), or (b) nqlite-zig's sidecar story wanting knot alongside the
   zig engine in-process. Phase-3's pre-defined scope: `/batch`, MCP,
   `predict_long` + `min_confidence`, shortlist, multi-checkpoint LRU,
   integrity check, CI matrix — same oracle method, same gate style, with
   the G4 fallback semantics already pinned.

## Consequences

- knot users see zero change; the evidence is now durable and citable
  (`topic-12` carries the full table).
- Promotion becomes a *smaller* future proposal: method validated, numbers
  recorded, gaps enumerated — the unknown surface is the gap list, not
  feasibility.
- The spike branch is evidence-grade only: it is not built or linted by
  `main` CI.

## Revisit when

- a phase-3 driver in (a) or (b) appears — propose phase-3 citing this
  record, **or**
- any gate in the table regresses when re-run against a newer knot
  (the scripts are the regression suite), **or**
- ADR-002's own revisit clause fires (measured seam need + ops evidence).
