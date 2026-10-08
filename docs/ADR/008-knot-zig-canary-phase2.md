# ADR-008: knot-in-zig canary — phase-2 gate (session wiring + parity + bench)

- Status: proposed
- Supersedes: —
- Related: [ADR-001](001-ort-first.md) (ORT-first runtime pin — the session this must honour), [ADR-005](005-artifacts-and-fixtures.md) (parity-fixture format), [ADR-006](006-optional-nqlite-adapters.md) (the load-test-ladder gate template), nqlite-zig ADR-001/002 (`docs/ADR/…` — seam scope, *link Rust, transport Zig*; unchanged by this record), workspace `discussions/topic-12.md` (full verdicts + charter deliberation)

## Context

A canary spike (M0, 2026-10-08, `/tmp/opencode/knot-zig-spike/`) tested
whether knot's observable surface can be reproduced byte-for-byte in zig.
Every gate compared against an oracle binary built from knot's own pins —
never against prose:

| Gate | Oracle | Result |
|---|---|---|
| Tokenizer probes (NFD, emoji, digits, newlines) | Rust `tokenizers =0.21.4`, knot's exact `encode(..., false)` | **29/29 exact** |
| knot fixture strings (golden/engine/parity/lang_cases/eval/massive_hi) | same oracle | **249/249 exact** |
| NFC corpus | Python `unicodedata` (Unicode 13.0.0) | **2137/2137** |
| ONNX session | Rust `ort =2.0.0-rc.10` → same static `.a`, C API 1.22.0 | **byte-identical** logits hex `9ca25f3facbba1bd` |
| Prompt bytes | three-way: Python `json.dumps` ⇄ `pyjson.rs` ⇄ zig | **55/55** |
| HTTP transport vs live `knot-serve` (15-case probe corpus) | knot-serve itself | **15/15** (12 byte-identical + 3 expected-differ: zig 501 stub vs knot 200) |

The two predeclared go/no-go bars — tokenizer byte-parity, then ORT link
parity — were both cleared. What the canary did **not** prove: no
inference is wired into the zig HTTP path (the 501 stub is where the
session would run), no end-to-end answers, no perf numbers, no ops/CI
story for the hand-fitted static-link recipe, single canary.

nqlite-zig ADR-001 chartered knot *out of its v1*; ADR-002 recorded the
seam decision (*link Rust, transport Zig*, revisit clause reserved for a
measured need + ops evidence). Those decide the **nqlite engine linkage**
— not knot's own implementation. This record does not touch either; it
charterizes the next, bounded unit of evidence so the scope question is
next answered with a product-level test rather than component gates.

## Decision

1. **Charter phase-2 canary (bounded):** wire the ORT C-API session into
   `main_http.zig`, replacing the `http skeleton: inference not wired`
   501 stub, and serve knot's own end-to-end cases through the zig path
   side-by-side with `knot-serve`.

2. **Exit gates (predeclared — thresholds may not be revised after the
   run; ADR-006's ladder is the gate-writing template):**

   | Gate | Pass condition |
   |---|---|
   | G1 transport regression | the phase-1 corpus stays **15/15** (3 expected-differ unchanged) |
   | G2 answer parity | knot's parity fixtures (`golden_english`, `engine_english`, `parity_english`, `lang_cases`) served by zig **byte-match** `knot-serve` answers; fixture set + N frozen *before* the first run |
   | G3 benchmark row | one row recorded against Rust knot (cold start, binary size, RSS, tok/s) — **measurement only, no threshold claim**; nqlite-zig's `docs/BENCHMARKING.md` is the template |
   | G4 issue #37 exercised | the single-checkpoint `multilingual` routing-500 case is *defined* in the zig path (fallback vs fail-loud per #37's option pick) — not silently divergent |

3. **Explicit non-goals:** no scope reversal (PRD §4 unchanged), no repo
   promotion, no training path, no candle runtime, no MCP surface, no
   change to the nqlite seam (ADR-002 stands). The spike stays in
   `/tmp/opencode/knot-zig-spike/` until gates decide otherwise.

## Consequences

- A bounded, oracle-driven unit of work converts "component parity" into
  "product parity — or a concrete gap list".
- If gates go **green**: the reverse/scope/promotion question returns to
  `discussions/topic-12.md` with end-to-end evidence *and* the first
  perf numbers (§4's untested hypotheses retired in either direction).
- If gates go **red**: the gap list is the answer; this ADR and its
  predecessors stand unchanged and the spike archives as evidence.
- knot's observable contract remains the oracle throughout: whatever
  ships, `/v1/systemone` byte behavior is pinned by fixtures, not by
  implementation language.

## Revisit when

- phase-2 gates are decided (green → an ADR on scope/promotion with the
  benchmark row attached; red → close the line), **or**
- ADR-002's own revisit clause fires (measured seam need + ops evidence).
