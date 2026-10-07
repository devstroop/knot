# ADR-006: optional persistence adapters (nqlite) outside the inference path

- Status: proposed
- Supersedes: —

## Context

Two facts meet. First, knot already has composition seams that were designed
for substitution: `Embedder` (`crates/knot/src/shortlist.rs`) parameterizes
shortlist ranking, and `Predictor` (`crates/knot-serve/src/lib.rs`) exists so
"tests inject a stub" — both admit a decorator/back-end without touching
prompt assembly, the `Runtime` trait, or any SPEC invariant. Second, the
workspace contains [devstroop/nqlite](https://github.com/devstroop/nqlite), a
deterministic, zero-model, single-file store (pure Rust, MSRV 1.82) that
already does durable records + graph + vectors + time in one transaction —
and a retrieval-side spike of it already lives **outside** the crates in
[`docs/GITA-DEMO.md`](../GITA-DEMO.md) (`scripts/gita_nqlite_spike.py`,
`gita_nqlite_demo.py`, PR #24): nqlite hybrid beat knot's BM25 baseline on
holdout citation recall (1.000 vs 0.938) while lacking abstention (0.000 vs
0.333) — a comparison, explicitly not a migration.

What was missing is the boundary: which of the four possible integration
directions are admissible, and under what rules, decided *before* the first
adapter PR. The full reasoning (including the directions that are forbidden)
is mirrored in nqlite's
[`docs/integrations.md`](https://github.com/devstroop/nqlite/blob/main/docs/integrations.md);
this ADR is knot's side of that agreement. Overlaps with that document are
pointers, not restatements — a future divergence between the two is a finding
to file, not a silent fork.

## Decision

**Problem statement.** The driver is *forensics — seam B*: reconstructing
*what state, checkpoint revision, and policy produced this answer* after the
response is gone. The consumer is the operator debugging a bad decision;
prompt-iteration and the product-improving N3 study come later. Doing nothing
leaves every decision unreconstructable post-response — the metric that makes
that concrete is time-to-explain-one-bad-decision.

**Where nqlite may appear:**

| Position | Rank | Rule |
|---|---|---|
| After inference (audit/telemetry) | permitted | seam B below — the seam this ADR ships |
| Around inference (decorator) | permitted, **default off** | rules 1–7 |
| Before inference (shortlist prep) | deferred | see Sequencing — no call site exists today |
| During inference (retrieval in the prompt path) | **forbidden** | see Explicitly out of scope |

**nqlite never enters knot's core.** `crates/knot` and `crates/knot-serve` gain
no nqlite dependency; the inference path (`router → prompt → runtime →
answers`) stays exactly as it is; `cargo check -p knot` keeps building without
a store, a lock, or a system dependency. No model or learned weight may move
the other way either — nqlite's write path stays zero-model by its own D1.

**Optional adapters are admissible, behind feature gates, at two existing
seams:**

- **A — shortlist back-end.** An `Embedder` implementation backed by a
  nqlite store: options as rows, top-k via `vector::similarity` instead of
  the in-memory cosine. Value appears only when the store is *voted*: `:voted`
  edges on options turn the pre-filter into a feedback-ranked shortlist
  (`::score`), which is agent-side learning, not engine learning.
- **B — decision audit ledger.** A `Predictor` decorator that appends
  `(state) -[:decided]-> (decision)` plus decision rows (answers, confidence,
  routing, checkpoint revision, prompt hash) **after** the response returns.
  Buys `AS OF` forensics on past decisions and per-question-type confidence
  calibration from logged outcomes — knot's evidence harness gains provenance,
  not just replay.

**Rules every adapter must satisfy:**

1. **Single opener.** nqlite enforces an exclusive per-file lock (its issue
   #84): one `Arc`-shared handle per process, never per-request opens.
2. **Off the hot path.** Writes happen after the response; the semaphore
   gate and `x-inference-time-ms` must never wait on a WAL append. Writes go
   to a dedicated writer task behind the one shared handle; request handlers
   never touch the store (serve's admission `Semaphore` stays inference-side).
3. **Retention named up front.** A per-request ledger is unbounded growth by
   default; the adapter's ADR/PR states the prune or snapshot policy before
   merge (nqlite tracks history growth as its issue #95).
4. **Feature-gated, fast-path-preserving.** The `cargo check -p knot` and
   `cargo test --workspace` (no checkpoint) paths stay dependency-clean;
   adapter tests skip without a store, like the parity suites skip without a
   checkpoint.
5. **Behavioral claims go through fixtures.** Any observable serve change
   (response fields, headers) needs COMPAT/SPEC movement plus a fixture, per
   AGENTS.md — an adapter that changes the wire is not an adapter.
6. **Dependency direction (no cycles).** The adapter crate depends on
   `knot` **core** and `nqlite` only — never on `knot-serve`. Seam B is
   therefore expressed against a `Predictor` moved into core: that move is
   justified plumbing (the trait is the engine contract; `SystemOneRequest`
   is a core type) and lands as its **own commit**, not smuggled inside an
   adapter PR. `knot-serve` may then depend on the adapter behind its
   optional feature (single binary, ADR-004) — the arrow never reverses, so
   no cycle exists even when the gate is open.
7. **Votes never enter parity.** SPEC §8's tuple (same checkpoint + same
   request → same decisions) stays untouched: golden and parity runs are
   always votes-off. Votes-on is deterministic *given a store snapshot* — a
   different contract, stated as such in the adapter PR. The scoring math is
   nqlite's (D9: Laplace `::score` over `:voted`), not knot's; the adapter
   only maps confidence ≥ τ to a vote. Kill switch: per-deployment config,
   default off (env override for CI).

**Explicitly out of scope:** nqlite as a required or default dependency;
long-document chunk memory behind `predict_long` (windowing and budgets
already cover it — PRD §4's "callers compose" rule applies); any persistence
on the `Runtime` trait path; using nqlite as an embedding source *for the
model* (embeddings remain BYO/caller-supplied, unchanged). The one revisit
form for the prompt path: retrieval results computed out-of-band and
**pinned as request inputs** (reproducible under nqlite's context-bytes +
checkpoint-revision rule) — the nqlite engine itself is never called there.

**Sequencing.** Serial, one PR stream: this ADR lands first, then (1) the
audit decorator, (2) prove it (fixtures + skip-without-store, plus a
manual/scheduled real-checkpoint tier mirroring `KNOT_MODEL_DIR`), (3)
shortlist **only if** a shortlist route gains a call site in `knot-serve` —
persistence buys nothing while `CachedEmbedder` owns warm-start, and a wire
change needs its own ADR + sign-off — (4) votes only after the N3 scoring
spec. Per-PR budget ≈ 400 LoC with fixture deltas called out; CI is the
second reviewer.

## Consequences

- **Positive:** decision provenance and calibration history for free-shaped
  queries (`AS OF`, `::score`, regression triples over logged decisions);
  shortlists that can be audited and feedback-ranked; a self-hosted audit
  story that hosted Jev cannot tell; zero change to parity fixtures or SPEC
  invariants for the adapter layer itself.
- **Negative / accepted costs:** a feature-gate surface to maintain; one
  process-wide lock held for the server lifetime; retention policy must be
  designed (nqlite #95 is upstream of this); MSRV floor for adapter builds
  is max(1.88, 1.82) = 1.88 — fine, but the direction is one-way (nqlite can
  never depend back on knot: 1.82 floor + zero-model contract). Under
  sustained load the bounded write queue overflows as a **counted incident**
  (never silent), and a crash between response and flush can lose the
  in-flight rows — accepted for evidence written off the hot path.
- **Neutral:** the Gita spike/demo stay where they are — outside the crates,
  comparison-only, per their own framing.
- **Compat matrix:** both sides Apache-2.0 (license-compatible); MSRV
  direction as above; nqlite format v3 (see References) — the adapter PR
  pins its minimum nqlite version.

## Open questions (resolve in the adapter's own ADR/PR)

1. Ledger schema: one row per predict vs state-keyed dedup; what exactly goes
   in `state_hash` vs full state.
2. Auto-vote policy if/when confidence feeds back into shortlist ranking:
   weight, decay, and the human-review threshold below which nothing is
   written as a vote.
3. Retention mechanics: prune-by-age vs snapshot windows (coordinate with
   nqlite #95 rather than inventing a second mechanism).
4. Gate name and default: adapter off by default; what turns it on for a
   deployment.
5. Durability tier under sustained load: bounded-queue-plus-counted-incident
   (evidence-grade, the rule-2 default) vs drop-on-full (telemetry) — settle
   with numbers from the adapter's own load test.

## References

- nqlite [`docs/integrations.md`](https://github.com/devstroop/nqlite/blob/main/docs/integrations.md)
  — the joint boundary discussion (directions, naming, third-project loop)
- [`GITA-DEMO.md`](../GITA-DEMO.md) — nqlite hybrid spike numbers +
  `scripts/gita_nqlite_spike.py` (retrieval-side precedent, outside crates)
- `docs/PRD.md` §4 (non-goals discipline), AGENTS.md (SPEC/ADR rules)
- nqlite issues: #84 (single-writer lock, fixed), #95 (history retention),
  #111 (`nql-server --db`, merged); nqlite format **v3** (lazy-history layout,
  PR #138; v2 files still load, pre-#133 readers reject v3 loudly)
- nqlite E10 harness (`nqlite-experiments`) — ledger semantics already
  validated: forensics `AS OF` sweep vs an exact Python model, precedent
  recall, vote reliability, rotation equivalence
