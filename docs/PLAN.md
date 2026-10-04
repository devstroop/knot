# oio Plan

Top-level milestone plan.

| ID | Milestone | Scope | Exit criteria |
|---|---|---|---|
| M0 | Contracts | Serde types for `/v1/systemone`, error taxonomy (422/413/415), fixture corpus harvested from `laya/tests` | Fixtures round-trip through types |
| M1 | Prompt layer | Port `build_head`/`build_sequence`/`state_room`/`window_budget`, render_options, noul labels, token cache + tokenizer lock | Unit tests byte-equal Laya's prompt token ids |
| M2 | ONNX runtime | Load Laya ONNX exports via `ort`, marker-position logits, temperatures, confidence; `ONNXAgent` parity | ≥99% decision agreement on fixtures |
| M3 | Router | Port `lang.py` script/stopword detection, checkpoint table, LRU, `OIO_DEFAULT_MODEL`, `route`/`route_batch` | Router picks same checkpoint as Laya on fixture inputs |
| M4 | Serve binary | axum: `/v1/systemone`, `/batch` ≤64, `/health`, `/models`, bearer auth, concurrency + budget caps, 413/422 semantics | Contract test suite mirrors `laya/tests/test_serve.py` shape |
| M5 | Long docs | Windowed `predict_long` with stride, offsets, max/most-confident window policy | Window-offset tests match Laya |
| M6 | Calibration + shortlist | Load `rl_agent_config.json`; embedding shortlist trait | Calibration temps match Laya config |
| M7 | MCP | stdio server with predict/route/batch tools | MCP local e2e passes |
| M8 | Native runtime | `Runtime` trait second impl (candle), CPU parity with ONNX path | Parity within fp tolerance on fixtures |

## Working rules

- One milestone per PR; CI gates: `cargo fmt --check`, `cargo clippy`, `cargo test`.
- Milestones keep the doc tables above in sync.
- Out-of-scope features are declared once, in `PRD.md` §4 with one-line
  rationales (evidence: `RESEARCH-COMPARE.md`); a milestone must not widen
  scope without editing that table first.
- Issues/bugs are not tracked here — only milestone exit criteria.

## Open decisions (next rounds)

- GPU timing (CUDA vs CPU measurements) — the ort CUDA EP itself landed
  behind `OIO_DEVICE=cuda` (SPEC §10); the numbers still owe a GPU runner run

Resolved: fixture format + export script → ADR-005; one binary vs two →
ADR-004; model cache layout → SPEC §9 (resolution order, `SHA256SUMS`);
device selection → SPEC §10 (explicit `cpu|cuda`, fail-fast).
