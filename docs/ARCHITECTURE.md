# knot Architecture

## Crates

```
knot/
├── crates/
│   ├── knot/              # library — wire contract + inference path
│   │   ├── src/
│   │   │   ├── lib.rs            # public API surface, feature gates
│   │   │   ├── protocol.rs       # wire types (/v1/systemone request/response)
│   │   │   ├── error.rs          # error taxonomy → HTTP codes
│   │   │   ├── integrity.rs      # SHA256SUMS verification at load (SPEC §9)
│   │   │   ├── prompt.rs         # prompt assembly + criteria render
│   │   │   ├── pyjson.rs         # Python-json byte-compat rendering (wire parity)
│   │   │   ├── router.rs         # lang/script detect + checkpoint LRU
│   │   │   ├── lang.rs           # lang/script detection
│   │   │   ├── lang_data.rs      # generated stopword tables (from laya/lang.py)
│   │   │   ├── runtime.rs        # Runtime trait, Calibration, ONNX impl
│   │   │   ├── candle_runtime.rs # native candle backend (feature "candle")
│   │   │   ├── shortlist.rs      # embedding shortlist + cache
│   │   │   ├── engine.rs         # predict / predict_long / predict_batch
│   │   │   └── predictor.rs      # Predictor trait + Engine impl (ADR-006 rule 6)
│   │   └── tests/                # parity + behaviour + evidence suites
│   │                              # (candle_parity, onnx_parity, lang_parity,
│   │                              #  evidence, runtime_bench, …; skip without
│   │                              #  KNOT_MODEL_DIR)
│   ├── knot-serve/        # binary + axum library
│   │   └── src/
│   │       ├── main.rs           # env config, runtime/device pick, --mcp dispatch
│   │       ├── lib.rs            # HTTP app: /v1/systemone(/batch), /health, /models
│   │       ├── mcp.rs            # JSON-RPC stdio MCP server
│   │       └── model.rs          # checkpoint directory resolution (SPEC §9)
│   └── knot-nqlite/       # optional audit-ledger adapter (feature "audit")
│       ├── src/lib.rs            # Predictor decorator → nqlite (ADR-006 seam B)
│       └── tests/audit.rs        # stub-predictor e2e (no model dir)
└── docs/
```

Dependency direction: `knot-serve` → `knot`; `knot-nqlite` → `knot` core
(**never** `knot-serve` — ADR-006 rule 6), and `knot-serve --features audit`
may depend on the adapter, so no cycle exists even behind the gate.

## One `predict()` data flow

```
HTTP /v1/systemone (or MCP knot_predict)
  → validate: limits/refusals/controls (422/413)
  → router.route(state) → checkpoint id
  → checkpoint(name): LRU touch, (re)load if not resident
  → prompt::build_sequence(tok, state, questions)
       (token cache, tokenizer lock, budgets, stats)
  → runtime.forward(input_ids, attention_mask,
        marker_pos, marker_mask, qtype)
  → scaled_softmax(logits / T_type) at marker positions
  → answers {choice|score|noul} + usage + routing (+ shortlist)
  → (feature "audit") after the response: AuditPredictor enqueues the
       decision to nqlite — off the hot path, bounded queue (ADR-006)
```

## Runtime trait

```rust
pub trait Runtime {
    fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        marker_pos: &[i64],
        marker_mask: &[bool],
        qtype: &[i64],
        batch: usize,
        seq_len: usize,
        num_markers: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)>;
    fn config(&self) -> &serde_json::Value;
    fn calibration(&self) -> &Calibration;
}
```

Impls: `OnnxRuntime` (feature `onnx`, `ort`, CPU) and `CandleRuntime`
(feature `candle`, candle 0.11, CPU). `Engine::load` / `Engine::load_candle`
pick the backend; `knot` selects at runtime with `KNOT_RUNTIME`.

## Concurrency

- Single shared tokenizer behind a lock (Laya's `_TOKENIZE_LOCK` semantics).
- Server: semaphore-bounded concurrency, token-budget cap; the
  `x-inference-time-ms` header is measured after the gate is acquired.
- Checkpoint residency: LRU bounded by `max_loaded` (default 2, `KNOT_MAX_LOADED`).
  Engine loads up to `max_loaded` eagerly, loads further models on first use,
  evicts on overflow; an in-flight prediction holds an `Arc` so it is never
  evicted out from under itself.

## Config (`KNOT_*`)

`KNOT_MODELS=name=/path,...` or `KNOT_MODEL_DIR`, `KNOT_CACHE_DIR`,
`KNOT_DEFAULT_MODEL`, `KNOT_MAX_LOADED`, `KNOT_AUTO_TASK`, `KNOT_API_KEY`,
`KNOT_MAX_CONCURRENT`, `KNOT_MAX_TOKEN_BUDGET`, `KNOT_HOST`, `KNOT_PORT`,
`KNOT_RUNTIME=onnx|candle`, `KNOT_DEVICE=cpu|cuda` (SPEC §10). With
`--features audit`: `KNOT_AUDIT_DB` (enables the ledger),
`KNOT_AUDIT_CAPACITY`, `KNOT_AUDIT_EXCERPT` (ADR-006).
