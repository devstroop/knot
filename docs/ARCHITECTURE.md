# oio Architecture

## Crates

```
oio-workspace/oio/
├── crates/
│   ├── oio/         # library
│   │   └── src/
│   │       ├── lib.rs            # public API surface, feature gates
│   │       ├── protocol.rs       # wire types (M0)
│   │       ├── error.rs          # error taxonomy → HTTP codes
│   │       ├── lang.rs           # lang/script detection (M3)
│   │       ├── prompt.rs         # prompt assembly + criteria render (M1)
│   │       ├── router.rs         # lang detect + checkpoint LRU (M3)
│   │       ├── runtime.rs        # Runtime trait, Calibration, ONNX impl (M2)
│   │       ├── candle_runtime.rs # native candle backend (M8, feature "candle")
│   │       ├── shortlist.rs      # embedding shortlist + cache (M6)
│   │       └── engine.rs         # predict/predict_long/predict_batch (M2.5+)
│   │   └── tests/                # parity + behaviour suites
│   └── oio-serve/   # binary + axum library
│       └── src/
│           ├── main.rs           # env config, --mcp dispatch (M4/M7)
│           ├── lib.rs            # HTTP app: /v1/systemone(/batch), /health, /models
│           └── mcp.rs            # JSON-RPC stdio MCP server (M7)
└── docs/
```

## One `predict()` data flow

```
HTTP /v1/systemone (or MCP oio_predict)
  → validate: limits/refusals/controls (422/413)          [M4/M7]
  → router.route(state) → checkpoint id                    [M3]
  → checkpoint(name): LRU touch, (re)load if not resident  [M3/PRD F6]
  → prompt::build_sequence(tok, state, questions)          [M1]
       (token cache, tokenizer lock, budgets, stats)
  → runtime.forward(input_ids, attention_mask,
        marker_pos, marker_mask, qtype)                    [M2/M8]
  → scaled_softmax(logits / T_type) at marker positions
  → answers {choice|score|noul} + usage + routing (+ shortlist, M6)
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
pick the backend; `oio-serve` selects at runtime with `OIO_RUNTIME`.

## Concurrency

- Single shared tokenizer behind a lock (Laya's `_TOKENIZE_LOCK` semantics).
- Server: semaphore-bounded concurrency, token-budget cap; the
  `x-inference-time-ms` header is measured after the gate is acquired.
- Checkpoint residency: LRU bounded by `max_loaded` (default 2, `OIO_MAX_LOADED`).
  Engine loads up to `max_loaded` eagerly, loads further models on first use,
  evicts on overflow; an in-flight prediction holds an `Arc` so it is never
  evicted out from under itself.

## Config (`OIO_*`)

`OIO_MODELS=name=/path,...` or `OIO_MODEL_DIR`, `OIO_CACHE_DIR`,
`OIO_DEFAULT_MODEL`, `OIO_MAX_LOADED`, `OIO_AUTO_TASK`, `OIO_API_KEY`,
`OIO_MAX_CONCURRENT`, `OIO_MAX_TOKEN_BUDGET`, `OIO_HOST`, `OIO_PORT`,
`OIO_RUNTIME=onnx|candle`.
