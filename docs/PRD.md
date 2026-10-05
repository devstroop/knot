# oio PRD

## 1. Problem

TypeSafe's Jev is a hosted decision API: great latency/accuracy, but closed weights,
per-token cost, and vendor lock-in. Laya proves an open, self-hostable equivalent exists
(Apache 2.0, fine-tuned checkpoint beats Jev's published typed-decisions score), but it is
Python-first and research-oriented: torch eager, FastAPI, broad surface area. Production
teams need a small, fast, strict, operable artifact.

## 2. Product

oio is a Rust implementation of the same decision-engine inference path, exposed as:

- `oio` library crate — prompt assembly, checkpoint router, ONNX inference, confidence,
  calibration, long-document windowing.
- `oio-serve` binary — Jev/Laya-compatible HTTP server (`POST /v1/systemone`, `/batch`,
  `/health`, `/models`) with auth, budgets, and concurrency controls.

## 3. Goals

| Area | Goal |
|---|---|
| Protocol | Byte-compatible request/response with Laya's `/v1/systemone` (answers `choice`/`score`/`noul`, usage block, routing block) |
| Parity | Numerical parity with Laya's ONNXAgent on frozen fixtures; decision agreement tracked per checkpoint |
| Latency | CPU p50 ≤ Laya's ONNXAgent (target: match or beat; measure, don't claim) |
| Strictness | Same 422/413 semantics; no silent truncation — report dropped state tokens |
| Operability | Env config (`OIO_*`), structured logs (tracing), pinned model revisions, digest verification, graceful shutdown |
| Extensibility | `Runtime` trait: ONNX first, candle native later, own kernels after |
| License/cost | Apache-2.0, zero per-token cost, CPU baseline (x86_64 Linux + macOS arm64), GPU opt-in via ort CUDA EP (`OIO_DEVICE=cuda`, x86_64 Linux) |
| Model development | Reproducible, optional training experiments that produce a canonical checkpoint for both configured inference runtimes |

## 4. Non-goals (out of scope)

Deliberate exclusions — each exists in Laya's ecosystem but outside oio's
remit (a local, hosted-free HTTP/MCP decision server). Evidence for every
row is in [`RESEARCH-COMPARE.md`](RESEARCH-COMPARE.md) ("Laya's experimental
layer vs oio scope").

| Excluded | Why |
|---|---|
| Hooks framework + hook MCP tools | oio refuses them by name on the MCP surface; a fixed inference path keeps responses auditable |
| `structured.decide` API, presets | preset sugar over the single `/v1/systemone` contract; callers compose the same behaviour themselves |
| Eval harnesses (`laya-evals` regression gate, per-language research reports) | oio gates on parity fixtures plus its own evidence harness (latency bench + golden replay); accuracy evals stay in Laya's research tree |
| RLCD/online reward training, production calibration, broad hyperparameter sweeps, and model-quality claims from synthetic smoke data | keep stage one to a bounded, supervised CPU experiment; require licensed labels and separate held-out evaluation before release |
| Compile fast-path (TileLang kernels, torch.compile) | no torch in the stack; kernels arrive later via the `Runtime` trait only if they pay for themselves |
| Integrations: langchain/langgraph, LlamaIndex, CrewAI, TypeScript SDK | Laya keeps them; oio is a base-URL drop-in behind any HTTP or MCP client |
| Automatic GPU selection (AMP, silent OOM→CPU fallback), WASM edge builds | CPU x86_64 Linux + Apple Silicon is the v1 baseline; ort CUDA EP is explicit opt-in behind `OIO_DEVICE=cuda` (SPEC §10) — automatic selection and silent fallback stay out |
| Intel Macs (`x86_64-apple-darwin`); CUDA outside x86_64 Linux | ONNX Runtime ships no prebuilt binaries for Intel Macs and ort's CUDA distributions are x86_64-Linux-only (ort-sys dist matrix); Apple Silicon stays on the CPU path (SPEC §10) |
| ONNX export scripts, TensorRT capacity sweeps | checkpoints ship as ready-made ONNX; export belongs to Laya's toolchain and TRT rides on ONNX Runtime later |
| Softlist retrieval training | the shortlist feature hook ships; training its retriever does not |

Not excluded — scheduled instead: `batch_size`/`sort_by_length` batch
controls (implemented, G3) and the evidence harness (latency bench + golden
replay), per the gap ranking at the end of `RESEARCH-COMPARE.md`.

## 5. Users

- Platform teams self-hosting decision endpoints for agents/workflow automation
- Teams migrating from hosted Jev who want a drop-in base URL change

## 6. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Accept `{state, questions, model?, max_len?}`; return typed answers + usage + routing |
| F2 | Question types `choice`, `score`, `noul`; `noul` custom true/false labels |
| F3 | Prompt format identical to Laya: `[CLS] <type> question: instr [SEP] [MASK] opt … [SEP] state [SEP]` |
| F4 | Token budgets: per-checkpoint `max_len`, `head_max_len`; per-option trim stats; state clamp stats |
| F5 | Confidence: `confidence` (1 − normalized entropy) and `answer_confidence` (max prob); per-type calibration temperatures from `rl_agent_config.json` |
| F6 | Router: script/stopword detection, checkpoint pick, `OIO_DEFAULT_MODEL` override, LRU `max_loaded=2` |
| F7 | `predict_long`-style windowing with stride overlap + offset reporting |
| F8 | `predict_batch` grouping by checkpoint + question schema |
| F9 | HTTP: `/v1/systemone`, `/batch` (≤64 states), `/health`, `/models`; bearer auth; 413 option guard (≤100 choices); 422 validation |
| F10 | Long-context checkpoints: `max_len=8192` opt-in |
| F11 | Shortlist hook: embedding-based pre-filter before typed head |
| F12 | MCP stdio server exposing predict/route/batch tools (v2 of serve crate) |

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | x86_64 Linux + macOS arm64 (Apple Silicon) v1 with a CPU-only baseline (no GPU required; opt-in CUDA per SPEC §10, x86_64 Linux only); release build with thin LTO |
| N2 | Bounded concurrency via semaphore; token-budget cap env |
| N3 | Tokenizer access serialized (mirrors Laya's shared-lock semantics) |
| N4 | No network at inference time; models fetched explicitly with pinned revision + digest check |
| N5 | Test gates: fixture parity vs Laya ONNXAgent, unit tests per prompt-budget rule, clippy/rustfmt clean |
| N6 | Training tools are optional and isolated from the Rust inference dependency graph; no training-time downloads or Python dependencies are required to build or serve OIO |

## 8. Success metrics (measurable, tracked in docs rounds)

- Decision agreement vs Laya ONNXAgent ≥ 99% on frozen fixtures (fp32)
- p50 single-question latency ≤ Laya ONNXAgent CPU on same hardware
- Zero silent truncation: every dropped-state call returns `state_tokens_dropped`
- 100% of F1–F9 covered by automated tests before v0.2

## 9. Risks

| Risk | Mitigation |
|---|---|
| Numerical drift ONNX vs torch | Parity fixture gate per checkpoint; block release on regression |
| Head token budget edge cases (option collision) | Port Laya's #538 guard + tests |
| Tokenizer thread-safety | Single shared tokenizer behind lock, token cache like Laya |
| `ort` download/version pinning in CI | Pin `ort` version, cache runtime artifacts |
