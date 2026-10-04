# Research comparison: oio, Jev, and Laya

Evidence behind `COMPAT.md`'s Jev-divergence section and the gap ranking at
the end. Three questions answered from primary sources: what the real hosted
Jev actually returns, how that differs from Laya's wire (which oio targets),
and which parts of Laya's experimental layer oio does not carry. The workspace
vendors the Laya checkout alongside `oio/`; every file:line below is readable
there. Nothing here is an issue tracker — it is the evidence.

## Evidence sources

| Source | What it proves | Limits |
|---|---|---|
| `laya/research/benchmarks/feishu_zh/results/v1/jev/raw.jsonl` (+ `metadata.json`) | **The only recorded hosted-Jev traffic**: 64 cases × 3 repeats × 2 modes against `POST api.typesafe.ai/v1/systemone`, `model: "jev-1.13.0"`, captured 2026-09-21 | Community diagnostic, N=64, Chinese workplace only; contributed snapshot |
| `…/feishu_zh/results/v1/laya/raw.jsonl` (+ `summary.json`) | Paired Laya answers to byte-identical requests (`request_sha256` matches): in-process `Agent.predict`, multilingual checkpoint `1c5edc17`, MPS, laya source `ef7d7d2` | Direct `laya.load()` bypasses `Router`, so no `routing`; snapshot predates `answer_confidence` — not the current serve wire |
| `laya/research/README.md:168-180`, `laya/BENCHMARKS.md` | Laya's own sweeps + the explicit statement that **Jev was never run directly** for the upstream suites; Jev figures are third-party | Different prompts and sample sizes — "indicative, not a controlled head-to-head" |
| `AbdelStark/jev-benchmarks` (cited in README) | Jev AG News 0.910, Banking77 0.870, DAIR Emotion 0.480 (Brier 0.846, zero prob on the true label for 16%) | Published third-party |
| `nibzard/decision-model-benchmark` (cited) | Jev ECE 0.246 (worst in that study), banking77 0.763, option-order flip rate 13%, p50 264–276 ms | Published third-party |
| `laya/research/scripts/bench_local.py:262` | Jev `jev-1.13.0` published typed-decisions: accuracy 0.727, soft accuracy 0.580, Brier 0.148 | Published third-party |
| `oio/docs/JEV-WIRE.md` | The **current** Laya serve/Router wire (read from this workspace's checkout) — oio's actual serialization target | Source-derived, not traffic-derived |

## Wire: recorded Jev vs Laya vs oio

Every row below was checked against the recordings (or, where marked, the
current source). Divergences 1–8 are why "Jev-compatible" for oio means
**request-level wire compatibility with the Laya implementation of that wire**,
not byte-identity with a `jev-1.13.0` response:

| # | Aspect | Jev (recorded) | Laya | oio |
|---|---|---|---|---|
| 1 | Top-level keys | `model, answers, usage` | same; `routing` appended by `Router.predict` (`router.py:947`) — absent only because the recording used `Agent.predict` directly | `model, answers, usage, routing` — serve parity |
| 2 | Choice answer key order | `type, choice, confidence, probabilities` | `type, choice, probabilities, confidence[, answer_confidence], action` (`JEV-WIRE.md`) | same as Laya |
| 3 | Noul answer | `{"type":"noul","noul":…}` only | `type, noul, confidence[, answer_confidence], action` | same as Laya |
| 4 | Numeric precision | 2 decimals in **768/768** probability values | 4 decimals in **768/768** | 4 decimals |
| 5 | `confidence` meaning | `round2((n·p_max − 1)/(n − 1))` — matches every sampled answer (p_max 0.53 → 0.37, 0.85 → 0.80, 0.99 → 0.99); not `p_max` itself | 1 − normalized entropy (near-uniform four-way gave 0.0456) | same as Laya; gate compares `answer_confidence` |
| 6 | `usage.output_tokens` | 46–68 (nonzero, generative) | 0 | 0 |
| 7 | Probability key order | **24 distinct orders across 192 answers** although the request's criteria order is fixed (`prompts.py:7`) | criteria order in **192/192** | criteria order (workspace `serde_json` `preserve_order` / `IndexMap`) |
| 8 | `model` id | `jev-1.13.0` | `laya-rl-agent` | `laya-rl-agent` |

Request shapes, question primitives, `option_order` handling and error codes
are shared — a Jev or Laya client repoints its base URL and keeps calling.

## Serve surface: confirmed identical vs remaining deltas

Confirmed identical at `develop` (`3628821`), serve.py ↔ `oio-serve`:

- All eight guard constants: 64 questions, 50 000 state chars, 64 batch
  states, 2 MiB body, 100 choice options, 32 score levels, 512 total options,
  concurrency 16, token budget 8192 (`serve.py:66-88` vs `lib.rs:19-27`).
- Admission control: `503 {"detail":"server busy, try again later"}` with
  `Retry-After: 1` on overflow (`serve.py:805-811` vs `lib.rs:476-486`).
- Batch response `{"results": [...], "total_usage": {...}}`, the
  `min_confidence` gate, empty-questions short-circuit, compact UTF-8 body.

Remaining deltas (ranked at the end of this file). Fixed since `3628821`:
G1 health detail payload, G2 `Server-Timing` header (`X-Inference-Time-Ms`
already existed), G5 `/models` documented as an oio extension.

| | Delta | Laya | oio at `3628821` | Status |
|---|---|---|---|---|
| G1 | `/health` detail payload | `revisions`, `checkpoint_devices`, `cpu_fallbacks` alongside `loaded`/`device` (`serve.py:755-796`, `docs/http-api.md`) | only `loaded`, `device` (hardcoded `"cpu"`), `device_is_preference` (`lib.rs`); unauth → `{"status":"ok"}` matches | fixed: full payload; `revisions` from the HF snapshot etag (the reviewed `55cf4c4e…` for `convaiinnovations/laya`), `null` for plain local dirs (Laya's own local-path value) |
| G2 | Latency headers | `Server-Timing: inference;dur=…`, `X-Inference-Time-Ms` (`serve.py:884-885, 1009-1010`) | `X-Inference-Time-Ms` only | fixed: both headers on predict + batch |
| G3 | Batch call controls `batch_size`, `sort_by_length` | validated, wrong type/range → **422** (`serve.py:112, 266-291`) | parsed and silently ignored | open (Step 3: validation + engine effect) |
| G4 | `lang_temperatures` | per-language temperature maps (`common.py:728`) | only the 3 per-type + per-bucket calibration (`runtime.rs:56-69`) | open |
| G5 | `GET /models` | not part of laya-serve | present (`lib.rs:416-424`) — an oio extension, documented as such | fixed: documented in COMPAT |

Note on MCP status: laya's `laya_status` tool reports the device trio
(`device`, `checkpoint_devices`, `device_is_preference`) that oio's own
`oio_status` does not carry. `oio_status` is oio's tool, not a wire promise —
recorded here for completeness, not as a compat gap.

## Quality and latency evidence

Paired community diagnostic (`feishu_zh`, N=64, 3 repeats; Jev hosted, Laya
local on MPS — the two halves of `results/v1/summary.json`):

| Mode | Jev accuracy | Laya accuracy | Jev p50 | Laya p50 | Repeat consistency (Jev/Laya) |
|---|---|---|---|---|---|
| choice | **1.000** | 0.313 | 253 ms | 151 ms | 63/64 · 64/64 |
| four_noul | **0.984** | 0.281 | 250 ms | 415 ms | 64/64 · 64/64 |

Published/third-party headline (`laya/BENCHMARKS.md`, Jev figures never run by
Laya's authors):

| | Laya | Jev (published) |
|---|---|---|
| typed-decisions (2,000) | **0.766** | 0.727 |
| AG News | **0.953** | 0.910 |
| DAIR Emotion | **0.600** | 0.480 |
| ECE after temperature fitting | **0.081** | 0.246 |
| p50 latency, 1 question | **32.8 ms** (T4) | 236–276 ms |

Reading: Laya (and therefore oio's architecture) wins the English suites,
calibration and latency; Jev wins the Chinese workplace diagnostic decisively
and is competitive on Banking77. Both ship over-confident (Laya refits to ECE
0.081; Jev's published ECE is 0.246 with a 13% option-order flip rate — the
research README's own framing). oio inherits Laya's quality profile from the
same weights but has measured nothing of its own yet: the parity suites prove
behavior against fixtures, not accuracy or latency. The evidence-harness gap
below is where that gets closed.

## Laya's experimental layer vs oio scope

| Area | Laya | oio |
|---|---|---|
| Serving spine: router, prompt, calibration, gate, long-doc window, batch, MCP subset, auth/caps | yes | **yes** |
| Hooks framework + hook MCP tools | yes | refused by name on MCP (`lib.rs:28-34`) |
| `structured.decide` API, presets | yes | no |
| Eval harnesses: `laya-evals` regression gate, independent `research/eval` per-language reports | yes | parity fixtures only |
| Finetuning + recipes (es_phone_turns 0.396 → 0.912) | yes | no |
| Compile fast-path | yes | no |
| Integrations: langchain/langgraph, LlamaIndex, CrewAI, TypeScript SDK | yes | no |
| GPU: CUDA/AMP, silent OOM→CPU fallback, length-sorted batching (measured 2.15× on 10k tickets, `research/README.md:66-71`) | yes | CPU only (ort + candle); length sorting not implemented |
| ONNX export scripts, TensorRT capacity sweeps | export scripts, TRT via ONNX Runtime | ort CPU, no export script |
| Research benches: feishu_zh, zh_short_commands, position sensitivity, latency, NVIDIA capacity | yes | latency bench pending (evidence harness) |

Most of the right-hand column is out of oio's scope by design (a local,
hosted-free HTTP/MCP decision server); `PRD.md` now states that explicitly.

## Gap ranking

1. **Serve wire parity** — G1 health payload, G2 timing headers, G3 batch
   controls (validate now, effect with #2), G5 document `/models`.
2. **`sort_by_length`** — laya's measured 2.15× with zero decision changes;
   G3's valid flag becomes real here.
3. **Scope declarations** — PRD/PLAN record the "no" column above.
4. **Evidence harness** — latency numbers and an end-to-end golden replay so
   oio measures itself, not just its fixtures.
