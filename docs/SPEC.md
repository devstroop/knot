# oio Spec — Invariants

Behavior contracts that code, tests, and docs must not contradict. When one of
these changes, update this file in the same change.

## 1. Wire

- `POST /v1/systemone` request: `{state, questions, model?, max_len?, ...}`.
- Response: `{model, answers, usage, routing}` (`shortlist` last when the
  shortlist runs), keys in that order, compact separators, raw UTF-8.
  `model` is the agent id `laya-rl-agent`; the routed checkpoint is
  `routing.model`/`routing.repo`.
- `routing` is always present: `{model, repo, reason, detection, workflow}`
  with `detection`/`workflow` nullable.
- `usage.output_tokens` is always `0` (non-autoregressive; no decoded
  tokens); a single-shot predict also always carries `state_tokens`,
  `state_tokens_dropped`, `truncated`, `truncated_questions`. Empty
  `questions` short-circuits before any tokenization/forward pass with
  empty `answers` and the two-key usage `{input_tokens: 0, output_tokens: 0}`.
- Typed answers: `choice: {choice, confidence, answer_confidence}`,
  `score: {score, confidence, answer_confidence}`,
  `noul: {noul, confidence, answer_confidence}`.
- `confidence` = 1 − normalized entropy (matches Laya, **not** Jev's formula).
- `answer_confidence` = probability of the reported answer. Gate on this.
- Batch: `POST /v1/systemone/batch`, `states[]` with length ≤ 64. Call
  controls validate like Laya's (`serve.py`): `batch_size` must be a
  positive integer (else 422, `"batch_size must be an integer"` or
  `"batch_size must be a positive integer, got N"`), `sort_by_length` must
  be a boolean (else 422, `"sort_by_length must be a boolean"`); `null` is
  "not set". `batch_size` caps states per forward pass (default: the whole
  group); `sort_by_length` stably sorts states by encoded length within
  windows of `8 × batch_size` before chunking, and only when
  `1 < batch_size < len(states)` — results always come back in input order.
  The MCP batch tool accepts and validates the same two arguments with
  Laya's MCP wording (`"batch_size must be a positive integer, got …"`).
- Predict responses carry both timing headers, Laya's pair: `Server-Timing:
  inference;dur=<ms>` and `X-Inference-Time-Ms: <ms>` (2 decimals, inference
  time only — gate wait excluded).
- `GET /health`: open liveness `{"status": "ok"}`; with the bearer (or no key
  configured) the detail payload adds `loaded`, `revisions` (artifact commit
  per resident checkpoint — HF snapshot etag when present, else `null`,
  matching Laya's local-path value), `device`, `device_is_preference`,
  `checkpoint_devices`, `cpu_fallbacks` — keyed by `loaded` names, Laya's
  field order.

## 2. Prompt assembly (from Laya `common.py`)

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

- Rendered options: `choice` → `"label: criterion"` (or the label alone);
  `score` → `"level i: criterion"`; `noul` → `"<label>: <criterion or default>"`.
- The special ids behind `[CLS]`/`[SEP]`/`[MASK]` resolve the way Laya's
  AutoTokenizer does: the declarations in `tokenizer_config.json` /
  `special_tokens_map.json` beside the vocabulary win; candidate strings
  (`[CLS]` → `<s>` → `<bos>`, …) are only a fallback for a vocabulary with no
  sibling config. A vocabulary may contain look-alike entries that are plain
  tokens — mmBERT ships `<s>`/`</s>` at 204/213 while its config declares
  `<bos>`/`<eos>` at 2/1 — and picking those would frame the sequence with
  content tokens.
- Each option span is `[MASK]` + ≤ 48 tokens of option text.
- Options collectively share `head_max_len` (192 for English checkpoint, 256 otherwise).
  Overflow → per-option cap (`tokens_per_option` in stats); if options no longer fit, 422.
- State tokens fill the remainder of `max_len`; overflow is dropped and reported via
  `usage.state_tokens_dropped` + `usage.truncated` — never silent.
- Options that collide to identical token spans after trimming are reported via
  `options` vs `options_distinct` stats (#538 guard).

## 3. Question types

- `choice`: `criteria` is a label→description map, or a list of labels that
  normalizes to `{str(label): None}` (labels required, descriptions
  stringified). List labels must be scalar, unique under Python equality
  (`1 == 1.0`, `True == 1`), and non-null — else 422.
- `score`: `criteria` is an ordered list (index = level); every level requires a description.
- `noul`: true/false labels default to `{true, false}`; custom labels must be distinct
  non-empty strings. `criteria` keys are `true`/`false` only (case-insensitive;
  `_to_internal` lowercases them), else 422.

## 4. Confidence & calibration

- Calibration: per-type temperatures from `rl_agent_config.json`; soft max applied to
  option logits before softmax. Fitting is out of scope (M6 consumes artifacts only).
- Abstention: callers should threshold `answer_confidence`, not `confidence`.
- When `min_confidence` is sent, the gate writes `low_confidence` (below the
  threshold), then `abstention` (`abstained`/`passed`/`unevaluated`) and
  `abstention_threshold` on **every** answer, in that order after the scored
  fields; gate confidence is `answer_confidence` else `confidence`, rounded
  to 4 decimals before comparison. `min_confidence == 0.0` skips only the
  flag pass; an absent `min_confidence` writes nothing.

## 5. Routing

- Script detection (Latin vs non-Latin) + Latin-script stopword heuristic picks the
  checkpoint (`english` / `multilingual` / `typed-decisions`); the detection
  reported on the wire is the rounded analysis (`non_latin_fraction` at 4
  decimals).
- `OIO_DEFAULT_MODEL` overrides the fallback; explicit `model` in the request wins;
  Jev ids (`jev-*`) mean "auto-route".
- `typed-decisions` is never auto-selected without `auto_task_detection`.

## 6. Limits & errors

- HTTP: ≤ 100 choice options per question (else 413); `MAX_CONCURRENT` and
  `MAX_TOKEN_BUDGET` caps (429/503 per server config); bearer auth when
  configured; `/health` liveness always open (detail payload requires the
  bearer, see §1).
- Library errors map: `InvalidRequest` → 422, `PayloadTooLarge` → 413,
  `Model` → 500.

## 7. Long documents

- `predict_long` windows with ≤ 50% overlap (default stride = window/2);
  window capped at smallest `state_room` across questions; explicit oversized window →
  clamp with warning; stride > window → error.

## 8. Determinism

- Same checkpoint + same request + fp32 ONNX → same decisions. Any numeric drift beyond
  fixture tolerance blocks release.

## 9. Model resolution & integrity

Checkpoint directories resolve at startup, first source that yields at least
one directory wins:

1. `OIO_MODELS=name=/path[,...]` — explicit map, declaration order kept.
2. `OIO_MODEL_DIR=/path` — shorthand for a single `english` checkpoint.
3. Cache root: `$OIO_CACHE_DIR`, else `$XDG_CACHE_HOME/oio`, else
   `~/.cache/oio` (ADR-002). Only directories named in `router::DEFAULT_MODELS`
   (`english`, `multilingual`) are picked up; stray subdirectories are ignored,
   and the order follows that table.
4. Nothing resolves → startup error naming all three sources.

The router's fallback model is `english`. When `english` is not among the
resolved directories and `OIO_DEFAULT_MODEL` is unset, the fallback becomes
the first resolved directory (source order above). `OIO_DEFAULT_MODEL`, when
set, is validated by name at startup and never second-guessed.

Integrity is verify-if-present: a `SHA256SUMS` manifest beside a checkpoint
(coreutils format — `<64-hex> <space>[`*`]<relative-path>`, `#` comments and
blank lines skipped) is checked when the checkpoint loads. Every listed file
must exist under the directory and hash-match; absolute paths or `..`
components are rejected. A mismatch or missing file is a `Model` error
(→ HTTP 500). No manifest → load proceeds untouched, mirroring
`snapshot_revision` returning `None` for hand-copied weights. Fetching and
manifest creation are explicit operator steps; inference never touches the
network (PRD N4).
