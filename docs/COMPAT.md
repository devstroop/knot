# knot — Jev/Laya Compatibility

## Position

- Laya is wire-compatible with TypeSafe's hosted Jev `/v1/systemone`.
- knot targets the same wire, so a Jev or Laya client repoints its base URL at `knot`.

## What stays identical

- Endpoint shapes and JSON field names of request/response.
- Question primitives and answer payloads (`choice`/`score`/`noul`).
- Request fields: `option_order` (validated permutation, un-permuted answers),
  dict/list/number `instructions`, `labels` for noul.
- HTTP error codes for oversized option lists and invalid requests.

## Response wire shape

knot serializes responses the way Starlette does for Laya: compact separators
(`{"k":v}`), raw UTF-8 (`ensure_ascii=False`), keys in Laya's insertion order
(workspace `serde_json` runs with `preserve_order`):

- Top level: `model`, `answers`, `usage`, `routing`, then `shortlist` when
  present. `model` is always the agent id `"laya-rl-agent"` (not the routed
  checkpoint — that lives in `routing.model`).
- Per answer: `type` first, then the answer fields in Laya's order —
  `choice`/`probabilities`/`confidence`/`answer_confidence`/`action`
  (score adds `legend` before `probabilities`; noul uses `noul`), `window`
  last when a long-doc scan names the deciding window. The confidence gate
  appends `low_confidence`, `abstention`, `abstention_threshold`, in that
  order.
- `routing` is always present with Laya's `RouteDecision` keys: `model`,
  `repo`, `reason`, `detection`, `workflow` — `repo` is the checkpoint id
  (`convaiinnovations/laya[/subfolder]`), `detection` the script/language
  analysis object (or `null`), `workflow` the matched typed-decisions
  workflow (or `null`).
- `usage` always carries `input_tokens` and `output_tokens`; a single-shot
  predict also carries `state_tokens`, `state_tokens_dropped`, `truncated`
  and `truncated_questions` (`options` only when the head budget collapsed
  option spans); a long-doc scan adds `windows`.
- Empty `questions` short-circuits before any tokenizer or forward pass:
  `{"model":"laya-rl-agent","answers":{},"usage":{"input_tokens":0,"output_tokens":0}}`
  plus `routing` — the same two-key usage Laya returns.

## Confidence gate (`min_confidence`)

Port of Laya `confidence.apply_confidence_gate`, applied on the HTTP and MCP
surfaces:

- `min_confidence` absent or `null`: nothing is written — an ungated call
  returns exactly what it returned before the gate existed.
- With a threshold `mc`: first the flag pass — `low_confidence: true` on
  answers whose gate confidence is `< mc`, skipped entirely when
  `mc == 0.0` — then the abstention pass, which writes `abstention`
  (`"abstained"`, `"passed"`, or `"unevaluated"` when the answer carries no
  usable confidence) plus `abstention_threshold: mc` on **every** answer.
- Gate confidence is `answer_confidence` if it is a finite number, else
  `confidence`, rounded to 4 decimals (Python `round(x, 4)`) before the
  comparison, so a boundary threshold compares the value Python would.

## Prompt-side rendering

`state`, criterion values and non-string `instructions` are rendered through
Python `json.dumps(..., ensure_ascii=False)` with its **default** separators
(`", "` / `": "`) — the spaced form — while the HTTP response body stays
compact. Scalar reprs follow Python: `True`/`False`, `1.0`, `1e-05`.

## Deliberate differences from Jev

| Area | Jev | Laya | knot |
|---|---|---|---|
| Options per question | cap 255 | shared `head_max_len` budget (192/256), trim, then 422; HTTP guard ≤100 (413) | same as Laya |
| `score` levels | null allowed, echoed in `legend` | description required, null → 422 | same as Laya |
| `confidence` | `(n·p_max − 1)/(n − 1)` | 1 − normalized entropy | same as Laya; gate on `answer_confidence` |
| Extra response fields | — | `routing`, `usage.state_tokens_*`, `answer_confidence`, action, gate fields | same as Laya |
| `usage.truncated` in long-doc scan | — | summed window cuts (a count; test `> 0`) | `bool`, OR over windows (single-shot Laya also sends a bool) |
| Choice criteria as a list of labels | — | accepted; normalizes to `{str(label): None}` | same; `choice` and `probabilities` keys use Python `str()`, so a non-string label echoes as its text (`"True"`, `"7"`) rather than the typed JSON value |
| Privacy/infra | hosted API | self-hosted, Apache 2.0 | self-hosted, Apache 2.0 |

## Recorded Jev responses

The only captured hosted-Jev traffic (`jev-1.13.0`, 192 recorded answers,
`laya/research/benchmarks/feishu_zh/results/v1/jev/raw.jsonl`) confirms the
table above and adds the response-level divergences. A Jev response and a
Laya/knot response are **different JSON** even to the same request:

- Jev sends `model, answers, usage` only — no `routing`, no
  `answer_confidence`, no `action`.
- Jev's answer key order is `type, choice, confidence, probabilities`
  (noul: just `type, noul`); Laya/knot use Laya's order with the extra fields.
- Jev rounds every probability to 2 decimals (768/768 recorded); Laya/knot to
  4 (768/768).
- Jev's `confidence` is `round2((n·p_max − 1)/(n − 1))` — re-verified against
  the recordings (p_max 0.53 → 0.37, 0.85 → 0.80) — while Laya/knot use
  1 − normalized entropy and gate on `answer_confidence`.
- Jev's `usage.output_tokens` counts generated tokens (46–68 recorded);
  Laya/knot always send 0.
- Jev's `probabilities` key order varies per response (24 distinct orders in
  192 answers despite a fixed request criteria order); Laya/knot always echo
  criteria order.

knot follows Laya in every row. Full evidence, serve-surface deltas and the
quality/latency numbers: `docs/RESEARCH-COMPARE.md`.

## HTTP surface parity (vs `laya-serve`)

- `GET /health`: liveness `{"status":"ok"}` is always open; the bearer (or
  no configured key) unlocks Laya's detail payload — `loaded`, `revisions`,
  `device`, `device_is_preference`, `checkpoint_devices`, `cpu_fallbacks`,
  in that order (`docs/http-api.md`). knot's values: `revisions` is the HF
  snapshot etag the checkpoint was downloaded at (e.g. Laya's reviewed
  `55cf4c4e…` for `convaiinnovations/laya`) or `null` for a hand-copied
  directory — the same `null` Laya reports for a local path;
  `checkpoint_devices` is the configured `KNOT_DEVICE` per resident name
  (`"cpu"` by default, `"cuda"` with a `cuda`-feature build — SPEC §10)
  and `cpu_fallbacks` is `{"count":0,"last_reason":null}`: knot never falls
  back silently — a device that cannot come up is a startup error.
- Predict responses carry both timing headers Laya sends: `Server-Timing:
  inference;dur=<ms>` and `X-Inference-Time-Ms: <ms>`.
- `/v1/systemone/batch` accepts Laya's batch call controls with Laya's
  validation: `batch_size` a positive integer (`"batch_size must be an
  integer"` / `"batch_size must be a positive integer, got N"`), 
  `sort_by_length` a boolean (`"sort_by_length must be a boolean"`), both
  `null`-tolerated — all three wrong-type/range cases are 422. The engine
  honours them the way `agent.py` does: chunking by `batch_size` (default
  the whole group), a stable ascending length sort inside a `chunk*8` window
  only when `1 < batch_size < n`, results written back to input positions.
  The MCP batch tool validates the same arguments with Laya's merged MCP
  wording (`"batch_size must be a positive integer, got {v!r}"`).
- `GET /models` is **knot's extension** — laya-serve has no such route
  (`docs/http-api.md` lists only `/health`, `/v1/systemone`,
  `/v1/systemone/batch`). Clients written against Jev/Laya never call it;
  it exists for local introspection.

## Porting checklist for an existing client

1. Repoint base URL to `knot`.
2. Replace any `confidence` threshold with `answer_confidence`; re-fit cutoffs
   at the option counts you use.
3. Cap choice options at 100 per question; narrow larger label sets with
   `predict_shortlist` before the typed head.
4. Ensure every `score` level has a description.
5. Pass `max_len=8192` for long documents on the multilingual checkpoint.
6. Read `model` for the agent id and `routing.model`/`routing.repo` for the
   routed checkpoint; `routing.detection`/`workflow` may be `null`.
7. Treat `abstention`/`abstention_threshold` as present on every answer when
   `min_confidence` was sent.
