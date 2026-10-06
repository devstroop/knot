# Jev/Laya wire-format research notes

Reference notes for the wire-compat hardening pass: what Laya actually does,
read out of the source at `laya/laya/` (the repo this workspace vendors
alongside `knot/`). `docs/COMPAT.md` states the knot-side contract; this file
records the evidence it is built from. Nothing here is an issue tracker — it
is the behaviour being ported.

## Response construction and key order

- `Agent.predict` builds the payload as `{"model": "laya-rl-agent",
  "answers": ..., "usage": ...}` (`agent.py:1409`, batch path
  `agent.py:1330`); `Router.predict` then appends `result["routing"] =
  dict(decision)` (`router.py:947`), and `Router.predict_batch` appends it
  per result (`router.py:1304`). `shortlist` (when the embedding shortlist
  runs) is appended after that.
- Per-answer dicts: `type` is serialized first, then the answer fields —
  choice: `choice`, `probabilities`, `confidence`, `answer_confidence`,
  `action`; score inserts `legend` before `probabilities`; noul: `noul`,
  `confidence`, `answer_confidence`, `action`. `window` (long-doc deciding
  window) follows, and gate fields (`low_confidence`, `abstention`,
  `abstention_threshold`) are appended last because they are assigned after
  the answer dict exists.
- `usage` in a single-shot predict (`agent.py:1389`): `input_tokens`,
  `output_tokens`, `state_tokens`, `state_tokens_dropped`, `truncated`,
  `truncated_questions`, then `options` only when the head budget actually
  collapsed option spans (`collapsed_options`).
- HTTP body: Starlette serializes with `json.dumps(..., ensure_ascii=False)`
  and compact separators — `{"k":v}` — so the wire bytes are compact, raw
  UTF-8, keys in dict insertion order. The spaced `", "` / `": "` separators
  are only the prompt-side rendering (below).

## Routing decision dict

- `RouteDecision` renders as `model`, `repo`, `reason`, `detection`,
  `workflow`, `workflow` last and `null` when no workflow matched.
- `repo` is `_repo_str(DEFAULT_MODELS[key])` (`router.py:63`): the repo for
  plain checkpoints, `repo/subfolder` for the subfoldered ones
  (`convaiinnovations/laya`, `.../laya/multilingual`,
  `.../laya/typed-decisions`).
- `detection` is `lang.analyse(state)` (never omitted; `null` when there is
  no analysis). `analyse` (`lang.py:644`) returns `script`,
  `script_profile`, `language`, `is_english`, `language_undecided`,
  `diacritic_rate`, `non_latin_fraction`, `mixed_segment` — that order.
- Script bookkeeping is first-appearance ordered with `latin` special-cased:
  `_script_counts` (`lang.py:287`) inserts `latin` last; `_profile_from_counts`
  (`lang.py:329`) lists `latin` first, then first-appearance order, dropping
  zero counts; `_script_from_counts` takes Python `max`, i.e. the first
  maximum in iteration order, so named scripts win ties against `latin`.
- `_analyse_text` rounds `non_latin = round(1.0 - prof.get("latin", 0.0), 4)`
  **before** comparing thresholds (`lang.py:571`), so the wire value and the
  comparison value are the same rounded number.

## Empty questions

- `predict_batch` with no question ids returns
  `{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0,
  "output_tokens": 0}}` per state (`agent.py:1331`) — two usage keys only —
  and routing is appended by the router as usual. No tokenization, no
  forward pass.

## Confidence gate

- `apply_confidence_gate(results, min_confidence)` (`confidence.py:104`):
  threshold `None` → nothing written at all; otherwise the
  `flag_low_confidence` pass (`confidence.py:70`) first — skipped for
  `mc == 0.0`, since nothing can fall below it — then the abstention pass
  writes `abstention` and `abstention_threshold = float(mc)` on **every**
  answer (`mc == 0.0` only skips the flag, not this).
- `abstention` values: `"abstained"` when the answer was flagged,
  `"unevaluated"` when the gate confidence is `None`, else `"passed"`.
- Gate confidence (`_gate_confidence`, `confidence.py:38`):
  `answer_confidence` if it is a finite non-bool number, else `confidence`,
  else `None` — and `round(x, 4)` is applied before the comparison, so a
  decimal threshold compares against the 4-decimal figure Python computed.
- `Agent.predict_long` calls the gate with `None` (`agent.py:1698`): a
  window loop has no single confidence to gate on, and every answer still
  reports that it ran ungated.

## predict_long aggregation

- Single-window short-circuit: `len(state_ids) <= budget` → a plain
  `system_one` plus `usage.windows = 1` (`agent.py:1616` area).
- Multi-window: usage is aggregated generically (`agent.py:1674` area) —
  numeric fields (including bools, which Python sums as ints, so
  `truncated` becomes a count 0..N) are summed across windows, dict fields
  (`options`) are shallow-merged, everything else (notably
  `truncated_questions`) is replaced by the last window's value; then
  `output_tokens` is forced to 0 and `windows = len(results)`.
- Answer choice: `noul` takes the strongest window (`max` of `noul`),
  `choice`/`score` take the most confident (`max` of `answer_confidence`);
  the deciding answer carries `window = {index, token_start, token_end,
  count}`.

## Prompt-side rendering

- `serialize_state`, criterion values and non-string `instructions` go
  through `json.dumps(..., ensure_ascii=False)` with Python's **default**
  separators — `", "` and `": "`. Strings pass through untouched;
  `null`/empty-container `instructions` become empty text and are rejected
  as 422s by `_check_question` (`agent.py:848` area).
- Python scalar reprs reach the prompt text: `True`/`False` (not
  `true`), `1.0` for floats, `1e-05`-style scientific notation.

## Question validation and normalization (`_check_question` / `_to_internal`)

- Question ids must be non-empty (`_check_question`, `agent.py:857`).
- Choice criteria are a dict **or** a list of labels; the list normalizes to
  `{str(label): None}`. Labels are checked structurally first (null and
  list/dict labels → 422), then for duplicates with Python equality — so
  `[1, 1.0]` and `[True, 1]` are one key. Both the option text and the
  answer/probability keys use `str(label)`, which is why a bool label echoes
  as `"True"`.
- Noul criteria must be a dict keyed (case-insensitively) only
  `true`/`false` (`agent.py:960` area); `_to_internal` then lowercases the
  keys, so `{"True": ...}` reaches `crit.get("true")` while any other key
  is a 422.
- Score criteria are a list, non-empty, no null levels.

## HTTP/serve surface notes

- `min_confidence` must be a float in `[0.0, 1.0]` (same 400/422 wording on
  the HTTP and MCP surfaces).
- Batch endpoint caps states at 64 and empty `states` is a 400.
- Response JSON uses compact separators; errors are `{"detail": ...}` with
  Laya's status mapping (413/415/422/503/…).
