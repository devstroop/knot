# ADR-005: checkpoint artifact layout and parity-fixture format

- Status: accepted
- Supersedes: —

## Context

PLAN carried the open decision "exact fixture format + export script for
checkpoint ONNX artifacts". PRD §4 already excludes exporter scripts (Laya's
toolchain owns export); what remained undefined was the concrete layout oio
consumes and the concrete shape of the parity corpus.

## Decision

- A **checkpoint directory** is Laya's export layout at a pinned snapshot
  revision: `laya.onnx` (+ external weights), `rl_agent_config.json`,
  `tokenizer/tokenizer.json` with sibling `tokenizer_config.json` /
  `special_tokens_map.json` (special-token ids resolve config-first —
  SPEC §2).
- **Export stays with Laya** (`laya/scripts/export_onnx.py`, ADR-001). oio
  ships no exporter; `scripts/demo.sh` may run an export locally as setup
  tooling, which is not a shipped artifact.
- **Parity fixtures** are committed JSON/JSONL under
  `crates/oio/tests/fixtures/`: request/response contracts
  (`systemone_*.json`), a labelled English corpus (`eval_english.jsonl`:
  `{state, questions, expected, tags, language}`), golden prompt/parity
  snapshots (`golden_english.json`, `parity_english.json`), and router cases
  (`lang_cases.json`). Harvested from `laya/tests` (M0) and replayed by
  `crates/oio/tests/evidence.rs`.

## Consequences

- Reproducibility comes from pinned revision + digest verification (PRD N4),
  not from an in-tree exporter.
- Fixture evolution is a test-side change gated by the same parity suites —
  no runtime contract moves with it.
