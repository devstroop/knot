# AGENTS.md

Context for AI coding assistants working in the oio repository.

## Do NOT

- Add a hosted-service dependency. Everything runs in the user's own process or hardware.
- Change SPEC.md invariants silently — code, tests, and docs move together.
- Edit a merged ADR; supersede it instead.
- Track individual issues/bugs in docs — PRD/PLAN/SPEC + ADR only.
- Introduce formatting-only diffs that obscure the real change.

## CI gates (run before declaring done)

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo check -p oio            # fast path: no ort/tokenizers/axum
```

The end-to-end parity suites (engine/longdoc/onnx_parity/candle_parity) skip
themselves when no checkpoint is available, so "green" is only meaningful with
one set:

```bash
OIO_MODEL_DIR=/path/to/laya-english cargo test --workspace
```

## Where to look

| Question | Read |
|---|---|
| What & why | `docs/PRD.md` |
| Milestones | `docs/PLAN.md` |
| Invariants | `docs/SPEC.md` |
| Jev/Laya compat | `docs/COMPAT.md` |
| Crate layout, data flow | `docs/ARCHITECTURE.md` |
| Why something was decided | `docs/ADR/` |
