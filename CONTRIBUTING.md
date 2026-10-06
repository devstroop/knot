# Contributing to knot

## Build and test gates

Every change must pass these before review:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p knot            # fast path: no ort/tokenizers/axum
```

The checkpoint-backed suites (`engine`, `longdoc`, `onnx_parity`,
`candle_parity`) skip themselves without a checkpoint; meaningful green
needs one set:

```bash
KNOT_MODEL_DIR=/path/to/laya-english cargo test --workspace
```

Python harnesses (training, demos, evals) run with the system Python where
possible; the torch/FastEmbed paths need their venvs (see `docs/TRAINING.md`
and `docs/GITA-DEMO.md`).

## Branches and pull requests

- Branch from `main` as `feat/<concise-topic>`; one concern per PR.
- CI (`.github/workflows/ci.yml`) runs fmt, Clippy, and tests on every push
  and pull request; GPU/parity jobs run where configured.
- PRs need a summary, validation evidence (commands run + results), and docs
  updates where user-facing behavior or contracts move.

## Docs rules

- `docs/SPEC.md` changes **before** code when behavior moves — code, tests,
  and docs move together.
- ADRs (`docs/ADR/`) are append-only: never edit a merged ADR, supersede it
  with a new one.
- Keep `docs/README.md` (the docs map) in sync with the tree in the same PR.
- Numbers published in docs carry their binding (commit, profile, machine,
  date); generated artifacts (`demo/`, eval reports) change only by
  re-running their scripts.
- No hosted-service dependencies: everything runs in the user's own process
  or hardware.

## License

Apache-2.0 (see `LICENSE`). Contributions are accepted under the same terms.
