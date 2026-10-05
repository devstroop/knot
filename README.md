# oio

[![CI](https://github.com/devstroop/oio/actions/workflows/ci.yml/badge.svg)](https://github.com/devstroop/oio/actions/workflows/ci.yml)

Production-oriented Rust decision engine — a self-hosted, open alternative to
TypeSafe's hosted Jev API, compatible with Laya's `/v1/systemone` wire protocol.

Non-autoregressive "System 1" decisions over typed questions (`choice` / `score` / `noul`)
in one forward pass. Laya (Python) is the research reference; oio targets
operability, latency, strictness, and deployability.

## Layout

| Path | Contents |
|---|---|
| `crates/oio` | Core library: wire protocol, prompt assembly, router, ONNX/candle runtimes |
| `crates/oio-serve` | HTTP server binary (`/v1/systemone`, `/batch`, `/health`, `/models`) + MCP stdio mode |
| `docs/` | PRD, plan, spec, compatibility, architecture, ADRs |
| `scripts/` | Checkpoint fixture tooling |
| `training/` | Optional research-only data preparation and frozen-head experiments |

## Requirements

- Rust stable **1.88 or newer** (the crate uses let-chains).
- Platforms: x86_64 Linux and macOS on Apple Silicon (arm64). Intel Macs
  have no ONNX Runtime prebuilts, and the `cuda` feature /
  `OIO_DEVICE=cuda` are x86_64-Linux-only (SPEC §10).
- Linux builds need a C/C++ toolchain, `pkg-config`, and OpenSSL headers for
  the build-time ONNX Runtime download (`build-essential pkg-config libssl-dev`
  on Ubuntu/Debian).
- A converted Laya checkpoint to **serve** or to run the parity suites; the
  library and the fast test suites build without one.

## Build

```bash
cargo check -p oio            # core (no model runtime)
cargo check -p oio-serve      # pulls ONNX Runtime + tokenizers
cargo check -p oio-serve --features cuda   # + ort CUDA EP (Turing sm_75+,
                                           #   CUDA 12 runtime, cuDNN 9)
```

## Run

```bash
# HTTP server on :8000
OIO_MODEL_DIR=/path/to/laya-english cargo run -p oio-serve

# same server on the GPU (needs the cuda feature build; fails fast if the
# device cannot come up — never a silent CPU downgrade)
OIO_MODEL_DIR=/path/to/laya-english OIO_DEVICE=cuda \
    cargo run -p oio-serve --features cuda

# MCP stdio server (same engine, --mcp)
OIO_MODEL_DIR=/path/to/laya-english cargo run -p oio-serve -- --mcp

curl -s localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "Customer was charged twice for order 1234 and wants it fixed",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "tech": "bugs, outages"
        }
      }
    }
  }'
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OIO_MODEL_DIR` | — | Checkpoint directory (single model) |
| `OIO_MODELS` | — | `name=/path[,name=/path...]` (multiple models; overrides `OIO_MODEL_DIR`) |
| `OIO_CACHE_DIR` | `~/.cache/oio` | Model cache root (else `$XDG_CACHE_HOME/oio`) — see Checkpoint |
| `OIO_DEFAULT_MODEL` | router fallback | Force the initial checkpoint |
| `OIO_RUNTIME` | `onnx` | `onnx` or `candle` (native CPU) |
| `OIO_DEVICE` | `cpu` | `cpu` or `cuda` (ort CUDA EP, needs a `cuda`-feature build — SPEC §10) |
| `OIO_ORT_INTRA_THREADS` | ONNX Runtime default | Optional positive integer for ORT intra-op CPU threads |
| `OIO_API_KEY` | unset | Require `Authorization: Bearer <key>` |
| `OIO_MAX_CONCURRENT` | `16` | In-flight predicts per process |
| `OIO_MAX_TOKEN_BUDGET` | `8192` | Token budget cap |
| `OIO_MAX_LOADED` | router default | LRU bound on resident checkpoints |
| `OIO_AUTO_TASK` | `false` | Auto-detect typed-decisions workflow |
| `OIO_HOST` | `0.0.0.0` | Bind address |
| `OIO_PORT` | `8000` | Bind port |

## Checkpoint

A checkpoint directory is a converted Laya ONNX export:

```
laya-english/
├── encoder/
├── laya.onnx
├── laya.onnx.data
├── model.safetensors
├── rl_agent_config.json
└── tokenizer/
```

Resolution order (SPEC §9): `OIO_MODELS` → `OIO_MODEL_DIR` → the cache
(`$OIO_CACHE_DIR`, else `$XDG_CACHE_HOME/oio`, else `~/.cache/oio`). The cache
holds one directory per canonical checkpoint name (`english`,
`multilingual`); stray subdirectories are ignored:

```
~/.cache/oio/
├── english/        # snapshot root (layout above)
└── multilingual/   # …/multilingual subdir of the same snapshot
```

Fetching is an explicit operator step; inference never touches the network:

```bash
huggingface-cli download convaiinnovations/laya --revision <pin> --local-dir /srv/laya
OIO_MODELS=english=/srv/laya,multilingual=/srv/laya/multilingual oio-serve
```

A coreutils `SHA256SUMS` manifest beside a checkpoint makes load verify every
listed file (mismatch → model error); without one, loading proceeds untouched:

```bash
cd /srv/laya && sha256sum laya.onnx laya.onnx.data rl_agent_config.json > SHA256SUMS
```

Without a checkpoint the workspace still builds and all non-parity tests run;
the engine/longdoc/onnx_parity/candle_parity suites skip themselves
(`OIO_MODEL_DIR` unset) rather than fail.

## Development

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo check -p oio            # fast path: no ort/tokenizers/axum

# full parity against the reference checkpoint
OIO_MODEL_DIR=/path/to/laya-english cargo test --workspace
```

CI runs the first three on every push and pull request
(`.github/workflows/ci.yml`).

## Docs

- [docs/PRD.md](docs/PRD.md) — product requirements
- [docs/PLAN.md](docs/PLAN.md) — milestone plan
- [docs/SPEC.md](docs/SPEC.md) — invariants
- [docs/COMPAT.md](docs/COMPAT.md) — Jev/Laya compatibility contract
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate layout and data flow
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) — paired ONNX/Candle benchmark method
- [docs/TRAINING.md](docs/TRAINING.md) — staged training pipeline, data policy, and smoke-test scope
- [docs/GITA-DEMO.md](docs/GITA-DEMO.md) — local Gita retrieval + typed-decision prototype
- [docs/JEV-WIRE.md](docs/JEV-WIRE.md) — reference notes on Laya's wire behaviour
- [docs/ADR/](docs/ADR/) — decision records

## License

Copyright 2026 devstroop. Licensed under the
[Apache License 2.0](LICENSE).
