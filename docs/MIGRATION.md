# Migrating from oio to knot

The decision engine previously named `oio` is now **knot**
([ADR-007](ADR/007-knot-naming.md)). This guide covers the mechanical
upgrade. Rule of thumb: the product prefix changed `OIO_` → `KNOT_`, the
binary `oio-serve` → `knot`, everything else about behavior is identical —
parity fixtures pass byte-identical across the rename.

## 1. Config vars (`OIO_*` → `KNOT_*`, hard cut, no shims)

Rename every variable; old names are not read.

| Before (`OIO_*`) | After (`KNOT_*`) | Notes |
|---|---|---|
| `OIO_MODELS` | `KNOT_MODELS` | Multiple checkpoints; overrides `KNOT_MODEL_DIR` |
| `OIO_MODEL_DIR` | `KNOT_MODEL_DIR` | Single checkpoint directory |
| `OIO_DEFAULT_MODEL` | `KNOT_DEFAULT_MODEL` | Force the initial checkpoint |
| `OIO_MAX_LOADED` | `KNOT_MAX_LOADED` | LRU bound on resident checkpoints |
| `OIO_AUTO_TASK` | `KNOT_AUTO_TASK` | Typed-decisions workflow detection |
| `OIO_DEVICE` | `KNOT_DEVICE` | `cpu` or `cuda` |
| `OIO_RUNTIME` | `KNOT_RUNTIME` | `onnx` or `candle` |
| `OIO_ORT_INTRA_THREADS` | `KNOT_ORT_INTRA_THREADS` | ORT intra-op threads |
| `OIO_CANDLE_PROFILE` | `KNOT_CANDLE_PROFILE` | Candle stage profiling |
| `OIO_API_KEY` | `KNOT_API_KEY` | Bearer auth |
| `OIO_MAX_CONCURRENT` | `KNOT_MAX_CONCURRENT` | In-flight predicts per process |
| `OIO_MAX_TOKEN_BUDGET` | `KNOT_MAX_TOKEN_BUDGET` | Token budget cap |
| `OIO_HOST` | `KNOT_HOST` | Bind address |
| `OIO_PORT` | `KNOT_PORT` | Bind port |
| `OIO_CACHE_DIR` | `KNOT_CACHE_DIR` | Model cache root override |
| `OIO_UPDATE_GOLDEN` | `KNOT_UPDATE_GOLDEN` | Rewrite `golden_english.json` (tests) |
| `OIO_BENCH_MODE` | `KNOT_BENCH_MODE` | Benchmark harness mode |
| `OIO_BENCH_ROUNDS` | `KNOT_BENCH_ROUNDS` | Benchmark harness rounds |
| `OIO_TEST_MODEL_DIR` | `KNOT_TEST_MODEL_DIR` | Checkpoint for cache parity tests |
| `OIO_TEST_LAYA_SOURCE` | `KNOT_TEST_LAYA_SOURCE` | Laya checkout for cache parity tests |

Demo scripts additionally renamed their overrides: `DEMO_OIO_CACHE` →
`DEMO_KNOT_CACHE`, `DEMO_OIO_BIN` → `DEMO_KNOT_BIN`, `DEMO_OIO_URL` →
`DEMO_KNOT_URL` (see `scripts/demo.sh`, `scripts/demo_engines.py`).

## 2. Binary and MCP tools

| Before | After |
|---|---|
| `oio-serve` binary (`target/{debug,release}/oio-serve`, `cargo run -p oio-serve`) | `knot` binary (`target/{debug,release}/knot`, `cargo run -p knot-serve`) |
| MCP tools `oio_status`, `oio_route`, `oio_predict`, `oio_predict_batch` | `knot_status`, `knot_route`, `knot_predict`, `knot_predict_batch` |
| MCP server name `oio-mcp` | `knot-mcp` |

Update MCP client configs to the new tool names; there is no aliasing.

## 3. Cache directory

Default moved `~/.cache/oio` → `~/.cache/knot` (`$KNOT_CACHE_DIR`, else
`$XDG_CACHE_HOME/knot`). Move your checkpoints over (or symlink the old path):

```bash
mkdir -p ~/.cache && mv ~/.cache/oio ~/.cache/knot
```

The training-data download default moved likewise to
`~/.cache/knot/training-data`.

## 4. Repository and crates

- GitHub: `devstroop/oio` → [`devstroop/knot`](https://github.com/devstroop/knot)
  (redirect preserved; update remotes: `git remote set-url origin
  https://github.com/devstroop/knot`).
- Crates: `oio` → `knot`, `oio-serve` → `knot-serve` (path/git dependencies
  follow the repo; bare `knot` is taken on crates.io by an unrelated project,
  so publishing that name is out of scope — see ADR-007).

## 5. Training data-plane identifiers (renamed)

The synthetic score source was initially frozen under its original name,
then renamed off `oio` as well (2026-10-07):

| Before | After |
|---|---|
| dataset `oio-authored/score-smoke` | `knot-authored/score-smoke` |
| revision `oio-training-v1` | `knot-training-v1` |
| internal `_oio_row_index` keys | `_knot_row_index` |

Prepared corpora and feature caches issued under the old IDs no longer
validate: re-run `python -m training.prepare_data` and
`python -m training.validate_data`, then rebuild the feature cache with
`python -m training.build_feature_cache`.

## 6. Deliberately unchanged

- Wire protocol (`/v1/systemone`, `/batch`, `/health`, `/models`) and all
  request/response semantics.
- Checkpoint formats and prepared-training-data layouts.
- Recorded evidence artifacts (`demo/results.json`) and merged ADRs keep
  original names as history.
