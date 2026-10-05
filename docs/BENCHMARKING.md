# OIO runtime benchmarks

## Purpose

`runtime_bench` compares OIO's ONNX Runtime and Candle backends using the same
Rust `Engine`, prompt/tokenizer path, English checkpoint, and CPU. It is an
engineering benchmark for deciding whether Candle should become the preferred
runtime for native OIO models. It is not an MLPerf result or a general model
quality evaluation.

The harness measures:

- Engine construction/load time for each backend (reported separately from
  inference).
- Warm, single-request `Engine::predict` latency over all 12 labelled English
  fixture requests, including shared request preparation and tokenization.
- Warm `predict_batch` latency and throughput at batch sizes 1, 2, and 4 for
  the same 12 states under one common question schema. This is a controlled
  throughput workload, distinct from the heterogeneous single-request fixture.
- Decision agreement and maximum absolute answer-probability difference
  between backends for the single-request fixture, plus maximum absolute
  continuous-score delta. It fails on choice or noul decision disagreement;
  score drift is reported separately because score answers are continuous.

Both models are loaded before timing. Each receives three warm-up fixture
passes. Timed single requests are paired and alternate backend order per round
to reduce order bias. Batch timings likewise alternate order and report both
per-call latency and states/second. Percentiles use nearest-rank p50/p95.
Build with `--release`; debug timings are not comparable.

The harness prints process CPU affinity and thread-related environment
settings. `OIO_ORT_INTRA_THREADS` explicitly controls ORT's intra-op pool;
`RAYON_NUM_THREADS` configures Candle's Rayon-backed CPU kernels at process
startup. Use separate processes for each setting because thread pools are
process-global. For example, compare both runtimes using 1, 2, and 4 worker
threads:

```bash
for n in 1 2 4; do
  OIO_MODEL_DIR="$HOME/.cache/oio/english" \
  OIO_ORT_INTRA_THREADS="$n" RAYON_NUM_THREADS="$n" \
  OIO_BENCH_MODE=single OIO_BENCH_ROUNDS=3 \
  cargo test -p oio --features onnx,candle --release --test runtime_bench \
    -- --ignored --nocapture
done
```

Set `OIO_CANDLE_PROFILE=1` to report average Candle tensor preparation,
ModernBERT encoder, custom decision-head, and classifier/output-copy time per
forward call. Profiling adds timing/locking overhead and should be used to
locate bottlenecks, not to publish final latency numbers. Set it in the
benchmark process; aggregate stage summaries are printed when the runtime is
dropped.

## Initial measurement

One run on 2026-10-05 used checkpoint revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, Rust 1.99.0, Linux 6.17.2, and
an Intel Xeon E5-2640 v4 at 2.40 GHz (20 cores / 40 threads, 64 GiB RAM).
Single-request latency used 20 rounds (240 requests per backend); batch
latencies used 10 rounds. Both backends used their current defaults.

| Workload | ORT mean / p50 / p95 | Candle mean / p50 / p95 | Throughput ORT / Candle |
|---|---:|---:|---:|
| Single request | 235 / 232 / 271 ms | 1,583 / 1,580 / 1,649 ms | — |
| 12-state batch, size 1 | 1,741 / 1,838 / 1,991 ms | 18,529 / 18,489 / 18,885 ms | 6.89 / 0.65 states/s |
| 12-state batch, size 2 | 1,487 / 1,286 / 1,799 ms | 11,177 / 11,153 / 11,377 ms | 8.07 / 1.07 states/s |
| 12-state batch, size 4 | 1,351 / 1,170 / 1,619 ms | 7,554 / 7,419 / 7,872 ms | 8.88 / 1.59 states/s |

Engine construction took 3.52 s for ORT and 2.84 s for Candle. Choice and
noul decisions agreed on all fixture cases; maximum probability delta was
0.0018 and maximum continuous score delta was 0.0022. Thus this particular
Candle implementation was about 6.7x slower for single requests and remained
slower at batch size 4 on this host, despite batch throughput improving as
batch size increased.

This is one machine, one model, one run, and a small fixture. ORT also logged
a CPU-affinity warning for an invalid affinity mask in this environment, though
inference completed. Treat the numbers as a useful first result—not a general
Candle-versus-ORT conclusion. Repeat on the target deployment hardware and
investigate equalized thread/affinity settings before selecting a default.

## Controlled four-CPU follow-up

A follow-up pinned each fresh benchmark process to CPUs `2,4,6,8` and matched
ORT intra-op threads and Candle/Rayon threads at 1, 2, then 4. Each configuration
ran three timed rounds (36 examples per backend), with the same warm-up and
agreement checks. The process was pinned to four distinct physical CPUs on one
socket, and the cgroup reports no CPU quota (`cpu.max=max 100000`). This is a
valid four-worker host-local comparison, though it remains only four of the
machine's 20 physical cores.

| Matched threads | ORT mean / p50 / p95 | Candle mean / p50 / p95 | Candle / ORT latency |
|---:|---:|---:|---:|
| 1 | 534 / 511 / 725 ms | 880 / 852 / 1,164 ms | 1.65x |
| 2 | 587 / 580 / 680 ms | 1,138 / 1,133 / 1,310 ms | 1.94x |
| 4 | 200 / 194 / 239 ms | 412 / 395 / 528 ms | 2.06x |

No ORT CPU-affinity warning appeared with explicit intra-op thread counts.
The four-thread run is much faster than the default and is the best of these
three settings for both backends. Setting equal thread counts therefore
narrows, but does not close, the gap.

With `OIO_CANDLE_PROFILE=1`, four-thread Candle time averaged approximately
0.01 ms tensor preparation, 385 ms ModernBERT encoder, 36 ms custom
decision-head, and 0.75 ms classifier/output-copy per call. Thus the encoder
accounts for roughly 91% of profiled Candle forward time and is the rational
first optimization target. The 1- and 2-thread profiles also showed encoder
time dominating. Profiling numbers include instrumentation and are not the
final latency figures above.

A longer four-thread run used 10 rounds (120 single requests per backend and
10 batches per batch size), with profiling disabled:

| Workload | ORT mean / p50 / p95 | Candle mean / p50 / p95 | Throughput ORT / Candle |
|---|---:|---:|---:|
| Single request | 196 / 193 / 220 ms | 401 / 400 / 433 ms | — |
| 12 states, batch size 1 | 1,987 / 2,010 / 2,067 ms | 4,333 / 4,351 / 4,456 ms | 6.04 / 2.77 states/s |
| 12 states, batch size 2 | 1,749 / 1,695 / 1,846 ms | 3,709 / 3,732 / 3,832 ms | 6.86 / 3.23 states/s |
| 12 states, batch size 4 | 1,586 / 1,577 / 1,679 ms | 3,362 / 3,375 / 3,425 ms | 7.57 / 3.57 states/s |

The longer run confirms about a 2x Candle single-request latency penalty on
this host under matched four-thread CPU settings. Candle attained about 46–47%
of ORT's throughput on these 12-state batches. Choice/noul decisions agreed;
the same maximum probability and score deltas were observed. Because the
benchmark is pinned to four cores on one socket, repeat on the intended
deployment host before making broader claims.

A separate three-round single-request run compiled all Rust code with
`-C target-cpu=native`, pinned the process to the same four CPUs, and matched
both backends at four threads. It measured ORT mean 193 ms and Candle mean
391 ms (Candle/ORT latency 2.03x), versus 196 ms and 401 ms in the longer
baseline run. This small difference is within the variability of these
separate runs and does not justify requiring host-specific native codegen.

## Run

The checkpoint directory must contain both `laya.onnx` (and its external data
file) and `model.safetensors`, along with OIO's usual tokenizer/config files.
The pinned English checkpoint in `~/.cache/oio/english` is suitable.

```bash
cd oio
OIO_MODEL_DIR="$HOME/.cache/oio/english" \
OIO_BENCH_ROUNDS=20 \
cargo test -p oio --features onnx,candle --release --test runtime_bench \
  -- --ignored --nocapture
```

An Intel MKL backend was explored as a possible CPU-kernel optimization, but
the optional MKL integration did not link in this environment: Candle 0.11's
MKL path referenced `hgemm_`, which was not provided by the bundled MKL
2020.1-3038006115 library. No MKL performance result was obtained, so it is
not included as a supported OIO feature.

The benchmark is ignored in regular test runs because it loads both large
models and consumes CPU for several minutes. Increase `OIO_BENCH_ROUNDS` for
more stable tail percentiles; use the same value and machine for comparisons.
Run at least three fresh processes and report the median of each run's summary
for a release decision.

Record the following with any published result:

- OIO commit, checkpoint revision/digest, Rust version, OS/kernel, CPU model,
  memory, and whether the machine was otherwise idle.
- Compiler profile and features, benchmark rounds, device, and thread/runtime
  settings. Do not compare release results to debug builds.
- Load time, single-request p50/p95/mean, batch p50/p95 and states/second, and
  backend agreement, maximum probability delta, and maximum score delta. The
  `candle_over_onnx_latency` ratio is greater than 1 when Candle is slower;
  `candle_over_onnx_throughput` is greater than 1 when Candle is faster. Keep
  the heterogeneous fixture latency separate from the homogeneous batch
  workload.

Run final comparisons both with defaults and with controlled thread-count
sweeps. For a target machine, pin the process to a valid CPU set with `taskset`
and report that affinity. Use separate processes for RSS comparisons and do
not infer a universal backend winner from one CPU/checkpoint.

## Model quality is a separate question

The 12-case `eval_english.jsonl` fixture measures existing-model parity and
sanity, not whether an OIO-trained model is better. A model-quality benchmark
needs a frozen, representative held-out dataset, a predeclared metric and
acceptance threshold, and explicit calibration/robustness measurements.
Keep training/evaluation data separate from model selection where possible.
Report accuracy by question type and class, calibration (e.g. Brier score or
ECE with bin/sample details), and confidence intervals, not only an aggregate
score.

## References and useful examples

- [MLPerf Inference: Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/)
  describes scenarios, load generation, latency/throughput metrics, and
  quality targets. Its [benchmark paper](https://arxiv.org/abs/1911.02549)
  explains the motivation for controlled, representative inference workloads.
  OIO's microbenchmark is much narrower and should not be presented as an
  MLPerf-compliant measurement.
- [PyTorch benchmark timer quick start](https://docs.pytorch.org/tutorials/recipes/recipes/timer_quick_start.html)
  demonstrates warm-up/repeated measurements and controlling thread count.
  The OIO harness similarly separates warm-up and timed work; backend default
  thread settings are recorded as a limitation rather than changed invisibly.
- [Criterion.rs user guide](https://bheisler.github.io/criterion.rs/book/)
  is a useful reference for statistical microbenchmark design. This first
  harness avoids a new dependency and reports reproducible raw summary
  statistics; use Criterion for smaller isolated kernels if those become the
  tuning target.
- [Candle examples](https://github.com/huggingface/candle/tree/main/candle-examples/examples)
  provide concrete Rust-native model loading/inference implementations across
  transformer and other model families. OIO's benchmark compares its actual
  Candle ModernBERT implementation to its actual ORT path, not those unrelated
  example models.
- [ONNX concepts](https://onnx.ai/onnx/intro/concepts.html) explains the
  portable inference graph and operator model underpinning the ORT path.
