# KNOT training pipeline

KNOT keeps ONNX Runtime and Candle as configurable inference engines. Training
is a separate, optional research workflow: it must not add Python or training
dependencies to the Rust serving binaries, and it must not make either runtime
mandatory.

## Stage-one scope

The first experiment is English typed decisions across multiple domains. It
starts from the pinned Laya checkpoint, freezes its ModernBERT encoder, and
trains only the decision head on CPU. It is a pipeline and compatibility
smoke test, not a claim that KNOT has trained a better model.

All three KNOT primitives are represented, with these important distinctions:

| Primitive | Source | Label meaning and limitation |
|---|---|---|
| `choice` | [MASSIVE English](https://huggingface.co/datasets/AmazonScience/massive) and [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) | Direct intent labels, using the datasets' official test splits |
| `noul` | Deterministic derivation from MASSIVE domain labels | Whether an utterance belongs to its annotated domain versus a different domain. These are derived binary examples, **not** independently annotated out-of-scope examples |
| `score` | OIO-authored synthetic smoke examples in `training/data/score_smoke.jsonl` | Exercises score serialization, tokenization, loss, and evaluation only. It is not public data or a quality benchmark |

The MASSIVE and BANKING77 dataset cards declare CC BY 4.0. Their data is
downloaded at immutable revisions when preparing a local run; source examples
are not checked into this repository. Retain attribution and the source,
revision, split, and derivation for each output row. Only English examples are
used in this experiment. The synthetic score file is authored for this
repository and covered by its Apache-2.0 license.

We did not find a non-social, ordered text-score dataset with a sufficiently
clear reuse license in the reviewed candidates. Synthetic score examples are
permitted only to smoke-test the pipeline. A broad-domain score quality claim
is blocked until a clearly licensed, genuinely ordered score dataset is
approved. Likewise, derived MASSIVE `noul` examples do not establish
out-of-scope detection quality.

## Data contract and splits

`training/prepare_data.py` writes local JSONL files with one question per row.
Each row contains:

- `case_id`, `state`, `questions`, `expected`: the Laya-compatible decision
  input and hard label.
- `split`: `train`, `validation`, `calibration`, or `test`.
- `source`: dataset ID, immutable revision, declared license, original split,
  source-row identity, and derivation.
- `language` and `tags`: slice/report metadata.

Dataset IDs are frozen data-plane identifiers: the synthetic source stays
`oio-authored/score-smoke` (revision `oio-training-v1`) even after the
product rename, so existing prepared corpora, manifests, and feature caches
keep validating. Do not "fix" these strings to match the product name.

The prepare step uses MASSIVE's official validation and test splits, carving
calibration rows from its training split. BANKING77 has no official validation
split, so validation and calibration are carved deterministically and
stratified from its official training split; its official test split remains
untouched. Derived binary examples stay with their original MASSIVE row's
split. Exact-text duplicates are assigned to the highest-priority split in
`test`, `validation`, `calibration`, `train` order, so lower-priority copies
are excluded and the official test inputs are never removed. Exclusion counts
are written to the manifest. The validator rejects remaining source-row or
exact-text overlap across splits, unknown license labels, invalid targets, and
missing primitive coverage in the training split.

Prepared files and run outputs live under ignored `training/out/`. The
validator, tests, and data preparation use only the Python standard library.
Preparation reads the versioned MASSIVE 1.1 archive and BANKING77 CSVs from a
pinned upstream Git commit; it does not execute remote dataset-loading code.
Checksums for the downloaded inputs and prepared splits are recorded in the
manifest. To prepare data, run:

```bash
python -m training.prepare_data --output-dir training/out/data
python -m training.validate_data training/out/data
```

The prepared-data manifest records the fixed revisions and SHA-256 of each
split. Do not publish the generated rows without checking the source license
and attribution obligations for the intended distribution.

## Frozen-head smoke run

The smoke trainer uses the Laya reference implementation of the exact typed
head and an already-downloaded, locally pinned KNOT-compatible checkpoint. It
does not fetch weights. Install the optional training requirements and make a
clean Laya source checkout available through `--laya-source`:

```bash
python -m pip install -r training/requirements.txt
python -m training.train_frozen_head \
  --data-dir training/out/data \
  --model-dir /path/to/pinned/laya-checkpoint \
  --laya-source /path/to/laya \
  --output-dir training/out/frozen-head-smoke \
  --max-train-per-label 1 \
  --epochs 1
```

Use the Laya source revision pinned in `training/train_frozen_head.py`; the
script refuses a different checkout rather than silently changing the head
implementation.

The encoder is held in evaluation mode with gradients disabled. The head uses
supervised cross-entropy over the existing typed option markers. The default
selection cap is deliberately small and deterministic. The output is an
experimental weights file and a run manifest, **not a deployable model**:
calibration has not been refit and final-test evaluation is not part of this
smoke command. Never replace a release checkpoint with this output.

The run manifest records the selected source IDs, seeds, split hashes, base
checkpoint path, and trainable parameter count. A later quality experiment
must separately define and freeze model-selection, calibration, and test
protocols; compare against the unchanged base checkpoint; report per-primitive
metrics and confidence calibration; and pass both ONNX and Candle parity gates
from the same canonical weights before any deployment consideration.

## Frozen-encoder feature cache

Stage one freezes the encoder, so its output for a case never changes. Reusing
it is what makes CPU training affordable: without the cache, every optimizer
step re-runs the encoder (tens of CPU-hours per epoch); with the cache, an
epoch only runs the two head layers.

Build it once per checkpoint/data combination:

```bash
python -m training.build_feature_cache \
  --data-dir training/out/data \
  --model-dir /path/to/pinned/laya-checkpoint \
  --laya-source /path/to/laya \
  --output-dir training/out/features \
  --split train --split validation --split calibration --split test
```

How it is guarded:

- **Contents:** encoder outputs only — captured before the head's type
  embedding — stored per case as a flat binary plus a JSON index keyed by
  `case_id`. `--dtype float16` is the default; storage scales with
  `cases × sequence length × hidden size` (about 0.68 MB per case at the
  current ~330-token sequences and hidden size 1024).
- **Provenance:** base weights SHA-256, Laya revision, and data-manifest
  SHA-256 are recorded in the index. A cache built from different weights,
  source revision, or prepared data is rejected as stale instead of being
  used. Because the trainer refuses to run with any encoder parameter
  unfrozen, the cached encoder and the live encoder are always the same
  encoder.
- **Verification:** after writing a split, the builder compares cached-feature
  logits against the live forward pass on 64 cases and refuses to publish a
  cache that predicts a different option. Trainers and evaluators re-verify on
  every run with `--verify-cases` (default 64, `0` disables). A mismatch means
  rebuild with `--dtype float32`; the error says so rather than continuing
  with disagreeing paths.

Use it wherever the encoder would otherwise run:

```bash
python -m training.train_frozen_head \
  ... --feature-cache training/out/features

python -m training.evaluate_checkpoint \
  ... --feature-cache training/out/features
```

Both record the cache directory and verification summary alongside their
metrics, so a report produced from cached features stays identifiable, and a
missing or partial cache is an error — never a silent fallback to live
encoder runs.

## Stage-one quality gates (predeclared)

The frozen-head smoke run proves the plumbing. The first quality experiment has to
prove a held-out gain against the unchanged base checkpoint. These gates are
declared before any full-corpus training run. Changing a threshold after
seeing test numbers invalidates the experiment; thresholds change only through
an explicit revision of this document that says so.

### Protocol

1. **Baseline first.** Evaluate the unchanged base checkpoint on the prepared
   `validation` split with `training/evaluate_checkpoint.py` and keep the JSON
   report. Every later comparison is made against that number, produced by the
   same evaluator on the same split.
2. **Select on validation only.** Epochs, learning rate, and checkpoint choice
   come from `validation` metrics. The `calibration` split is reserved for
   fitting confidence calibration after the recipe is frozen. The `test` split
   is evaluated exactly twice — base and trained — and only after the recipe,
   seed, and epoch count are frozen.
3. **Report per primitive and per source.** Aggregate accuracy can hide a
   regression in MASSIVE or BANKING77, so both slices are reported next to the
   overall numbers.
4. **Calibration is measured, not assumed.** Brier score and 10-bin expected
   calibration error come from the evaluator; any calibration refit on the
   `calibration` split is recorded separately and never fit on test.

### Gates

A stage-one run counts as a held-out gain only if all of these hold on the
`test` split, base versus trained, from the same evaluator:

| Gate | Requirement |
|---|---|
| G1 `choice` | trained choice accuracy ≥ base choice accuracy + 0.01 absolute |
| G2 no source regression | neither MASSIVE nor BANKING77 choice accuracy falls more than 0.005 below its base value |
| G3 `noul` | trained noul Brier ≤ base Brier and trained noul ECE ≤ base ECE. Both may stay poor: the labels are derived, so this gate only forbids worsening |
| G4 `score` | reported for completeness, excluded from pass/fail — the labels are OIO-authored synthetic |
| G5 protocol | test evaluated only after freezing; calibration fit only on `calibration`; both reports carry the split and weights hashes |
| G6 deployment | out of scope for stage one: deployability still requires ONNX and Candle parity from the same canonical weights |

If a gate fails, the experiment failed. Do not retune the thresholds to match
the result; change the recipe — data volume, epochs, learning rate, what is
unfrozen — and rerun.

### Measuring a checkpoint

```bash
python -m training.evaluate_checkpoint \
  --data-dir training/out/data \
  --model-dir /path/to/pinned/laya-checkpoint \
  --laya-source /path/to/laya \
  --split validation \
  --batch-size 16 --threads 16 \
  --output training/out/baseline-validation.json
```

Pass `--weights path/to/experimental-model.safetensors` to evaluate trained
weights on top of the same base checkpoint. The report records the evaluated
weights SHA-256, base weights SHA-256, data-manifest hash, Laya revision, batch
size, thread count, wall-clock seconds, and metrics: overall accuracy, then per
primitive count/accuracy/mean confidence/Brier/10-bin ECE (plus MAE for
`score`) and per-source accuracy. `--split test` reports carry a protocol note
reminding the caller of rule 2 above.

`--max-cases N` runs a fast smoke check; those reports are marked
`smoke_limited` and must never be quoted as results.

### Recorded baseline (validation)

Produced from the unchanged pinned base checkpoint before any training run:
weights `891102d372688fc2…`, data manifest `5add84afe7ff7932…`, batch 16, 16
threads, 3,097 s wall for all 7,087 cases.

| Slice | Cases | Accuracy | Mean confidence | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| `choice`, all | 3,030 | 0.425 | 0.549 | 0.779 | 0.159 |
| `choice`, MASSIVE | 2,028 | 0.453 | | | |
| `choice`, BANKING77 | 1,002 | 0.369 | | | |
| `noul`, derived | 4,056 | 0.689 | 0.937 | 0.544 | 0.247 |
| `score`, synthetic | 1 | n=1, not meaningful | | | |
| overall | 7,087 | 0.576 | | | |

What this fixes for later comparisons:

- The base checkpoint reaches 0.425/0.369 choice accuracy on the two intent
  sources, so G1 and G2 have measurable headroom. These — not published
  dataset numbers — are what trained runs are compared against.
- The `score` slice has exactly one validation case: the synthetic corpus is
  far too small to say anything about score quality, which is why G4 keeps
  score out of pass/fail and why real score supervision stays the blocker
  rather than a score threshold.
- `noul` shows 0.94 mean confidence against 0.69 accuracy (ECE 0.247): the
  base checkpoint is already overconfident on derived binary labels, so G3
  forbidding any worsening is a real constraint, not a formality.
- Forward-only throughput was 2.3 rows/s (7,087 rows in 3,097 s). Training
  adds backward passes, so a full 38,988-row epoch is on the order of ten or
  more CPU-hours on this class of host — see the cost note below.

The full JSON report — per-source slices, weights and data hashes, wall clock
— lives in the ignored `training/out/baseline-validation.json`. Reproduce it
with the command above; do not edit these numbers by hand.

Note on cost: the encoder is frozen but still evaluated on every training
step, so a full-corpus CPU epoch is measured in hours. Record the selected data
volume and wall-clock time in the run manifest so runs stay comparable, and
treat a subset run as a declared subset — never report it as a full-corpus
result.

## Future stages

1. **Data and smoke pipeline:** licensed adapters, provenance/split validation,
   and this frozen-head CPU run.
2. **Quality experiment:** execute the gates above — baseline, frozen recipe,
   held-out test — and obtain approved real `score` supervision, which still
   blocks any score-quality claim.
3. **Canonical artifact:** version the training checkpoint and derive both
   ONNX and Candle-compatible artifacts from the same weights; retain
   `KNOT_RUNTIME=onnx|candle`.
4. **Small student:** only after stage two demonstrates a repeatable held-out
   gain, compare distillation into a smaller model compatible with both
   runtimes.

Full encoder fine-tuning, RLCD, teacher-generated labels, private/user data,
unlicensed corpora, and training a model from scratch are out of scope for this
first smoke experiment.

## Dataset references

- MASSIVE: [CC BY 4.0 dataset card](https://huggingface.co/datasets/AmazonScience/massive),
  official [1.1 archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz).
- BANKING77: [CC BY 4.0 dataset card](https://huggingface.co/datasets/PolyAI/banking77),
  upstream Git commit `9d081458ff52e53cf7e848f414e6e9344e4e6696` in
  [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets).
- Base checkpoint source: Laya's Apache-2.0 model card, pinned to the
  checkpoint already used by this workspace; provide that local checkpoint
  explicitly to the trainer rather than downloading a floating revision.
