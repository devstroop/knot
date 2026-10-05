# OIO training pipeline

OIO keeps ONNX Runtime and Candle as configurable inference engines. Training
is a separate, optional research workflow: it must not add Python or training
dependencies to the Rust serving binaries, and it must not make either runtime
mandatory.

## Stage-one scope

The first experiment is English typed decisions across multiple domains. It
starts from the pinned Laya checkpoint, freezes its ModernBERT encoder, and
trains only the decision head on CPU. It is a pipeline and compatibility
smoke test, not a claim that OIO has trained a better model.

All three OIO primitives are represented, with these important distinctions:

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
head and an already-downloaded, locally pinned OIO-compatible checkpoint. It
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

## Future stages

1. **Data and smoke pipeline:** licensed adapters, provenance/split validation,
   and this frozen-head CPU run.
2. **Quality experiment:** obtain approved real score supervision and define
   data volume, validation/calibration policy, held-out metrics, and baselines.
3. **Canonical artifact:** version the training checkpoint and derive both
   ONNX and Candle-compatible artifacts from the same weights; retain
   `OIO_RUNTIME=onnx|candle`.
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
