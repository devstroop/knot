#!/usr/bin/env python3
"""Evaluate a local OIO checkpoint on one prepared split and write a JSON report.

Used to record the unchanged base-checkpoint baseline and, after a training
recipe is frozen, to evaluate the experimental weights against the same split.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data_contract import (
        DataError,
        SPLITS,
        read_split,
        sha256_file,
        validate_directory,
    )
    from training.eval_metrics import evaluate_model, summarize
    from training.train_frozen_head import PINNED_LAYA_REVISION, load_model, source_revision
else:
    from .data_contract import (
        DataError,
        SPLITS,
        read_split,
        sha256_file,
        validate_directory,
    )
    from .eval_metrics import evaluate_model, summarize
    from .train_frozen_head import PINNED_LAYA_REVISION, load_model, source_revision


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="Local checkpoint; no network access is used")
    parser.add_argument("--laya-source", required=True, help="Laya source checkout at the pinned revision")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--weights", help="Optional experimental safetensors overriding the base weights")
    parser.add_argument("--output", required=True, type=Path, help="Report path (JSON)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Evaluate only the first N rows; the report is marked as smoke-limited",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress after this many evaluated cases (0 disables)",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must not be negative")

    data_dir = Path(args.data_dir)
    laya_source = Path(args.laya_source).resolve()
    model_dir = Path(args.model_dir).resolve()
    try:
        counts = validate_directory(data_dir)
        laya_revision = source_revision(laya_source)
        rows = read_split(data_dir / ("%s.jsonl" % args.split))
    except (DataError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    if laya_revision != PINNED_LAYA_REVISION:
        parser.error(
            "Laya source revision %s does not match pinned %s"
            % (laya_revision, PINNED_LAYA_REVISION)
        )

    smoke_limited = args.max_cases is not None and args.max_cases < len(rows)
    if smoke_limited:
        rows = rows[:args.max_cases]

    import torch
    import transformers
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    torch, model, tokenizer, _, encode = load_model(model_dir, laya_source)
    weights_path = model_dir / "model.safetensors"
    if args.weights:
        weights_path = Path(args.weights).resolve()
        model.load_state_dict(load_file(str(weights_path)), strict=True)
    items = [encode(row) for row in rows]
    if not items:
        parser.error("%s split produced no encoded cases" % args.split)

    metrics = summarize(evaluate_model(
        model, items, tokenizer, args.batch_size, torch,
        progress_every=args.progress_every or None,
    ))
    report = {
        "experiment": "stage-one-checkpoint-eval",
        "split": args.split,
        "smoke_limited": smoke_limited,
        "split_counts": counts,
        "evaluated_case_count": len(items),
        "model_dir": str(model_dir),
        "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
        "evaluated_weights_sha256": sha256_file(weights_path),
        "evaluated_weights_path": str(weights_path),
        "laya_source_revision": laya_revision,
        "training_data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "device": "cpu",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "wall_clock_seconds": round(time.monotonic() - started, 3),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "calibration_applied": False,
        "metrics": metrics,
    }
    if args.split == "test":
        report["protocol_note"] = (
            "Test evaluations must follow the frozen stage-one protocol in "
            "docs/TRAINING.md: run them only after the recipe is frozen."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "split": report["split"],
        "smoke_limited": report["smoke_limited"],
        "evaluated_case_count": report["evaluated_case_count"],
        "accuracy": metrics["accuracy"],
        "by_primitive": {
            name: {key: round(value, 4) if isinstance(value, float) else value
                   for key, value in summary.items()}
            for name, summary in metrics["by_primitive"].items()
        },
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
