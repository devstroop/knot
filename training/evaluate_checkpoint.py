#!/usr/bin/env python3
"""Evaluate a local KNOT checkpoint on one prepared split and write a JSON report.

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
        "--feature-cache",
        type=Path,
        help="Directory from training.build_feature_cache; reads cached encoder features",
    )
    parser.add_argument(
        "--verify-cases",
        type=int,
        default=None,
        help="With --feature-cache, check cached logits against the live forward pass "
        "(default: 64, 0 disables)",
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
    verify_cases = args.verify_cases
    if verify_cases is None:
        verify_cases = 64 if args.feature_cache else 0
    if verify_cases < 0:
        parser.error("--verify-cases must not be negative")
    if args.verify_cases is not None and not args.feature_cache:
        parser.error("--verify-cases requires --feature-cache")

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

    if args.feature_cache:
        if __package__ in (None, ""):
            from training.feature_cache import (
                CacheError,
                FeatureCache,
                cached_forward_fn,
                verify_cached_forward,
            )
        else:
            from .feature_cache import (
                CacheError,
                FeatureCache,
                cached_forward_fn,
                verify_cached_forward,
            )

    torch.set_num_threads(args.threads)
    torch, model, tokenizer, _, encode = load_model(model_dir, laya_source)
    weights_path = model_dir / "model.safetensors"
    if args.weights:
        weights_path = Path(args.weights).resolve()
        model.load_state_dict(load_file(str(weights_path)), strict=True)
    items = [encode(row) for row in rows]
    if not items:
        parser.error("%s split produced no encoded cases" % args.split)

    forward_fn = None
    cache_report = None
    if args.feature_cache:
        provenance = {
            "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
            "laya_source_revision": laya_revision,
            "training_data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        }
        cache_dir = args.feature_cache.resolve()
        try:
            cache = FeatureCache(
                cache_dir / ("%s.bin" % args.split),
                split=args.split,
                expected_provenance=provenance,
            )
        except CacheError as exc:
            parser.error(str(exc))
        missing = cache.missing(item["case_id"] for item in items)
        if missing:
            cache.close()
            parser.error(
                "feature cache is missing %d case(s), first %r; rebuild it"
                % (len(missing), missing[0])
            )
        verification = None
        if verify_cases:
            try:
                verification = verify_cached_forward(
                    model, items, cache, tokenizer, args.batch_size, torch,
                    limit=min(verify_cases, len(items)),
                )
            except CacheError as exc:
                cache.close()
                parser.error(str(exc))
        forward_fn = cached_forward_fn(model, cache, torch)
        cache_report = {
            "directory": str(cache_dir),
            "split": args.split,
            "cases": len(cache),
            "verification": verification,
        }

    metrics = summarize(evaluate_model(
        model, items, tokenizer, args.batch_size, torch,
        progress_every=args.progress_every or None,
        forward_fn=forward_fn,
    ))
    if forward_fn is not None:
        cache.close()
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
        "feature_cache": cache_report,
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
        "feature_cache": report["feature_cache"],
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
