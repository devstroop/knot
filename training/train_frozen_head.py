#!/usr/bin/env python3
"""Run a CPU-only supervised smoke experiment with the ModernBERT encoder frozen."""

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data_contract import (
        DataError,
        PRIMITIVES,
        primitive_and_label,
        read_split,
        sha256_file,
        validate_directory,
    )
    from training.eval_metrics import collate, evaluate_model, summarize
else:
    from .data_contract import (
        DataError,
        PRIMITIVES,
        primitive_and_label,
        read_split,
        sha256_file,
        validate_directory,
    )
    from .eval_metrics import collate, evaluate_model, summarize

PINNED_LAYA_REVISION = "4aa6761be8173de4ce6d92c31b3e40b6eaf59a7c"
PINNED_BASE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def select_rows(rows, max_per_label, seed):
    groups = defaultdict(list)
    for row in rows:
        primitive, label = primitive_and_label(row, row["case_id"])
        label_key = json.dumps(label, sort_keys=True, ensure_ascii=False)
        groups[(primitive, label_key)].append(row)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        selected.extend(group[:max_per_label])
    selected.sort(key=lambda row: row["case_id"])
    missing = PRIMITIVES - {
        primitive_and_label(row, row["case_id"])[0] for row in selected
    }
    if missing:
        raise DataError("training selection is missing primitive(s): %s" % ", ".join(sorted(missing)))
    return selected


def load_model(model_dir, laya_source):
    sys.path.insert(0, str(laya_source))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import QTYPES, build_model, build_sequence

    model_dir = Path(model_dir)
    required = (
        model_dir / "model.safetensors",
        model_dir / "rl_agent_config.json",
        model_dir / "encoder" / "config.json",
        model_dir / "tokenizer" / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise DataError("base checkpoint is incomplete: " + ", ".join(missing))
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir / "tokenizer", local_files_only=True
    )
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.float()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.to(torch.device("cpu"))
    model.train()
    model.encoder.eval()
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("encoder freeze failed")

    def encode(row):
        qid, question = next(iter(row["questions"].items()))
        primitive = question["type"]
        criterion = {
            "t": primitive,
            "ins": question["instructions"],
            "crit": question["criteria"],
        }
        if "labels" in question:
            criterion["labels"] = question["labels"]
        ids, markers = build_sequence(
            tokenizer,
            row["state"],
            criterion,
            cfg.get("max_len", 1024),
            cfg.get("head_max_len", 256),
        )
        if not markers:
            raise DataError("%s produced no option markers" % row["case_id"])
        target = row["expected"][qid]
        if primitive == "choice":
            target_index = list(question["criteria"]).index(str(target))
        elif primitive == "score":
            target_index = int(target)
        else:
            target_index = 1 if target else 0
        if target_index < 0 or target_index >= len(markers):
            raise DataError("%s target does not fit its marker count" % row["case_id"])
        return {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[primitive],
            "target_index": target_index,
            "primitive": primitive,
            "source": row["source"]["dataset_id"],
            "case_id": row["case_id"],
        }

    return torch, model, tokenizer, cfg, encode


def evaluate(model, items, tokenizer, batch_size, torch, progress_every=None,
             forward_fn=None):
    """Return aggregated held-out metrics for the encoded items."""
    return summarize(evaluate_model(
        model, items, tokenizer, batch_size, torch,
        progress_every=progress_every, forward_fn=forward_fn,
    ))


def source_revision(path):
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Laya source checkout must be clean")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="Local checkpoint; no network access is used")
    parser.add_argument("--laya-source", required=True, help="Laya source checkout at the pinned revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-revision", default=PINNED_BASE_REVISION)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-per-label", type=int, default=1)
    parser.add_argument("--max-validation-per-label", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Directory from training.build_feature_cache; skips the frozen encoder while training",
    )
    parser.add_argument(
        "--verify-cases",
        type=int,
        default=None,
        help="With --feature-cache, check cached logits against the live forward pass "
        "(default: 64, 0 disables)",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_train_per_label < 1:
        parser.error("epochs, batch-size, and max-train-per-label must be positive")
    if args.max_validation_per_label < 1 or args.threads < 1:
        parser.error("max-validation-per-label and threads must be positive")
    verify_cases = args.verify_cases
    if verify_cases is None:
        verify_cases = 64 if args.feature_cache else 0
    if verify_cases < 0:
        parser.error("--verify-cases must not be negative")
    if args.verify_cases is not None and not args.feature_cache:
        parser.error("--verify-cases requires --feature-cache")
    if args.base_revision != PINNED_BASE_REVISION:
        parser.error("base-revision must match the pinned local checkpoint revision")

    data_dir = Path(args.data_dir)
    laya_source = Path(args.laya_source).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir)
    try:
        counts = validate_directory(data_dir)
        laya_revision = source_revision(laya_source)
    except (DataError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    if laya_revision != PINNED_LAYA_REVISION:
        parser.error(
            "Laya source revision %s does not match pinned %s"
            % (laya_revision, PINNED_LAYA_REVISION)
        )

    import torch
    import torch.nn.functional as functional
    import transformers
    import safetensors
    from safetensors.torch import save_file

    if args.feature_cache:
        if __package__ in (None, ""):
            from training.feature_cache import (
                CacheError,
                FeatureCache,
                cached_forward_fn,
                forward_from_cached,
                stack_features,
                verify_cached_forward,
            )
        else:
            from .feature_cache import (
                CacheError,
                FeatureCache,
                cached_forward_fn,
                forward_from_cached,
                stack_features,
                verify_cached_forward,
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_rows = select_rows(read_split(data_dir / "train.jsonl"), args.max_train_per_label, args.seed)
    validation_rows = select_rows(
        read_split(data_dir / "validation.jsonl"),
        args.max_validation_per_label,
        args.seed + 1,
    )
    torch, model, tokenizer, _, encode = load_model(model_dir, laya_source)
    train_items = [encode(row) for row in train_rows]
    validation_items = [encode(row) for row in validation_rows]

    feature_caches = {"train": None, "validation": None}
    cache_verification = None
    if args.feature_cache:
        provenance = {
            "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
            "laya_source_revision": laya_revision,
            "training_data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        }
        cache_dir = args.feature_cache.resolve()
        try:
            train_cache = FeatureCache(
                cache_dir / "train.bin", split="train", expected_provenance=provenance
            )
            validation_cache = FeatureCache(
                cache_dir / "validation.bin",
                split="validation",
                expected_provenance=provenance,
            )
        except CacheError as exc:
            parser.error(str(exc))
        missing = (
            train_cache.missing(item["case_id"] for item in train_items)
            + validation_cache.missing(item["case_id"] for item in validation_items)
        )
        if missing:
            train_cache.close()
            validation_cache.close()
            parser.error(
                "feature cache is missing %d selected case(s), first %r; rebuild it"
                % (len(missing), missing[0])
            )
        if verify_cases:
            try:
                cache_verification = verify_cached_forward(
                    model, train_items, train_cache, tokenizer,
                    args.batch_size, torch,
                    limit=min(verify_cases, len(train_items)),
                )
            except CacheError as exc:
                train_cache.close()
                validation_cache.close()
                parser.error(str(exc))
        feature_caches = {"train": train_cache, "validation": validation_cache}

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable or any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("expected only non-encoder parameters to be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_items)
        model.train()
        model.encoder.eval()
        for start in range(0, len(train_items), args.batch_size):
            chunk = train_items[start:start + args.batch_size]
            ids, attention, positions, mask, labels, qtypes = collate(
                chunk, tokenizer.pad_token_id, torch
            )
            optimizer.zero_grad(set_to_none=True)
            if feature_caches["train"] is not None:
                features = stack_features(chunk, feature_caches["train"], torch)
                logits = forward_from_cached(
                    model, features, attention, positions, mask, qtypes, torch
                )
            else:
                logits, _ = model(ids, attention, positions, mask, qtypes)
            loss = functional.cross_entropy(logits.masked_fill(~mask, -1e4), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

    validation_forward = (
        cached_forward_fn(model, feature_caches["validation"], torch)
        if feature_caches["validation"] is not None
        else None
    )
    validation_metrics = evaluate(
        model, validation_items, tokenizer, args.batch_size, torch,
        progress_every=1000, forward_fn=validation_forward,
    )
    for cache in feature_caches.values():
        if cache is not None:
            cache.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "experimental-model.safetensors"
    save_file(
        {
            name: value.detach().half().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(weights_path),
    )
    source_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    run_manifest = {
        "experiment": "frozen-modernbert-head-smoke",
        "deployable": False,
        "base_checkpoint_revision": args.base_revision,
        "base_checkpoint_path": str(model_dir),
        "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
        "laya_source_revision": laya_revision,
        "training_data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "split_counts": counts,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_per_label": args.max_train_per_label,
        "max_validation_per_label": args.max_validation_per_label,
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "encoder_frozen": True,
        "trainable_parameter_count": trainable_count,
        "wall_clock_seconds": round(time.monotonic() - started, 3),
        "feature_cache": (
            {
                "directory": str(args.feature_cache.resolve()),
                "verification": cache_verification,
            }
            if args.feature_cache
            else None
        ),
        "train_case_ids": [item["case_id"] for item in train_items],
        "validation_case_ids": [item["case_id"] for item in validation_items],
        "validation_metrics_smoke_only": validation_metrics,
        "test_evaluated": False,
        "calibration_refit": False,
        "source_split_policy": source_manifest["split_policy"],
        "weights_sha256": sha256_file(weights_path),
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Saved non-deployable smoke weights to %s" % weights_path)
    print("Validation smoke metrics: %s" % json.dumps(validation_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
