#!/usr/bin/env python3
"""Build the frozen-encoder feature cache for prepared splits.

Runs the (frozen) encoder once per case and stores its hidden states so
training and evaluation can reuse them instead of paying the encoder cost on
every step. The cache records the base weights, Laya revision, and data
manifest hashes, so a stale cache is rejected rather than silently used.
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
    from training.eval_metrics import collate
    from training.feature_cache import (
        CacheError,
        CacheWriter,
        DTYPES,
        FeatureCache,
        index_path_for,
        verify_cached_forward,
    )
    from training.train_frozen_head import PINNED_LAYA_REVISION, load_model, source_revision
else:
    from .data_contract import (
        DataError,
        SPLITS,
        read_split,
        sha256_file,
        validate_directory,
    )
    from .eval_metrics import collate
    from .feature_cache import (
        CacheError,
        CacheWriter,
        DTYPES,
        FeatureCache,
        index_path_for,
        verify_cached_forward,
    )
    from .train_frozen_head import PINNED_LAYA_REVISION, load_model, source_revision

DEFAULT_SPLITS = ("train", "validation")


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="Local checkpoint; no network access is used")
    parser.add_argument("--laya-source", required=True, help="Laya source checkout at the pinned revision")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split", action="append", choices=SPLITS,
        help="Split to cache (repeatable); default: train and validation",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--verify-cases", type=int, default=64,
        help="Compare cached logits against the live forward pass (0 disables)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Cache only the first N cases per split; for smoke checks, not real runs",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must not be negative")
    if args.verify_cases < 0:
        parser.error("--verify-cases must not be negative")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    splits = tuple(dict.fromkeys(args.split or DEFAULT_SPLITS))

    data_dir = Path(args.data_dir)
    laya_source = Path(args.laya_source).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = args.output_dir.resolve()
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
    existing = [
        str(path) for split in splits
        for path in (output_dir / ("%s.bin" % split), index_path_for(output_dir / ("%s.bin" % split)))
        if path.exists()
    ]
    if existing and not args.force:
        parser.error("cache files already exist (%s); pass --force to rebuild" % ", ".join(existing))

    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch, model, tokenizer, cfg, encode = load_model(model_dir, laya_source)
    provenance = {
        "base_model_sha256": sha256_file(model_dir / "model.safetensors"),
        "laya_source_revision": laya_revision,
        "training_data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
    }
    hidden_size = model.encoder.config.hidden_size
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    built = {}
    for split in splits:
        rows = read_split(data_dir / ("%s.jsonl" % split))
        if args.max_cases is not None and args.max_cases < len(rows):
            rows = rows[:args.max_cases]
        items = [encode(row) for row in rows]
        bin_path = output_dir / ("%s.bin" % split)
        writer = CacheWriter(
            bin_path, split, args.dtype, hidden_size, provenance
        )
        next_report = args.progress_every
        try:
            with torch.no_grad():
                for start in range(0, len(items), args.batch_size):
                    chunk = items[start:start + args.batch_size]
                    ids, attention, _, _, _, _ = collate(
                        chunk, tokenizer.pad_token_id, torch
                    )
                    hidden = model.encoder(
                        input_ids=ids, attention_mask=attention
                    ).last_hidden_state.detach().cpu().numpy()
                    for index, item in enumerate(chunk):
                        length = len(item["ids"])
                        writer.append(
                            [item["case_id"]],
                            hidden[index:index + 1, :length, :],
                        )
                    completed = start + len(chunk)
                    if args.progress_every and completed >= next_report:
                        print(
                            "%s: cached %d/%d cases" % (split, completed, len(items)),
                            flush=True,
                        )
                        next_report = ((completed // args.progress_every) + 1) * args.progress_every
            index = writer.close()
        except BaseException:
            writer.abort()
            raise
        if args.verify_cases:
            cache = FeatureCache(
                bin_path, split=split, expected_provenance=provenance
            )
            verification = verify_cached_forward(
                model, items, cache, tokenizer, args.batch_size, torch,
                limit=min(args.verify_cases, len(items)),
            )
            cache.close()
            print("%s: verified %s" % (split, json.dumps(verification)), flush=True)
        built[split] = {
            "cases": index["case_count"],
            "bytes": bin_path.stat().st_size,
            "index": str(index_path_for(bin_path)),
        }

    print(json.dumps({
        "splits": built,
        "dtype": args.dtype,
        "hidden_size": hidden_size,
        "provenance": provenance,
        "split_counts": counts,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "wall_clock_seconds": round(time.monotonic() - started, 3),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
