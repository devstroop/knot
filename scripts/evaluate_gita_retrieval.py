#!/usr/bin/env python3
"""Evaluate local Gita retrieval separately from OIO decisions."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gita_decision_demo import (
        BM25Index,
        SemanticIndex,
        evaluate_retrieval,
        git_revision,
        load_passages,
        sha256_file,
    )
else:
    from .gita_decision_demo import (
        BM25Index,
        SemanticIndex,
        evaluate_retrieval,
        git_revision,
        load_passages,
        sha256_file,
    )

DEFAULT_GITA_REPO = Path(__file__).resolve().parents[2] / "gita"
DEFAULT_EVAL_SET = Path(__file__).with_name("gita_retrieval_eval.jsonl")


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d is invalid JSON: %s" % (path, line_number, exc)) from exc
    if not cases:
        raise ValueError("%s contains no evaluation cases" % path)
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gita-repo", type=Path, default=DEFAULT_GITA_REPO)
    parser.add_argument("--author", default="Swami Sivananda")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=("bm25", "semantic"), default="bm25")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="FastEmbed model for semantic retrieval",
    )
    parser.add_argument(
        "--min-query-overlap",
        type=int,
        default=1,
        help="BM25 only: require this many distinct query terms in a passage (default: 1)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Only evaluate results at or above this retriever-specific score",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > 100:
        parser.error("--top-k must be between 1 and 100")
    if args.min_query_overlap < 1:
        parser.error("--min-query-overlap must be positive")
    if args.retriever == "semantic" and args.min_query_overlap != 1:
        parser.error("--min-query-overlap is available only with --retriever bm25")
    try:
        passages = load_passages(args.gita_repo, args.author)
        cases = load_cases(args.eval_set)
        index = (
            BM25Index(passages)
            if args.retriever == "bm25"
            else SemanticIndex(passages, args.model)
        )
        report = evaluate_retrieval(
            index,
            cases,
            args.top_k,
            args.min_query_overlap,
            args.min_score,
        )
        data_dir = args.gita_repo / "data"
        report["retriever"] = args.retriever
        if args.retriever == "semantic":
            report["embedding_model"] = args.model
        report["corpus"] = {
            "repository": "https://github.com/itsalfredashu/gita",
            "revision": git_revision(args.gita_repo),
            "translation_author": args.author,
            "verse_json_sha256": sha256_file(data_dir / "verse.json"),
            "translation_json_sha256": sha256_file(data_dir / "translation.json"),
        }
        report["evaluation_set"] = {
            "path": str(args.eval_set),
            "sha256": sha256_file(args.eval_set),
        }
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
