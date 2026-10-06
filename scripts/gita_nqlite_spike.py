#!/usr/bin/env python3
"""Spike: drive nqlite hybrid retrieval over the Gita corpus and score it.

Owns a single `nql-server --stdio` subprocess: ingests the 701-verse English
translation (with local MiniLM embeddings), then runs the knot English
retrieval eval sets through nqlite's hybrid query and reports the same
Recall/MRR metrics knot uses. Answerable metrics go through knot's
`evaluate_retrieval` via a small index shim; unanswerable abstention is
measured separately with a BM25-positive companion query, because nqlite
always returns its top-k rows (fused RRF scores are never zero).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gita_decision_demo import evaluate_retrieval, load_passages
    from evaluate_gita_retrieval import load_cases
else:
    from .gita_decision_demo import evaluate_retrieval, load_passages
    from .evaluate_gita_retrieval import load_cases

DEFAULT_GITA_REPO = Path(__file__).resolve().parents[2] / "gita"
DEFAULT_SERVER = Path(__file__).resolve().parents[2] / "nqlite" / "target" / "debug" / "nql-server"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

ROW_RE = re.compile(r"verse:(\d+) score=([0-9]+\.[0-9]+)")
# NOTE: the pattern scans whole SELECT lines including field blobs, so a verse
# text containing a literal "verse:<n> score=<s>" fragment would parse as a
# spurious row. The Sivananda translations contain no such fragment; a
# production client should parse the row structure instead of regexing.


def escape_nql_string(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


class NqliteServer:
    """A single long-lived `nql-server --stdio` subprocess."""

    def __init__(self, server_path):
        try:
            self.process = subprocess.Popen(
                [str(server_path), "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise SystemExit("could not start nql-server %s: %s" % (server_path, exc))

    def execute(self, program):
        """Send one nql program; return (select_lines, ok, error)."""
        self.process.stdin.write(program + "\n")
        self.process.stdin.flush()
        selects = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("nql-server closed stdout")
            line = line.rstrip("\n")
            if line == "OK":
                return selects, True, None
            if line.startswith("ERR"):
                return selects, False, line
            selects.append(line)

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.wait(timeout=30)


def parse_rows(select_lines):
    """Parse `verse:<n> score=<s>` pairs from SELECT output lines."""
    rows = []
    for line in select_lines:
        for match in ROW_RE.finditer(line):
            rows.append((int(match.group(1)), float(match.group(2))))
    return rows


def embed_texts(model, texts):
    return [list(map(float, vector)) for vector in model.embed(texts)]


def ingest(server, model, passages):
    selects, ok, error = server.execute("CREATE TABLE verse VECTOR<f32, 384>")
    if not ok:
        raise RuntimeError("CREATE TABLE failed: %s" % error)
    texts = [passage.text for passage in passages]
    vectors = embed_texts(model, texts)
    for index, (passage, vector) in enumerate(zip(passages, vectors), 1):
        statement = (
            'INSERT INTO verse:%d { "citation": "%s", "author": "%s", "text": "%s" } EMBED [%s]'
            % (
                index,
                escape_nql_string(passage.citation),
                escape_nql_string(passage.author),
                escape_nql_string(passage.text),
                ", ".join("%.6f" % value for value in vector),
            )
        )
        _, ok, error = server.execute(statement)
        if not ok:
            raise RuntimeError("INSERT verse:%d failed: %s" % (index, error))
    return len(passages)


class NqliteHybridIndex:
    """knot `evaluate_retrieval` shim over nqlite hybrid queries."""

    def __init__(self, server, model, passages):
        self.server = server
        self.model = model
        self.passages = list(passages)

    def search(self, query, limit):
        (vector,) = embed_texts(self.model, [query])
        program = (
            'SELECT * FROM verse WHERE ::bm25(text, "%s") '
            "AND vector::similarity(embedding, [%s]) AND k = %d"
            % (
                escape_nql_string(query),
                ", ".join("%.6f" % value for value in vector),
                limit,
            )
        )
        selects, ok, error = self.server.execute(program)
        if not ok:
            raise RuntimeError("hybrid query failed: %s" % error)
        ranked = []
        for record_number, fused_score in parse_rows(selects)[:limit]:
            ranked.append((fused_score, self.passages[record_number - 1]))
        return ranked


def bm25_max_score(server, query, limit):
    program = 'SELECT * FROM verse WHERE ::bm25(text, "%s") AND k = %d' % (
        escape_nql_string(query),
        limit,
    )
    selects, ok, error = server.execute(program)
    if not ok:
        raise RuntimeError("bm25 query failed: %s" % error)
    scores = [score for _, score in parse_rows(selects)]
    return max(scores) if scores else 0.0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gita-repo", type=Path, default=DEFAULT_GITA_REPO)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--author", default="Swami Sivananda")
    parser.add_argument("--eval-set", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    scripts_dir = Path(__file__).resolve().parent
    eval_sets = args.eval_set or [
        scripts_dir / "gita_retrieval_eval.jsonl",
        scripts_dir / "gita_retrieval_holdout.jsonl",
    ]

    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL_NAME)
    passages = load_passages(args.gita_repo, args.author)
    server = NqliteServer(args.server)
    try:
        ingested = ingest(server, model, passages)
        index = NqliteHybridIndex(server, model, passages)
        reports = {}
        for eval_set in eval_sets:
            cases = load_cases(eval_set)
            report = evaluate_retrieval(index, cases, args.top_k)
            no_evidence = 0
            unanswerable = 0
            for case in cases:
                if not case["answerable"]:
                    unanswerable += 1
                    if bm25_max_score(server, case["query"], args.top_k) <= 0.0:
                        no_evidence += 1
            report["nqlite_bm25_positive_evidence_rate"] = (
                1.0 - no_evidence / unanswerable if unanswerable else None
            )
            report["nqlite_unanswerable_no_evidence_rate"] = (
                no_evidence / unanswerable if unanswerable else None
            )
            reports[str(eval_set)] = report
    finally:
        server.close()

    print(json.dumps({
        "engine": "nqlite",
        "retriever": "hybrid-bm25-minilm-rrf",
        "embedding_model": MODEL_NAME,
        "ingested_passages": ingested,
        "top_k": args.top_k,
        "eval_sets": reports,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
