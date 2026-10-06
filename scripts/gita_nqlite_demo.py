#!/usr/bin/env python3
"""Ask OIO to judge Gita passages retrieved from nqlite.

Same typed-decision contract as `gita_decision_demo.py`, but retrieval comes
from a long-lived `nql-server --stdio` subprocess (hybrid `::bm25` plus
MiniLM cosine, fused with RRF) instead of the hand-rolled Python indexes.
Ranking uses the full query text, exactly as the comparison spike measured;
the abstention decision uses a companion BM25 query over the query's content
terms only, which reproduces oio's own stopword-aware no-evidence semantics.
Retrieval plumbing (server ownership, ingest, row parsing) is shared with
`gita_nqlite_spike.py`.
"""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gita_decision_demo import (
        TOKEN_RE,
        STOP_WORDS,
        build_request,
        call_oio,
        clean_text,
        corpus_metadata,
        format_excerpts,
        load_passages,
    )
    from gita_nqlite_spike import (
        DEFAULT_GITA_REPO,
        DEFAULT_SERVER,
        MODEL_NAME,
        NqliteServer,
        embed_texts,
        escape_nql_string,
        ingest,
        parse_rows,
    )
else:
    from .gita_decision_demo import (
        TOKEN_RE,
        STOP_WORDS,
        build_request,
        call_oio,
        clean_text,
        corpus_metadata,
        format_excerpts,
        load_passages,
    )
    from .gita_nqlite_spike import (
        DEFAULT_GITA_REPO,
        DEFAULT_SERVER,
        MODEL_NAME,
        NqliteServer,
        embed_texts,
        escape_nql_string,
        ingest,
        parse_rows,
    )


def content_terms(query):
    """Query tokens with English stopwords removed (unstemmed, for nqlite)."""
    return [
        token
        for token in TOKEN_RE.findall(clean_text(query).casefold())
        if token not in STOP_WORDS
    ]


def retrieve(server, model, passages, query, top_k):
    """Hybrid nqlite retrieval on the full query text; returns (score, passage)."""
    (vector,) = embed_texts(model, [query])
    program = (
        'SELECT * FROM verse WHERE ::bm25(text, "%s") '
        "AND vector::similarity(embedding, [%s]) AND k = %d"
        % (
            escape_nql_string(query),
            ", ".join("%.6f" % value for value in vector),
            top_k,
        )
    )
    selects, ok, error = server.execute(program)
    if not ok:
        raise RuntimeError("nqlite hybrid query failed: %s" % error)
    return [
        (score, passages[record_number - 1])
        for record_number, score in parse_rows(selects)[:top_k]
    ]


def check_evidence(server, query, top_k):
    """Companion BM25 query over content terms.

    Returns (has_evidence, max_bm25_score). A query with no content terms,
    or with no positively scoring row, carries no retrieval evidence and OIO
    is not called.
    """
    terms = content_terms(query)
    if not terms:
        return False, 0.0
    program = 'SELECT * FROM verse WHERE ::bm25(text, "%s") AND k = %d' % (
        escape_nql_string(" ".join(terms)),
        top_k,
    )
    selects, ok, error = server.execute(program)
    if not ok:
        raise RuntimeError("nqlite evidence query failed: %s" % error)
    scores = [score for _, score in parse_rows(selects)]
    best = max(scores) if scores else 0.0
    return best > 0.0, best


def retriever_report(best_bm25):
    return {
        "engine": "nqlite",
        "mode": "hybrid-bm25-minilm-rrf",
        "evidence_rule": "bm25-positive-on-content-terms",
        "bm25_max_score": round(best_bm25, 4),
        "embedding_model": MODEL_NAME,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or decision to ground in the Gita")
    parser.add_argument("--gita-repo", type=Path, default=DEFAULT_GITA_REPO)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--author", default="Swami Sivananda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("choice", "score", "noul"), default="choice")
    parser.add_argument("--oio-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > 20:
        parser.error("--top-k must be between 1 and 20")
    if args.mode == "choice" and args.top_k < 2:
        parser.error("choice mode requires --top-k of at least 2")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    from fastembed import TextEmbedding

    try:
        passages = load_passages(args.gita_repo, args.author)
        corpus = corpus_metadata(args.gita_repo, args.author)
        if not content_terms(args.query):
            print(json.dumps({
                "query": args.query,
                "mode": args.mode,
                "decision": None,
                "retrieved_passages": [],
                "no_evidence": True,
                "retriever": retriever_report(0.0),
                "corpus": corpus,
                "caveat": (
                    "The query has no content terms after stopword removal; "
                    "OIO was not called and no decision was made."
                ),
            }, ensure_ascii=False, indent=2))
            return 0
        model = TextEmbedding(MODEL_NAME)
        server = NqliteServer(args.server)
        try:
            ingest(server, model, passages)
            results = retrieve(server, model, passages, args.query, args.top_k)
            has_evidence, best_bm25 = check_evidence(server, args.query, args.top_k)
        finally:
            server.close()
        retriever = retriever_report(best_bm25)
        if not has_evidence:
            print(json.dumps({
                "query": args.query,
                "mode": args.mode,
                "decision": None,
                "retrieved_passages": [],
                "no_evidence": True,
                "retriever": retriever,
                "corpus": corpus,
                "caveat": (
                    "No indexed English translation matched the query's content "
                    "terms; OIO was not called and no decision was made."
                ),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.mode == "choice" and len(results) < 2:
            print(json.dumps({
                "query": args.query,
                "mode": args.mode,
                "decision": None,
                "retrieved_passages": format_excerpts(results),
                "no_evidence": False,
                "insufficient_candidates": True,
                "retriever": retriever,
                "corpus": corpus,
                "caveat": (
                    "Choice mode requires at least two retrieved candidates; "
                    "OIO was not called and no decision was made."
                ),
            }, ensure_ascii=False, indent=2))
            return 0
        request, excerpts = build_request(args.query, results, args.mode)
        answer = call_oio(args.oio_url, request, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    output = {
        "query": args.query,
        "mode": args.mode,
        "decision": answer,
        "retrieved_passages": excerpts,
        "retriever": retriever,
        "corpus": corpus,
        "caveat": (
            "Decision is based only on the retrieved translations. It is not a "
            "religious ruling or a substitute for consulting the source in context."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
