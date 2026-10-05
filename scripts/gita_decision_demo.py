#!/usr/bin/env python3
"""Retrieve Bhagavad Gita verses, then ask a running OIO server to judge them."""

import argparse
import collections
import hashlib
import html
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOP_WORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "here", "how", "i", "if", "in", "into", "is", "it",
    "its", "may", "me", "more", "most", "my", "of", "on", "or", "our", "she",
    "say", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "was", "we", "were", "what",
    "text", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your",
}
DEFAULT_GITA_REPO = Path(__file__).resolve().parents[2] / "gita"


def normalize_token(token):
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass(frozen=True)
class Passage:
    chapter: str
    verse: str
    author: str
    text: str
    tokens: tuple

    @property
    def citation(self):
        return "BG %s.%s" % (self.chapter, self.verse)

    @property
    def label(self):
        return "%s (%s)" % (self.citation, self.author)


def clean_text(value):
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def tokenize(value):
    return tuple(
        normalized
        for token in TOKEN_RE.findall(clean_text(value).casefold())
        if token not in STOP_WORDS
        for normalized in (normalize_token(token),)
        if normalized
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_passages(gita_repo, author):
    data_dir = Path(gita_repo) / "data"
    verses_path = data_dir / "verse.json"
    translations_path = data_dir / "translation.json"
    for path in (verses_path, translations_path):
        if not path.is_file():
            raise ValueError("missing Gita corpus file: %s" % path)
    try:
        verses = json.loads(verses_path.read_text(encoding="utf-8"))
        translations = json.loads(translations_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not load Gita verse/translation JSON: %s" % exc) from exc
    if not isinstance(verses, list) or not isinstance(translations, list):
        raise ValueError("Gita verse.json and translation.json must contain arrays")

    verse_by_id = {}
    for verse in verses:
        if not isinstance(verse, dict) or "id" not in verse:
            raise ValueError("Gita verse record is missing its id")
        verse_by_id[str(verse["id"])] = verse

    passages = []
    authors = set()
    seen_verses = set()
    for translation in translations:
        if not isinstance(translation, dict) or translation.get("lang", "").casefold() != "english":
            continue
        translation_author = clean_text(translation.get("authorName"))
        if not translation_author:
            continue
        authors.add(translation_author)
        if author and translation_author.casefold() != author.casefold():
            continue
        verse = verse_by_id.get(str(translation.get("verse_id")))
        text = clean_text(translation.get("description"))
        if verse is None or not text:
            continue
        key = (str(verse.get("id")), translation_author)
        if key in seen_verses:
            continue
        seen_verses.add(key)
        tokens = tokenize(text)
        if not tokens:
            continue
        passages.append(Passage(
            chapter=str(verse.get("chapter_number", verse.get("chapter_id", ""))),
            verse=str(verse.get("verse_number", "")),
            author=translation_author,
            text=text,
            tokens=tokens,
        ))
    if author and not any(name.casefold() == author.casefold() for name in authors):
        raise ValueError(
            "English translation author %r not found; available: %s"
            % (author, ", ".join(sorted(authors)))
        )
    if not passages:
        raise ValueError("no indexed English translations found in %s" % data_dir)
    return passages


class BM25Index:
    def __init__(self, passages, k1=1.5, b=0.75):
        if not passages:
            raise ValueError("cannot index an empty passage list")
        self.passages = list(passages)
        self.k1 = k1
        self.b = b
        self.lengths = [len(passage.tokens) for passage in self.passages]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.frequencies = []
        document_frequency = collections.Counter()
        for passage in self.passages:
            frequencies = collections.Counter(passage.tokens)
            self.frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())
        count = len(self.passages)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query, limit=5):
        if limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable word")
        query_terms = set(query_tokens)
        results = []
        for index, passage in enumerate(self.passages):
            frequencies = self.frequencies[index]
            score = 0.0
            for token in query_terms:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                score += self.idf[token] * frequency * (self.k1 + 1) / denominator
            if score > 0:
                results.append((score, passage))
        results.sort(key=lambda item: (-item[0], item[1].chapter, item[1].verse, item[1].author))
        if not results or results[0][0] <= 0:
            return []
        return results[:limit]


def git_revision(repo):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_request(query, results, mode):
    if mode not in {"choice", "score", "noul"}:
        raise ValueError("mode must be choice, score, or noul")
    if not results:
        raise ValueError("cannot ask OIO to decide without retrieved evidence")
    if mode == "choice" and len(results) < 2:
        raise ValueError("choice mode requires at least two retrieved candidates")
    excerpts = format_excerpts(results)
    evidence = "\n".join(
        "[%s, %s] %s" % (
            excerpt["citation"], excerpt["author"], excerpt["text"],
        )
        for excerpt in excerpts
    )
    state = "Question: %s\nRetrieved source passages:\n%s" % (query, evidence)
    if mode == "choice":
        criteria = {
            excerpt["citation"] + " — " + excerpt["author"]: excerpt["text"]
            for excerpt in excerpts
        }
        question = {
            "type": "choice",
            "instructions": (
                "Choose the cited Bhagavad Gita passage that is most relevant to the "
                "question. Judge only the provided passages; do not infer unsupported facts."
            ),
            "criteria": criteria,
        }
    elif mode == "score":
        question = {
            "type": "score",
            "instructions": (
                "How strongly do the retrieved passages directly support an answer to "
                "the question? Evaluate the supplied text only."
            ),
            "criteria": [
                "The passages do not support an answer.",
                "The passages are loosely related but give little direct support.",
                "The passages provide relevant support but leave important gaps.",
                "The passages directly and clearly support an answer.",
            ],
        }
    else:
        question = {
            "type": "noul",
            "instructions": (
                "Can the question be answered from the retrieved passages without "
                "adding unsupported information?"
            ),
            "criteria": {
                "false": "No; the passages do not provide enough direct evidence.",
                "true": "Yes; the passages provide sufficient direct evidence.",
            },
        }
    return {
        "state": state,
        "questions": {"gita_decision": question},
    }, excerpts


def format_excerpts(results):
    return [
        {
            "citation": passage.citation,
            "author": passage.author,
            "text": passage.text[:360].rsplit(" ", 1)[0] + (
                "..." if len(passage.text) > 360 else ""
            ),
            "retrieval_score": round(score, 4),
        }
        for score, passage in results
    ]


def call_oio(base_url, request, timeout):
    url = base_url.rstrip("/") + "/v1/systemone"
    body = json.dumps(request, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("OIO returned HTTP %d: %s" % (exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("could not reach OIO at %s: %s" % (url, exc.reason)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("OIO returned invalid JSON: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OIO response must be a JSON object")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("OIO response is missing its answers object")
    answer = answers.get("gita_decision")
    if not isinstance(answer, dict):
        raise RuntimeError("OIO response is missing answers.gita_decision")
    return answer


def corpus_metadata(gita_repo, author):
    data_dir = Path(gita_repo) / "data"
    return {
        "repository": "https://github.com/itsalfredashu/gita",
        "revision": git_revision(gita_repo),
        "translation_author": author,
        "verse_json_sha256": sha256_file(data_dir / "verse.json"),
        "translation_json_sha256": sha256_file(data_dir / "translation.json"),
    }


def evaluate_retrieval(index, cases, top_k, min_query_overlap=1):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if min_query_overlap < 1:
        raise ValueError("min_query_overlap must be positive")
    seen_ids = set()
    seen_queries = set()
    answerable_total = 0
    relevant_found = 0
    relevant_total = 0
    reciprocal_rank_sum = 0.0
    unanswerable_total = 0
    unanswerable_no_evidence = 0
    evaluated = []
    available_citations = {passage.citation for passage in index.passages}

    for row_number, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError("evaluation row %d must be an object" % row_number)
        case_id = case.get("case_id")
        query = case.get("query")
        expected = case.get("relevant_citations")
        answerable = case.get("answerable")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("evaluation row %d needs a non-empty case_id" % row_number)
        if case_id in seen_ids:
            raise ValueError("duplicate evaluation case_id %r" % case_id)
        seen_ids.add(case_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("evaluation row %s needs a non-empty query" % case_id)
        if query in seen_queries:
            raise ValueError("duplicate evaluation query in case %r" % case_id)
        seen_queries.add(query)
        if not isinstance(answerable, bool):
            raise ValueError("evaluation row %s answerable must be boolean" % case_id)
        if not isinstance(expected, list) or any(
            not isinstance(citation, str) or not citation for citation in expected
        ):
            raise ValueError("evaluation row %s relevant_citations must be a list of citations" % case_id)
        if len(set(expected)) != len(expected):
            raise ValueError("evaluation row %s repeats a relevant citation" % case_id)
        if answerable != bool(expected):
            raise ValueError(
                "evaluation row %s answerable must agree with relevant_citations" % case_id
            )
        unknown = set(expected) - available_citations
        if unknown:
            raise ValueError(
                "evaluation row %s cites verses missing from this author index: %s"
                % (case_id, ", ".join(sorted(unknown)))
            )

        if min_query_overlap == 1:
            retrieved = index.search(query, top_k)
        else:
            query_terms = set(tokenize(query))
            ranked = index.search(query, len(index.passages))
            retrieved = [
                (score, passage)
                for score, passage in ranked
                if len(query_terms.intersection(passage.tokens)) >= min_query_overlap
            ][:top_k]
        citations = [passage.citation for _, passage in retrieved]
        first_relevant_rank = next(
            (rank for rank, citation in enumerate(citations, 1) if citation in expected),
            None,
        )
        found = sum(citation in set(citations) for citation in expected)
        if answerable:
            answerable_total += 1
            relevant_total += len(expected)
            relevant_found += found
            if first_relevant_rank is not None:
                reciprocal_rank_sum += 1.0 / first_relevant_rank
        else:
            unanswerable_total += 1
            if not retrieved:
                unanswerable_no_evidence += 1

        evaluated.append({
            "case_id": case_id,
            "query": query,
            "answerable": answerable,
            "relevant_citations": expected,
            "retrieved_citations": citations,
            "retrieved_scores": [round(score, 4) for score, _ in retrieved],
            "relevant_retrieved": found,
            "relevant_total": len(expected),
            "first_relevant_rank": first_relevant_rank,
            "no_evidence": not retrieved,
        })

    if not evaluated:
        raise ValueError("evaluation set contains no cases")
    return {
        "top_k": top_k,
        "minimum_query_overlap": min_query_overlap,
        "case_count": len(evaluated),
        "answerable_case_count": answerable_total,
        "answerable_citation_recall_at_k": (
            relevant_found / relevant_total if relevant_total else None
        ),
        "answerable_case_recall_at_k": (
            sum(case["first_relevant_rank"] is not None for case in evaluated if case["answerable"])
            / answerable_total
            if answerable_total
            else None
        ),
        "answerable_mrr_at_k": (
            reciprocal_rank_sum / answerable_total if answerable_total else None
        ),
        "unanswerable_case_count": unanswerable_total,
        "unanswerable_no_evidence_rate": (
            unanswerable_no_evidence / unanswerable_total if unanswerable_total else None
        ),
        "cases": evaluated,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or decision to ground in the Gita")
    parser.add_argument("--gita-repo", type=Path, default=DEFAULT_GITA_REPO)
    parser.add_argument("--author", default="Swami Sivananda", help="English translation author")
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

    try:
        passages = load_passages(args.gita_repo, args.author)
        results = BM25Index(passages).search(args.query, args.top_k)
        corpus = corpus_metadata(args.gita_repo, args.author)
        if not results:
            print(json.dumps({
                "query": args.query,
                "mode": args.mode,
                "decision": None,
                "retrieved_passages": [],
                "no_evidence": True,
                "corpus": corpus,
                "caveat": (
                    "No indexed English translation matched the query terms; OIO was not "
                    "called and no decision was made."
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
                "corpus": corpus,
                "caveat": (
                    "Choice mode requires at least two positive-scoring passages; "
                    "OIO was not called and no decision was made."
                ),
            }, ensure_ascii=False, indent=2))
            return 0
        request, excerpts = build_request(args.query, results, args.mode)
        answer = call_oio(args.oio_url, request, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    data_dir = args.gita_repo / "data"
    output = {
        "query": args.query,
        "mode": args.mode,
        "decision": answer,
        "retrieved_passages": excerpts,
        "corpus": corpus,
        "caveat": (
            "Decision is based only on the retrieved translations. It is not a "
            "religious ruling or a substitute for consulting the source in context."
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
