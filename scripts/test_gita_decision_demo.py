import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gita_decision_demo import (
    BM25Index,
    HybridIndex,
    Passage,
    SemanticIndex,
    build_request,
    evaluate_retrieval,
    load_passages,
    main,
    tokenize,
)
from evaluate_gita_retrieval import DEFAULT_GITA_REPO, load_cases
from evaluate_gita_retrieval import main as evaluate_main

SCRIPTS_DIR = Path(__file__).resolve().parent
HAVE_CORPUS = (DEFAULT_GITA_REPO / "data" / "translation.json").is_file()


def passage(chapter, verse, text):
    return Passage(
        chapter=str(chapter),
        verse=str(verse),
        author="Test Translator",
        text=text,
        tokens=tokenize(text),
    )


class GitaDecisionDemoTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            passage("2", "47", "You have a right to action but not to its fruits."),
            passage("6", "5", "Elevate yourself through the power of your own mind."),
            passage("12", "13", "One who is free from hatred and friendly to all beings."),
        ]
        self.index = BM25Index(self.passages)

    def test_retriever_prioritizes_matching_passage(self):
        ranked = self.index.search("action mind", limit=2)
        self.assertEqual(ranked[0][1].citation, "BG 2.47")
        self.assertEqual(len(ranked), 2)

    def test_retriever_normalizes_common_english_inflections(self):
        self.assertEqual(tokenize("acting attachments"), ("act", "attachment"))
        index = BM25Index([passage("2", "47", "Act without attachment.")])
        ranked = index.search("acting without attachments", limit=1)
        self.assertGreater(ranked[0][0], 0)

    def test_retriever_rejects_empty_query_and_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "searchable word"):
            self.index.search("the and to")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.index.search("action", limit=0)

    def test_zero_score_search_returns_no_evidence(self):
        self.assertEqual(self.index.search("postgresql transaction isolation"), [])
        with self.assertRaisesRegex(ValueError, "without retrieved evidence"):
            build_request(
                "Which database should I use?",
                self.index.search("postgresql transaction isolation"),
                "choice",
            )

    def test_retrieval_eval_separates_answerable_recall_and_unanswerable_hits(self):
        cases = [
            {
                "case_id": "action",
                "query": "action fruits",
                "answerable": True,
                "relevant_citations": ["BG 2.47", "BG 6.5"],
            },
            {
                "case_id": "no-match",
                "query": "postgresql transaction isolation",
                "answerable": False,
                "relevant_citations": [],
            },
        ]
        report = evaluate_retrieval(self.index, cases, top_k=1)
        self.assertEqual(report["answerable_citation_recall_at_k"], 0.5)
        self.assertEqual(report["answerable_case_recall_at_k"], 1.0)
        self.assertEqual(report["answerable_mrr_at_k"], 1.0)
        self.assertEqual(report["unanswerable_no_evidence_rate"], 1.0)

    def test_positive_results_exclude_zero_score_ties(self):
        ranked = self.index.search("mind", limit=10)
        self.assertTrue(ranked)
        self.assertTrue(all(score > 0 for score, _ in ranked))

    def test_unanswerable_eval_case_with_retrieval_hit_counts_as_miss(self):
        cases = [{
            "case_id": "misleading-overlap",
            "query": "action results",
            "answerable": False,
            "relevant_citations": [],
        }]
        report = evaluate_retrieval(self.index, cases, top_k=3)
        self.assertEqual(report["unanswerable_no_evidence_rate"], 0.0)

    def test_retrieval_eval_minimum_query_overlap_filter(self):
        cases = [
            {
                "case_id": "answerable",
                "query": "action fruits",
                "answerable": True,
                "relevant_citations": ["BG 2.47"],
            },
            {
                "case_id": "generic-overlap",
                "query": "action results",
                "answerable": False,
                "relevant_citations": [],
            },
        ]
        report = evaluate_retrieval(
            self.index, cases, top_k=3, min_query_overlap=2
        )
        self.assertEqual(report["minimum_query_overlap"], 2)
        self.assertEqual(report["answerable_citation_recall_at_k"], 1.0)
        self.assertEqual(report["unanswerable_no_evidence_rate"], 1.0)

    def test_retrieval_eval_minimum_score_filter(self):
        cases = [
            {
                "case_id": "answerable",
                "query": "mind",
                "answerable": True,
                "relevant_citations": ["BG 6.5"],
            },
            {
                "case_id": "unanswerable",
                "query": "minds",
                "answerable": False,
                "relevant_citations": [],
            },
        ]
        report = evaluate_retrieval(self.index, cases, top_k=3, min_score=100.0)
        self.assertEqual(report["minimum_score"], 100.0)
        self.assertEqual(report["answerable_citation_recall_at_k"], 0.0)
        self.assertEqual(report["unanswerable_no_evidence_rate"], 1.0)

    def test_holdout_set_is_balanced_and_disjoint_from_development_set(self):
        holdout = load_cases(SCRIPTS_DIR / "gita_retrieval_holdout.jsonl")
        development = load_cases(SCRIPTS_DIR / "gita_retrieval_eval.jsonl")
        self.assertEqual(len(holdout), 24)
        self.assertEqual(sum(case["answerable"] for case in holdout), 12)
        self.assertEqual(sum(not case["answerable"] for case in holdout), 12)
        self.assertFalse(
            {case["case_id"] for case in holdout}
            & {case["case_id"] for case in development}
        )
        self.assertFalse(
            {case["query"] for case in holdout}
            & {case["query"] for case in development}
        )

    def test_devanagari_tokens_keep_whole_words(self):
        self.assertEqual(
            tokenize("धर्मक्षेत्रे कुरुक्षेत्र कर्मफल"),
            ("धर्मक्षेत्रे", "कुरुक्षेत्र", "कर्मफल"),
        )
        self.assertEqual(tokenize("और फल की इच्छा नहीं"), ("फल", "इच्छा"))
        index = BM25Index([passage("2", "47", "कर्म करने में अधिकार है, फल में नहीं।")])
        ranked = index.search("कर्म और फल की इच्छा", limit=1)
        self.assertGreater(ranked[0][0], 0)

    def test_hindi_holdout_is_balanced_and_disjoint_from_other_sets(self):
        holdout = load_cases(SCRIPTS_DIR / "gita_retrieval_holdout_hi.jsonl")
        others = load_cases(SCRIPTS_DIR / "gita_retrieval_holdout.jsonl") + load_cases(
            SCRIPTS_DIR / "gita_retrieval_eval.jsonl"
        )
        self.assertEqual(len(holdout), 24)
        self.assertEqual(sum(case["answerable"] for case in holdout), 12)
        self.assertEqual(sum(not case["answerable"] for case in holdout), 12)
        self.assertTrue(all(case["case_id"].startswith("hi-") for case in holdout))
        self.assertEqual(
            len({case["case_id"] for case in holdout}), len(holdout)
        )
        self.assertEqual(len({case["query"] for case in holdout}), len(holdout))
        self.assertFalse(
            {case["case_id"] for case in holdout}
            & {case["case_id"] for case in others}
        )
        self.assertFalse(
            {case["query"] for case in holdout} & {case["query"] for case in others}
        )

    @unittest.skipUnless(HAVE_CORPUS, "local Gita clone is not available")
    def test_hindi_holdout_citations_exist_in_hindi_index(self):
        passages = load_passages(DEFAULT_GITA_REPO, "Swami Tejomayananda", "hindi")
        self.assertEqual(len(passages), 701)
        self.assertTrue(all(p.author == "Swami Tejomayananda" for p in passages))
        available = {p.citation for p in passages}
        for case in load_cases(SCRIPTS_DIR / "gita_retrieval_holdout_hi.jsonl"):
            unknown = set(case["relevant_citations"]) - available
            self.assertFalse(unknown, "%s cites %s" % (case["case_id"], sorted(unknown)))

    @unittest.skipUnless(HAVE_CORPUS, "local Gita clone is not available")
    def test_load_passages_rejects_unknown_language_and_author(self):
        with self.assertRaisesRegex(ValueError, "language must be english or hindi"):
            load_passages(DEFAULT_GITA_REPO, "Swami Sivananda", "sanskrit")
        with self.assertRaisesRegex(ValueError, "Hindi translation author 'Nobody'"):
            load_passages(DEFAULT_GITA_REPO, "Nobody", "hindi")

    def test_hindi_evaluation_requires_bm25(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                evaluate_main([
                    "--gita-repo", str(DEFAULT_GITA_REPO),
                    "--language", "hindi",
                    "--retriever", "semantic",
                ])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("English-only", stderr.getvalue())

    @unittest.skipUnless(HAVE_CORPUS, "local Gita clone is not available")
    def test_hindi_evaluation_report_records_language_and_author(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            evaluate_main([
                "--gita-repo", str(DEFAULT_GITA_REPO),
                "--language", "hindi",
                "--eval-set", str(SCRIPTS_DIR / "gita_retrieval_holdout_hi.jsonl"),
                "--top-k", "5",
            ])
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["language"], "hindi")
        self.assertEqual(report["author"], "Swami Tejomayananda")
        self.assertEqual(report["corpus"]["language"], "hindi")
        self.assertEqual(report["retriever"], "bm25")
        self.assertEqual(report["case_count"], 24)

    def test_semantic_index_ranks_cosine_similarities(self):
        class FakeTextEmbedding:
            def __init__(self, model_name):
                self.model_name = model_name

            def embed(self, texts):
                for text in texts:
                    lowered = text.lower()
                    if "mind" in lowered:
                        yield [0.0, 2.0]
                    elif "action" in lowered:
                        yield [3.0, 0.0]
                    else:
                        yield [1.0, 1.0]

        fake_fastembed = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": fake_fastembed}):
            index = SemanticIndex(self.passages, "test-model")
            ranked = index.search("mind", limit=3)
        self.assertEqual(index.model_name, "test-model")
        self.assertEqual(ranked[0][1].citation, "BG 6.5")
        self.assertAlmostEqual(ranked[0][0], 1.0)
        self.assertAlmostEqual(ranked[1][0], 0.70710678)

    def test_semantic_index_reports_missing_optional_dependency(self):
        with patch.dict(sys.modules, {"fastembed": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-gita-semantic"):
                SemanticIndex(self.passages)

    def test_hybrid_index_fuses_ranks_and_omits_zero_score_lexical_passages(self):
        class FakeSemanticIndex:
            def __init__(self, passages):
                self.passages = list(passages)

            def search(self, query, limit):
                return [
                    (0.9, self.passages[1]),
                    (0.8, self.passages[0]),
                    (0.7, self.passages[2]),
                ][:limit]

        semantic = FakeSemanticIndex(self.passages)
        index = HybridIndex(
            self.index,
            semantic,
            semantic_weight=0.5,
            reciprocal_rank_constant=1,
        )
        ranked = index.search("action", limit=3)
        self.assertEqual(ranked[0][1].citation, "BG 2.47")
        self.assertGreater(ranked[0][0], ranked[1][0])
        self.assertEqual(len(ranked), 3)

    def test_hybrid_index_rejects_invalid_configuration(self):
        class FakeSemanticIndex:
            def __init__(self, passages):
                self.passages = list(passages)

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            HybridIndex(self.index, FakeSemanticIndex(self.passages), 1.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            HybridIndex(
                self.index,
                FakeSemanticIndex(self.passages),
                reciprocal_rank_constant=0,
            )

    def test_hybrid_index_requires_identical_passage_sets(self):
        class FakeSemanticIndex:
            def __init__(self, passages):
                self.passages = list(passages)

        with self.assertRaisesRegex(ValueError, "same passages"):
            HybridIndex(
                self.index,
                FakeSemanticIndex(self.passages[:-1]),
            )

    def test_semantic_index_rejects_empty_embedding_results(self):
        class EmptyTextEmbedding:
            def __init__(self, model_name):
                pass

            def embed(self, texts):
                return iter(())

        fake_fastembed = types.SimpleNamespace(TextEmbedding=EmptyTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": fake_fastembed}):
            with self.assertRaisesRegex(RuntimeError, "no passage embeddings"):
                SemanticIndex(self.passages)

    def test_main_skips_oio_when_retrieval_has_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            data = repo / "data"
            data.mkdir()
            (data / "verse.json").write_text(json.dumps([{
                "id": "1",
                "chapter_number": "1",
                "verse_number": "1",
            }]), encoding="utf-8")
            (data / "translation.json").write_text(json.dumps([{
                "lang": "english",
                "authorName": "Test Translator",
                "verse_id": "1",
                "description": "A teaching about action.",
            }]), encoding="utf-8")
            stdout = io.StringIO()
            with patch("gita_decision_demo.call_oio") as call_oio:
                with contextlib.redirect_stdout(stdout):
                    status = main([
                        "postgresql transaction isolation",
                        "--gita-repo", str(repo),
                        "--author", "Test Translator",
                    ])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertTrue(payload["no_evidence"])
            self.assertIsNone(payload["decision"])
            call_oio.assert_not_called()

    def test_choice_payload_retains_citations_and_authorship(self):
        ranked = self.index.search("action mind", limit=2)
        request, excerpts = build_request("How should I act?", ranked, "choice")
        question = request["questions"]["gita_decision"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(len(question["criteria"]), 2)
        self.assertTrue(all("Test Translator" in label for label in question["criteria"]))
        self.assertTrue(all(item["author"] == "Test Translator" for item in excerpts))

    def test_choice_requires_two_candidates_but_score_accepts_one(self):
        ranked = self.index.search("action", limit=1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_request("How should I act?", ranked, "choice")
        request, excerpts = build_request("How should I act?", ranked, "score")
        self.assertEqual(request["questions"]["gita_decision"]["type"], "score")
        self.assertEqual(len(excerpts), 1)

    def test_main_skips_oio_when_choice_has_only_one_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            data = repo / "data"
            data.mkdir()
            (data / "verse.json").write_text(json.dumps([{
                "id": "1",
                "chapter_number": "2",
                "verse_number": "47",
            }]), encoding="utf-8")
            (data / "translation.json").write_text(json.dumps([{
                "lang": "english",
                "authorName": "Test Translator",
                "verse_id": "1",
                "description": "A teaching about action.",
            }]), encoding="utf-8")
            stdout = io.StringIO()
            with patch("gita_decision_demo.call_oio") as call_oio:
                with contextlib.redirect_stdout(stdout):
                    status = main([
                        "action",
                        "--gita-repo", str(repo),
                        "--author", "Test Translator",
                    ])
            payload = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertIsNone(payload["decision"])
            self.assertFalse(payload["no_evidence"])
            self.assertTrue(payload["insufficient_candidates"])
            self.assertEqual(len(payload["retrieved_passages"]), 1)
            call_oio.assert_not_called()

    def test_main_allows_single_candidate_for_score_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            data = repo / "data"
            data.mkdir()
            (data / "verse.json").write_text(json.dumps([{
                "id": "1",
                "chapter_number": "2",
                "verse_number": "47",
            }]), encoding="utf-8")
            (data / "translation.json").write_text(json.dumps([{
                "lang": "english",
                "authorName": "Test Translator",
                "verse_id": "1",
                "description": "A teaching about action.",
            }]), encoding="utf-8")
            stdout = io.StringIO()
            with patch("gita_decision_demo.call_oio", return_value={"score": 2}) as call_oio:
                with contextlib.redirect_stdout(stdout):
                    status = main([
                        "action",
                        "--gita-repo", str(repo),
                        "--author", "Test Translator",
                        "--mode", "score",
                        "--top-k", "1",
                    ])
            payload = json.loads(stdout.getvalue())
            self.assertIsNone(status)
            self.assertEqual(payload["decision"], {"score": 2})
            self.assertEqual(len(payload["retrieved_passages"]), 1)
            call_oio.assert_called_once()

    def test_score_and_noul_payloads_have_valid_shapes(self):
        ranked = self.index.search("mind", limit=2)
        score_request, _ = build_request("How should I train the mind?", ranked, "score")
        noul_request, _ = build_request("Is this supported?", ranked, "noul")
        score = score_request["questions"]["gita_decision"]
        noul = noul_request["questions"]["gita_decision"]
        self.assertEqual(score["type"], "score")
        self.assertEqual(len(score["criteria"]), 4)
        self.assertEqual(noul["type"], "noul")
        self.assertEqual(set(noul["criteria"]), {"false", "true"})
        self.assertIn("Retrieved source passages", noul_request["state"])


if __name__ == "__main__":
    unittest.main()
