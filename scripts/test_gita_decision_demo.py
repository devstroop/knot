import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gita_decision_demo import (
    BM25Index,
    Passage,
    build_request,
    evaluate_retrieval,
    main,
    tokenize,
)


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
