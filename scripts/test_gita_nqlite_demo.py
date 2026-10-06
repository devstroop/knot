import contextlib
import io
import json
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from gita_decision_demo import Passage, tokenize
from gita_nqlite_demo import (
    check_evidence,
    content_terms,
    main,
    retriever_report,
    retrieve,
)


def passage(chapter, verse, text):
    return Passage(
        chapter=str(chapter),
        verse=str(verse),
        author="Test Translator",
        text=text,
        tokens=tokenize(text),
    )


class FakeEmbedding:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        for _ in texts:
            yield [0.5] * 384


FAKE_FASTEMBED = types.SimpleNamespace(TextEmbedding=FakeEmbedding)


class FakeServer:
    """Canned `nql-server --stdio` responses for hybrid and BM25 programs."""

    def __init__(self, hybrid_rows=(), bm25_rows=()):
        self.hybrid_rows = list(hybrid_rows)
        self.bm25_rows = list(bm25_rows)
        self.programs = []
        self.closed = False

    def execute(self, program):
        self.programs.append(program)
        if "vector::similarity" in program:
            rows = self.hybrid_rows
        elif "::bm25" in program:
            rows = self.bm25_rows
        else:
            return [], True, None
        body = "; ".join(
            "verse:%d score=%.4f {citation=\"BG 0.%d\"}" % (number, score, number)
            for number, score in rows
        )
        return (
            ["SELECT verse (%d rows): %s" % (len(rows), body)],
            True,
            None,
        )

    def close(self):
        self.closed = True


class NqliteDemoTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            passage("2", "47", "You have a right to action but not to its fruits."),
            passage("6", "5", "Elevate yourself through the power of your own mind."),
            passage("12", "13", "One who is free from hatred and friendly to all beings."),
        ]
        self.corpus = {"test": True}

    def run_main(self, argv, server):
        stdout = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"fastembed": FAKE_FASTEMBED}))
            stack.enter_context(
                patch("gita_nqlite_demo.NqliteServer", return_value=server)
            )
            stack.enter_context(
                patch("gita_nqlite_demo.load_passages", return_value=self.passages)
            )
            stack.enter_context(
                patch("gita_nqlite_demo.corpus_metadata", return_value=self.corpus)
            )
            stack.enter_context(
                patch("gita_nqlite_demo.ingest", return_value=len(self.passages))
            )
            with contextlib.redirect_stdout(stdout):
                with patch(
                    "gita_nqlite_demo.call_knot", return_value={"ok": True}
                ) as call_knot:
                    status = main(argv)
        return status, json.loads(stdout.getvalue()), call_knot

    def test_content_terms_drop_stopwords_unstemmed(self):
        self.assertEqual(
            content_terms("acting without attachments"),
            ["acting", "without", "attachments"],
        )
        self.assertEqual(content_terms("what is it"), [])
        self.assertEqual(content_terms("  "), [])

    def test_retriever_report_shape(self):
        report = retriever_report(0.123456)
        self.assertEqual(report["engine"], "nqlite")
        self.assertEqual(report["bm25_max_score"], 0.1235)
        self.assertIn("MiniLM", report["embedding_model"])

    def test_retrieve_maps_verse_ids_to_passages(self):
        server = FakeServer(hybrid_rows=[(3, 0.02), (701, 0.019)])
        ranked = retrieve(server, FakeEmbedding("m"), self.passages * 234, "mind", 2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][1].citation, "BG 12.13")
        self.assertAlmostEqual(ranked[0][0], 0.02)

    def test_check_evidence_positive_zero_and_empty(self):
        server = FakeServer(bm25_rows=[(1, 0.5)])
        self.assertEqual(check_evidence(server, "action fruits", 5), (True, 0.5))
        server = FakeServer(bm25_rows=[(1, 0.0)])
        self.assertEqual(check_evidence(server, "action fruits", 5), (False, 0.0))
        server = FakeServer(bm25_rows=[])
        has_evidence, best = check_evidence(server, "action fruits", 5)
        self.assertFalse(has_evidence)
        self.assertEqual(best, 0.0)
        quiet = FakeServer(bm25_rows=[(1, 0.5)])
        has_evidence, best = check_evidence(quiet, "what is it", 5)
        self.assertFalse(has_evidence)
        self.assertEqual(best, 0.0)
        self.assertEqual(
            [program for program in quiet.programs if "::bm25" in program], []
        )

    def test_zero_evidence_skips_knot(self):
        server = FakeServer(hybrid_rows=[(1, 0.02)], bm25_rows=[(1, 0.0)])
        status, payload, call_knot = self.run_main(["how should I act?"], server)
        self.assertEqual(status, 0)
        self.assertTrue(payload["no_evidence"])
        self.assertIsNone(payload["decision"])
        self.assertEqual(payload["retrieved_passages"], [])
        self.assertEqual(payload["retriever"]["engine"], "nqlite")
        call_knot.assert_not_called()
        self.assertTrue(server.closed)

    def test_stopword_only_query_needs_no_server_queries(self):
        server = FakeServer()
        status, payload, call_knot = self.run_main(["what is it"], server)
        self.assertEqual(status, 0)
        self.assertTrue(payload["no_evidence"])
        self.assertIsNone(payload["decision"])
        call_knot.assert_not_called()
        self.assertEqual(server.programs, [])

    def test_choice_needs_two_candidates(self):
        server = FakeServer(hybrid_rows=[(1, 0.02)], bm25_rows=[(1, 0.5)])
        status, payload, call_knot = self.run_main(["how should I act?"], server)
        self.assertEqual(status, 0)
        self.assertTrue(payload["insufficient_candidates"])
        self.assertFalse(payload["no_evidence"])
        self.assertIsNone(payload["decision"])
        self.assertEqual(len(payload["retrieved_passages"]), 1)
        call_knot.assert_not_called()

    def test_choice_calls_knot_with_cited_excerpts(self):
        server = FakeServer(
            hybrid_rows=[(1, 0.02), (2, 0.019)], bm25_rows=[(1, 0.5)]
        )
        status, payload, call_knot = self.run_main(["how should I act?"], server)
        self.assertIsNone(status)
        self.assertEqual(payload["decision"], {"ok": True})
        self.assertEqual(len(payload["retrieved_passages"]), 2)
        self.assertEqual(
            [item["citation"] for item in payload["retrieved_passages"]],
            ["BG 2.47", "BG 6.5"],
        )
        self.assertTrue(
            all(item["author"] == "Test Translator" for item in payload["retrieved_passages"])
        )
        call_knot.assert_called_once()

    def test_score_accepts_single_candidate(self):
        server = FakeServer(hybrid_rows=[(2, 0.02)], bm25_rows=[(2, 0.5)])
        status, payload, call_knot = self.run_main(
            ["how calm is the mind?", "--mode", "score", "--top-k", "1"], server
        )
        self.assertIsNone(status)
        self.assertEqual(payload["decision"], {"ok": True})
        self.assertEqual(len(payload["retrieved_passages"]), 1)
        call_knot.assert_called_once()

    def test_cli_validation(self):
        server = FakeServer()
        with self.assertRaises(SystemExit) as raised:
            self.run_main(["q", "--top-k", "0"], server)
        self.assertEqual(raised.exception.code, 2)
        with self.assertRaises(SystemExit) as raised:
            self.run_main(["q", "--mode", "choice", "--top-k", "1"], server)
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
