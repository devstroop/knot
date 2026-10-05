import json
import tempfile
import unittest
from pathlib import Path

from training.data_contract import (
    DataError,
    SOURCES,
    SOURCE_INPUT_SHA256,
    SPLITS,
    primitive_and_label,
    sha256_file,
    validate_directory,
)
from training.prepare_data import deduplicate_by_state, make_row, stratified_split


def example(split, primitive, suffix):
    source_id = "%s-%s-%s" % (split, primitive, suffix)
    if primitive == "choice":
        question = {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": "A", "b": "B"},
        }
        target = "a"
    elif primitive == "score":
        question = {
            "type": "score",
            "instructions": "Rate it.",
            "criteria": ["low", "high"],
        }
        target = 1
    else:
        question = {
            "type": "noul",
            "instructions": "Is it true?",
            "criteria": {"false": "No", "true": "Yes"},
        }
        target = True
    return make_row(
        "%s-%s-%s" % (split, primitive, suffix),
        "Unique state for %s %s %s." % (split, primitive, suffix),
        primitive,
        question,
        target,
        split,
        "oio-authored/score-smoke",
        "authored",
        source_id,
        "Unit-test-only example.",
    )


class DataPipelineTests(unittest.TestCase):
    def test_primitive_targets_are_checked_against_question_schema(self):
        for primitive in ("choice", "score", "noul"):
            row = example("train", primitive, "valid")
            self.assertEqual(primitive_and_label(row, "case"), (primitive, row["expected"]["decision"]))

        invalid_score = example("train", "score", "invalid")
        invalid_score["expected"]["decision"] = 2
        with self.assertRaisesRegex(DataError, "outside its ordered levels"):
            primitive_and_label(invalid_score, "case")

    def test_stratified_split_is_reproducible_and_keeps_every_row_once(self):
        rows = [{"id": str(index)} for index in range(30)]
        labels = ["a" if index % 2 == 0 else "b" for index in range(30)]
        first = stratified_split(
            rows, labels, 71, {"validation": 0.1, "calibration": 0.1, "train": 0.8}
        )
        second = stratified_split(
            rows, labels, 71, {"validation": 0.1, "calibration": 0.1, "train": 0.8}
        )
        self.assertEqual(first, second)
        indices = [index for group in first.values() for index, _ in group]
        self.assertEqual(sorted(indices), list(range(len(rows))))
        self.assertEqual(len(set(indices)), len(rows))

    def test_exact_text_duplicates_keep_test_and_drop_lower_priority_rows(self):
        rows = {
            split: [example(split, primitive, "only") for primitive in ("choice", "score", "noul")]
            for split in SPLITS
        }
        rows["train"][0]["state"] = rows["test"][0]["state"]
        rows["validation"][1]["state"] = rows["test"][0]["state"]
        exclusions = deduplicate_by_state(rows)
        self.assertEqual(exclusions, {"train": 1, "validation": 1})
        self.assertEqual(len(rows["test"]), 3)
        self.assertTrue(any(row["case_id"] == "test-choice-only" for row in rows["test"]))

    def test_directory_manifest_and_all_primitive_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            hashes = {}
            for split in SPLITS:
                rows = [example(split, primitive, "only") for primitive in ("choice", "score", "noul")]
                path = data_dir / (split + ".jsonl")
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                hashes[split] = sha256_file(path)
            (data_dir / "manifest.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "sources": SOURCES,
                    "source_input_sha256": SOURCE_INPUT_SHA256,
                    "split_sha256": hashes,
                    "split_counts": {split: 3 for split in SPLITS},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                validate_directory(data_dir),
                {"train": 3, "validation": 3, "calibration": 3, "test": 3},
            )

    def test_directory_rejects_cross_split_source_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            hashes = {}
            for split in SPLITS:
                rows = [example(split, primitive, "only") for primitive in ("choice", "score", "noul")]
                if split == "test":
                    rows[0]["source"]["source_id"] = "train-choice-only"
                path = data_dir / (split + ".jsonl")
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                hashes[split] = sha256_file(path)
            (data_dir / "manifest.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "sources": SOURCES,
                    "source_input_sha256": SOURCE_INPUT_SHA256,
                    "split_sha256": hashes,
                    "split_counts": {split: 3 for split in SPLITS},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataError, "also occurs in train"):
                validate_directory(data_dir)

    def test_approved_sources_are_revision_and_license_pinned(self):
        self.assertEqual(SOURCES["AmazonScience/massive"]["license"], "cc-by-4.0")
        self.assertEqual(SOURCES["PolyAI/banking77"]["license"], "cc-by-4.0")


if __name__ == "__main__":
    unittest.main()
