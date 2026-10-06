import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
    from training.feature_cache import (
        CacheError,
        CacheWriter,
        FeatureCache,
        index_path_for,
    )
except ImportError:  # numpy and the training extras are optional
    np = None
    CacheError = CacheWriter = FeatureCache = index_path_for = None

NUMPY_REQUIRED = "install training/requirements.txt to run feature-cache tests"

PROVENANCE = {
    "base_model_sha256": "b" * 64,
    "laya_source_revision": "c" * 40,
    "training_data_manifest_sha256": "d" * 64,
}
HIDDEN = 8


def sample_rows():
    return [
        ("case-a", np.arange(3 * HIDDEN, dtype=np.float32).reshape(3, HIDDEN)),
        ("case-b", np.ones((5, HIDDEN), dtype=np.float32) * 2.0),
        ("case-c", np.full((1, HIDDEN), 3.0, dtype=np.float32)),
    ]


def write_cache(directory, dtype="float16", provenance=None, rows=None):
    bin_path = Path(directory) / "train.bin"
    writer = CacheWriter(
        bin_path, "train", dtype, HIDDEN, provenance if provenance is not None else PROVENANCE
    )
    for case_id, array in (rows if rows is not None else sample_rows()):
        writer.append([case_id], array[None, :, :])
    index = writer.close()
    return bin_path, index


@unittest.skipUnless(np is not None, NUMPY_REQUIRED)
class CacheWriterTests(unittest.TestCase):
    def test_roundtrip_preserves_values_lengths_and_provenance(self):
        rows = sample_rows()
        with TemporaryDirectory() as directory:
            bin_path, index = write_cache(directory)
            self.assertEqual(index["schema_version"], 1)
            self.assertEqual(index["split"], "train")
            self.assertEqual(index["dtype"], "float16")
            self.assertEqual(index["hidden_size"], HIDDEN)
            self.assertEqual(index["case_count"], 3)
            self.assertEqual(index["provenance"], PROVENANCE)
            self.assertEqual(len(index["file_sha256"]), 64)

            cache = FeatureCache(bin_path, split="train", expected_provenance=PROVENANCE)
            self.assertEqual(len(cache), 3)
            for case_id, array in rows:
                stored = cache.get(case_id)
                self.assertEqual(stored.shape, array.shape)
                np.testing.assert_allclose(
                    stored.astype(np.float32), array, rtol=1e-3, atol=1e-3
                )
            self.assertEqual(cache.missing(["case-a", "nope"]), ["nope"])
            cache.close()

    def test_float32_roundtrip_is_exact(self):
        rows = sample_rows()
        with TemporaryDirectory() as directory:
            bin_path, _ = write_cache(directory, dtype="float32")
            cache = FeatureCache(bin_path, split="train")
            for case_id, array in rows:
                np.testing.assert_array_equal(cache.get(case_id), array)
            cache.close()

    def test_rejects_duplicate_case_ids_and_bad_shapes(self):
        with TemporaryDirectory() as directory:
            bin_path = Path(directory) / "train.bin"
            writer = CacheWriter(bin_path, "train", "float16", HIDDEN, PROVENANCE)
            writer.append(["case-a"], np.zeros((1, 2, HIDDEN), dtype=np.float32))
            with self.assertRaisesRegex(CacheError, "duplicate"):
                writer.append(["case-a"], np.zeros((1, 2, HIDDEN), dtype=np.float32))
            with self.assertRaisesRegex(CacheError, "expected"):
                writer.append(["case-b"], np.zeros((1, 2, HIDDEN + 1), dtype=np.float32))
            writer.abort()
            self.assertFalse(bin_path.exists())
            self.assertFalse(index_path_for(bin_path).exists())

    def test_rejects_unknown_dtype_and_incomplete_provenance(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CacheError, "dtype"):
                CacheWriter(Path(directory) / "a.bin", "train", "int8", HIDDEN, PROVENANCE)
            incomplete = dict(PROVENANCE)
            incomplete.pop("laya_source_revision")
            with self.assertRaisesRegex(CacheError, "provenance"):
                CacheWriter(Path(directory) / "b.bin", "train", "float16", HIDDEN, incomplete)


@unittest.skipUnless(np is not None, NUMPY_REQUIRED)
class FeatureCacheTests(unittest.TestCase):
    def test_missing_index_reports_how_to_build_it(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CacheError, "build_feature_cache"):
                FeatureCache(Path(directory) / "train.bin")

    def test_rejects_stale_provenance_and_wrong_split(self):
        with TemporaryDirectory() as directory:
            bin_path, _ = write_cache(directory)
            stale = dict(PROVENANCE)
            stale["base_model_sha256"] = "e" * 64
            with self.assertRaisesRegex(CacheError, "stale"):
                FeatureCache(bin_path, split="train", expected_provenance=stale)
            with self.assertRaisesRegex(CacheError, "expected"):
                FeatureCache(bin_path, split="validation")
            self._rewrite_index(bin_path, {"schema_version": 99})
            with self.assertRaisesRegex(CacheError, "unsupported"):
                FeatureCache(bin_path, split="train")

    def test_rejects_truncated_data(self):
        with TemporaryDirectory() as directory:
            bin_path, _ = write_cache(directory)
            with bin_path.open("r+b") as handle:
                handle.truncate(4)
            cache = FeatureCache(bin_path, split="train")
            with self.assertRaisesRegex(CacheError, "truncated"):
                cache.get("case-a")
            cache.close()

    def test_rejects_unknown_case_id(self):
        with TemporaryDirectory() as directory:
            bin_path, _ = write_cache(directory)
            cache = FeatureCache(bin_path, split="train")
            with self.assertRaisesRegex(CacheError, "not in feature cache"):
                cache.get("case-z")
            cache.close()

    def _rewrite_index(self, bin_path, changes):
        index_path = index_path_for(bin_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index.update(changes)
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return index_path


TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.environ.get("OIO_TEST_MODEL_DIR")
LAYA_SOURCE = os.environ.get("OIO_TEST_LAYA_SOURCE")
DATA_FILE = Path(__file__).resolve().parents[1] / "out" / "data" / "validation.jsonl"


@unittest.skipUnless(
    np is not None and TORCH_AVAILABLE and MODEL_DIR and LAYA_SOURCE and DATA_FILE.is_file(),
    "install training/requirements.txt and set OIO_TEST_MODEL_DIR, "
    "OIO_TEST_LAYA_SOURCE with prepared data to run cache parity tests",
)
class CachedForwardParityTests(unittest.TestCase):
    def test_cached_features_match_the_live_forward_pass(self):
        from training.data_contract import read_split
        from training.eval_metrics import collate
        from training.feature_cache import CacheWriter, FeatureCache, verify_cached_forward
        from training.train_frozen_head import load_model

        torch, model, tokenizer, _, encode = load_model(MODEL_DIR, LAYA_SOURCE)
        torch.set_num_threads(4)
        rows = read_split(DATA_FILE)[:8]
        items = [encode(row) for row in rows]
        hidden_size = model.encoder.config.hidden_size

        with TemporaryDirectory() as directory:
            bin_path = Path(directory) / "validation.bin"
            writer = CacheWriter(
                bin_path, "validation", "float16", hidden_size, {
                    "base_model_sha256": "x" * 64,
                    "laya_source_revision": "y" * 40,
                    "training_data_manifest_sha256": "z" * 64,
                }
            )
            model.eval()
            with torch.no_grad():
                ids, attention, _, _, _, _ = collate(items, tokenizer.pad_token_id, torch)
                hidden = model.encoder(
                    input_ids=ids, attention_mask=attention
                ).last_hidden_state.detach().cpu().numpy()
            for index, item in enumerate(items):
                length = len(item["ids"])
                writer.append([item["case_id"]], hidden[index:index + 1, :length, :])
            writer.close()

            cache = FeatureCache(
                bin_path, split="validation", expected_provenance={
                    "base_model_sha256": "x" * 64,
                    "laya_source_revision": "y" * 40,
                    "training_data_manifest_sha256": "z" * 64,
                }
            )
            summary = verify_cached_forward(
                model, items, cache, tokenizer, 4, torch, limit=len(items)
            )
            cache.close()
        self.assertEqual(summary["prediction_mismatches"], 0)
        self.assertEqual(summary["cases"], len(items))
        self.assertLess(summary["max_abs_logit_diff"], 0.5)


if __name__ == "__main__":
    unittest.main()
