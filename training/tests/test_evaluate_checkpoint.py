import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from training.evaluate_checkpoint import main


class EvaluateCheckpointCliTests(unittest.TestCase):
    def base_arguments(self, data_dir):
        return [
            "--data-dir", str(data_dir),
            "--model-dir", "/nonexistent-model",
            "--laya-source", "/nonexistent-laya",
            "--split", "validation",
            "--output", "out.json",
        ]

    def test_rejects_invalid_batch_size_threads_and_case_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.base_arguments(Path(directory))
            for flag, value in (
                ("--batch-size", "0"),
                ("--threads", "0"),
                ("--max-cases", "0"),
                ("--progress-every", "-1"),
            ):
                with self.subTest(flag=flag):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            main(arguments + [flag, str(value)])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertTrue(stderr.getvalue().strip())

    def test_rejects_missing_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(self.base_arguments(missing))
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("missing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
