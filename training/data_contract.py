#!/usr/bin/env python3
"""Shared validation for the optional OIO training-data pipeline."""

import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "calibration", "test")
PRIMITIVES = {"choice", "score", "noul"}
SOURCES = {
    "AmazonScience/massive": {
        "revision": "MASSIVE-1.1",
        "license": "cc-by-4.0",
    },
    "PolyAI/banking77": {
        "revision": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
        "license": "cc-by-4.0",
    },
    "oio-authored/score-smoke": {
        "revision": "oio-training-v1",
        "license": "apache-2.0",
    },
}
SOURCE_INPUT_SHA256 = {
    "massive_archive_sha256": "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
    "banking77_train_sha256": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "banking77_test_sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


class DataError(ValueError):
    """Raised when a prepared corpus violates the training-data contract."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primitive_and_label(row, where):
    if not isinstance(row, dict):
        raise DataError("%s must be an object" % where)
    questions = row.get("questions")
    expected = row.get("expected")
    if not isinstance(questions, dict) or len(questions) != 1:
        raise DataError("%s must contain exactly one question" % where)
    if not isinstance(expected, dict) or len(expected) != 1:
        raise DataError("%s must contain exactly one expected label" % where)

    qid, question = next(iter(questions.items()))
    if not isinstance(question, dict):
        raise DataError("%s question must be an object" % where)
    primitive = question.get("type")
    if primitive not in PRIMITIVES:
        raise DataError("%s has unsupported primitive %r" % (where, primitive))
    if qid not in expected:
        raise DataError("%s expected label does not match its question" % where)
    label = expected[qid]
    criteria = question.get("criteria")

    if primitive == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise DataError("%s choice criteria must be a non-empty object" % where)
        if str(label) not in criteria:
            raise DataError("%s choice target %r is not a criterion" % (where, label))
    elif primitive == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DataError("%s score criteria must be an ordered list of at least 2 levels" % where)
        if type(label) is not int:
            raise DataError("%s score target must be an integer level index" % where)
        if not 0 <= label < len(criteria):
            raise DataError("%s score target is outside its ordered levels" % where)
    else:
        if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
            raise DataError("%s noul criteria must define false and true" % where)
        if not isinstance(label, bool):
            raise DataError("%s noul target must be boolean" % where)
    return primitive, label


def read_split(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataError("%s:%d is invalid JSON: %s" % (path, line_no, exc)) from exc
    return rows


def validate_rows(rows, split, case_ids, source_splits, text_splits):
    if not rows:
        raise DataError("%s split is empty" % split)
    coverage = set()
    for index, row in enumerate(rows, 1):
        where = "%s row %d" % (split, index)
        if not isinstance(row, dict):
            raise DataError("%s must be an object" % where)
        if row.get("split") != split:
            raise DataError("%s has split %r, expected %r" % (where, row.get("split"), split))
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise DataError("%s needs a non-empty case_id" % where)
        if case_id in case_ids:
            raise DataError("duplicate case_id %r" % case_id)
        case_ids.add(case_id)

        state = row.get("state")
        if not isinstance(state, str) or not state.strip():
            raise DataError("%s needs a non-empty text state" % where)
        questions = row.get("questions")
        expected = row.get("expected")
        if not isinstance(questions, dict) or not isinstance(expected, dict):
            raise DataError("%s needs questions and expected objects" % where)
        primitive, _ = primitive_and_label(row, where)
        coverage.add(primitive)

        source = row.get("source")
        if not isinstance(source, dict):
            raise DataError("%s needs source provenance" % where)
        dataset_id = source.get("dataset_id")
        if dataset_id not in SOURCES:
            raise DataError("%s has unapproved source %r" % (where, dataset_id))
        approved = SOURCES[dataset_id]
        if source.get("revision") != approved["revision"]:
            raise DataError("%s source revision is not pinned to the approved revision" % where)
        if str(source.get("license", "")).lower() != approved["license"]:
            raise DataError("%s source license does not match its approved license" % where)
        if not isinstance(source.get("source_id"), str) or not source["source_id"]:
            raise DataError("%s needs a source row identity" % where)
        if not isinstance(source.get("derivation"), str) or not source["derivation"].strip():
            raise DataError("%s needs a label/data derivation description" % where)

        source_key = (dataset_id, source["source_id"])
        previous = source_splits.setdefault(source_key, split)
        if previous != split:
            raise DataError("%s source row %r also occurs in %s" % (where, source_key, previous))
        text_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
        previous = text_splits.setdefault(text_hash, split)
        if previous != split:
            raise DataError("%s exact state text also occurs in %s" % (where, previous))

    missing = PRIMITIVES - coverage
    if missing:
        raise DataError("%s split is missing primitive(s): %s" % (split, ", ".join(sorted(missing))))


def validate_directory(data_dir):
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DataError("missing %s" % manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError("invalid manifest JSON: %s" % exc) from exc
    if manifest.get("schema_version") != 1:
        raise DataError("manifest schema_version must be 1")
    if manifest.get("sources") != SOURCES:
        raise DataError("manifest source revisions/licenses do not match the approved sources")
    if manifest.get("source_input_sha256") != SOURCE_INPUT_SHA256:
        raise DataError("downloaded input SHA-256 values do not match the pinned data")

    case_ids = set()
    source_splits = {}
    text_splits = {}
    counts = {}
    for split in SPLITS:
        path = data_dir / (split + ".jsonl")
        if not path.is_file():
            raise DataError("missing split file %s" % path)
        declared_hash = manifest.get("split_sha256", {}).get(split)
        if declared_hash != sha256_file(path):
            raise DataError("%s does not match the manifest SHA-256" % path)
        rows = read_split(path)
        validate_rows(rows, split, case_ids, source_splits, text_splits)
        counts[split] = len(rows)
        if manifest.get("split_counts", {}).get(split) != counts[split]:
            raise DataError("%s row count does not match the manifest" % split)
    return counts
