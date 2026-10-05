#!/usr/bin/env python3
"""Download pinned public data and build leakage-checked OIO JSONL splits."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data_contract import (
        SOURCES,
        SOURCE_INPUT_SHA256,
        SPLITS,
        DataError,
        sha256_file,
        validate_directory,
    )
else:
    from .data_contract import (
        SOURCES,
        SOURCE_INPUT_SHA256,
        SPLITS,
        DataError,
        sha256_file,
        validate_directory,
    )

MASSIVE_ID = "AmazonScience/massive"
BANKING_ID = "PolyAI/banking77"
SEED = 20260611
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)
BANKING_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
BANKING_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    + BANKING_COMMIT
    + "/banking_data"
)
BANKING_TRAIN_URL = BANKING_BASE_URL + "/train.csv"
BANKING_TEST_URL = BANKING_BASE_URL + "/test.csv"
SCORE_FIXTURE = Path(__file__).parent / "data" / "score_smoke.jsonl"


def download(url, path, expected_sha256):
    path = Path(path)
    if path.is_file():
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise DataError(
                "%s has SHA-256 %s, expected %s"
                % (path, actual_sha256, expected_sha256)
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".", suffix=".download", dir=path.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            request = Request(url, headers={"User-Agent": "OIO-training-data/1"})
            with urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        actual_sha256 = sha256_file(temporary_path)
        if actual_sha256 != expected_sha256:
            raise DataError(
                "downloaded %s has SHA-256 %s, expected %s"
                % (url, actual_sha256, expected_sha256)
            )
        os.replace(temporary_path, path)
    except (DataError, OSError, URLError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, DataError):
            raise
        raise RuntimeError("failed to download %s: %s" % (url, exc)) from exc
    return path


def row_identity(row, index, split):
    value = row.get("id")
    if value is None:
        value = "row-%d" % index
    return "%s:%s" % (split, value)


def stratified_split(rows, labels, seed, fractions):
    if len(rows) != len(labels):
        raise DataError("cannot stratify rows and labels with different lengths")
    groups = defaultdict(list)
    for index, (row, label) in enumerate(zip(rows, labels)):
        groups[label].append((index, row))
    rng = random.Random(seed)
    outputs = {name: [] for name in fractions}
    last_split = next(reversed(fractions))
    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        start = 0
        for name, fraction in fractions.items():
            if name == last_split:
                selected = group[start:]
            else:
                count = max(1, round(len(group) * fraction))
                count = min(count, max(0, len(group) - start - 1))
                selected = group[start:start + count]
                start += count
            outputs[name].extend(
                (index, dict(row, _oio_row_index=index)) for index, row in selected
            )
    for name in outputs:
        outputs[name].sort(key=lambda item: item[0])
    return outputs


def read_massive(archive_path):
    rows = {name: [] for name in ("train", "validation", "test")}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if PurePosixPath(member.name).parts[-2:] == ("data", "en-US.jsonl")
        ]
        if len(members) != 1 or not members[0].isfile():
            raise DataError("MASSIVE 1.1 archive must contain exactly one data/en-US.jsonl file")
        member = members[0]
        if member.size > 64 * 1024 * 1024:
            raise DataError("MASSIVE English data file is unexpectedly large")
        stream = archive.extractfile(member)
        if stream is None:
            raise DataError("could not read MASSIVE English data file")
        with stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError("MASSIVE en-US.jsonl:%d: %s" % (line_no, exc)) from exc
                partition = row.get("partition")
                split = {"train": "train", "dev": "validation", "test": "test"}.get(partition)
                if split is None:
                    raise DataError("unknown MASSIVE partition %r" % partition)
                if not all(isinstance(row.get(key), str) and row[key] for key in
                           ("id", "scenario", "intent", "utt")):
                    raise DataError("MASSIVE row %d is missing a required string field" % line_no)
                rows[split].append(row)
    if any(not rows[name] for name in rows):
        raise DataError("MASSIVE 1.1 is missing an English train, dev, or test partition")
    return rows


def read_banking(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise DataError("BANKING77 CSV must have text and category columns")
        for index, row in enumerate(reader):
            text, label = row.get("text"), row.get("category")
            if not text or not label:
                raise DataError("BANKING77 row %d is missing text or category" % (index + 1))
            rows.append({"text": text, "label": label, "_oio_row_index": index})
    if not rows:
        raise DataError("BANKING77 source CSV is empty")
    return rows


def make_row(case_id, state, primitive, question, target, split, dataset_id,
             original_split, source_id, derivation):
    source_info = SOURCES[dataset_id]
    return {
        "case_id": case_id,
        "state": state,
        "questions": {"decision": question},
        "expected": {"decision": target},
        "split": split,
        "source": {
            "dataset_id": dataset_id,
            "revision": source_info["revision"],
            "license": source_info["license"],
            "original_split": original_split,
            "source_id": source_id,
            "derivation": derivation,
        },
        "language": "en",
        "tags": ["primitive:" + primitive, "source:" + dataset_id.rsplit("/", 1)[-1]],
    }


def choice_row(state, label, labels, case_id, dataset_id, split, original_split, source_id):
    criteria = {name: name.replace("_", " ") for name in labels}
    question = {
        "type": "choice",
        "instructions": "Which intent best describes the user's utterance?",
        "criteria": criteria,
    }
    return make_row(
        case_id, state, "choice", question, label, split, dataset_id, original_split,
        source_id, "Directly uses the source dataset's intent label.",
    )


def massive_rows(rows, source_split, output_split, intent_names, scenario_names):
    records = []
    domain_ids = {name: index for index, name in enumerate(scenario_names)}
    for position, item in enumerate(rows):
        index = item.get("_oio_row_index", position)
        state = item["utt"].strip()
        intent = item["intent"]
        scenario = item["scenario"]
        source_id = row_identity(item, index, source_split)
        prefix = "massive:%s:%s" % (source_split, source_id)
        records.append(choice_row(
            state, intent, intent_names, prefix + ":choice", MASSIVE_ID,
            output_split, source_split, source_id,
        ))

        scenario_index = domain_ids[scenario]
        negative_domain = scenario_names[(scenario_index + 1) % len(scenario_names)]
        for target_domain, target in ((scenario, True), (negative_domain, False)):
            question = {
                "type": "noul",
                "instructions": "Is this utterance about the %s domain?" % target_domain.replace("_", " "),
                "criteria": {
                    "false": "The utterance is about a different domain.",
                    "true": "The utterance is about this domain.",
                },
            }
            records.append(make_row(
                prefix + ":noul:" + ("positive" if target else "negative"),
                state, "noul", question, target, output_split, MASSIVE_ID,
                source_split, source_id,
                "Derived deterministically from the MASSIVE scenario/domain label; "
                "not an independent out-of-scope annotation.",
            ))
    return records


def banking_rows(rows, labels, output_split, original_split):
    records = []
    for position, value in enumerate(rows):
        index, item = value if (
            isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)
        ) else (position, value)
        source_id = row_identity(item, item.get("_oio_row_index", index), original_split)
        records.append(choice_row(
            item["text"].strip(), item["label"], labels,
            "banking77:%s:%s:choice" % (original_split, source_id), BANKING_ID,
            output_split, original_split, source_id,
        ))
    return records


def load_synthetic_rows():
    rows = []
    with open(SCORE_FIXTURE, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataError("%s:%d: %s" % (SCORE_FIXTURE, line_no, exc)) from exc
    return rows


def deduplicate_by_state(rows_by_split):
    priority = ("test", "validation", "calibration", "train")
    owner = {}
    for split in priority:
        for row in rows_by_split[split]:
            state_hash = hashlib.sha256(row["state"].encode("utf-8")).hexdigest()
            owner.setdefault(state_hash, split)

    exclusions = defaultdict(int)
    for split in SPLITS:
        kept = []
        for row in rows_by_split[split]:
            state_hash = hashlib.sha256(row["state"].encode("utf-8")).hexdigest()
            if owner[state_hash] == split:
                kept.append(row)
            else:
                exclusions[split] += 1
        rows_by_split[split] = kept
    return dict(exclusions)


def build(output_dir, cache_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    massive_path = download(
        MASSIVE_URL,
        Path(cache_dir) / "amazon-massive-dataset-1.1.tar.gz",
        SOURCE_INPUT_SHA256["massive_archive_sha256"],
    )
    banking_train_path = download(
        BANKING_TRAIN_URL,
        Path(cache_dir) / "banking77-train.csv",
        SOURCE_INPUT_SHA256["banking77_train_sha256"],
    )
    banking_test_path = download(
        BANKING_TEST_URL,
        Path(cache_dir) / "banking77-test.csv",
        SOURCE_INPUT_SHA256["banking77_test_sha256"],
    )
    massive = read_massive(massive_path)
    banking_train = read_banking(banking_train_path)
    banking_test = read_banking(banking_test_path)

    intent_names = sorted({row["intent"] for row in massive["train"]})
    scenario_names = sorted({row["scenario"] for row in massive["train"]})
    if len(intent_names) != 60 or len(scenario_names) != 18:
        raise DataError(
            "MASSIVE English schema changed: expected 60 intents and 18 domains, got %d/%d"
            % (len(intent_names), len(scenario_names))
        )
    massive_labels = [row["intent"] for row in massive["train"]]
    massive_split = stratified_split(
        massive["train"], massive_labels, SEED, {"calibration": 0.1, "train": 0.9}
    )

    banking_labels = sorted({row["label"] for row in banking_train})
    bank_targets = [row["label"] for row in banking_train]
    bank_split = stratified_split(
        banking_train, bank_targets, SEED,
        {"validation": 0.1, "calibration": 0.1, "train": 0.8},
    )

    rows_by_split = {name: [] for name in SPLITS}
    for source_split, output_split in (("train", "train"), ("calibration", "calibration")):
        rows_by_split[output_split].extend(
            massive_rows(
                [row for _, row in massive_split[source_split]],
                "train",
                output_split,
                intent_names,
                scenario_names,
            )
        )
    rows_by_split["validation"].extend(
        massive_rows(massive["validation"], "dev", "validation", intent_names, scenario_names)
    )
    rows_by_split["test"].extend(
        massive_rows(massive["test"], "test", "test", intent_names, scenario_names)
    )

    for split_name in ("train", "validation", "calibration"):
        rows_by_split[split_name].extend(
            banking_rows(bank_split[split_name], banking_labels, split_name, "train")
        )
    rows_by_split["test"].extend(
        banking_rows(banking_test, banking_labels, "test", "test")
    )
    for row in load_synthetic_rows():
        rows_by_split[row["split"]].append(row)

    exact_text_exclusions = deduplicate_by_state(rows_by_split)
    for split_name in SPLITS:
        rows_by_split[split_name].sort(key=lambda row: row["case_id"])
        path = output_dir / (split_name + ".jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows_by_split[split_name]:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    input_hashes = {
        "massive_archive_sha256": sha256_file(massive_path),
        "banking77_train_sha256": sha256_file(banking_train_path),
        "banking77_test_sha256": sha256_file(banking_test_path),
    }
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "sources": SOURCES,
        "source_urls": {
            "massive": MASSIVE_URL,
            "banking77_train": BANKING_TRAIN_URL,
            "banking77_test": BANKING_TEST_URL,
        },
        "source_input_sha256": input_hashes,
        "split_policy": {
            "massive": "official dev/test; stratified 10% of official train reserved for calibration",
            "banking77": "official test unchanged; stratified 10% validation and 10% calibration carved from official train",
            "score": "OIO-authored synthetic smoke rows use their explicit fixed split assignments",
            "exact_text_overlap": "keep the highest-priority split in test, validation, calibration, train order; never remove official test rows",
        },
        "exact_text_overlap_excluded_rows": exact_text_exclusions,
        "split_counts": {name: len(rows_by_split[name]) for name in SPLITS},
        "split_sha256": {
            name: sha256_file(output_dir / (name + ".jsonl")) for name in SPLITS
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validate_directory(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="training/out/data")
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "oio" / "training-data"),
    )
    args = parser.parse_args()
    try:
        manifest = build(args.output_dir, args.cache_dir)
    except (DataError, OSError, RuntimeError, tarfile.TarError) as exc:
        parser.error(str(exc))
    print("Prepared rows: " + ", ".join(
        "%s=%d" % item for item in manifest["split_counts"].items()
    ))
    print("Manifest: %s" % (Path(args.output_dir) / "manifest.json"))


if __name__ == "__main__":
    main()
