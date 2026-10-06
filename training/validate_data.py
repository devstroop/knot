#!/usr/bin/env python3
"""Validate a prepared KNOT corpus and its provenance/split manifest."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.data_contract import DataError, validate_directory
else:
    from .data_contract import DataError, validate_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing the four JSONL splits and manifest")
    args = parser.parse_args()
    try:
        counts = validate_directory(args.data_dir)
    except (DataError, OSError) as exc:
        parser.error(str(exc))
    print("Validated rows: " + ", ".join("%s=%d" % item for item in counts.items()))


if __name__ == "__main__":
    main()
