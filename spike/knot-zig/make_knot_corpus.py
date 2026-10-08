#!/usr/bin/env python3
"""Extract a large tokenizer-parity corpus from knot's own fixtures.

Walks knot's real wire/golden fixtures (states, instructions, criteria, ids —
every string value) and emits a deduplicated, deterministically ordered
strings file for the Rust oracle → zig parity pipeline. These are the exact
kinds of text knot tokenizes in production (English prompts, Hindi demo case,
unicode edge cases inside recorded responses).

  out: knot_strings.json  (list of strings, cap KNOT_CORPUS_CAP)
Skips: empty strings, strings over MAX_LEN (huge states would make the
debug-build NFC linear scans glacial; counted and reported).
"""

import json
import pathlib
import sys
import unicodedata

KNOT = pathlib.Path("/root/ai-workspace/knot")
SOURCES = [
    "crates/knot/tests/fixtures/golden_english.json",
    "crates/knot/tests/fixtures/engine_english.json",
    "crates/knot/tests/fixtures/parity_english.json",
    "crates/knot/tests/fixtures/lang_cases.json",
    "crates/knot/tests/fixtures/systemone_request_minimal.json",
    "crates/knot/tests/fixtures/systemone_request_multilingual.json",
    "crates/knot/tests/fixtures/systemone_response_choice.json",
    "crates/knot/tests/fixtures/systemone_response_all_types.json",
    "crates/knot/tests/fixtures/eval_english.jsonl",
    "demo/cases/massive_hi.json",
]
OUT = pathlib.Path(__file__).resolve().parent / "knot_strings.json"
CAP = 400
MAX_LEN = 4000


def walk(node, out: set[str]) -> None:
    if isinstance(node, str):
        if node:
            out.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            walk(v, out)
    elif isinstance(node, list):
        for v in node:
            walk(v, out)


def main() -> None:
    found: set[str] = set()
    for rel in SOURCES:
        p = KNOT / rel
        if not p.exists():
            print(f"MISSING {rel}", file=sys.stderr)
            continue
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    walk(json.loads(line), found)
        else:
            walk(json.loads(p.read_text()), found)

    too_long = sum(1 for s in found if len(s) > MAX_LEN)
    kept = sorted(
        (s for s in found if 0 < len(s) <= MAX_LEN),
        key=lambda s: (-len(s), s),  # longest first = richest tokenization
    )[:CAP]
    # deterministic final order (longest-first is informative, keep it stable)
    non_ascii = sum(1 for s in kept if any(ord(c) > 127 for c in s))
    scripts = sorted(
        {
            unicodedata.category(chr(cp))[0]
            for s in kept
            for cp in map(ord, s)
            if ord(c := chr(cp)) > 127
        }
    )
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=0))
    print(
        f"found={len(found)} kept={len(kept)} too_long_skipped={too_long}\n"
        f"max_len={max(map(len, kept))} non_ascii_strings={non_ascii} "
        f"non-latin categories={scripts}\n-> {OUT}"
    )


if __name__ == "__main__":
    main()
