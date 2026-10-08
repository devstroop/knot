"""Tiny differential driver: same request corpus at knot and the zig
skeleton; compare status+body for every case. Inference cases are allowed
to differ (expected_differ) — everything else must match byte-for-byte.

Usage: python3 http_diff.py <knot_base> <skeleton_base>
"""
import json
import sys

import http_probe  # reuse CASES + send()

EXPECTED_DIFFER = {
    # knot runs real inference (200 + answer); skeleton has no engine (501)
    "minimal_ok", "wrong_content_type", "no_content_type",
}  # contract: everything else must match status+body byte-for-byte


def main():
    knot_base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
    skel_base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8125"
    rows, fails = [], 0
    for name, method, path, body, ctype in http_probe.CASES:
        a = http_probe.send(knot_base, method, path, body, ctype)
        b = http_probe.send(skel_base, method, path, body, ctype)
        same = a["status"] == b["status"] and a["body"] == b["body"]
        if not same and name in EXPECTED_DIFFER:
            verdict = "expected-differ"
        elif same:
            verdict = "MATCH"
        else:
            verdict = "FAIL"
            fails += 1
        rows.append({"case": name, "verdict": verdict,
                     "knot": a["status"], "skel": b["status"],
                     "knot_body": a["body"], "skel_body": b["body"]})
        print(f"{name:22s} {verdict:16s} knot={a['status']} skel={b['status']}")
        if verdict == "FAIL":
            print(f"    knot: {a['body'][:100]!r}")
            print(f"    skel: {b['body'][:100]!r}")
    print(f"result: {len(rows) - fails}/{len(rows)} transport cases match")
    with open("knot_http_diff.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
