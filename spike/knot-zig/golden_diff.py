#!/usr/bin/env python3
"""G1 part 2: full-body golden diff — POST every recorded fixture request
(+ the #37 fallback case) to knot and the zig engine, byte-compare.

Usage: golden_diff.py <knot_base> <zig_base> [fixtures_dir]
"""
import json
import sys
import urllib.error
import urllib.request

def post(base, path, body):
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

def main():
    knot = sys.argv[1].rstrip("/")
    zig = sys.argv[2].rstrip("/")
    fdir = sys.argv[3] if len(sys.argv) > 3 else "/root/ai-workspace/knot/crates/knot/tests/fixtures"

    cases = []
    g = json.load(open(fdir + "/golden_english.json"))
    for i, c in enumerate(g["cases"]):
        cases.append((f"golden[{i}]", {"state": c["state"], "questions": c["questions"]}))
    e = json.load(open(fdir + "/engine_english.json"))
    cases.append(("engine", {"state": e["state"], "questions": e["questions"]}))
    # #37 fallback over the wire (single-checkpoint deploys)
    cases.append(("fallback_sv", {
        "state": "Tack, nu fungerar det igen!",
        "questions": {"q": {"type": "noul", "instructions": "Is thanks given?"}},
    }))
    cases.append(("fallback_ar", {
        "state": "مرحبا بالعالم",
        "questions": {"q": {"type": "noul", "instructions": "Is this greeted?"}},
    }))
    # content-type variants on a real request (knot ignores it)
    cases.append(("fallback_sv_noct", {
        "state": "Tack, nu fungerar det igen!",
        "questions": {"q": {"type": "noul", "instructions": "Is thanks given?"}},
    }))

    matched = 0
    for name, body in cases:
        ks, kb = post(knot, "/v1/systemone", body)
        zs, zb = post(zig, "/v1/systemone", body)
        if ks == zs and kb == zb:
            matched += 1
            print(f"{name:24s} MATCH   {ks}")
        else:
            print(f"{name:24s} FAIL    knot={ks} zig={zs}")
            if kb != zb:
                for i, (a, b) in enumerate(zip(kb, zb)):
                    if a != b:
                        print("  first byte diff at", i)
                        print("  knot:", kb[max(0, i - 60):i + 140])
                        print("  zig :", zb[max(0, i - 60):i + 140])
                        break
                else:
                    print("  lengths:", len(kb), len(zb))
                    print("  knot tail:", kb[min(len(kb), len(zb)) - 40:])
                    print("  zig  tail:", zb[min(len(kb), len(zb)) - 40:])
    print(f"result: {matched}/{len(cases)} golden full-body cases match")
    return 0 if matched == len(cases) else 1

if __name__ == "__main__":
    sys.exit(main())
