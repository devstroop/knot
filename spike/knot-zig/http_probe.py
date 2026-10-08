"""Capture knot's exact transport-error surface as the skeleton's target.

Sends a fixed request corpus at a base URL and records {status, body,
content_type} per case -> knot_http_baseline.json. The zig skeleton must
reproduce every NON-inference case byte-for-byte; inference cases are
expected to differ (skeleton returns 501) and are marked accordingly.

Usage: python3 http_probe.py http://127.0.0.1:8123 [out.json]
"""
import json
import sys
import urllib.error
import urllib.request

CASES = [
    ("health", "GET", "/health", b"", "application/json"),
    ("models", "GET", "/models", b"", "application/json"),
    ("post_on_health", "POST", "/health", b"{}", "application/json"),
    ("get_on_systemone", "GET", "/v1/systemone", b"", "application/json"),
    ("unknown_path", "POST", "/v1/nope", b"{}", "application/json"),
    ("empty_body", "POST", "/v1/systemone", b"", "application/json"),
    ("bad_json", "POST", "/v1/systemone", b"{not json", "application/json"),
    ("missing_state", "POST", "/v1/systemone", b'{"questions": {}}', "application/json"),
    ("missing_questions", "POST", "/v1/systemone", b'{"state": "hi"}', "application/json"),
    ("bad_type_field", "POST", "/v1/systemone",
     b'{"state": "hi", "questions": {"q": {"type": "banana"}}}', "application/json"),
    ("wrong_content_type", "POST", "/v1/systemone", b'{"state": "hi", "questions": {}}', "text/plain"),
    ("no_content_type", "POST", "/v1/systemone", b'{"state": "hi", "questions": {}}', None),
    ("state_null", "POST", "/v1/systemone", b'{"state": null, "questions": {}}', "application/json"),
    ("batch_missing_body", "POST", "/v1/systemone/batch", b"{}", "application/json"),
    ("minimal_ok", "POST", "/v1/systemone",
     b'{"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}',
     "application/json"),
]


def send(base, method, path, body, ctype):
    req = urllib.request.Request(base + path, data=body if method == "POST" else None,
                                 method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    req.add_header("User-Agent", "curl/8.5.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8", "replace"),
                    "headers": {k.lower(): v for k, v in resp.headers.items()
                                if k.lower() in ("content-type", "allow", "content-length")}}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode("utf-8", "replace"),
                "headers": {k.lower(): v for k, v in exc.headers.items()
                            if k.lower() in ("content-type", "allow", "content-length")}}
    except Exception as exc:  # noqa: BLE001
        return {"status": None, "body": f"TRANSPORT: {exc}", "headers": {}}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "knot_http_baseline.json"
    results = {}
    for name, method, path, body, ctype in CASES:
        results[name] = send(base, method, path, body, ctype)
        print(f"{name:22s} {results[name]['status']} {results[name]['body'][:70]!r}")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
