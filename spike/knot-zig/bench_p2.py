#!/usr/bin/env python3
"""ADR-008 gate G3: one benchmark row — cold start, binary size, peak RSS,
throughput, latency — zig engine vs Rust knot, same workload.

Usage: bench_p2.py <zig_bin> <knot_bin> <model_dir> [request_json]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HEALTH = "/health"
PREDICT = "/v1/systemone"

def wait_health(base, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + HEALTH, timeout=1) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("no health")

def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1

def post(base, body):
    req = urllib.request.Request(
        base + PREDICT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def run(name, cmd, env, port, body, iters=30):
    e = {**os.environ, **env}
    p = subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        cold = wait_health(base)
        # warm
        post(base, body)
        lat0 = time.time()
        in_tok = 0
        for _ in range(iters):
            resp = json.loads(post(base, body))
            in_tok += resp["usage"]["input_tokens"]
        total = time.time() - lat0
        hwm = rss_kb(p.pid)
        return {
            "name": name,
            "cold_s": cold,
            "rss_mb": hwm / 1024.0,
            "ms_per_req": 1000.0 * total / iters,
            "tok_per_s": in_tok / total,
            "in_tok": in_tok // iters,
        }
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()

def main():
    zig_bin, knot_bin, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    req_path = sys.argv[4] if len(sys.argv) > 4 else None
    if req_path:
        req = json.load(open(req_path))
    else:
        req = {
            "state": "I was charged twice for the same order, please refund me. "
                     "The invoice number is 88213 and the amount was $49.99 on March 3.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What is the user asking for?",
                    "criteria": {
                        "billing": "refunds or charges",
                        "technical": "bugs or errors",
                        "sales": "pricing or purchase",
                    },
                },
                "urgent": {"type": "noul", "instructions": "Is the user angry?"},
            },
        }

    rows = []
    rows.append(run(
        "zig (release)",
        [zig_bin, "8133", model_dir],
        {},
        8133, req,
    ))
    rows.append(run(
        "rust knot (release)",
        [knot_bin],
        {"KNOT_MODEL_DIR": model_dir, "KNOT_PORT": "8133", "KNOT_RUNTIME": "onnx"},
        8133, req,
    ))

    sizes = {n: os.path.getsize(p) for n, p in (("zig", zig_bin), ("knot", knot_bin))}
    print(f"workload: {rows[0]['in_tok']} input tokens/request, {30} sequential requests")
    print(f"{'':22s} {'cold start':>10s} {'binary MB':>10s} {'peak RSS MB':>12s} {'ms/req':>9s} {'tok/s':>9s}")
    for r, sz in zip(rows, (sizes["zig"], sizes["knot"])):
        print(f"{r['name']:22s} {r['cold_s']:9.3f}s {sz/1e6:9.2f} {r['rss_mb']:11.1f} {r['ms_per_req']:8.1f} {r['tok_per_s']:8.0f}")
    z, k = rows
    print(f"zig/rust: cold ×{z['cold_s']/k['cold_s']:.2f} · size ×{sizes['zig']/sizes['knot']:.2f} · "
          f"rss ×{z['rss_mb']/k['rss_mb']:.2f} · latency ×{z['ms_per_req']/k['ms_per_req']:.2f} · "
          f"throughput ×{z['tok_per_s']/k['tok_per_s']:.2f}")

if __name__ == "__main__":
    main()
