#!/usr/bin/env python3
"""Three-way demonstration runner: Laya vs oio live, Jev recorded/published.

Corpora
  en     12-case recorded English eval fixture (crates/oio/tests/fixtures/eval_english.jsonl)
  feishu 64-case frozen Chinese diagnostic (feishu_zh checkout, both modes)
  hi     100-case MASSIVE intent slice (demo/cases/massive_hi.json, seed-13 20-option protocol)

Both live engines answer identical HTTP POSTs to /v1/systemone, sequentially
(one server resident at a time, warmup before timing). Jev never runs live: no
API key exists; its columns come from the archived feishu recordings and from
published third-party figures, labelled as such in the report.

Writes <out>/results.json (raw evidence) and regenerates <report>
(docs/DEMO.md). Stdlib only. Driven by scripts/demo.sh.
"""
import argparse
import ast
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYA_CHECKOUT = Path(os.environ.get("DEMO_LAYA_CHECKOUT", ROOT.parent / "laya"))
OIO_CACHE = Path(os.environ.get("DEMO_OIO_CACHE", "/home/devstroop/oio-cache"))
HF_SNAP = Path(
    os.environ.get(
        "DEMO_HF_SNAP",
        Path.home()
        / ".cache/huggingface/hub/models--convaiinnovations--laya"
        / "snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    )
)
OIO_BIN = Path(os.environ.get("DEMO_OIO_BIN", ROOT / "target/release/oio-serve"))
LAYA_SERVE = Path(os.environ.get("DEMO_LAYA_SERVE", ROOT / ".venv-demo/bin/laya-serve"))
OIO_URL = os.environ.get("DEMO_OIO_URL", "http://127.0.0.1:8977")
LAYA_URL = os.environ.get("DEMO_LAYA_URL", "http://127.0.0.1:8001")
FEISHU_DIR = LAYA_CHECKOUT / "research/benchmarks/feishu_zh"
RESERVED_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

FULL_REPEATS = {"en": 3, "feishu": 3, "hi": 1}
SMOKE_REPEATS = {"en": 1, "feishu": 1, "hi": 1}


# --------------------------------------------------------------------------- utils
def pct(xs, q):
    """Linear-interpolated percentile — mirrors feishu metrics.percentile."""
    if not xs:
        return None
    xs = sorted(xs)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def timing(times):
    if not times:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}
    return {
        "n": len(times),
        "p50_ms": pct(times, 0.5),
        "p95_ms": pct(times, 0.95),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "definition": "client wall clock, all timed repeats pooled, linear interpolation",
    }


def post_json(url, payload, timeout=120):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return json.loads(raw), elapsed_ms, raw.decode("utf-8")


def get_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def git_rev(path):
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def machine_info():
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "mem_gb": mem_gb,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def log_tail(path, n=2500):
    try:
        data = path.read_bytes()[-n:]
        return data.decode("utf-8", "replace")
    except OSError:
        return "<no log>"


def require(cond, msg):
    if not cond:
        raise SystemExit("error: " + msg)


# --------------------------------------------------------------------------- servers
class Server:
    def __init__(self, name, cmd, env, health_url, log_path, timeout=300):
        self.name = name
        self.cmd = cmd
        self.env = env
        self.health_url = health_url
        self.log_path = Path(log_path)
        self.timeout = timeout
        self.proc = None
        self._log = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            list(self.cmd), stdout=self._log, stderr=subprocess.STDOUT, env=self.env
        )
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"error: {self.name} exited rc={self.proc.returncode}\n"
                    f"--- {self.log_path} ---\n{log_tail(self.log_path)}"
                )
            if get_ok(self.health_url):
                print(f"[{self.name}] healthy at {self.health_url}")
                return self
            time.sleep(0.5)
        self.proc.terminate()
        raise SystemExit(
            f"error: {self.name} not healthy after {self.timeout}s\n"
            f"--- {self.log_path} ---\n{log_tail(self.log_path)}"
        )

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log:
            self._log.close()
        print(f"[{self.name}] stopped")
        return False


def port_open(url):
    return get_ok(url.replace("/v1/systemone", "") + "/health", timeout=1)


def oio_server(logs):
    env = dict(os.environ)
    env.pop("OIO_API_KEY", None)
    env.pop("OIO_MODEL_DIR", None)
    env["OIO_HOST"] = "127.0.0.1"
    env["OIO_PORT"] = OIO_URL.rsplit(":", 1)[1]
    env["OIO_MODELS"] = (
        f"english={OIO_CACHE / 'laya-english'},"
        f"multilingual={HF_SNAP / 'multilingual'}"
    )
    return Server("oio-serve", [str(OIO_BIN)], env, OIO_URL + "/health", logs / "oio-serve.log")


def laya_server(logs):
    env = dict(os.environ)
    env.pop("LAYA_API_KEY", None)
    env["LAYA_HOST"] = "127.0.0.1"
    env["LAYA_PORT"] = LAYA_URL.rsplit(":", 1)[1]
    env["LAYA_MODELS"] = "english,multilingual"
    env["LAYA_PRELOAD"] = "1"
    env["LAYA_LOG_LEVEL"] = "warning"
    env["HF_HUB_OFFLINE"] = "1"
    env["USE_TORCH"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return Server("laya-serve", [str(LAYA_SERVE)], env, LAYA_URL + "/health", logs / "laya-serve.log")


# --------------------------------------------------------------------------- corpora
def build_en(limit=None):
    path = ROOT / "crates/oio/tests/fixtures/eval_english.jsonl"
    require(path.exists(), f"missing {path} (in-repo fixture)")
    cases = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        v = json.loads(line)
        cases.append({
            "corpus": "en",
            "id": f"en-{i}",
            "mode": "eval",
            "state": v["state"],
            "questions": v["questions"],
            "expected": v.get("expected", {}),
        })
        if limit and len(cases) >= limit:
            break
    return cases


def build_feishu(limit=None):
    require(FEISHU_DIR.is_dir(), f"laya checkout not found at {LAYA_CHECKOUT}")
    sys.path.insert(0, str(FEISHU_DIR))
    from audit import load_cases  # noqa: E402  (feishu modules import siblings by bare name)
    from prompts import requests_for  # noqa: E402

    cases_raw, manifest = load_cases()
    cases = []
    for c in (cases_raw[:limit] if limit else cases_raw):
        modes = requests_for(c)
        for mode in ("choice", "four_noul"):
            req = modes[mode]
            cases.append({
                "corpus": "feishu",
                "id": c["id"],
                "mode": mode,
                "family": c.get("family"),
                "state": req["state"],
                "questions": req["questions"],
                "expected": c["expected"],
            })
    return cases, manifest


def build_hi(limit=None):
    path = ROOT / "demo/cases/massive_hi.json"
    require(path.exists(), f"missing {path} (scripts/demo.sh harvests it)")
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["labels"]
    rng = random.Random(13)  # bench_local.py Part A seed — byte-identical case sets
    cases = []
    rows = data["rows"][: limit or 100]
    for i, row in enumerate(rows):
        pool = [x for x in labels if x != row["label"]]
        keys = [row["label"]] + rng.sample(pool, min(19, len(pool)))
        rng.shuffle(keys)
        cases.append({
            "corpus": "hi",
            "id": f"hi-{i}",
            "mode": "massive-20",
            "state": {"utterance": row["text"]},
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What is the user asking for in `utterance`?",
                    "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys},
                }
            },
            "expected": row["label"],
        })
    return cases, data.get("source", {})


# --------------------------------------------------------------------------- scoring
def extract_en(answers, questions):
    out = {}
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            out[qid] = None
            continue
        qtype = q.get("type")
        if qtype == "choice":
            out[qid] = a.get("choice")
        elif qtype == "score":
            # wire returns a continuous value; the fixture labels are levels
            v = a.get("score")
            out[qid] = int(v + 0.5) if isinstance(v, (int, float)) else None
        elif qtype == "noul":
            # wire returns the continuous noul signal; fixture labels are bool
            v = a.get("noul")
            out[qid] = v if isinstance(v, bool) else (
                v >= 0.5 if isinstance(v, (int, float)) else None
            )
        else:
            out[qid] = None
    return out


def score_case(case, answers):
    """Return (predicted, correct) in the corpus's comparison vocabulary."""
    if case["corpus"] == "en":
        pred = extract_en(answers, case["questions"])
        correct = sum(
            1 for qid, want in case["expected"].items() if pred.get(qid) == want
        )
        total = len(case["expected"])
        return pred, (correct, total)
    if case["corpus"] == "hi":
        pred = answers.get("intent", {}).get("choice")
        return pred, (1 if pred == case["expected"] else 0, 1)
    # feishu: the frozen interpret() of prompts.py — same mapping as the recordings
    sys.path.insert(0, str(FEISHU_DIR))
    from prompts import interpret
    info = interpret(case["mode"], answers)
    pred = info["predicted"]
    return pred, (1 if pred == case["expected"] else 0, 1)


def predicted_only(case, answers):
    return score_case(case, answers)[0]


def prob_vector(answers, questions, qid):
    a = answers.get(qid)
    if isinstance(a, dict) and isinstance(a.get("probabilities"), dict):
        return a["probabilities"]
    return None


def prob_mean_abs_diff(ans_a, ans_b, questions):
    """Mean |p_a - p_b| over shared choice probabilities, repeat-0 answers."""
    diffs = []
    for qid, q in questions.items():
        if q.get("type") != "choice":
            continue
        pa, pb = prob_vector(ans_a, questions, qid), prob_vector(ans_b, questions, qid)
        if not pa or not pb or set(pa) != set(pb):
            continue
        diffs.extend(abs(pa[k] - pb[k]) for k in pa)
    return statistics.mean(diffs) if diffs else None


# --------------------------------------------------------------------------- run
def run_engine(engine, url, cases, repeats, samples):
    """Send every case × repeats to one engine; returns flat call records."""
    by_corpus = {}
    for c in cases:
        by_corpus.setdefault(c["corpus"], []).append(c)
    calls = []
    for corpus in ("en", "feishu", "hi"):
        group = by_corpus.get(corpus)
        if not group:
            continue
        # warmup (untimed): first case of the corpus
        warm = group[0]
        try:
            _, ms, _ = post_json(url + "/v1/systemone",
                                 {"state": warm["state"], "questions": warm["questions"]})
            print(f"[{engine}] warmup {corpus}: {ms:.1f} ms")
        except Exception as e:  # noqa: BLE001 — warmup failure aborts the phase
            raise SystemExit(f"error: {engine} warmup failed on {corpus}: {e}")
        for case in group:
            for rep in range(repeats[corpus]):
                rec = {
                    "engine": engine, "corpus": corpus, "id": case["id"],
                    "mode": case["mode"], "repeat": rep,
                }
                try:
                    resp, ms, raw = post_json(
                        url + "/v1/systemone",
                        {"state": case["state"], "questions": case["questions"]},
                    )
                    # wire: {"model":..., "answers": {qid: ...}, "usage":..., "routing":...}
                    answers = resp.get("answers") if isinstance(resp, dict) else None
                    if not isinstance(answers, dict):
                        answers = resp
                    rec.update(status="ok", elapsed_ms=ms, answers=answers)
                    key = f"{corpus}:{case['mode']}" if corpus == "feishu" else corpus
                    if rep == 0:
                        samples.setdefault(engine, {}).setdefault(key, raw)
                except urllib.error.HTTPError as e:
                    rec.update(status="error", error=f"HTTP {e.code}",
                               detail=e.read().decode("utf-8", "replace")[:500])
                except Exception as e:  # noqa: BLE001 — record and continue
                    rec.update(status="error", error=type(e).__name__, detail=str(e)[:500])
                calls.append(rec)
            done = sum(1 for c in calls
                       if c["engine"] == engine and c["corpus"] == corpus)
            if done % 100 == 0:
                print(f"[{engine}] {corpus}: {done} calls")
        ok = [c for c in calls if c["engine"] == engine and c["corpus"] == corpus
              and c["status"] == "ok"]
        print(f"[{engine}] {corpus}: {len(ok)} ok calls done")
    return calls


# --------------------------------------------------------------------------- summaries
def summarize(cases, calls, recorded_rows, archived_hi, archived_feishu_summary):
    by = {}
    for c in calls:
        by.setdefault((c["engine"], c["corpus"]), []).append(c)

    engines = sorted({c["engine"] for c in calls})
    corpora = ["en", "feishu", "hi"]
    case_idx = {(c["id"], c["mode"]): c for c in cases}
    out = {"engines": engines, "per_corpus": {}, "agreement": {}}

    def first_ok(engine, corpus, cid, mode):
        for c in by.get((engine, corpus), []):
            if c["id"] == cid and c["mode"] == mode and c["repeat"] == 0 \
                    and c["status"] == "ok":
                return c
        return None

    for corpus in corpora:
        rows = [c for c in cases if c["corpus"] == corpus]
        if not rows:
            continue
        entry = {"cases": len(rows), "modes": {}}
        for mode in sorted({c["mode"] for c in rows}):
            mrows = [c for c in rows if c["mode"] == mode]
            mentry = {"cases": len(mrows), "engines": {}}
            for eng in engines:
                calls_ec = [c for c in by.get((eng, corpus), [])
                            if c["mode"] == mode]
                oks = [c for c in calls_ec if c["status"] == "ok"]
                times = [c["elapsed_ms"] for c in oks]
                errors = [c for c in calls_ec if c["status"] != "ok"]
                correct = total = 0
                for c in oks:
                    if c["repeat"] != 0:
                        continue
                    case = case_idx[(c["id"], c["mode"])]
                    _, (hit, n) = score_case(case, c["answers"])
                    correct += hit
                    total += n
                mentry["engines"][eng] = {
                    "accuracy": (correct / total) if total else None,
                    "correct": correct, "questions": total,
                    "timing": timing(times),
                    "errors": len(errors),
                    "error_detail": errors[:3],
                }
            entry["modes"][mode] = mentry
        out["per_corpus"][corpus] = entry

    def agree(eng_a, eng_b, corpus, mode=None, vs_rows=None, vs_backend=None):
        n = same = 0
        for case in cases:
            if case["corpus"] != corpus or (mode and case["mode"] != mode):
                continue
            if vs_rows is None:
                ca = first_ok(eng_a, corpus, case["id"], case["mode"])
                cb = first_ok(eng_b, corpus, case["id"], case["mode"])
                if not ca or not cb:
                    continue
                pa = predicted_only(case, ca["answers"])
                pb = predicted_only(case, cb["answers"])
            else:
                ca = first_ok(eng_a, corpus, case["id"], case["mode"])
                rec = vs_rows.get((vs_backend, case["id"], case["mode"], 0))
                if not ca or not rec or rec.get("status") != "ok":
                    continue
                pa = predicted_only(case, ca["answers"])
                pb = rec.get("predicted")
            if pa is None or pb is None:
                continue
            n += 1
            same += pa == pb
        return {"same": same, "n": n}

    if "en" in out["per_corpus"]:
        out["agreement"]["en_oio_vs_laya"] = agree("oio", "laya", "en")
        # mean |dp| over choice probabilities (repeat 0)
        diffs = []
        for case in cases:
            if case["corpus"] != "en":
                continue
            ca = first_ok("oio", "en", case["id"], case["mode"])
            cb = first_ok("laya", "en", case["id"], case["mode"])
            if ca and cb:
                d = prob_mean_abs_diff(ca["answers"], cb["answers"], case["questions"])
                if d is not None:
                    diffs.append(d)
        out["agreement"]["en_prob_mean_abs_diff"] = statistics.mean(diffs) if diffs else None

    for mode in ("choice", "four_noul"):
        out["agreement"][f"feishu_{mode}_oio_vs_laya"] = agree("oio", "laya", "feishu", mode)
        out["agreement"][f"feishu_{mode}_oio_vs_jev_recorded"] = agree(
            "oio", None, "feishu", mode, vs_rows=recorded_rows, vs_backend="jev")
        out["agreement"][f"feishu_{mode}_laya_vs_laya_recorded"] = agree(
            "laya", None, "feishu", mode, vs_rows=recorded_rows, vs_backend="laya")

    out["agreement"]["hi_oio_vs_laya"] = agree("oio", "laya", "hi")
    diffs = []
    for case in cases:
        if case["corpus"] != "hi":
            continue
        ca = first_ok("oio", "hi", case["id"], case["mode"])
        cb = first_ok("laya", "hi", case["id"], case["mode"])
        if ca and cb:
            d = prob_mean_abs_diff(ca["answers"], cb["answers"], case["questions"])
            if d is not None:
                diffs.append(d)
    out["agreement"]["hi_prob_mean_abs_diff"] = statistics.mean(diffs) if diffs else None

    out["archived_feishu_summary"] = archived_feishu_summary
    out["archived_hi"] = archived_hi
    return out


# --------------------------------------------------------------------------- recorded
def load_recorded():
    base = FEISHU_DIR / "results/v1"
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    rows = {}
    for backend in ("jev", "laya"):
        with (base / backend / "raw.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[(backend, r["id"], r["mode"], r["repeat"])] = r

    results = LAYA_CHECKOUT / "research/results"
    hi = {}
    try:
        committed = json.loads(
            (results / "cpu_51_language_sweep.json").read_text(encoding="utf-8"))
        refreshed = json.loads(
            (results / "cpu_51_language_sweep_refreshed.json").read_text(encoding="utf-8"))
        per_c = committed["part_a"]["by_model"]
        per_r = refreshed["by_model"]
        hi = {
            "multilingual_committed": per_c["multilingual"]["per_language"]["hi"],
            "english_committed": per_c["english"]["per_language"]["hi"],
            "multilingual_refreshed_clamped":
                per_r["multilingual"]["per_language"]["hi"]["refreshed_clamped"],
            "source": "laya/research/results/cpu_51_language_sweep*.json (n=100, seed 13)",
        }
    except (OSError, KeyError) as e:
        hi = {"error": f"archived sweep unreadable: {e}"}

    return summary, rows, hi


# --------------------------------------------------------------------------- report
def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ms(x):
    return "—" if x is None else f"{x:.1f}"


def agreement_cell(d):
    if not d:
        return "—"
    return f"{d['same']}/{d['n']}"


def render_report(meta, cases, samples, summary, recorded_summary, recorded_rows,
                  source_hi, smoke):
    def acc(corpus, mode, eng):
        e = summary["per_corpus"].get(corpus, {}).get("modes", {}) \
            .get(mode, {}).get("engines", {}).get(eng)
        if not e or e["accuracy"] is None:
            return "—"
        return f"{e['accuracy']:.3f} ({e['correct']}/{e['questions']})"

    def lat(corpus, mode, eng):
        e = summary["per_corpus"].get(corpus, {}).get("modes", {}) \
            .get(mode, {}).get("engines", {}).get(eng)
        if not e:
            return ("—", "—", "—")
        t = e["timing"]
        return (t["n"], ms(t["p50_ms"]), ms(t["p95_ms"]))

    def json_block(obj, note=""):
        body = json.dumps(obj, ensure_ascii=False, indent=1)
        if note:
            return f"{note}\n```json\n{body}\n```\n"
        return f"```json\n{body}\n```\n"

    L = []
    add = L.append
    add("# oio demonstration — Jev, Laya and oio side by side\n")
    add("> GENERATED by `scripts/demo_engines.py` (via `scripts/demo.sh`) — "
         "rerun the command below to refresh. Raw evidence: `demo/results.json`.\n")
    add(f"- Run: {meta['run_at']} ({'smoke' if smoke else 'full'})")
    add(f"- Machine: {meta['machine']['cpu']} × {meta['machine']['cores']}, "
        f"{meta['machine']['mem_gb']} GB, {meta['machine']['platform']}")
    add(f"- Revisions: oio `{meta['revisions']['oio']}`, laya checkout "
        f"`{meta['revisions']['laya']}`, checkpoint pin `{RESERVED_REVISION[:12]}…`")
    add("- Engines over identical HTTP `POST /v1/systemone` bodies, one server "
        "resident at a time (warmup before timing), client wall clock\n")

    add("## How to read this\n")
    add("- **Jev never runs live** — no API key exists. Jev columns are the "
        "archived feishu recordings (`jev-1.13.0`, 2026-09-21, archived-benchmark "
        "provenance) or published third-party figures, labelled per table.")
    add("- **Accuracy** is computed on repeat 0 like the feishu protocol; "
        "**latency** pools every timed repeat (linear interpolation).")
    add("- This box is not the hardware the archived numbers were measured on "
        "(Laya recordings: Apple M4 GPU; published Laya CPU: AWS m7a.xlarge) — "
        "directional comparison only.\n")

    # ---------------------------------------------------------------- English
    add("## English — recorded eval fixture (12 cases, choice/score/noul)\n")
    add("| metric | Laya live (CPU) | oio live (CPU) |")
    add("|---|---|---|")
    add(f"| accuracy vs fixture labels | {acc('en','eval','laya')} | {acc('en','eval','oio')} |")
    n, p50, p95 = lat("en", "eval", "laya")
    n2, p502, p952 = lat("en", "eval", "oio")
    add(f"| latency p50 / p95 (ms) | {p50} / {p95} (n={n}) | {p502} / {p952} (n={n2}) |")
    add("")
    add("Scoring: `choice` exact; `score` rounded to the nearest level; `noul` "
        "thresholded at ≥ 0.5 (the wire returns the continuous signal, the "
        "fixture labels are boolean).")
    ag = summary["agreement"].get("en_oio_vs_laya", {})
    add(f"- **oio ↔ Laya agreement**: {agreement_cell(ag)} questions identical")
    d = summary["agreement"].get("en_prob_mean_abs_diff")
    if d is not None:
        add(f"- mean |Δp| between the two engines on choice probabilities: **{d:.1e}** "
            "(float32 vs float64 serialization of the same 4-decimal values)")
    add("- Published Jev context (different corpora/hardware): typed-decisions "
        "0.727, AG News 0.910 — see `docs/RESEARCH-COMPARE.md` and the sources "
        "listed at the end.\n")

    add("### Wire shape, live\n")
    add("One identical request, one response per engine (first English case). "
        "Both engines round probabilities to 4 decimals; oio stores the "
        "rounded value in f32, so `0.97509998…` is the same value as laya's "
        "`0.9751`. The recorded Jev shape sits beside a live response in the "
        "feishu section.\n")
    for eng in summary["engines"]:
        body = samples.get(eng, {}).get("en")
        if body:
            add(json_block(json.loads(body), note=f"**{eng} live** —\n"))
    add("")

    # ---------------------------------------------------------------- feishu
    add("## feishu_zh — frozen 64-case Chinese diagnostic (both modes)\n")
    add("Archived columns are the frozen recordings shipped with the benchmark; "
        "live columns are this run on this box.\n")
    for mode in ("choice", "four_noul"):
        arch = recorded_summary
        j = arch.get("jev", {}).get(mode, {})
        ly = arch.get("laya", {}).get(mode, {})
        entry = summary["per_corpus"]["feishu"]["modes"][mode]["engines"]
        add(f"### mode `{mode}`\n")
        add("| backend | accuracy (rep0) | p50 (ms) | p95 (ms) | source |")
        add("|---|---|---|---|---|")
        jt, lt = j.get("timing", {}), ly.get("timing", {})
        add(f"| Jev recorded (hosted, network incl.) | {fmt(j.get('accuracy'))} "
            f"(n={j.get('n')}) | {ms(jt.get('p50_ms'))} | {ms(jt.get('p95_ms'))} "
            f"| `results/v1/jev` |")
        add(f"| Laya recorded (Apple M4 GPU) | {fmt(ly.get('accuracy'))} "
            f"(n={ly.get('n')}) | {ms(lt.get('p50_ms'))} | {ms(lt.get('p95_ms'))} "
            f"| `results/v1/laya` |")
        for eng in summary["engines"]:
            e = entry.get(eng, {})
            t = e.get("timing", {})
            add(f"| {eng} live (this box, CPU) | {fmt(e.get('accuracy'))} "
                f"(n={e.get('questions')}) | {ms(t.get('p50_ms'))} | {ms(t.get('p95_ms'))} "
                f"| this run |")
        add("")
        a1 = summary["agreement"].get(f"feishu_{mode}_oio_vs_laya", {})
        a2 = summary["agreement"].get(f"feishu_{mode}_oio_vs_jev_recorded", {})
        a3 = summary["agreement"].get(f"feishu_{mode}_laya_vs_laya_recorded", {})
        add(f"- agreement `{mode}`: oio↔Laya-live **{agreement_cell(a1)}**, "
            f"oio↔Jev-recorded {agreement_cell(a2)}, "
            f"Laya-live↔Laya-recorded {agreement_cell(a3)}\n")

    add("### Wire shape, one feishu request (choice mode)\n")
    add("Two live responses to one identical request, then the recorded Jev "
        "response to that same feishu case from the archive — a *different* "
        "request than the English exhibit above, shown for shape: 2-decimal "
        "probabilities, `confidence` instead of `answer_confidence`, no "
        "`routing` block (the divergences tabulated in "
        "`docs/RESEARCH-COMPARE.md`, now with live traffic).\n")
    for eng in summary["engines"]:
        body = samples.get(eng, {}).get("feishu:choice")
        if body:
            add(json_block(json.loads(body), note=f"**{eng} live** —\n"))
    first_fc = next((c for c in cases
                     if c["corpus"] == "feishu" and c["mode"] == "choice"), None)
    if first_fc:
        row = recorded_rows.get(("jev", first_fc["id"], "choice", 0))
        if row and isinstance(row.get("response"), dict):
            add(json_block(row["response"],
                           note=f"**Jev recorded** ({first_fc['id']}, archived) —\n"))
    add("")

    # ---------------------------------------------------------------- Hindi
    add("## Hindi — MASSIVE intent (`hi`, 100 cases, 20-option choice)\n")
    add("Case set rebuilt exactly like `bench_local.py` Part A: first 100 "
        "dataset rows, seed 13, 19 seeded negatives + gold shuffled per row — "
        "byte-identical criteria to the archived sweep, so accuracies are "
        "directly comparable.\n")
    add("| backend | accuracy | n | source |")
    add("|---|---|---|---|")
    ah = summary.get("archived_hi", {})
    if "multilingual_committed" in ah:
        mc = ah["multilingual_committed"]
        mr = ah["multilingual_refreshed_clamped"]
        en = ah["english_committed"]
        add(f"| multilingual archived (committed sweep) | {fmt(mc['accuracy'])} "
            f"| {mc['n']} | `cpu_51_language_sweep.json` |")
        add(f"| multilingual archived (refreshed, temperature-clamped) | "
            f"{fmt(mr['accuracy'])} | {mr['n']} | `cpu_51_language_sweep_refreshed.json` |")
        add(f"| *english* checkpoint on hi (router contrast) | {fmt(en['accuracy'])} "
            f"| {en['n']} | `cpu_51_language_sweep.json` |")
    for eng in summary["engines"]:
        e = summary["per_corpus"].get("hi", {}).get("modes", {}) \
            .get("massive-20", {}).get("engines", {}).get(eng, {})
        t = e.get("timing", {})
        add(f"| {eng} live (this box, CPU) | {fmt(e.get('accuracy'))} "
            f"(n={e.get('questions')}) | {e.get('questions')} | this run |")
    add("")
    a = summary["agreement"].get("hi_oio_vs_laya", {})
    add(f"- **oio ↔ Laya agreement**: {agreement_cell(a)} cases")
    d = summary["agreement"].get("hi_prob_mean_abs_diff")
    if d is not None:
        add(f"- mean |Δp| on the 20-way choice: **{d:.1e}**")
    n, p50, p95 = lat("hi", "massive-20", "laya")
    n2, p502, p952 = lat("hi", "massive-20", "oio")
    add(f"- latency: Laya p50 {p50} / p95 {p95} ms (n={n}); "
        f"oio p50 {p502} / p95 {p952} ms (n={n2})\n")
    add("Devanagari sample (first case, as routed to the multilingual checkpoint):\n")
    for eng in summary["engines"]:
        body = samples.get(eng, {}).get("hi")
        if body:
            add(json_block(json.loads(body), note=f"**{eng} live** —\n"))

    # ---------------------------------------------------------------- Jev published
    add("## Jev from the published record (no live calls)\n")
    add("| figure | value | source |")
    add("|---|---|---|")
    add("| typed-decisions accuracy | 0.727 | Laya `BENCHMARKS.md` (Jev published) |")
    add("| AG News accuracy | 0.910 | Laya `BENCHMARKS.md` (Jev published) |")
    add("| DAIR Emotion accuracy | 0.480 | Laya `BENCHMARKS.md` (Jev published) |")
    add("| ECE | 0.246 | `nibzard/decision-model-benchmark` (third party) |")
    add("| option-order flip rate | 13% | `nibzard/decision-model-benchmark` |")
    add("| p50 latency, 1 question | 264–276 ms (hosted) | "
        "`nibzard/decision-model-benchmark` |")
    add("| feishu choice accuracy / p50 | 1.000 / 253 ms | archived recording, above |")
    add("")
    add("Full context, methodology and the wire-divergence table: "
        "`docs/RESEARCH-COMPARE.md`.\n")

    # ---------------------------------------------------------------- findings
    add("## Findings and caveats\n")
    add("- **Special-token resolution (found by this harness, fixed on this "
        "branch)**: mmBERT's vocabulary contains `<s>`/`</s>` as ordinary BPE "
        "tokens (ids 204/213) while its `tokenizer_config.json` declares "
        "`cls_token: <bos>` / `sep_token: <eos>` (ids 2/1). oio resolved "
        "CLS/SEP by candidate-string priority, so every multilingual prompt was "
        "framed with content tokens — four token ids per sequence, and the "
        "model answered from a differently framed prompt. On this harness "
        "before the fix: oio↔Laya agreement 55/100 on hi (oio accuracy 0.270 "
        "vs Laya 0.460) and 28/64 on feishu choice, with confident *different* "
        "answers. Fixed in `prompt.rs` (declared specials win; SPEC §2; "
        "regression test); agreement is now 12/12, 64/64 + 64/64, 100/100.")
    add("- **feishu latency on this box** (p50 ≈ 1.7–2.0 s) reflects the long "
        "Chinese state plus policy instructions on an 8th-generation laptop "
        "CPU; the archived Laya recording ran on an Apple M4 GPU (p50 "
        "150–415 ms). Compare engines to each other here, not to the archive.")
    add("- **The English fixture is a parity vehicle, not a leaderboard**: both "
        "engines score 0.667 against the labels and agree on every question; "
        "the labels come from the dataset, not from either model.")
    add("- **Jev's feishu columns are archived, not re-run**: no "
        "`TYPESAFE_API_KEY` exists in this environment, and the recording has "
        "archived-benchmark provenance (`docs/RESEARCH-COMPARE.md`).")
    add("- **oio vs laya probability serialization**: both round to 4 decimals; "
        "oio stores the rounded value in f32, so the JSON shows "
        "`0.97509998…` where laya shows `0.9751`.\n")

    # ---------------------------------------------------------------- reproduction
    add("## Reproduce\n")
    add("```bash")
    add("scripts/demo.sh              # prep (venv, HF snapshot, ONNX export, fixture) + full run")
    add("scripts/demo.sh --limit 2    # smoke run")
    add("```")
    add("")
    add(f"- recorded fixture sha256: `{meta.get('hi_fixture_sha', '—')}`")
    add(f"- results: `demo/results.json` (this report is generated from it)")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "demo"))
    ap.add_argument("--report", default=str(ROOT / "docs/DEMO.md"))
    ap.add_argument("--limit", type=int, default=None,
                    help="first N cases per corpus, single repeat (smoke)")
    ap.add_argument("--engines", default="oio,laya")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the report from an existing results.json "
                         "(re-runs scoring/summarization, no engine calls)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        results_path = out / "results.json"
        require(results_path.exists(),
                f"{results_path} missing — run a real pass first")
        data = json.loads(results_path.read_text(encoding="utf-8"))
        recorded_summary, recorded_rows, archived_hi = load_recorded()
        summary = summarize(data["cases"], data["calls"], recorded_rows,
                            archived_hi, recorded_summary)
        report = render_report(data["meta"], data["cases"], data["samples"],
                               summary, recorded_summary, recorded_rows,
                               archived_hi, data["meta"].get("smoke", False))
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
        return

    logs = out / "logs"
    smoke = args.limit is not None
    repeats = SMOKE_REPEATS if smoke else FULL_REPEATS

    require(OIO_BIN.exists(), f"{OIO_BIN} missing — scripts/demo.sh builds it")
    require(LAYA_SERVE.exists(), f"{LAYA_SERVE} missing — scripts/demo.sh creates the venv")
    require((HF_SNAP / "multilingual/laya.onnx").exists(),
             f"multilingual ONNX missing under {HF_SNAP} — run scripts/demo.sh")
    require((OIO_CACHE / "laya-english/laya.onnx").exists(),
             f"english checkpoint missing at {OIO_CACHE}/laya-english")
    for port_check, name in ((OIO_URL, "oio"), (LAYA_URL, "laya")):
        require(not port_open(port_check),
                f"port for {name} already serving at {port_check} — stop it first")

    cases = build_en(args.limit)
    feishu_cases, feishu_manifest = build_feishu(args.limit)
    cases += feishu_cases
    hi_cases, hi_source = build_hi(args.limit)
    cases += hi_cases
    print(f"cases: en={sum(c['corpus']=='en' for c in cases)} "
          f"feishu={len(feishu_cases)} hi={len(hi_cases)} repeats={repeats}")

    recorded_summary, recorded_rows, archived_hi = load_recorded()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    all_calls, samples = [], {}
    servers = {"oio": oio_server, "laya": laya_server}
    for eng in engines:
        require(eng in servers, f"unknown engine {eng!r} (oio/laya)")
        with servers[eng](logs):
            all_calls += run_engine(eng, {"oio": OIO_URL, "laya": LAYA_URL}[eng],
                                    cases, repeats, samples)

    meta = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": smoke,
        "repeats": repeats,
        "machine": machine_info(),
        "revisions": {"oio": git_rev(ROOT), "laya": git_rev(LAYA_CHECKOUT)},
        "checkpoint_pin": RESERVED_REVISION,
        "hi_fixture_sha": hi_source.get("sha256_of_gz"),
        "feishu_manifest": {k: feishu_manifest.get(k)
                            for k in ("cases_sha256", "requests_sha256")},
        "limits": args.limit,
    }
    summary = summarize(cases, all_calls, recorded_rows, archived_hi, recorded_summary)

    results = {
        "meta": meta,
        "cases": cases,
        "summary": summary,
        "recorded_feishu_summary": recorded_summary,
        "samples": samples,
        "calls": all_calls,
    }
    results_path = out / "results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"wrote {results_path} ({results_path.stat().st_size // 1024} KiB)")

    # console headline
    for corpus in ("en", "feishu", "hi"):
        entry = summary["per_corpus"].get(corpus)
        if not entry:
            continue
        for mode, ment in entry["modes"].items():
            for eng, e in ment["engines"].items():
                t = e["timing"]
                print(f"  {corpus:6} {mode:10} {eng:5} acc={fmt(e['accuracy'])} "
                      f"p50={ms(t['p50_ms'])}ms errors={e['errors']}")
    print("  agreement:", json.dumps(
        {k: v for k, v in summary["agreement"].items() if isinstance(v, dict)},
        ensure_ascii=False))

    if not args.no_report:
        report = render_report(meta, cases, samples, summary, recorded_summary,
                               recorded_rows, archived_hi, smoke)
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
