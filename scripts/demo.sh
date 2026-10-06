#!/usr/bin/env bash
# One-command demonstration: prep the environment, then run the three-way
# comparison (Laya vs knot live, Jev recorded/published) and regenerate
# docs/DEMO.md.
#
#   scripts/demo.sh              # full run
#   scripts/demo.sh --limit 2    # smoke run (first 2 cases per corpus, 1 repeat)
#
# Idempotent: every expensive step checks for its artifact first.
# Environment: user-local only (uv-managed venv + interpreters, HuggingFace
# cache, the knot-cache checkpoints). Nothing touches the system Python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
VENV="$ROOT/.venv-demo"
LAYA="${DEMO_LAYA_CHECKOUT:-$ROOT/../laya}"
KNOT_CACHE="${DEMO_KNOT_CACHE:-/home/devstroop/knot-cache}"
HF_SNAP="${DEMO_HF_SNAP:-$HOME/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851}"
FIXTURE="$ROOT/demo/cases/massive_hi.json"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

step() { printf '\n== %s ==\n' "$*"; }

step "venv"
if [ ! -x "$VENV/bin/python" ]; then
    "$UV" venv "$VENV" --python 3.13
fi
if ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
    "$UV" pip install --python "$VENV/bin/python" \
        torch --index-url https://download.pytorch.org/whl/cpu
fi
if ! "$VENV/bin/python" -c "import laya, fastapi, onnxruntime, onnxscript" >/dev/null 2>&1; then
    "$UV" pip install --python "$VENV/bin/python" -e "$LAYA[serve,onnx]"
fi
"$VENV/bin/python" -c "import torch, laya, fastapi, onnxruntime; print('venv ok, torch', torch.__version__)"

step "huggingface snapshot (english + multilingual @ pinned revision)"
if [ ! -f "$HF_SNAP/model.safetensors" ] || [ ! -f "$HF_SNAP/multilingual/model.safetensors" ]; then
    "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snap = snapshot_download(
    "convaiinnovations/laya",
    revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    allow_patterns=[
        "*.json", "*.py", "*.safetensors",
        "tokenizer/*", "encoder/*",
        "multilingual/*.json", "multilingual/*.py", "multilingual/*.safetensors",
        "multilingual/tokenizer/*", "multilingual/encoder/*",
    ],
    ignore_patterns=["typed-decisions/*"],
)
print("snapshot:", snap)
PY
fi
[ -f "$HF_SNAP/model.safetensors" ] || { echo "error: snapshot missing at $HF_SNAP" >&2; exit 1; }
[ -f "$HF_SNAP/multilingual/model.safetensors" ] || { echo "error: multilingual missing at $HF_SNAP" >&2; exit 1; }

step "multilingual ONNX export (knot consumes ONNX; the hub ships torch only)"
if [ ! -f "$HF_SNAP/multilingual/laya.onnx" ]; then
    HF_HUB_OFFLINE=1 USE_TORCH=1 TOKENIZERS_PARALLELISM=false "$VENV/bin/python" - <<PY
import sys
sys.path.insert(0, "$LAYA")
from scripts.export_onnx import export_to_onnx
export_to_onnx("$HF_SNAP/multilingual", "$HF_SNAP/multilingual/laya.onnx")
PY
fi
ls -la "$HF_SNAP/multilingual/" | grep -E "laya\.onnx" || { echo "error: export produced no laya.onnx" >&2; exit 1; }

step "MASSIVE hi fixture (first 100 rows, seed-13 protocol source data)"
if [ ! -f "$FIXTURE" ]; then
    mkdir -p "$(dirname "$FIXTURE")"
    python3 - "$FIXTURE" <<'PY'
import gzip, hashlib, json, sys, urllib.request
out = sys.argv[1]
url = ("https://huggingface.co/datasets/mteb/amazon_massive_intent/"
       "resolve/main/test/hi.json.gz")
raw = urllib.request.urlopen(url, timeout=120).read()
rows = [json.loads(line) for line in gzip.decompress(raw).decode("utf-8").splitlines()]
labels = sorted({r["label_text"] for r in rows})
fixture = {
    "source": {
        "dataset": "mteb/amazon_massive_intent",
        "file": "test/hi.json.gz",
        "url": url,
        "sha256_of_gz": hashlib.sha256(raw).hexdigest(),
        "rows_total": len(rows),
        "labels": len(labels),
        "note": "First 100 rows verbatim (bench_local.py Part A, seed 13, n_options 20).",
    },
    "labels": labels,
    "rows": [{"text": r["text"], "label": r["label_text"]} for r in rows[:100]],
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"wrote {out}: {len(fixture['rows'])} rows, {len(labels)} labels")
PY
fi

step "knot (release)"
(cd "$ROOT" && cargo build --release -p knot-serve)

step "run demo${*:+ ($*)}"
mkdir -p "$ROOT/demo/logs"
"$VENV/bin/python" "$ROOT/scripts/demo_engines.py" \
    --out "$ROOT/demo" \
    --report "$ROOT/docs/DEMO.md" \
    "$@"
