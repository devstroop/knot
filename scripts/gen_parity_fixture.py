"""Generate M2 parity fixture: Laya torch Agent vs ONNX path.

Usage:
    /path/to/laya-venv/bin/python scripts/gen_parity_fixture.py \
        --model /path/to/checkpoint --out crates/knot/tests/fixtures/parity_english.json
"""

import argparse
import json

import torch

from laya.agent import Agent
from laya.common import QTYPES, collate_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agent = Agent(args.model, compile=False, device="cpu")

    questions = {
        "dept": {
            "t": "choice",
            "ins": "which team handles this?",
            "crit": {"billing": "invoices, payments, refunds", "tech": "bugs, outages", "other": None},
        },
        "urgency": {
            "t": "score",
            "ins": "how urgent is this?",
            "crit": ["not urgent", "soon", "blocking"],
        },
        "churn": {"t": "noul", "ins": "does the user threaten to cancel?", "crit": {}},
    }
    state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel."

    internal = {qid: q for qid, q in questions.items()}
    items = agent._encode_state(state, list(internal.keys()), internal, max_len=512, head_max_len=192)
    batch = collate_items([items], agent.tok.pad_token_id)
    logits, act = agent.model(
        batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"]
    )

    fixture = {
        "items": [
            {"ids": it["ids"], "markers": it["markers"], "qtype": it["qtype"], "qid": qid}
            for it, qid in zip(items, internal.keys())
        ],
        "collated": {
            "input_ids": batch["input_ids"].tolist(),
            "attention_mask": batch["attention_mask"].tolist(),
            "marker_pos": batch["marker_pos"].tolist(),
            "marker_mask": batch["marker_mask"].tolist(),
            "qtype": batch["qtype"].tolist(),
        },
        "logits": logits.detach().cpu().tolist(),
        "act_logits": act.detach().cpu().tolist(),
        "temperatures": agent.temperature,
        "temperatures_by_options": agent.temperature_by_options,
        "pad_token_id": agent.tok.pad_token_id,
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
