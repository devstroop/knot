#!/usr/bin/env python3
"""Held-out metric aggregation for OIO training experiments.

Aggregation and calibration math are pure Python so they can be unit-tested
without torch; the model forward pass takes torch as an explicit argument,
matching the rest of the training package.
"""

import math

PRIMITIVES = ("choice", "score", "noul")
ECE_BINS = 10


def ece(confidences, hits, bins=ECE_BINS):
    """Return the expected calibration error over equal-width confidence bins."""
    if len(confidences) != len(hits):
        raise ValueError("confidences and hits must have the same length")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not confidences:
        return None
    total = len(confidences)
    buckets = [[] for _ in range(bins)]
    for confidence, hit in zip(confidences, hits):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite probability")
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, bool(hit)))
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        mean_accuracy = sum(1.0 if hit else 0.0 for _, hit in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_confidence - mean_accuracy)
    return error


def _record(source, primitive, correct, confidence, brier, abs_error=None):
    if primitive not in PRIMITIVES:
        raise ValueError("unknown primitive %r" % (primitive,))
    return {
        "source": source,
        "primitive": primitive,
        "correct": bool(correct),
        "confidence": confidence,
        "brier": brier,
        "abs_error": abs_error,
    }


def summarize(records):
    """Aggregate per-case records into overall, per-primitive, per-source metrics."""
    records = list(records)
    if not records:
        raise ValueError("no evaluation records")
    required = {"source", "primitive", "correct", "confidence", "brier", "abs_error"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("record %d is malformed" % index)
        if record["primitive"] not in PRIMITIVES:
            raise ValueError("record %d has unknown primitive %r"
                             % (index, record["primitive"]))

    def accuracy_of(group):
        return sum(1 for record in group if record["correct"]) / len(group)

    overall = {
        "count": len(records),
        "accuracy": accuracy_of(records),
    }

    by_primitive = {}
    for primitive in PRIMITIVES:
        group = [r for r in records if r["primitive"] == primitive]
        if not group:
            continue
        summary = {
            "count": len(group),
            "accuracy": accuracy_of(group),
            "mean_confidence": sum(r["confidence"] for r in group) / len(group),
            "brier": sum(r["brier"] for r in group) / len(group),
            "ece": ece([r["confidence"] for r in group], [r["correct"] for r in group]),
        }
        if primitive == "score":
            errors = [r["abs_error"] for r in group if r["abs_error"] is not None]
            if errors:
                summary["mae"] = sum(errors) / len(errors)
        by_primitive[primitive] = summary

    by_source = {}
    for source in sorted({r["source"] for r in records}):
        group = [r for r in records if r["source"] == source]
        source_summary = {
            "count": len(group),
            "accuracy": accuracy_of(group),
            "by_primitive": {},
        }
        for primitive in PRIMITIVES:
            primitive_group = [r for r in group if r["primitive"] == primitive]
            if not primitive_group:
                continue
            source_summary["by_primitive"][primitive] = {
                "count": len(primitive_group),
                "accuracy": accuracy_of(primitive_group),
            }
        by_source[source] = source_summary

    return {
        "case_count": overall["count"],
        "accuracy": overall["accuracy"],
        "by_primitive": by_primitive,
        "by_source": by_source,
    }


def evaluate_model(model, items, tokenizer, batch_size, torch, progress_every=None):
    """Run the model over encoded items and return one metric record per case."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = []
    next_report = progress_every
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ids, attention, positions, mask, labels, qtypes = collate(
                chunk, tokenizer.pad_token_id, torch
            )
            logits, _ = model(ids, attention, positions, mask, qtypes)
            probabilities = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
            predictions = probabilities.argmax(-1).tolist()
            for row_index, item in enumerate(chunk):
                valid = mask[row_index]
                row = probabilities[row_index][valid]
                target = item["target_index"]
                one_hot = torch.zeros_like(row)
                one_hot[target] = 1.0
                brier = float(((row - one_hot) ** 2).sum().item())
                confidence = float(row.max().item())
                prediction = predictions[row_index]
                records.append(_record(
                    item["source"],
                    item["primitive"],
                    prediction == target,
                    confidence,
                    brier,
                    abs(prediction - target) if item["primitive"] == "score" else None,
                ))
            completed = start + len(chunk)
            if progress_every and completed >= next_report:
                print("evaluated %d/%d cases" % (completed, len(items)), flush=True)
                next_report = ((completed // progress_every) + 1) * progress_every
    model.train()
    model.encoder.eval()
    return records


def collate(batch, pad_id, torch):
    batch_size = len(batch)
    seq_len = max(len(item["ids"]) for item in batch)
    option_count = max(len(item["markers"]) for item in batch)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention = torch.zeros((batch_size, seq_len), dtype=torch.long)
    marker_pos = torch.zeros((batch_size, option_count), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, option_count), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    qtypes = torch.zeros(batch_size, dtype=torch.long)
    for index, item in enumerate(batch):
        length = len(item["ids"])
        input_ids[index, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[index, :length] = 1
        count = len(item["markers"])
        marker_pos[index, :count] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[index, :count] = True
        labels[index] = item["target_index"]
        qtypes[index] = item["qtype"]
    return input_ids, attention, marker_pos, marker_mask, labels, qtypes
