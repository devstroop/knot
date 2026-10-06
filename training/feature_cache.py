#!/usr/bin/env python3
"""Frozen-encoder feature cache: compute encoder hidden states once per split.

Stage-one training freezes the ModernBERT encoder, so its output for a case
never changes. Storing it once avoids re-running the encoder on every
optimizer step, which is what makes a full-corpus CPU epoch affordable.

The cache holds only encoder outputs, before the decision head's type
embedding, so the head keeps training normally on top of cached features.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.eval_metrics import collate
else:
    from .eval_metrics import collate

SCHEMA_VERSION = 1
DTYPES = {"float16": np.float16, "float32": np.float32}
PROVENANCE_KEYS = (
    "base_model_sha256",
    "laya_source_revision",
    "training_data_manifest_sha256",
)


class CacheError(ValueError):
    """Raised when a feature cache is missing, stale, or incompatible."""


def index_path_for(bin_path):
    return Path(str(bin_path) + ".index.json")


class CacheWriter:
    """Append-only writer for one split's features plus its sidecar index."""

    def __init__(self, bin_path, split, dtype, hidden_size, provenance):
        if dtype not in DTYPES:
            raise CacheError("dtype must be one of %s" % ", ".join(sorted(DTYPES)))
        if hidden_size < 1:
            raise CacheError("hidden_size must be positive")
        missing = [key for key in PROVENANCE_KEYS if not provenance.get(key)]
        if missing:
            raise CacheError("cache provenance is missing %s" % ", ".join(missing))
        self.bin_path = Path(bin_path)
        self.index_path = index_path_for(self.bin_path)
        self.split = split
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.provenance = {key: provenance[key] for key in PROVENANCE_KEYS}
        self._np_dtype = DTYPES[dtype]
        self._handle = open(self.bin_path, "wb")
        self._digest = hashlib.sha256()
        self._cases = []
        self._seen = set()
        self._elements = 0
        self._closed = False

    def append(self, case_ids, array):
        if self._closed:
            raise CacheError("cache writer is already closed")
        array = np.asarray(array)
        if array.ndim != 3 or array.shape[2] != self.hidden_size:
            raise CacheError(
                "expected (batch, length, %d) features, got shape %s"
                % (self.hidden_size, (array.shape,))
            )
        if len(case_ids) != array.shape[0]:
            raise CacheError("case_ids and feature batch disagree")
        if array.dtype != self._np_dtype:
            array = array.astype(self._np_dtype)
        for index, case_id in enumerate(case_ids):
            if case_id in self._seen:
                raise CacheError("duplicate case_id %r" % case_id)
            self._seen.add(case_id)
            row = np.ascontiguousarray(array[index])
            payload = row.tobytes()
            self._handle.write(payload)
            self._digest.update(payload)
            self._cases.append([case_id, self._elements, int(row.shape[0])])
            self._elements += int(row.shape[0]) * self.hidden_size

    def close(self):
        if self._closed:
            return None
        self._handle.flush()
        self._handle.close()
        self._closed = True
        index = {
            "schema_version": SCHEMA_VERSION,
            "split": self.split,
            "dtype": self.dtype,
            "hidden_size": self.hidden_size,
            "case_count": len(self._cases),
            "file_sha256": self._digest.hexdigest(),
            "provenance": self.provenance,
            "cases": self._cases,
        }
        self.index_path.write_text(
            json.dumps(index, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        return index

    def abort(self):
        if not self._closed:
            self._handle.close()
            self._closed = True
        self.bin_path.unlink(missing_ok=True)
        self.index_path.unlink(missing_ok=True)


class FeatureCache:
    """Random access to one split's cached encoder features."""

    def __init__(self, bin_path, split=None, expected_provenance=None):
        self.bin_path = Path(bin_path)
        self.index_path = index_path_for(self.bin_path)
        if not self.index_path.is_file():
            raise CacheError(
                "missing feature cache index %s; run training.build_feature_cache"
                % self.index_path
            )
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError("could not read cache index %s: %s" % (self.index_path, exc))
        if index.get("schema_version") != SCHEMA_VERSION:
            raise CacheError("unsupported feature cache schema in %s" % self.index_path)
        if split is not None and index.get("split") != split:
            raise CacheError(
                "cache index %s holds split %r, expected %r"
                % (self.index_path, index.get("split"), split)
            )
        if index.get("dtype") not in DTYPES:
            raise CacheError("cache index %s has unknown dtype" % self.index_path)
        if not self.bin_path.is_file():
            raise CacheError("missing feature cache data %s" % self.bin_path)
        for key, expected in (expected_provenance or {}).items():
            actual = index.get("provenance", {}).get(key)
            if expected and actual != expected:
                raise CacheError(
                    "feature cache %s is stale: %s is %r, expected %r; rebuild it"
                    % (self.index_path, key, actual, expected)
                )
        cases = index.get("cases")
        if not isinstance(cases, list) or not cases:
            raise CacheError("feature cache %s has no cases" % self.index_path)
        self.hidden_size = int(index["hidden_size"])
        self._np_dtype = DTYPES[index["dtype"]]
        self._itemsize = np.dtype(self._np_dtype).itemsize
        self.split = index.get("split")
        self.provenance = index.get("provenance", {})
        self._lookup = {}
        for case_id, offset_scalars, length in cases:
            if case_id in self._lookup:
                raise CacheError("cache index repeats case_id %r" % case_id)
            if length < 1 or offset_scalars < 0:
                raise CacheError("cache index has invalid span for %r" % case_id)
            self._lookup[case_id] = (int(offset_scalars) * self._itemsize, int(length))
        self._handle = open(self.bin_path, "rb")

    def __len__(self):
        return len(self._lookup)

    def __contains__(self, case_id):
        return case_id in self._lookup

    def missing(self, case_ids):
        return [case_id for case_id in case_ids if case_id not in self._lookup]

    def get(self, case_id):
        try:
            byte_offset, length = self._lookup[case_id]
        except KeyError:
            raise CacheError("case_id %r is not in feature cache %s"
                             % (case_id, self.index_path))
        byte_count = length * self.hidden_size * self._itemsize
        self._handle.seek(byte_offset)
        payload = bytearray(self._handle.read(byte_count))
        if len(payload) != byte_count:
            raise CacheError("feature cache %s is truncated" % self.bin_path)
        return np.frombuffer(payload, dtype=self._np_dtype).reshape(length, self.hidden_size)

    def close(self):
        self._handle.close()


def stack_features(items, cache, torch):
    """Stack per-case features into a padded float32 batch tensor."""
    if not items:
        raise CacheError("cannot stack an empty feature batch")
    arrays = []
    for item in items:
        array = cache.get(item["case_id"])
        if array.shape[0] != len(item["ids"]):
            raise CacheError(
                "case %r has %d cached positions but %d tokens; rebuild the cache"
                % (item["case_id"], array.shape[0], len(item["ids"]))
            )
        arrays.append(array)
    max_length = max(array.shape[0] for array in arrays)
    batch = torch.zeros(
        (len(arrays), max_length, cache.hidden_size), dtype=torch.float32
    )
    for index, array in enumerate(arrays):
        batch[index, :array.shape[0]] = torch.from_numpy(
            array.astype(np.float32, copy=False)
        )
    return batch


def forward_from_cached(model, features, attention, marker_pos, marker_mask, qtypes, torch):
    """Decision head forward pass over cached encoder features.

    Mirrors Laya's ``DecisionModel.forward`` minus the encoder step: the same
    type embedding, head layers, marker gather, and scorer, with the padding
    mask taken from the collated attention mask.
    """
    h = features + model.type_emb(qtypes)[:, None, :]
    pad = ~attention.bool()
    if model.head is not None:
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    marker_hidden = torch.gather(h, 1, idx)
    logits = model.scorer(marker_hidden).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def cached_forward_fn(model, cache, torch):
    """Build an ``evaluate_model`` forward function that reads cached features."""

    def forward_fn(ids, attention, positions, mask, qtypes, items):
        return forward_from_cached(
            model, stack_features(items, cache, torch),
            attention, positions, mask, qtypes, torch,
        )

    return forward_fn


def verify_cached_forward(model, items, cache, tokenizer, batch_size, torch, limit=64):
    """Compare cached-feature logits against the live encoder forward pass.

    Returns a summary dict; raises CacheError when any case predicts a
    different option, because that would make cached training and live
    evaluation disagree.
    """
    if limit < 1:
        raise CacheError("verification limit must be positive")
    sample = items[:limit]
    if not sample:
        raise CacheError("nothing to verify")
    model.eval()
    max_abs_diff = 0.0
    mismatches = []
    with torch.no_grad():
        for start in range(0, len(sample), batch_size):
            chunk = sample[start:start + batch_size]
            ids, attention, positions, mask, labels, qtypes = collate(
                chunk, tokenizer.pad_token_id, torch
            )
            live, _ = model(ids, attention, positions, mask, qtypes)
            cached = forward_from_cached(
                model, stack_features(chunk, cache, torch),
                attention, positions, mask, qtypes, torch,
            )
            max_abs_diff = max(
                max_abs_diff, float((live - cached).abs().max().item())
            )
            live_prediction = live.argmax(-1)
            cached_prediction = cached.argmax(-1)
            for index, item in enumerate(chunk):
                if live_prediction[index] != cached_prediction[index]:
                    mismatches.append(item["case_id"])
    model.train()
    model.encoder.eval()
    if mismatches:
        raise CacheError(
            "cached features disagree with the live forward pass on %d of %d "
            "cases (first: %s); rebuild the cache with --dtype float32"
            % (len(mismatches), len(sample), mismatches[0])
        )
    return {
        "cases": len(sample),
        "max_abs_logit_diff": max_abs_diff,
        "prediction_mismatches": 0,
    }
