# ADR-001: ort-first runtime, ONNX artifacts reused from Laya

- Status: accepted
- Supersedes: —

## Context

Laya is research-oriented (torch eager, broad features). oio needs a production
runtime quickly with minimal numerical divergence from Laya's ONNX exports.

## Decision

Use the `ort` crate (ONNX Runtime) loading the same ONNX exports Laya produces
(`scripts/export_onnx.py`). Feature-gate behind `oio/onnx`. Defer candle/native
implementation to a later ADR; it must satisfy the same `Runtime` trait and pass
the same fixture parity gate.

## Consequences

- CPU-only baseline ships without libtorch or CUDA.
- Parity testing is meaningful from M2 onward.
- Native-runtime work is isolated behind `Runtime`, not a rewrite.
