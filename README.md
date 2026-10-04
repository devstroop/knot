# oio

Production-oriented Rust decision engine — a self-hosted, open alternative to
TypeSafe's hosted Jev API, compatible with Laya's `/v1/systemone` wire protocol.

Non-autoregressive "System 1" decisions over typed questions (`choice` / `score` / `noul`)
in one forward pass. Laya (Python) is the research reference; oio targets
operability, latency, strictness, and deployability.

## Layout

| Path | Contents |
|---|---|
| `crates/oio` | Core library: wire protocol, prompt assembly, router, ONNX runtime |
| `crates/oio-serve` | HTTP server binary (`/v1/systemone`, `/batch`, `/health`, `/models`) |
| `docs/` | PRD, plan, spec, compatibility, architecture, ADRs |
| `scripts/` | Checkpoint export / fixture tooling |

## Docs

- [docs/PRD.md](docs/PRD.md) — product requirements
- [docs/PLAN.md](docs/PLAN.md) — milestone plan
- [docs/SPEC.md](docs/SPEC.md) — invariants
- [docs/COMPAT.md](docs/COMPAT.md) — Jev/Laya compatibility contract

## Build

```bash
cargo check -p oio            # core (no model runtime)
cargo check -p oio-serve      # pulls ONNX Runtime + tokenizers
```
