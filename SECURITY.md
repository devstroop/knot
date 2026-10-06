# Security policy

## Supported versions

`knot` is pre-1.0 research software. Only the latest `main` receives fixes;
no older series is maintained.

## Reporting a vulnerability

Use **GitHub private vulnerability reporting** (the repository's Security
tab → "Report a vulnerability"). Do not open public issues for suspected
vulnerabilities. Include the commit tested, the configuration (env vars,
checkpoint revision if relevant), and reproduction steps.

## Deployment notes (operator responsibility)

- `knot` binds `0.0.0.0:8000` by default — set `KNOT_HOST`/`KNOT_PORT` and
  front it appropriately for your network.
- Set `KNOT_API_KEY` to require `Authorization: Bearer <key>` on the HTTP
  surface. There is no auth on the MCP stdio transport by design (it inherits
  the caller's process boundary).
- Checkpoints are trust roots: pin revisions and ship a `SHA256SUMS`
  manifest so loading verifies every file (see README § Checkpoint).
- Inference never touches the network; models are fetched only by explicit
  operator steps.
