#![cfg(feature = "onnx")]
//! Evidence harness (RESEARCH-COMPARE "Gap ranking" #4): knot measures itself.
//!
//! Two checkpoint-gated checks over the recorded English corpus harvested
//! from `laya/research/evals/fixture.jsonl` (12 cases covering
//! `choice`/`score`/`noul`). The `expected` labels ride along for human
//! context only — both checks are regression-vs-self, not accuracy claims.
//!
//! * `golden_replay_matches_recorded_knot_responses` runs in the default
//!   gate: it replays the corpus and compares each response against
//!   `fixtures/golden_english.json`. Those goldens are **knot-recorded**, not
//!   Laya-recorded — this machine has no runnable Laya environment (the
//!   system Python lacks both `onnxruntime` and `torch`). Regenerate with
//!   `KNOT_UPDATE_GOLDEN=1` after an intentional behaviour change, review the
//!   diff, commit it.
//! * `latency_bench` is `#[ignore]`d so the gate stays stable; run it
//!   on demand for the p50/p95/mean figures quoted in
//!   `docs/RESEARCH-COMPARE.md`:
//!
//! ```text
//! KNOT_MODEL_DIR=/path/to/checkpoint cargo test -p knot --features onnx \
//!     --test evidence -- --ignored --nocapture
//! ```
//!
//! Both skip themselves when `KNOT_MODEL_DIR` is unset (AGENTS.md gate rules).

use std::path::PathBuf;
use std::time::Instant;

use knot::engine::Engine;
use knot::protocol::SystemOneRequest;
use knot::router::Router;
use serde_json::Value;

/// Wire tolerances match the parity suites: choices/key sets exact,
/// probabilities and confidences within 1e-3 (relative for large values).
const TOL: f64 = 1e-3;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

struct Case {
    state: String,
    questions: Value,
}

fn load_cases() -> Vec<Case> {
    let text = std::fs::read_to_string(fixture("eval_english.jsonl")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            Case {
                state: v["state"].as_str().unwrap().to_string(),
                questions: v["questions"].clone(),
            }
        })
        .collect()
}

fn request(case: &Case) -> SystemOneRequest {
    serde_json::from_value(serde_json::json!({
        "state": case.state,
        "questions": case.questions,
    }))
    .unwrap()
}

/// Engine for a case's checkpoint; skips (returns `None`) without a model.
fn engine_or_skip() -> Option<Engine> {
    let Ok(dir) = std::env::var("KNOT_MODEL_DIR") else {
        eprintln!("skip: KNOT_MODEL_DIR not set");
        return None;
    };
    let dir = PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return None;
    }
    match Engine::load(Router::new(), &[("english", &dir)]) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skip: engine failed to load: {e}");
            None
        }
    }
}

fn assert_close(got: &Value, want: &Value, path: &str) {
    match (got, want) {
        (Value::Object(g), Value::Object(w)) => {
            assert_eq!(
                g.keys().collect::<Vec<_>>(),
                w.keys().collect::<Vec<_>>(),
                "{path} keys"
            );
            for (k, wv) in w {
                assert_close(&g[k], wv, &format!("{path}.{k}"));
            }
        }
        (Value::Array(g), Value::Array(w)) => {
            assert_eq!(g.len(), w.len(), "{path} length");
            for (i, wv) in w.iter().enumerate() {
                assert_close(&g[i], wv, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(g), Value::Number(w)) => {
            let g = g.as_f64().unwrap();
            let w = w.as_f64().unwrap();
            assert!(
                (g - w).abs() <= TOL * (1.0 + w.abs()),
                "{path}: got {g}, want {w} (tol {TOL})"
            );
        }
        (g, w) => assert_eq!(g, w, "{path}"),
    }
}

#[test]
fn golden_replay_matches_recorded_knot_responses() {
    let Some(engine) = engine_or_skip() else {
        return;
    };
    let cases = load_cases();

    if std::env::var_os("KNOT_UPDATE_GOLDEN").is_some() {
        let mut out = serde_json::json!({
            "note": "knot-recorded goldens (not Laya: no runnable Laya env on the \
                     recording machine). Regenerate with KNOT_UPDATE_GOLDEN=1 after \
                     an intentional behaviour change; review the diff.",
            "source": "tests/fixtures/eval_english.jsonl (laya/research/evals/fixture.jsonl)",
            "cases": [],
        });
        for case in &cases {
            let res = engine.predict(&request(case)).unwrap();
            out["cases"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "state": case.state,
                    "questions": case.questions,
                    "response": res,
                }));
        }
        std::fs::write(
            fixture("golden_english.json"),
            serde_json::to_string_pretty(&out).unwrap(),
        )
        .unwrap();
        eprintln!(
            "golden_english.json rewritten with {} cases — review the diff and commit",
            cases.len()
        );
        return;
    }

    let recorded: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("golden_english.json")).unwrap())
            .unwrap();

    let recorded_cases = recorded["cases"].as_array().unwrap();
    assert_eq!(
        recorded_cases.len(),
        cases.len(),
        "golden/corpus size drift"
    );
    for (i, case) in cases.iter().enumerate() {
        let rec = &recorded_cases[i];
        assert_eq!(
            rec["state"],
            Value::String(case.state.clone()),
            "case {i} state drift"
        );
        assert_eq!(rec["questions"], case.questions, "case {i} questions drift");
        let got = serde_json::to_value(engine.predict(&request(case)).unwrap()).unwrap();
        assert_close(&got, &rec["response"], &format!("case {i}"));
    }
}

/// Warm in-process `Engine::predict` latency over the recorded English
/// corpus. Measures the engine only (no HTTP/auth/serde-on-the-wire costs —
/// `knot-serve` timings are observable live via `Server-Timing`).
#[test]
#[ignore = "latency bench: run with --ignored --nocapture under KNOT_MODEL_DIR"]
fn latency_bench() {
    let Some(engine) = engine_or_skip() else {
        return;
    };
    let cases = load_cases();
    let rounds: usize = std::env::var("KNOT_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    for _ in 0..3 {
        for case in &cases {
            engine.predict(&request(case)).unwrap();
        }
    }

    let mut samples_ms: Vec<f64> = Vec::with_capacity(rounds * cases.len());
    for _ in 0..rounds {
        for case in &cases {
            let t0 = Instant::now();
            engine.predict(&request(case)).unwrap();
            samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    let mean = samples_ms.iter().sum::<f64>() / n as f64;
    let pct = |p: f64| {
        let idx = ((n as f64) * p).ceil() as usize;
        samples_ms[idx.saturating_sub(1).min(n - 1)]
    };
    let questions: usize = cases
        .iter()
        .map(|c| c.questions.as_object().map_or(0, serde_json::Map::len))
        .sum();
    eprintln!(
        "knot in-process latency — {} cases, {} questions, {n} samples ({} rounds), warm, CPU",
        cases.len(),
        questions,
        rounds
    );
    eprintln!(
        "mean {mean:.2} ms | p50 {:.2} ms | p95 {:.2} ms | min {:.2} ms | max {:.2} ms",
        pct(0.50),
        pct(0.95),
        samples_ms[0],
        samples_ms[n - 1],
    );
}
