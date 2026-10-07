//! End-to-end audit-ledger test: stub predictor + real nqlite store.
//!
//! No model dir, no network — the ADR-006 default gate (like the parity
//! suites without `KNOT_MODEL_DIR`).

use std::collections::HashMap;

use knot::Predictor;
use knot::protocol::{SystemOneRequest, SystemOneResponse};
use knot_nqlite::{AuditConfig, AuditPredictor};

/// Canned predictor — response must pass through the decorator untouched.
#[derive(Clone)]
struct Stub;

impl Predictor for Stub {
    fn predict(&self, _req: &SystemOneRequest) -> knot::Result<SystemOneResponse> {
        // Usage/Routing have no Default — mirror the wire shape via serde.
        Ok(SystemOneResponse {
            model: "stub".into(),
            answers: Default::default(),
            usage: serde_json::from_value(serde_json::json!({
                "input_tokens": 0, "output_tokens": 0
            }))
            .expect("usage json"),
            routing: serde_json::from_value(serde_json::json!({
                "model": "stub", "repo": "stub", "reason": "test"
            }))
            .expect("routing json"),
            shortlist: None,
        })
    }

    fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        _opts: knot::engine::BatchOpts,
    ) -> knot::Result<Vec<SystemOneResponse>> {
        Ok(states
            .iter()
            .map(|_| Self.predict(&template).unwrap())
            .collect())
    }

    fn loaded(&self) -> Vec<&'static str> {
        vec!["stub-model"]
    }

    fn route(
        &self,
        _state: &serde_json::Value,
        _questions: Option<&HashMap<String, serde_json::Value>>,
        _model: Option<&str>,
        _task: Option<&str>,
        _lang: Option<&str>,
        _lang_guess: Option<&str>,
    ) -> knot::Result<serde_json::Value> {
        Ok(serde_json::json!({"model": "stub"}))
    }
}

fn request() -> SystemOneRequest {
    serde_json::from_value(serde_json::json!({
        "state": {"doc": "the quarterly numbers look off"},
        "questions": {}
    }))
    .expect("request json")
}

/// Same input → same hash as the decorator (identity, not content).
fn state_hash(req: &SystemOneRequest) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(req.state.to_string().as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn responses_pass_through_and_ledger_persists_after_flush() {
    let dir = std::env::temp_dir().join(format!("knot-audit-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ledger.nql");

    let cfg = AuditConfig {
        path: path.clone(),
        capacity: 8,
        excerpt_chars: 16, // deliberately shorter than the state sentence
    };

    let expected = Stub.predict(&request()).unwrap();
    let audit = AuditPredictor::new(Stub, cfg).expect("open ledger");

    // Two decisions; responses must be byte-identical to the bare stub's.
    for _ in 0..2 {
        let got = audit.predict(&request()).unwrap();
        assert_eq!(
            serde_json::to_string(&got).unwrap(),
            serde_json::to_string(&expected).unwrap(),
            "decorator must not alter the response"
        );
    }
    // Batch: one parent row, response unchanged.
    let batch = audit
        .predict_batch(
            &[serde_json::json!({"a": 1}), serde_json::json!({"b": 2})],
            request(),
            knot::engine::BatchOpts::default(),
        )
        .unwrap();
    assert_eq!(batch.len(), 2);

    let (written, dropped) = audit.stats();
    assert_eq!(dropped, 0, "small batch must not overflow");
    assert!(
        written <= 3,
        "at most one row enqueued per call (got {written})"
    );

    drop(audit); // Drop joins the writer → everything enqueued is flushed

    // Re-open the store directly (single-writer: the decorator is gone).
    let mut db = nqlite::Database::open(&path).expect("reopen ledger");
    let plan = nql::parse("SELECT * FROM decision;").expect("parse");
    let res = db.execute(&plan).expect("select");
    let rows = &res[0].rows;
    assert_eq!(rows.len(), 3, "2 predicts + 1 batch row (drained queue)");

    // Provenance edge exists: state -> :decided -> decision (ADR-006 seam B).
    let state_id = format!("state:s{}", &state_hash(&request())[..16]);
    let edge = nql::parse(&format!("MATCH ({state_id}) -> :decided;")).expect("parse match");
    let res = db.execute(&edge).expect("match");
    // Both predicts shared the same state → two :decided edges (the batch
    // used a different state node).
    assert_eq!(res[0].rows.len(), 2, "state -> decided edges stored");

    // Hash-only storage: every row's excerpt is truncated to the limit —
    // the full state sentence must never be persisted (ADR-006 privacy line).
    let all = nql::parse("SELECT * FROM decision;").unwrap();
    let res = db.execute(&all).unwrap();
    let excerpts: Vec<String> = res[0]
        .rows
        .iter()
        .filter_map(|r| match r.record.body.get("excerpt") {
            Some(nqlite::Value::Str(s)) => Some(s.as_str()),
            _ => None,
        })
        .map(str::to_owned)
        .collect();
    assert_eq!(excerpts.len(), 3, "every decision row carries an excerpt");
    for e in &excerpts {
        assert!(!e.contains("look off"), "excerpt must be truncated: {e}");
        assert!(
            !e.contains("quarterly numbers"),
            "full state must not be stored: {e}"
        );
    }
    // Row ORDER is hash-dependent (ids salt with time+pid → stable per store,
    // not across runs/platforms) — assert on the set, not the position.
    assert!(
        excerpts
            .iter()
            .any(|e| e.starts_with("{\"doc\":") && e.contains("the quar")),
        "predict-row excerpt keeps the leading chars: {excerpts:?}"
    );
    assert!(
        excerpts.iter().any(|e| e.starts_with("{\"batch\":")),
        "batch row carries its combined state excerpt: {excerpts:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn disabled_without_env() {
    // SAFETY: test-local; no other thread reads the environment here.
    unsafe { std::env::remove_var("KNOT_AUDIT_DB") };
    assert!(
        AuditConfig::from_env().is_none(),
        "off by default (open Q4)"
    );
}
