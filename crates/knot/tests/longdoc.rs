#![cfg(feature = "onnx")]
//! M5: `Engine::predict_long` windowed scans vs Laya `Agent.predict_long`.
//!
//! Run: KNOT_MODEL_DIR=/path/to/checkpoint cargo test -p knot --features onnx --test longdoc

use knot::engine::Engine;
use knot::protocol::{Answer, SystemOneRequest};
use knot::router::Router;

fn engine() -> Option<Engine> {
    let Ok(dir) = std::env::var("KNOT_MODEL_DIR") else {
        eprintln!("skip: KNOT_MODEL_DIR not set");
        return None;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return None;
    }
    Some(Engine::load(Router::new(), &[("english", &dir)]).unwrap())
}

fn questions() -> serde_json::Value {
    serde_json::json!({
        "churn": {
            "type": "noul",
            "instructions": "Will this customer churn?",
            "criteria": {"false": "no churn", "true": "will churn"},
            "labels": {"false": "no", "true": "yes"}
        },
        "dept": {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": {"billing": "billing issues", "sales": "sales questions"}
        }
    })
}

#[test]
fn long_state_scans_windows_and_attributes() {
    let Some(engine) = engine() else { return };
    // A state far longer than one window (english head_max_len=192, max_len=512).
    let state = "The customer is unhappy about a double charge and wants a refund. ".repeat(40);
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": state,
        "questions": questions(),
    }))
    .unwrap();
    let res = engine.predict_long(&req, None, None).unwrap();
    let windows = res.usage.windows.expect("windows present");
    assert!(windows > 1, "expected a multi-window scan, got {windows}");
    match &res.answers["churn"] {
        Answer::Noul { window, .. } => {
            let w = window.as_ref().expect("window attribution");
            assert_eq!(w.count, windows);
            assert!(w.token_start < w.token_end);
            assert!(w.index < windows);
        }
        other => panic!("expected noul, got {other:?}"),
    }
    match &res.answers["dept"] {
        Answer::Choice { window, .. } => {
            let w = window.as_ref().expect("window attribution");
            assert_eq!(w.count, windows);
        }
        other => panic!("expected choice, got {other:?}"),
    }
    // Summed numeric fields across windows (Laya semantics).
    assert!(res.usage.input_tokens > 0);
    assert!(res.usage.state_tokens.unwrap_or(0) > 0);
}

#[test]
fn short_state_single_window() {
    let Some(engine) = engine() else { return };
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": "I was charged twice this month, please refund.",
        "questions": questions(),
    }))
    .unwrap();
    let res = engine.predict_long(&req, None, None).unwrap();
    assert_eq!(res.usage.windows, Some(1));
    // Single-window path: no per-answer window attribution (Laya shape).
    assert!(matches!(
        res.answers["churn"],
        Answer::Noul { window: None, .. }
    ));
}

#[test]
fn stride_past_window_is_an_error() {
    let Some(engine) = engine() else { return };
    let state = "word ".repeat(2000);
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": state,
        "questions": questions(),
    }))
    .unwrap();
    let err = engine.predict_long(&req, Some(64), Some(128)).unwrap_err();
    assert!(
        matches!(err, knot::error::Error::InvalidRequest(_)),
        "got {err:?}"
    );
}
