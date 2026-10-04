//! M0: wire fixtures must round-trip through protocol types.

use oio::protocol::{Answer, QuestionType, SystemOneRequest, SystemOneResponse};

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn request_minimal_parses() {
    let req: SystemOneRequest =
        serde_json::from_str(&fixture("systemone_request_minimal.json")).unwrap();
    assert_eq!(req.model.as_deref(), Some("jev-1"));
    assert_eq!(req.questions["dept"].r#type, QuestionType::Choice);
}

#[test]
fn request_multilingual_all_question_types() {
    let req: SystemOneRequest =
        serde_json::from_str(&fixture("systemone_request_multilingual.json")).unwrap();
    assert_eq!(req.questions["department"].r#type, QuestionType::Choice);
    assert_eq!(req.questions["urgency"].r#type, QuestionType::Score);
    assert_eq!(req.questions["churn_risk"].r#type, QuestionType::Noul);
}

#[test]
fn response_choice_parses_probabilities() {
    let res: SystemOneResponse =
        serde_json::from_str(&fixture("systemone_response_choice.json")).unwrap();
    assert_eq!(res.usage.output_tokens, 0);
    match &res.answers["dept"] {
        Answer::Choice {
            choice,
            probabilities,
            ..
        } => {
            assert_eq!(choice, "billing");
            assert!((probabilities["billing"] - 0.94).abs() < 1e-6);
        }
        other => panic!("expected choice, got {other:?}"),
    }
}

/// Laya's response dict emits `model`, `answers`, `usage`, then `routing`
/// appended by the router; the routing dict is
/// `model`/`repo`/`reason`/`detection`/`workflow` with `detection: null`
/// when there is no analysis (`router.py` `RouteDecision.as_dict`).
#[test]
fn response_wire_key_order_matches_laya() {
    let res: SystemOneResponse =
        serde_json::from_str(&fixture("systemone_response_choice.json")).unwrap();
    let wire = serde_json::to_string(&res).unwrap();
    assert!(
        wire.starts_with(r#"{"model":"laya-rl-agent","answers":"#),
        "wire must start with model then answers: {wire}"
    );
    assert!(
        wire.contains(r#""usage":{"input_tokens":42,"output_tokens":0}"#)
            && wire.ends_with(r#","workflow":null}}"#),
        "usage before routing, routing last: {wire}"
    );

    let value: serde_json::Value = serde_json::from_str(&wire).unwrap();
    let routing = value["routing"].as_object().expect("routing object");
    assert_eq!(
        routing.keys().collect::<Vec<_>>(),
        vec!["model", "repo", "reason", "detection", "workflow"],
        "routing key order"
    );
    assert_eq!(routing["detection"], serde_json::Value::Null);
}

#[test]
fn response_all_types_round_trip() {
    let src = fixture("systemone_response_all_types.json");
    let res: SystemOneResponse = serde_json::from_str(&src).unwrap();
    let out = serde_json::to_string(&res).unwrap();
    let res2: SystemOneResponse = serde_json::from_str(&out).unwrap();
    assert_eq!(res.answers.len(), res2.answers.len());
    match &res2.answers["urgency"] {
        Answer::Score { score, legend, .. } => {
            assert!((score - 1.7).abs() < 1e-6);
            assert_eq!(legend["2"], "blocking");
        }
        other => panic!("expected score, got {other:?}"),
    }
    match &res2.answers["churn_risk"] {
        Answer::Noul { noul, .. } => assert!((noul - 0.13).abs() < 1e-6),
        other => panic!("expected noul, got {other:?}"),
    }
}
