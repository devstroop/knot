#![cfg(feature = "onnx")]
//! M2.5: end-to-end Engine parity vs Laya's Agent.predict fixture.
//!
//! Run: OIO_MODEL_DIR=/path/to/checkpoint cargo test -p oio --features onnx --test engine

use oio::engine::Engine;
use oio::protocol::{Answer, SystemOneRequest};
use oio::router::Router;

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/engine_english.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn engine_matches_laya_predict() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return;
    }
    let fix = fixture();
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": fix["state"],
        "questions": fix["questions"],
    }))
    .unwrap();
    let engine = Engine::load(Router::new(), &[("english", &dir)]).unwrap();
    let res = engine.predict(&req).unwrap();
    let expected = &fix["result"];

    // Choice answer parity
    match &res.answers["dept"] {
        Answer::Choice {
            choice,
            probabilities,
            answer_confidence,
            ..
        } => {
            assert_eq!(
                choice,
                expected["answers"]["dept"]["choice"].as_str().unwrap()
            );
            assert!(
                (answer_confidence
                    - expected["answers"]["dept"]["answer_confidence"]
                        .as_f64()
                        .unwrap() as f32)
                    .abs()
                    < 2e-3
            );
            assert!(
                (probabilities["billing"]
                    - expected["answers"]["dept"]["probabilities"]["billing"]
                        .as_f64()
                        .unwrap() as f32)
                    .abs()
                    < 2e-3
            );
        }
        other => panic!("expected choice, got {other:?}"),
    }
    match &res.answers["urgency"] {
        Answer::Score { score, .. } => {
            let e = expected["answers"]["urgency"]["score"].as_f64().unwrap() as f32;
            assert!((score - e).abs() < 2e-3);
        }
        other => panic!("expected score, got {other:?}"),
    }
    match &res.answers["churn"] {
        Answer::Noul { noul, .. } => {
            let e = expected["answers"]["churn"]["noul"].as_f64().unwrap() as f32;
            assert!((noul - e).abs() < 2e-3);
        }
        other => panic!("expected noul, got {other:?}"),
    }
    assert!(!res.answers.is_empty());
    assert_eq!(res.usage.output_tokens, 0);
    assert_eq!(res.routing.model, "english");

    // Batch collation (review #9): N states go through one collated forward
    // per model and must equal each state's own single predict, with usage
    // accumulated per item rather than across the batch.
    let state_b = serde_json::json!(
        "Hello, I want to change my plan to premium. Please confirm the new monthly price."
    );
    let req_b = SystemOneRequest {
        state: state_b.clone(),
        ..req.clone()
    };
    let single_b = engine.predict(&req_b).unwrap();
    let batch = engine
        .predict_batch(
            &[req.state.clone(), state_b],
            req.clone(),
            oio::engine::BatchOpts::default(),
        )
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_same_answers(&batch[0].answers, &res.answers, "item 0");
    assert_same_answers(&batch[1].answers, &single_b.answers, "item 1");
    assert_eq!(batch[0].usage.input_tokens, res.usage.input_tokens);
    assert_eq!(batch[1].usage.input_tokens, single_b.usage.input_tokens);
    assert_eq!(batch[0].usage.state_tokens, res.usage.state_tokens);
    assert_eq!(batch[1].usage.state_tokens, single_b.usage.state_tokens);
    assert_eq!(
        batch[0].usage.truncated_questions,
        res.usage.truncated_questions
    );
    assert_eq!(
        batch[1].usage.truncated_questions,
        single_b.usage.truncated_questions
    );
    assert_eq!(batch[0].routing.model, res.routing.model);
    assert_eq!(batch[0].routing.reason, res.routing.reason);
    assert_eq!(batch[1].routing.model, single_b.routing.model);
    assert_eq!(batch[1].routing.reason, single_b.routing.reason);
}

#[test]
fn unknown_model_passthrough_auto_routes() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        return;
    }
    let fix = fixture();
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": fix["state"],
        "questions": fix["questions"],
        "model": "jev-1",
    }))
    .unwrap();
    let engine = Engine::load(Router::new(), &[("english", &dir)]).unwrap();
    let res = engine.predict(&req).unwrap();
    assert_eq!(res.routing.model, "english");
}

// Regression (review #6): the router LRU (PRD F6, `max_loaded`) was dead
// code — every checkpoint stayed resident forever and `OIO_MAX_LOADED` did
// nothing. The engine now evicts beyond `max_loaded` and reloads on demand.
#[test]
fn lru_evicts_and_reloads_checkpoints() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return;
    }
    let fix = fixture();
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": fix["state"],
        "questions": fix["questions"],
    }))
    .unwrap();
    let mut req_alt = req.clone();
    req_alt.model = Some("multilingual".into());

    let mut router = Router::new();
    router.max_loaded = 1;
    // second model points at the same directory; only residency matters here
    let engine = Engine::load(router, &[("english", &dir), ("multilingual", &dir)]).unwrap();
    // boot loads up to max_loaded only
    assert_eq!(engine.resident(), vec!["english"]);

    engine.predict(&req).unwrap();
    assert_eq!(engine.resident(), vec!["english"]);

    // using the other model evicts the LRU entry
    engine.predict(&req_alt).unwrap();
    assert_eq!(engine.resident(), vec!["multilingual"]);

    // and coming back reloads it, evicting the other
    engine.predict(&req).unwrap();
    assert_eq!(engine.resident(), vec!["english"]);
}

// Regression (review #11): option_order was silently dropped on the wire
// (serde ignored the unknown field), so callers' slot order never reached the
// model and probabilities attached to the wrong options. Now: validated as a
// permutation, rendered into the prompt, un-permuted back (Laya unpermute_probs).
#[test]
fn option_order_is_validated_and_unpermuting_identity_matches() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return;
    }
    let fix = fixture();
    let base: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": fix["state"],
        "questions": fix["questions"],
    }))
    .unwrap();
    let engine = Engine::load(Router::new(), &[("english", &dir)]).unwrap();
    let canonical = engine.predict(&base).unwrap();

    // identity order must be exactly the canonical answer
    let mut identity = base.clone();
    for q in identity.questions.values_mut() {
        let n = match &q.criteria {
            serde_json::Value::Object(m) => m.len(),
            serde_json::Value::Array(a) => a.len(),
            _ => 2,
        };
        q.option_order = Some((0..n).collect());
    }
    let ident_res = engine.predict(&identity).unwrap();
    for (qid, a) in &canonical.answers {
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(&ident_res.answers[qid]).unwrap(),
            "qid {qid}"
        );
    }

    // a non-permutation is a caller error, not a silent mis-assignment
    // (right length: all-zeros is only a permutation for n == 1)
    let mut bad = base.clone();
    for q in bad.questions.values_mut() {
        let n = match &q.criteria {
            serde_json::Value::Object(m) => m.len().max(2),
            serde_json::Value::Array(a) => a.len().max(2),
            _ => 2,
        };
        q.option_order = Some(vec![0; n]);
    }
    let err = engine.predict(&bad).unwrap_err();
    assert!(
        err.to_string().contains("permutation"),
        "unexpected error: {err}"
    );

    // wrong length likewise
    let mut short = base.clone();
    for q in short.questions.values_mut() {
        q.option_order = Some(vec![0]);
    }
    let err = engine.predict(&short).unwrap_err();
    assert!(err.to_string().contains("one index per option"), "{err}");

    // a real permutation: same option keys, probabilities still a distribution
    let mut rev = base.clone();
    for q in rev.questions.values_mut() {
        if let serde_json::Value::Object(m) = &q.criteria {
            let n = m.len();
            q.option_order = Some((0..n).rev().collect());
        }
    }
    let rev_res = engine.predict(&rev).unwrap();
    for (qid, a) in &canonical.answers {
        if let Answer::Choice {
            probabilities: p0, ..
        } = a
        {
            match &rev_res.answers[qid] {
                Answer::Choice {
                    probabilities: p1, ..
                } => {
                    let keys0: Vec<_> = p0.keys().cloned().collect();
                    let keys1: Vec<_> = p1.keys().cloned().collect();
                    assert_eq!(keys0, keys1, "option keys must stay in caller order");
                    let sum: f32 = p1.values().sum();
                    assert!((sum - 1.0).abs() < 1e-2, "probs sum {sum}");
                }
                other => panic!("expected choice, got {other:?}"),
            }
        }
    }
}

/// Laya's empty-questions path returns `{"model": "laya-rl-agent", "answers": {},
/// "usage": {input_tokens: 0, output_tokens: 0}}` with routing appended and no
/// tokenizer/forward work (`agent.py` `if not ids`). `&[]` means no checkpoint
/// is ever touched — if this test loads a model, the short-circuit is broken.
#[test]
fn empty_questions_returns_laya_empty_shape() {
    let engine = Engine::load(Router::new(), &[]).unwrap();
    let req: SystemOneRequest =
        serde_json::from_value(serde_json::json!({"state": "hello", "questions": {}})).unwrap();
    let res = engine.predict(&req).unwrap();
    assert_eq!(res.model, "laya-rl-agent");
    assert!(res.answers.is_empty());
    assert_eq!(res.usage.input_tokens, 0);
    assert_eq!(res.usage.output_tokens, 0);
    assert!(res.usage.state_tokens.is_none());
    assert!(res.usage.windows.is_none());
    assert_eq!(res.routing.repo, "convaiinnovations/laya");
    assert!(res.routing.workflow.is_none());

    let wire = serde_json::to_string(&res).unwrap();
    assert!(
        wire.starts_with(
            r#"{"model":"laya-rl-agent","answers":{},"usage":{"input_tokens":0,"output_tokens":0}"#
        ),
        "empty response key order: {wire}"
    );

    let batch = engine
        .predict_batch(
            &[serde_json::json!("hello")],
            req.clone(),
            oio::engine::BatchOpts::default(),
        )
        .unwrap();
    assert_eq!(batch.len(), 1);
    assert!(batch[0].answers.is_empty());
    assert_eq!(batch[0].usage.input_tokens, 0);
}

fn assert_same_answers(
    got: &indexmap::IndexMap<String, Answer>,
    want: &indexmap::IndexMap<String, Answer>,
    item: &str,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{item}"
    );
    for (qid, w) in want {
        let g = &got[qid];
        match (g, w) {
            (
                Answer::Choice {
                    choice: gc,
                    probabilities: gp,
                    answer_confidence: gu,
                    ..
                },
                Answer::Choice {
                    choice: wc,
                    probabilities: wp,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert_eq!(gc, wc, "{item} question {qid}: choice");
                assert!(
                    (gu - wu).abs() < 1e-3,
                    "{item} question {qid}: confidence {gu} vs {wu}"
                );
                for (k, v) in wp {
                    assert!(
                        (gp[k] - v).abs() < 1e-3,
                        "{item} question {qid}: prob {k} {} vs {v}",
                        gp[k]
                    );
                }
            }
            (
                Answer::Score {
                    score: gs,
                    answer_confidence: gu,
                    ..
                },
                Answer::Score {
                    score: ws,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert!(
                    (gs - ws).abs() < 1e-3,
                    "{item} question {qid}: score {gs} vs {ws}"
                );
                assert!((gu - wu).abs() < 1e-3, "{item} question {qid}: confidence");
            }
            (
                Answer::Noul {
                    noul: gn,
                    answer_confidence: gu,
                    ..
                },
                Answer::Noul {
                    noul: wn,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert!(
                    (gn - wn).abs() < 1e-3,
                    "{item} question {qid}: noul {gn} vs {wn}"
                );
                assert!((gu - wu).abs() < 1e-3, "{item} question {qid}: confidence");
            }
            (g, w) => panic!("{item} question {qid}: variant mismatch\n got {g:?}\nwant {w:?}"),
        }
    }
}
