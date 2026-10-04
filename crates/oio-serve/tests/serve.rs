//! Contract tests for the oio HTTP surface, mirroring laya/tests/test_serve.py
//! in shape: 200 wire shape, 401/400/413/422/503 semantics, batch ≤ 64.

use std::sync::Arc;

use indexmap::IndexMap;

use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use oio::protocol::{Action, Answer, Routing, SystemOneRequest, SystemOneResponse, Usage};
use oio_serve::{Predictor, ServeConfig, build_app};

struct Stub(&'static str);

impl Predictor for Stub {
    fn device(&self) -> &'static str {
        self.0
    }

    fn predict(&self, req: &SystemOneRequest) -> oio::Result<SystemOneResponse> {
        let mut answers = IndexMap::new();
        for (qid, q) in &req.questions {
            answers.insert(
                qid.clone(),
                match q.r#type {
                    oio::protocol::QuestionType::Choice => Answer::Choice {
                        choice: "yes".into(),
                        probabilities: IndexMap::from([("yes".into(), 0.9)]),
                        confidence: 0.9,
                        answer_confidence: 0.9,
                        action: Action {
                            act_probability: 0.1,
                        },
                        window: None,
                    },
                    oio::protocol::QuestionType::Score => Answer::Score {
                        score: 1.0,
                        legend: IndexMap::from([("0".into(), "low".into())]),
                        probabilities: IndexMap::from([("0".into(), 0.9)]),
                        confidence: 0.9,
                        answer_confidence: 0.9,
                        action: Action {
                            act_probability: 0.1,
                        },
                        window: None,
                    },
                    oio::protocol::QuestionType::Noul => Answer::Noul {
                        noul: 0.8,
                        confidence: 0.8,
                        answer_confidence: 0.8,
                        action: Action {
                            act_probability: 0.1,
                        },
                        window: None,
                    },
                },
            );
        }
        Ok(SystemOneResponse {
            model: oio::protocol::AGENT_MODEL.into(),
            answers,
            usage: Usage {
                input_tokens: 7,
                output_tokens: 0,
                state_tokens: Some(3),
                state_tokens_dropped: Some(0),
                truncated: Some(false),
                truncated_questions: Some(vec![]),
                options: None,
                windows: None,
            },
            routing: Routing {
                model: "english".into(),
                repo: "convaiinnovations/laya".into(),
                reason: "stub".into(),
                detection: None,
                workflow: None,
            },
            shortlist: None,
        })
    }

    fn predict_batch(
        &self,
        states: &[Value],
        template: SystemOneRequest,
        _opts: oio::engine::BatchOpts,
    ) -> oio::Result<Vec<SystemOneResponse>> {
        Ok(states
            .iter()
            .map(|s| {
                self.predict(&SystemOneRequest {
                    state: s.clone(),
                    ..template.clone()
                })
                .unwrap()
            })
            .collect())
    }

    fn loaded(&self) -> Vec<&'static str> {
        vec!["english"]
    }

    fn route(
        &self,
        _state: &Value,
        _questions: Option<&std::collections::HashMap<String, Value>>,
        _model: Option<&str>,
        _task: Option<&str>,
        _lang: Option<&str>,
        _lang_guess: Option<&str>,
    ) -> oio::Result<Value> {
        Ok(json!({"model": "english", "reason": "stub"}))
    }
}

async fn call(
    app: axum::Router,
    method: &str,
    path: &str,
    body: Value,
) -> (u16, Value, axum::http::HeaderMap) {
    let req = axum::http::Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        headers,
    )
}

fn app(config: ServeConfig) -> axum::Router {
    build_app(Arc::new(Stub("cpu")), config)
}

fn simple_q() -> Value {
    json!({"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y", "no": "n"}}})
}

#[tokio::test]
async fn systemone_ok() {
    let (st, body, _headers) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 200);
    assert_eq!(body["answers"]["q"]["type"], "choice");
    assert_eq!(body["usage"]["output_tokens"], 0);
    assert_eq!(body["routing"]["model"], "english");
}

#[tokio::test]
async fn missing_questions_is_400() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "hi"}),
    )
    .await;
    assert_eq!(st, 400);
    assert!(body["detail"].as_str().unwrap().contains("questions"));
}

#[tokio::test]
async fn missing_state_is_400() {
    let (st, _, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"questions": {}}),
    )
    .await;
    assert_eq!(st, 400);
}

#[tokio::test]
async fn too_many_questions_is_413() {
    let mut qs = serde_json::Map::new();
    for i in 0..65 {
        qs.insert(
            format!("q{i}"),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": "b"}}),
        );
    }
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": qs}),
    )
    .await;
    assert_eq!(st, 413);
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("too many questions")
    );
}

#[tokio::test]
async fn too_many_choice_options_is_413() {
    let mut crit = serde_json::Map::new();
    for i in 0..101 {
        crit.insert(format!("k{i}"), json!("v"));
    }
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": crit}}}),
    )
    .await;
    assert_eq!(st, 413);
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("too many choice options")
    );
}

#[tokio::test]
async fn bad_budget_param_is_422() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q(), "max_len": "abc"}),
    )
    .await;
    assert_eq!(st, 422);
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("must be an integer")
    );
}

#[tokio::test]
async fn budget_over_cap_is_422() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q(), "max_len": 99999}),
    )
    .await;
    assert_eq!(st, 422);
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .contains("exceeds server limit")
    );
}

#[tokio::test]
async fn hooks_refused_are_422() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q(), "hooks": ["x"]}),
    )
    .await;
    assert_eq!(st, 422);
    assert!(body["detail"].as_str().unwrap().contains("hooks"));
}

#[tokio::test]
async fn auth_required_when_key_set() {
    let config = ServeConfig {
        api_key: Some("secret".into()),
        ..Default::default()
    };
    let app = app(config);
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .body(axum::body::Body::from(
            json!({"state": "s", "questions": simple_q()}).to_string(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status().as_u16(), 401);
}

#[tokio::test]
async fn auth_ok_with_bearer() {
    let config = ServeConfig {
        api_key: Some("secret".into()),
        ..Default::default()
    };
    let app = app(config);
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("authorization", "Bearer secret")
        .body(axum::body::Body::from(
            json!({"state": "s", "questions": simple_q()}).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
}

#[tokio::test]
async fn health_open_without_auth_when_key_set() {
    let config = ServeConfig {
        api_key: Some("secret".into()),
        ..Default::default()
    };
    let (st, body, _) = call(app(config), "GET", "/health", json!({})).await;
    assert_eq!(st, 200);
    assert_eq!(body["status"], "ok");
    // without the bearer: liveness only, no checkpoint names or hardware
    assert_eq!(
        body.as_object().map(|o| o.len()),
        Some(1),
        "unauth health must not leak detail: {body}"
    );
}

#[tokio::test]
async fn health_detail_payload_matches_laya_shape() {
    let (st, body, _) = call(app(ServeConfig::default()), "GET", "/health", json!({})).await;
    assert_eq!(st, 200);
    let keys: Vec<&str> = body
        .as_object()
        .expect("object")
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert_eq!(
        keys,
        [
            "status",
            "loaded",
            "revisions",
            "device",
            "device_is_preference",
            "checkpoint_devices",
            "cpu_fallbacks"
        ],
        "field order follows laya's /health"
    );
    assert_eq!(body["status"], json!("ok"));
    assert_eq!(body["loaded"], json!(["english"]));
    // stubs report no artifact commit — the local-path value, like laya
    assert_eq!(body["revisions"], json!({"english": null}));
    assert_eq!(body["device"], json!("cpu"));
    assert_eq!(body["device_is_preference"], json!(false));
    assert_eq!(body["checkpoint_devices"], json!({"english": "cpu"}));
    assert_eq!(
        body["cpu_fallbacks"],
        json!({"english": {"count": 0, "last_reason": null}})
    );
}

#[tokio::test]
async fn health_detail_reports_the_configured_device() {
    // SPEC §10: `/health` reports the device checkpoints really load onto.
    let app = build_app(Arc::new(Stub("cuda")), ServeConfig::default());
    let (st, body, _) = call(app, "GET", "/health", json!({})).await;
    assert_eq!(st, 200);
    assert_eq!(body["device"], json!("cuda"));
    assert_eq!(body["checkpoint_devices"], json!({"english": "cuda"}));
    assert_eq!(body["device_is_preference"], json!(false));
    assert_eq!(
        body["cpu_fallbacks"],
        json!({"english": {"count": 0, "last_reason": null}}),
        "a fallback never happens silently — failures are load errors"
    );
}

#[tokio::test]
async fn predict_responses_carry_laya_timing_headers() {
    let (st, _, headers) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 200);
    let timing = headers
        .get("server-timing")
        .expect("Server-Timing header")
        .to_str()
        .unwrap();
    let dur = timing
        .strip_prefix("inference;dur=")
        .unwrap_or_else(|| panic!("Server-Timing shape: {timing}"));
    dur.parse::<f64>().expect("dur is a number");
    headers
        .get("x-inference-time-ms")
        .expect("X-Inference-Time-Ms header")
        .to_str()
        .unwrap()
        .parse::<f64>()
        .expect("milliseconds are a number");
}

#[tokio::test]
async fn batch_responses_carry_laya_timing_headers() {
    let (st, _, headers) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone/batch",
        json!({"states": ["a"], "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 200);
    assert!(
        headers
            .get("server-timing")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("inference;dur="))
    );
    assert!(headers.get("x-inference-time-ms").is_some());
}

#[tokio::test]
async fn models_needs_auth_and_lists() {
    let config = ServeConfig {
        api_key: Some("secret".into()),
        ..Default::default()
    };
    let app = app(config);
    let req = axum::http::Request::builder()
        .method("GET")
        .uri("/models")
        .header("authorization", "Bearer secret")
        .body(axum::body::Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["models"], json!(["english"]));
}

#[tokio::test]
async fn batch_ok() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone/batch",
        json!({"states": ["a", "b"], "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 200);
    assert_eq!(body["results"].as_array().unwrap().len(), 2);
    assert_eq!(body["total_usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn batch_over_64_is_413() {
    let states: Vec<String> = (0..65).map(|i| format!("s{i}")).collect();
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone/batch",
        json!({"states": states, "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 413);
    assert!(body["detail"].as_str().unwrap().contains("too many states"));
}

#[tokio::test]
async fn batch_empty_states_is_400() {
    let (st, _, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone/batch",
        json!({"states": [], "questions": simple_q()}),
    )
    .await;
    assert_eq!(st, 400);
}

#[tokio::test]
async fn low_confidence_flag() {
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q(), "min_confidence": 0.95}),
    )
    .await;
    assert_eq!(st, 200);
    assert_eq!(body["answers"]["q"]["low_confidence"], json!(true));
    assert_eq!(body["answers"]["q"]["abstention"], json!("abstained"));
    assert_eq!(body["answers"]["q"]["abstention_threshold"], json!(0.95));
    // laya writes low_confidence, then abstention, then the threshold
    let wire = serde_json::to_string(&body).unwrap();
    let flag = wire.find("low_confidence").unwrap();
    let abst = wire.find("\"abstention\"").unwrap();
    assert!(flag < abst, "key order: {wire}");

    // above the threshold: no flag, but every answer still reports the gate
    let (st, body, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": simple_q(), "min_confidence": 0.5}),
    )
    .await;
    assert_eq!(st, 200);
    assert!(body["answers"]["q"].get("low_confidence").is_none());
    assert_eq!(body["answers"]["q"]["abstention"], json!("passed"));
    assert_eq!(body["answers"]["q"]["abstention_threshold"], json!(0.5));
}

#[tokio::test]
async fn batch_controls_validate_like_laya() {
    let q = simple_q();
    // Wrong type -> "must be an integer" (booleans, floats and strings are
    // all non-integers to Python's isinstance check too).
    for bad in [json!(true), json!(2.5), json!("3")] {
        let (st, body, _) = call(
            app(ServeConfig::default()),
            "POST",
            "/v1/systemone/batch",
            json!({"states": ["a"], "questions": q, "batch_size": bad}),
        )
        .await;
        assert_eq!(st, 422, "batch_size {bad}");
        assert_eq!(body["detail"], json!("batch_size must be an integer"));
    }
    // Present but non-positive -> Laya's %r wording.
    for (bad, want) in [
        (json!(0), "batch_size must be a positive integer, got 0"),
        (json!(-1), "batch_size must be a positive integer, got -1"),
    ] {
        let (st, body, _) = call(
            app(ServeConfig::default()),
            "POST",
            "/v1/systemone/batch",
            json!({"states": ["a"], "questions": q, "batch_size": bad}),
        )
        .await;
        assert_eq!(st, 422, "batch_size {bad}");
        assert_eq!(body["detail"], json!(want));
    }
    // sort_by_length must be a boolean.
    for bad in [json!("yes"), json!(1), json!(0)] {
        let (st, body, _) = call(
            app(ServeConfig::default()),
            "POST",
            "/v1/systemone/batch",
            json!({"states": ["a"], "questions": q, "sort_by_length": bad}),
        )
        .await;
        assert_eq!(st, 422, "sort_by_length {bad}");
        assert_eq!(body["detail"], json!("sort_by_length must be a boolean"));
    }
    // Valid values — and null, which means "not set" — all pass.
    for (bs, sbl) in [
        (json!(2), json!(true)),
        (json!(1), json!(false)),
        (json!(null), json!(null)),
    ] {
        let (st, _, _) = call(
            app(ServeConfig::default()),
            "POST",
            "/v1/systemone/batch",
            json!({"states": ["a", "b"], "questions": q, "batch_size": bs, "sort_by_length": sbl}),
        )
        .await;
        assert_eq!(st, 200, "batch_size={bs} sort_by_length={sbl}");
    }
    // The single-shot endpoint has no batch controls (laya doesn't read them
    // there either): an invalid value is simply not a batch-argument error.
    let (st, _, _) = call(
        app(ServeConfig::default()),
        "POST",
        "/v1/systemone",
        json!({"state": "s", "questions": q, "batch_size": 0}),
    )
    .await;
    assert_eq!(st, 200);
}
