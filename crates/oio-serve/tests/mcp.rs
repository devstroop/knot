//! M7: MCP stdio server — tool surface + JSON-RPC framing over a real stream.

use std::sync::Arc;

use indexmap::IndexMap;

use serde_json::{Value, json};

use oio::protocol::{Action, Answer, Routing, SystemOneRequest, SystemOneResponse, Usage};
use oio_serve::Predictor;
use oio_serve::mcp::{handle_message, run_stdio};

// Reuse the HTTP test stub via the public trait: same wire surface.
struct Stub;

impl Predictor for Stub {
    fn predict(&self, req: &SystemOneRequest) -> oio::Result<SystemOneResponse> {
        let mut answers = IndexMap::new();
        for qid in req.questions.keys() {
            answers.insert(
                qid.clone(),
                Answer::Choice {
                    choice: "yes".into(),
                    probabilities: IndexMap::from([("yes".into(), 0.9)]),
                    confidence: 0.9,
                    answer_confidence: 0.9,
                    action: Action {
                        act_probability: 0.1,
                    },
                    window: None,
                },
            );
        }
        Ok(SystemOneResponse {
            model: oio::protocol::AGENT_MODEL.into(),
            answers,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 0,
                state_tokens: Some(2),
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

fn predictor() -> Arc<dyn Predictor> {
    Arc::new(Stub)
}

#[test]
fn initialize_and_tools_list() {
    let p = predictor();
    let init = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
    )
    .unwrap();
    assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(init["result"]["serverInfo"]["name"], "oio-mcp");
    let list = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    .unwrap();
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec![
            "oio_status",
            "oio_route",
            "oio_predict",
            "oio_predict_batch"
        ]
    );
}

#[test]
fn predict_tool_round_trip() {
    let p = predictor();
    let resp = handle_message(
        &p,
        &json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "oio_predict", "arguments": {
                "state": "I need a refund",
                "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
            }},
        }),
    )
    .unwrap();
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["answers"]["q"]["type"], "choice");
    assert_eq!(payload["routing"]["model"], "english");
}

#[test]
fn route_batch_status_tools() {
    let p = predictor();
    let r = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "oio_route", "arguments": {"state": "hello"}}}),
    )
    .unwrap();
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["model"], "english");

    let b = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "oio_predict_batch", "arguments": {
            "states": ["a", "b"],
            "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
        }}}),
    )
    .unwrap();
    let text = b["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["results"].as_array().unwrap().len(), 2);
    assert_eq!(payload["total_usage"]["output_tokens"], 0);

    let s = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "oio_status", "arguments": {}}}),
    )
    .unwrap();
    let text = s["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["loaded"], json!(["english"]));
}

#[test]
fn tool_error_is_is_error() {
    let p = predictor();
    let r = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "oio_predict", "arguments": {"questions": {}}}}),
    )
    .unwrap();
    assert_eq!(r["result"]["isError"], true);
}

#[test]
fn unknown_method_and_tool() {
    let p = predictor();
    let r = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 8, "method": "bogus/method"}),
    )
    .unwrap();
    assert_eq!(r["error"]["code"], -32601);
    let t = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}),
    )
    .unwrap();
    assert_eq!(t["result"]["isError"], true);
}

#[tokio::test]
async fn stdio_e2e_over_duplex() {
    let p = predictor();
    let (client_tx, server_rx) = tokio::io::duplex(64 * 1024);
    let (server_tx, client_rx) = tokio::io::duplex(64 * 1024);
    let server = tokio::spawn(async move { run_stdio(p, server_rx, server_tx).await });

    use tokio::io::AsyncWriteExt;
    let mut client_tx = client_tx;
    client_tx
        .write_all(
            b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"oio_status\",\"arguments\":{}}}\n",
        )
        .await
        .unwrap();
    client_tx.shutdown().await.unwrap();

    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut lines = BufReader::new(client_rx).lines();
    let mut count = 0;
    let mut saw_status = false;
    while let Some(line) = lines.next_line().await.unwrap() {
        let v: Value = serde_json::from_str(&line).unwrap();
        if v["id"] == 3 {
            let text = v["result"]["content"][0]["text"].as_str().unwrap();
            let payload: Value = serde_json::from_str(text).unwrap();
            assert_eq!(payload["status"], "ok");
            saw_status = true;
        }
        count += 1;
    }
    assert_eq!(count, 3, "notification must not get a response");
    assert!(saw_status);
    server.await.unwrap().unwrap();
}

// Regression (review #4): min_confidence was accepted by oio_predict /
// oio_predict_batch and silently ignored; it must flag low-confidence answers
// exactly like the HTTP surface does.
#[test]
fn min_confidence_is_applied_on_mcp_tools() {
    let p = predictor();
    let call = |args: Value| {
        handle_message(
            &p,
            &json!({
                "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": "oio_predict", "arguments": args},
            }),
        )
        .unwrap()
    };
    // stub answer_confidence is 0.9
    let resp = call(json!({
        "state": "I need a refund",
        "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
        "min_confidence": 0.95,
    }));
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["answers"]["q"]["low_confidence"], true);
    assert_eq!(payload["answers"]["q"]["abstention"], "abstained");
    assert_eq!(payload["answers"]["q"]["abstention_threshold"], 0.95);

    let resp = call(json!({
        "state": "I need a refund",
        "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
        "min_confidence": 0.5,
    }));
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert!(payload["answers"]["q"].get("low_confidence").is_none());
    assert_eq!(payload["answers"]["q"]["abstention"], "passed");
    assert_eq!(payload["answers"]["q"]["abstention_threshold"], 0.5);

    // out-of-range is a caller error, same as HTTP
    let resp = call(json!({
        "state": "x",
        "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
        "min_confidence": 2.0,
    }));
    assert_eq!(resp["result"]["isError"], true);
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("min_confidence must be a float in [0.0, 1.0]"),
        "{text}"
    );
}

// Regression (review #7): the HTTP layer capped questions/state/options but
// oio_predict accepted anything; both surfaces now share the same rules.
#[test]
fn mcp_shares_http_request_limits() {
    let p = predictor();
    let mut questions = json!({});
    for i in 0..65 {
        questions[format!("q{i}")] = json!({"type": "noul", "instructions": "ok?"});
    }
    let r = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
        "params": {"name": "oio_predict", "arguments": {
            "state": "x", "questions": questions,
        }}}),
    )
    .unwrap();
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("too many questions"), "{text}");

    let r = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 12, "method": "tools/call",
        "params": {"name": "oio_predict", "arguments": {
            "state": "x".repeat(50_001),
            "questions": {"q": {"type": "noul", "instructions": "ok?"}},
        }}}),
    )
    .unwrap();
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("state too large"), "{text}");
}

#[test]
fn batch_tool_validates_batch_controls_like_laya_mcp() {
    let p = predictor();
    let bad_size = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 40, "method": "tools/call",
        "params": {"name": "oio_predict_batch", "arguments": {
            "states": ["a"],
            "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
            "batch_size": 0,
        }}}),
    )
    .unwrap();
    assert_eq!(bad_size["result"]["isError"], json!(true));
    let text = bad_size["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "batch_size must be a positive integer, got 0");

    let bad_bool = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 41, "method": "tools/call",
        "params": {"name": "oio_predict_batch", "arguments": {
            "states": ["a"],
            "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
            "batch_size": true,
        }}}),
    )
    .unwrap();
    let text = bad_bool["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "batch_size must be a positive integer, got True");

    let bad_sort = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {"name": "oio_predict_batch", "arguments": {
            "states": ["a"],
            "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
            "sort_by_length": "yes",
        }}}),
    )
    .unwrap();
    assert_eq!(bad_sort["result"]["isError"], json!(true));
    let text = bad_sort["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "sort_by_length must be a boolean");

    // Valid controls run.
    let ok = handle_message(
        &p,
        &json!({"jsonrpc": "2.0", "id": 43, "method": "tools/call",
        "params": {"name": "oio_predict_batch", "arguments": {
            "states": ["a", "b"],
            "questions": {"q": {"type": "choice", "instructions": "ok?", "criteria": {"yes": "y"}}},
            "batch_size": 1,
            "sort_by_length": true,
        }}}),
    )
    .unwrap();
    assert!(
        ok.get("result")
            .and_then(|r| r.get("isError"))
            .is_none_or(|v| *v == json!(false))
    );
    let payload: Value =
        serde_json::from_str(ok["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(payload["results"].as_array().unwrap().len(), 2);
}
