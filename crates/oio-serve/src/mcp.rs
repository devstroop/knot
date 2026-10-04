//! MCP stdio server (JSON-RPC 2.0, newline-delimited) exposing the decision
//! tools over the same `Predictor` interface the HTTP server uses.

use std::collections::HashMap;
use std::io;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use oio::protocol::SystemOneRequest;

use crate::Predictor;

fn tools() -> Vec<(&'static str, &'static str, serde_json::Value)> {
    vec![
        (
            "oio_status",
            "Report loaded checkpoints and version.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }),
        ),
        (
            "oio_route",
            "Route a state to a checkpoint without running inference.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "state": {},
                    "questions": { "type": "object" },
                    "model": { "type": "string" },
                    "task": { "type": "string" },
                    "lang": { "type": "string" },
                    "lang_guess": { "type": "string" },
                },
                "required": ["state"],
                "additionalProperties": false,
            }),
        ),
        (
            "oio_predict",
            "Score a state across typed questions (choice/score/noul answers).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "state": {},
                    "questions": { "type": "object" },
                    "model": { "type": "string" },
                    "max_len": { "type": "integer" },
                    "head_max_len": { "type": "integer" },
                    "task": { "type": "string" },
                    "lang": { "type": "string" },
                    "lang_guess": { "type": "string" },
                    "min_confidence": { "type": "number" },
                },
                "required": ["state", "questions"],
                "additionalProperties": false,
            }),
        ),
        (
            "oio_predict_batch",
            "Score several states over the same questions; one result per state.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "states": { "type": "array" },
                    "questions": { "type": "object" },
                    "model": { "type": "string" },
                    "max_len": { "type": "integer" },
                    "head_max_len": { "type": "integer" },
                    "task": { "type": "string" },
                    "lang": { "type": "string" },
                    "lang_guess": { "type": "string" },
                    "min_confidence": { "type": "number" },
                    "batch_size": { "type": "integer" },
                    "sort_by_length": { "type": "boolean" },
                },
                "required": ["states", "questions"],
                "additionalProperties": false,
            }),
        ),
    ]
}

fn tool_list() -> serde_json::Value {
    serde_json::json!({
        "tools": tools().iter().map(|(name, desc, schema)| {
            serde_json::json!({
                "name": name,
                "description": desc,
                "inputSchema": schema,
            })
        }).collect::<Vec<_>>()
    })
}

fn tool_error(msg: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": msg.into()}],
        "isError": true,
    })
}

/// Python `repr` for the scalar in laya's MCP error wording
/// (`f"batch_size must be a positive integer, got {v!r}"`).
fn py_repr(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Bool(b) => String::from(if *b { "True" } else { "False" }),
        serde_json::Value::String(s) => format!("'{s}'"),
        other => other.to_string(),
    }
}

/// Batch call controls for the MCP surface, validated the way laya's
/// `_validate_batch_size` does — one merged message for every wrong shape,
/// Python `repr` of the offending value.
fn batch_opts_of(args: &serde_json::Value) -> Result<oio::engine::BatchOpts, String> {
    let batch_size = match args.get("batch_size") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_i64() {
            Some(n) if n >= 1 => Some(n as usize),
            _ => {
                return Err(format!(
                    "batch_size must be a positive integer, got {}",
                    py_repr(v)
                ));
            }
        },
    };
    let sort_by_length = match args.get("sort_by_length") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => return Err("sort_by_length must be a boolean".into()),
    };
    Ok(oio::engine::BatchOpts {
        batch_size,
        sort_by_length,
    })
}

fn ok_text(payload: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": payload.to_string()}],
    })
}

/// Same rule as the HTTP surface: `min_confidence` in [0.0, 1.0] or null.
fn min_confidence_of(args: &serde_json::Value) -> Result<Option<f64>, String> {
    match args.get("min_confidence") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && (0.0..=1.0).contains(&f) => Ok(Some(f)),
            _ => Err(format!(
                "min_confidence must be a float in [0.0, 1.0], got {v}"
            )),
        },
    }
}

fn build_request(args: &serde_json::Value) -> Result<SystemOneRequest, String> {
    let state = args
        .get("state")
        .cloned()
        .ok_or_else(|| "'state' is required".to_string())?;
    let questions = args
        .get("questions")
        .cloned()
        .ok_or_else(|| "'questions' is required".to_string())?;
    serde_json::from_value(serde_json::json!({
        "state": state,
        "questions": questions,
        "model": args.get("model").cloned().unwrap_or(serde_json::Value::Null),
        "max_len": args.get("max_len").cloned().unwrap_or(serde_json::Value::Null),
        "head_max_len": args.get("head_max_len").cloned().unwrap_or(serde_json::Value::Null),
        "task": args.get("task").cloned().unwrap_or(serde_json::Value::Null),
        "lang": args.get("lang").cloned().unwrap_or(serde_json::Value::Null),
        "lang_guess": args.get("lang_guess").cloned().unwrap_or(serde_json::Value::Null),
        "min_confidence": args.get("min_confidence").cloned().unwrap_or(serde_json::Value::Null),
    }))
    .map_err(|e| format!("invalid request: {e}"))
}

fn call_tool(
    predictor: &Arc<dyn Predictor>,
    name: &str,
    args: &serde_json::Value,
) -> serde_json::Value {
    match name {
        "oio_status" => ok_text(serde_json::json!({
            "status": "ok",
            "loaded": predictor.loaded(),
            "version": env!("CARGO_PKG_VERSION"),
        })),
        "oio_route" => {
            let state = match args.get("state") {
                Some(s) => s,
                None => return tool_error("'state' is required"),
            };
            let questions: Option<HashMap<String, serde_json::Value>> = match args.get("questions")
            {
                Some(q) => match serde_json::from_value(q.clone()) {
                    Ok(m) => Some(m),
                    Err(_) => return tool_error("'questions' must be an object"),
                },
                None => None,
            };
            match predictor.route(
                state,
                questions.as_ref(),
                args.get("model").and_then(|v| v.as_str()),
                args.get("task").and_then(|v| v.as_str()),
                args.get("lang").and_then(|v| v.as_str()),
                args.get("lang_guess").and_then(|v| v.as_str()),
            ) {
                Ok(d) => ok_text(d),
                Err(e) => tool_error(e.to_string()),
            }
        }
        "oio_predict" => {
            // min_confidence was accepted here but never applied (the engine
            // ignores the field); apply the same flagging the HTTP path does.
            let req = match build_request(args) {
                Ok(r) => r,
                Err(e) => return tool_error(e),
            };
            // Same acceptance rules as the HTTP surface (review #7).
            let raw_questions =
                serde_json::to_value(&req.questions).unwrap_or(serde_json::Value::Null);
            if let Err((_, msg)) = crate::check_request_limits_inner(&req.state, &raw_questions) {
                return tool_error(msg);
            }
            let minc = match min_confidence_of(args) {
                Ok(m) => m,
                Err(e) => return tool_error(e),
            };
            match predictor.predict(&req) {
                Ok(res) => {
                    let mut value = serde_json::to_value(res).unwrap_or(serde_json::Value::Null);
                    crate::apply_confidence_gate(&mut value, minc);
                    ok_text(value)
                }
                Err(e) => tool_error(e.to_string()),
            }
        }
        "oio_predict_batch" => {
            let states = match args.get("states").and_then(|v| v.as_array()) {
                Some(arr) if !arr.is_empty() => arr.clone(),
                _ => return tool_error("'states' must be a non-empty list"),
            };
            if states.len() > 64 {
                return tool_error(format!("too many states in batch ({} > 64)", states.len()));
            }
            // Same acceptance rules as the HTTP surface (review #7).
            let raw_questions = args
                .get("questions")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            for s in &states {
                if let Err((_, msg)) = crate::check_request_limits_inner(s, &raw_questions) {
                    return tool_error(msg);
                }
            }
            let request_like = serde_json::json!({
                "state": null,
                "questions": args.get("questions").cloned().unwrap_or(serde_json::Value::Null),
                "model": args.get("model").cloned().unwrap_or(serde_json::Value::Null),
                "max_len": args.get("max_len").cloned().unwrap_or(serde_json::Value::Null),
                "head_max_len": args.get("head_max_len").cloned().unwrap_or(serde_json::Value::Null),
                "task": args.get("task").cloned().unwrap_or(serde_json::Value::Null),
                "lang": args.get("lang").cloned().unwrap_or(serde_json::Value::Null),
                "lang_guess": args.get("lang_guess").cloned().unwrap_or(serde_json::Value::Null),
                "min_confidence": null,
            });
            let minc = match min_confidence_of(args) {
                Ok(m) => m,
                Err(e) => return tool_error(e),
            };
            let opts = match batch_opts_of(args) {
                Ok(o) => o,
                Err(e) => return tool_error(e),
            };
            let template: SystemOneRequest = match serde_json::from_value(request_like) {
                Ok(t) => t,
                Err(e) => return tool_error(format!("invalid request: {e}")),
            };
            match predictor.predict_batch(&states, template, opts) {
                Ok(results) => {
                    let total: usize = results.iter().map(|r| r.usage.input_tokens).sum();
                    let mut value = serde_json::json!({
                        "results": results,
                        "total_usage": {"input_tokens": total, "output_tokens": 0},
                    });
                    if let Some(rows) = value.get_mut("results").and_then(|v| v.as_array_mut()) {
                        for row in rows {
                            crate::apply_confidence_gate(row, minc);
                        }
                    }
                    ok_text(value)
                }
                Err(e) => tool_error(e.to_string()),
            }
        }
        _ => tool_error(format!("unknown tool {name:?}")),
    }
}

/// Handle one JSON-RPC message; returns the response value (or None for
/// notifications and unknown methods without id).
pub fn handle_message(
    predictor: &Arc<dyn Predictor>,
    msg: &serde_json::Value,
) -> Option<serde_json::Value> {
    let method = msg.get("method").and_then(|v| v.as_str());
    let id = msg.get("id").cloned();
    match method {
        Some("initialize") => Some(json_rpc(
            id,
            Ok(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "oio-mcp", "version": env!("CARGO_PKG_VERSION")},
            })),
        )),
        Some("notifications/initialized") | Some("notifications/cancelled") => None,
        Some("ping") => Some(json_rpc(id, Ok(serde_json::json!({})))),
        Some("tools/list") => Some(json_rpc(id, Ok(tool_list()))),
        Some("tools/call") => {
            let params = msg
                .get("params")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            Some(json_rpc(id, Ok(call_tool(predictor, name, &args))))
        }
        Some(other) => Some(json_rpc(
            id,
            Err((-32601, format!("method not found: {other}"))),
        )),
        None => id
            .is_some()
            .then(|| json_rpc(id, Err((-32600, "invalid request".into())))),
    }
}

fn json_rpc(
    id: Option<serde_json::Value>,
    result: Result<serde_json::Value, (i64, String)>,
) -> serde_json::Value {
    match result {
        Ok(v) => {
            serde_json::json!({"jsonrpc": "2.0", "id": id.unwrap_or(serde_json::Value::Null), "result": v})
        }
        Err((code, message)) => serde_json::json!({
            "jsonrpc": "2.0",
            "id": id.unwrap_or(serde_json::Value::Null),
            "error": {"code": code, "message": message},
        }),
    }
}

/// Run the stdio loop: one newline-delimited JSON-RPC message per line.
pub async fn run_stdio<R, W>(predictor: Arc<dyn Predictor>, reader: R, writer: W) -> io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    let mut out = writer;
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        let response = match parsed {
            Ok(msg) => handle_message(&predictor, &msg),
            Err(_) => Some(json_rpc(None, Err((-32700, "parse error".into())))),
        };
        if let Some(resp) = response {
            let mut bytes = resp.to_string().into_bytes();
            bytes.push(b'\n');
            out.write_all(&bytes).await?;
            out.flush().await?;
        }
    }
    Ok(())
}
