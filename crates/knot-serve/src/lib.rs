//! knot-serve — Jev/Laya-compatible HTTP surface over the knot engine.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub mod mcp;
pub mod model;

use knot::error::Error;
use knot::protocol::SystemOneRequest;

// The inference contract lives in core (ADR-006 rule 6); re-exported here
// so `knot_serve::Predictor` keeps working for servers, tests, and stubs.
pub use knot::Predictor;

pub const MAX_QUESTIONS: usize = 64;
pub const MAX_STATE_CHARS: usize = 50_000;
pub const MAX_BATCH_STATES: usize = 64;
pub const MAX_CHOICE_OPTIONS: usize = 100;
pub const MAX_SCORE_LEVELS: usize = 32;
pub const MAX_TOTAL_OPTIONS: usize = 512;
pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_CONCURRENT: usize = 16;
pub const DEFAULT_MAX_TOKEN_BUDGET: usize = 8192;

const REFUSALS: [&str; 5] = [
    "hooks",
    "on_predict_start",
    "on_predict_end",
    "hooks_raise",
    "hooks_timeout",
];

#[derive(Clone)]
pub struct ServeConfig {
    pub api_key: Option<String>,
    pub max_concurrent: usize,
    pub max_token_budget: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            max_concurrent: DEFAULT_MAX_CONCURRENT,
            max_token_budget: DEFAULT_MAX_TOKEN_BUDGET,
        }
    }
}

struct AppState {
    predictor: Arc<dyn Predictor>,
    config: ServeConfig,
    admission: Arc<tokio::sync::Semaphore>,
    gate: Arc<tokio::sync::Mutex<()>>,
}

fn json_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({"detail": detail.into()}))).into_response()
}

fn authorized(headers: &HeaderMap, api_key: &Option<String>) -> bool {
    match api_key {
        None => true,
        Some(key) => {
            let expected = format!("Bearer {key}");
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|v| constant_time_eq(v.as_bytes(), expected.as_bytes()))
                .unwrap_or(false)
        }
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn state_len(state: &serde_json::Value) -> usize {
    match state {
        serde_json::Value::String(s) => s.chars().count(),
        other => serde_json::to_string(other)
            .map(|s| s.chars().count())
            .unwrap_or(0),
    }
}

/// Shared acceptance rules for every surface (HTTP and MCP). Returns the
/// status and message the HTTP layer uses; MCP reports just the message.
pub(crate) fn check_request_limits_inner(
    state: &serde_json::Value,
    questions: &serde_json::Value,
) -> std::result::Result<(), (StatusCode, String)> {
    if state.is_null() {
        return Err((StatusCode::BAD_REQUEST, "'state' is required".into()));
    }
    let questions = match questions.as_object() {
        Some(q) => q,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                "'questions' must be an object".into(),
            ));
        }
    };
    if questions.len() > MAX_QUESTIONS {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("too many questions ({} > {MAX_QUESTIONS})", questions.len()),
        ));
    }
    let mut total = 0usize;
    for (qid, q) in questions {
        if let Some(qobj) = q.as_object() {
            let qtype = qobj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let crit = qobj.get("criteria");
            let count = match (qtype, crit) {
                ("choice", Some(serde_json::Value::Object(m))) => m.len(),
                ("choice", Some(serde_json::Value::Array(a))) => a.len(),
                ("score", Some(serde_json::Value::Array(a))) => a.len(),
                _ => 0,
            };
            total += count;
            if qtype == "choice" && count > MAX_CHOICE_OPTIONS {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("too many choice options for {qid:?} ({count} > {MAX_CHOICE_OPTIONS})"),
                ));
            }
            if qtype == "score" && count > MAX_SCORE_LEVELS {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("too many score levels for {qid:?} ({count} > {MAX_SCORE_LEVELS})"),
                ));
            }
        }
    }
    if total > MAX_TOTAL_OPTIONS {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})"),
        ));
    }
    let slen = state_len(state);
    if slen > MAX_STATE_CHARS {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("state too large ({slen} > {MAX_STATE_CHARS} chars)"),
        ));
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn check_request_limits(
    state: &serde_json::Value,
    questions: &serde_json::Value,
) -> std::result::Result<(), Response> {
    check_request_limits_inner(state, questions).map_err(|(code, msg)| json_error(code, msg))
}

#[allow(clippy::result_large_err)]
fn check_refusals(body: &serde_json::Value) -> std::result::Result<(), Response> {
    let given: Vec<&str> = REFUSALS
        .iter()
        .copied()
        .filter(|k| body.get(k).map(|v| !v.is_null()).unwrap_or(false))
        .collect();
    if !given.is_empty() {
        return Err(json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "{} run inside the server process and cannot be sent to this endpoint; \
                 install them where knot-serve runs, or drop them",
                given.join(", ")
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn validate_budget(
    body: &serde_json::Value,
    key: &str,
    cap: usize,
) -> std::result::Result<Option<usize>, Response> {
    match body.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(0) => Err(json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("{key} must be a positive integer"),
            )),
            Some(n) if n as usize > cap => Err(json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("{key} exceeds server limit ({n} > {cap})"),
            )),
            Some(n) => Ok(Some(n as usize)),
            None => Err(json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("{key} must be an integer"),
            )),
        },
    }
}

#[allow(clippy::result_large_err)]
fn validate_lang_param(
    body: &serde_json::Value,
    key: &str,
) -> std::result::Result<Option<String>, Response> {
    match body.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("{key} must be a language code string such as \"de\", or null"),
        )),
    }
}

#[allow(clippy::result_large_err)]
fn validate_min_confidence(body: &serde_json::Value) -> std::result::Result<Option<f64>, Response> {
    match body.get("min_confidence") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && (0.0..=1.0).contains(&f) => Ok(Some(f)),
            _ => Err(json_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("min_confidence must be a float in [0.0, 1.0], got {v}"),
            )),
        },
    }
}

/// Laya's batch call controls (`serve.py` `_validate_batch_size_param` /
/// `_validate_sort_by_length_param`): `batch_size` must be a positive
/// integer, `sort_by_length` a boolean; both accept `null` as "not set",
/// each failure a 422 carrying Laya's exact wording. `false` arrives as
/// "the caller did not ask" — Laya only forwards `True` too.
#[allow(clippy::result_large_err)]
fn batch_opts(body: &serde_json::Value) -> std::result::Result<knot::engine::BatchOpts, Response> {
    let batch_size = match body.get("batch_size") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_i64() {
            Some(n) if n >= 1 => Some(n as usize),
            Some(n) => {
                return Err(json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("batch_size must be a positive integer, got {n}"),
                ));
            }
            None => {
                return Err(json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "batch_size must be an integer",
                ));
            }
        },
    };
    let sort_by_length = match body.get("sort_by_length") {
        None | Some(serde_json::Value::Null) => false,
        Some(v) => match v.as_bool() {
            Some(b) => b,
            None => {
                return Err(json_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "sort_by_length must be a boolean",
                ));
            }
        },
    };
    Ok(knot::engine::BatchOpts {
        batch_size,
        sort_by_length,
    })
}

fn map_error(e: Error) -> Response {
    if let Error::CheckpointUnavailable(_) = &e {
        // Issue #37: routed to a checkpoint this deployment does not
        // configure, with no fallback source — 503 says so plainly instead
        // of the generic 500.
        return json_error(StatusCode::SERVICE_UNAVAILABLE, e.to_string());
    }
    match e {
        Error::InvalidRequest(msg) => json_error(StatusCode::UNPROCESSABLE_ENTITY, msg),
        Error::PayloadTooLarge(msg) => json_error(StatusCode::PAYLOAD_TOO_LARGE, msg),
        _ => {
            tracing::error!(error = %e, "inference failed");
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "inference failed")
        }
    }
}

/// The number the abstention gate compares against, rounded to the 4 decimals
/// Laya stores (`round(x, 4)`) so a boundary threshold compares the same value
/// Python would: the wire holds an f32-rounded figure whose f64 reading can sit
/// a hair off the decimal Laya compares.
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// `answer_confidence` if it is a usable finite number, else `confidence`,
/// else None (laya `confidence._gate_confidence`).
fn gate_confidence(answer: &serde_json::Value) -> Option<f64> {
    answer
        .get("answer_confidence")
        .and_then(|v| v.as_f64())
        .filter(|f| f.is_finite())
        .or_else(|| {
            answer
                .get("confidence")
                .and_then(|v| v.as_f64())
                .filter(|f| f.is_finite())
        })
        .map(round4)
}

/// Port of laya `confidence.apply_confidence_gate`: with `min_confidence`
/// unset nothing is written (an ungated call returns exactly the payload it
/// returned before); with it set, `flag_low_confidence` marks the answers
/// below the threshold, then *every* answer reports `abstention`
/// (`passed`/`abstained`/`unevaluated`) plus `abstention_threshold`, in that
/// key order after the flag.
pub(crate) fn apply_confidence_gate(value: &mut serde_json::Value, min_confidence: Option<f64>) {
    let Some(mc) = min_confidence else {
        return;
    };
    let Some(answers) = value.get_mut("answers").and_then(|a| a.as_object_mut()) else {
        return;
    };
    if mc != 0.0 {
        for ans in answers.values_mut() {
            if gate_confidence(ans).is_some_and(|c| c < mc)
                && let Some(obj) = ans.as_object_mut()
            {
                obj.insert("low_confidence".into(), serde_json::Value::Bool(true));
            }
        }
    }
    for ans in answers.values_mut() {
        let conf = gate_confidence(ans);
        let flagged = ans
            .get("low_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Some(obj) = ans.as_object_mut() {
            let state = if flagged {
                "abstained"
            } else if conf.is_none() {
                "unevaluated"
            } else {
                "passed"
            };
            obj.insert("abstention".into(), serde_json::json!(state));
            obj.insert("abstention_threshold".into(), serde_json::json!(mc));
        }
    }
}

#[allow(clippy::result_large_err)]
async fn read_body(req: Request) -> std::result::Result<serde_json::Value, Response> {
    let (_parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body too large",
            ));
        }
    };
    serde_json::from_slice(&bytes)
        .map_err(|_| json_error(StatusCode::BAD_REQUEST, "request body must be valid JSON"))
}

async fn health(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let mut obj = serde_json::json!({"status": "ok"});
    if authorized(&headers, &state.config.api_key) {
        // Laya's detail payload, field for field (`serve.py` /health): the
        // device knot actually loaded checkpoints onto (SPEC §10), zero CPU
        // fallbacks (a provider failure is a load error, never a silent
        // downgrade), and the artifact commit per resident checkpoint, or
        // null for a local dir.
        let names = state.predictor.loaded();
        let device = state.predictor.device();
        obj["loaded"] = serde_json::json!(&names);
        let mut revisions = serde_json::Map::new();
        let mut devices = serde_json::Map::new();
        let mut fallbacks = serde_json::Map::new();
        for name in &names {
            revisions.insert(
                (*name).to_string(),
                state
                    .predictor
                    .revision(name)
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
            devices.insert((*name).to_string(), serde_json::json!(device));
            fallbacks.insert(
                (*name).to_string(),
                serde_json::json!({"count": 0, "last_reason": serde_json::Value::Null}),
            );
        }
        let resident_device = names.first().map(|_| device);
        obj["revisions"] = serde_json::Value::Object(revisions);
        obj["device"] = serde_json::json!(resident_device.unwrap_or(device));
        obj["device_is_preference"] = serde_json::json!(resident_device.is_none());
        obj["checkpoint_devices"] = serde_json::Value::Object(devices);
        obj["cpu_fallbacks"] = serde_json::Value::Object(fallbacks);
    }
    Json(obj).into_response()
}

async fn models(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&headers, &state.config.api_key) {
        return json_error(StatusCode::UNAUTHORIZED, "invalid or missing bearer token");
    }
    Json(serde_json::json!({"models": state.predictor.loaded()})).into_response()
}

struct ParsedControls {
    model: Option<String>,
    max_len: Option<usize>,
    head_max_len: Option<usize>,
    task: Option<String>,
    lang: Option<String>,
    lang_guess: Option<String>,
    min_confidence: Option<f64>,
}

#[allow(clippy::result_large_err)]
fn parse_controls(
    body: &serde_json::Value,
    budget_cap: usize,
) -> std::result::Result<ParsedControls, Response> {
    if body
        .get("task")
        .map(|v| !v.is_null() && !v.is_string())
        .unwrap_or(false)
    {
        return Err(json_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "task must be a string",
        ));
    }
    Ok(ParsedControls {
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        max_len: validate_budget(body, "max_len", budget_cap)?,
        head_max_len: validate_budget(body, "head_max_len", budget_cap)?,
        task: body
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        lang: validate_lang_param(body, "lang")?,
        lang_guess: validate_lang_param(body, "lang_guess")?,
        min_confidence: validate_min_confidence(body)?,
    })
}

async fn systemone(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Response {
    if !authorized(&headers, &state.config.api_key) {
        return json_error(StatusCode::UNAUTHORIZED, "invalid or missing bearer token");
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            let mut resp = json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server busy, try again later",
            );
            resp.headers_mut()
                .insert("retry-after", "1".parse().unwrap());
            return resp;
        }
    };
    let out = async {
        let body = read_body(req).await?;
        if !body.is_object() || body.get("questions").is_none() {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                "request body must be an object with a 'questions' field",
            ));
        }
        let state_value = body
            .get("state")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let questions = body
            .get("questions")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        check_request_limits(&state_value, &questions)?;
        check_refusals(&body)?;
        let controls = parse_controls(&body, state.config.max_token_budget)?;

        let req = SystemOneRequest {
            state: state_value,
            questions: serde_json::from_value(questions).map_err(|_| {
                json_error(
                    StatusCode::BAD_REQUEST,
                    "'questions' must match the request schema",
                )
            })?,
            model: controls.model,
            max_len: controls.max_len,
            head_max_len: controls.head_max_len,
            task: controls.task,
            lang: controls.lang,
            lang_guess: controls.lang_guess,
            min_confidence: None,
        };

        let gate = state.gate.clone();
        let predictor = state.predictor.clone();
        // Start the clock only after the gate is acquired: the header
        // reports inference time, not time spent queued behind it.
        let (inference, infer_ms) = {
            let _guard = gate.lock().await;
            let t0 = Instant::now();
            let inference = tokio::task::spawn_blocking(move || predictor.predict(&req))
                .await
                .map_err(|_| json_error(StatusCode::INTERNAL_SERVER_ERROR, "inference failed"))?;
            (inference, t0.elapsed().as_secs_f64() * 1000.0)
        };
        match inference {
            Ok(res) => {
                let mut value = serde_json::to_value(&res).unwrap_or(serde_json::Value::Null);
                apply_confidence_gate(&mut value, controls.min_confidence);
                let mut resp = Json(value).into_response();
                resp.headers_mut().insert(
                    "server-timing",
                    format!("inference;dur={infer_ms:.2}").parse().unwrap(),
                );
                resp.headers_mut().insert(
                    "x-inference-time-ms",
                    format!("{infer_ms:.2}").parse().unwrap(),
                );
                Ok(resp)
            }
            Err(e) => Ok(map_error(e)),
        }
    }
    .await;
    drop(permit);
    out.unwrap_or_else(|e| e)
}

async fn systemone_batch(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
    req: Request,
) -> Response {
    if !authorized(&headers, &state.config.api_key) {
        return json_error(StatusCode::UNAUTHORIZED, "invalid or missing bearer token");
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            let mut resp = json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server busy, try again later",
            );
            resp.headers_mut()
                .insert("retry-after", "1".parse().unwrap());
            return resp;
        }
    };
    let out = async {
        let body = read_body(req).await?;
        if !body.is_object() || body.get("states").is_none() || body.get("questions").is_none() {
            return Err(json_error(
                StatusCode::BAD_REQUEST,
                "request body must be an object with 'states' and 'questions' fields",
            ));
        }
        let states = match body.get("states").and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => arr.clone(),
            _ => {
                return Err(json_error(
                    StatusCode::BAD_REQUEST,
                    "'states' must be a non-empty list",
                ));
            }
        };
        if states.len() > MAX_BATCH_STATES {
            return Err(json_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "too many states in batch ({} > {MAX_BATCH_STATES})",
                    states.len()
                ),
            ));
        }
        let questions = body
            .get("questions")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        for s in &states {
            check_request_limits(s, &questions)?;
        }
        check_refusals(&body)?;
        let controls = parse_controls(&body, state.config.max_token_budget)?;
        let batch_opts = batch_opts(&body)?;

        let gate = state.gate.clone();
        let predictor = state.predictor.clone();
        let (inference, infer_ms) = {
            let _guard = gate.lock().await;
            let t0 = Instant::now();
            let states_clone = states.clone();
            let template = SystemOneRequest {
                state: serde_json::Value::Null,
                questions: serde_json::from_value(questions.clone()).map_err(|_| {
                    json_error(
                        StatusCode::BAD_REQUEST,
                        "'questions' must match the request schema",
                    )
                })?,
                model: controls.model.clone(),
                max_len: controls.max_len,
                head_max_len: controls.head_max_len,
                task: controls.task.clone(),
                lang: controls.lang.clone(),
                lang_guess: controls.lang_guess.clone(),
                min_confidence: None,
            };
            let inference = tokio::task::spawn_blocking(move || {
                predictor.predict_batch(&states_clone, template, batch_opts)
            })
            .await
            .map_err(|_| json_error(StatusCode::INTERNAL_SERVER_ERROR, "inference failed"))?;
            (inference, t0.elapsed().as_secs_f64() * 1000.0)
        };
        match inference {
            Ok(results) => {
                let total: usize = results.iter().map(|r| r.usage.input_tokens).sum();
                let results: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        let mut v = serde_json::to_value(r).unwrap_or(serde_json::Value::Null);
                        apply_confidence_gate(&mut v, controls.min_confidence);
                        v
                    })
                    .collect();
                let mut resp = Json(serde_json::json!({
                    "results": results,
                    "total_usage": {"input_tokens": total, "output_tokens": 0},
                }))
                .into_response();
                resp.headers_mut().insert(
                    "server-timing",
                    format!("inference;dur={infer_ms:.2}").parse().unwrap(),
                );
                resp.headers_mut().insert(
                    "x-inference-time-ms",
                    format!("{infer_ms:.2}").parse().unwrap(),
                );
                Ok(resp)
            }
            Err(e) => Ok(map_error(e)),
        }
    }
    .await;
    drop(permit);
    out.unwrap_or_else(|e| e)
}

pub fn build_app(predictor: Arc<dyn Predictor>, config: ServeConfig) -> Router {
    let state = Arc::new(AppState {
        predictor,
        admission: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent)),
        gate: Arc::new(tokio::sync::Mutex::new(())),
        config,
    });
    Router::new()
        .route("/health", get(health))
        .route("/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/systemone/batch", post(systemone_batch))
        .with_state(state)
}

#[cfg(test)]
mod gate_tests {
    use super::apply_confidence_gate;

    fn answer(
        confidence: Option<serde_json::Value>,
        answer_confidence: Option<f64>,
    ) -> serde_json::Value {
        let mut a = serde_json::json!({"type": "noul", "noul": 0.5});
        if let Some(c) = confidence {
            a["confidence"] = c;
        }
        if let Some(ac) = answer_confidence {
            a["answer_confidence"] = serde_json::json!(ac);
        }
        a
    }

    fn responses_with(answers: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"model": "laya-rl-agent", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}})
    }

    #[test]
    fn no_threshold_writes_nothing() {
        let mut v = responses_with(serde_json::json!({"q": answer(Some(0.9.into()), Some(0.9))}));
        let before = v.to_string();
        apply_confidence_gate(&mut v, None);
        assert_eq!(v.to_string(), before);
    }

    #[test]
    fn below_threshold_abstains_with_flag_in_laya_key_order() {
        let mut v = responses_with(serde_json::json!({"q": answer(Some(0.9.into()), Some(0.9))}));
        apply_confidence_gate(&mut v, Some(0.95));
        let a = &v["answers"]["q"];
        assert_eq!(a["low_confidence"], serde_json::json!(true));
        assert_eq!(a["abstention"], serde_json::json!("abstained"));
        assert_eq!(a["abstention_threshold"], serde_json::json!(0.95));
        let wire = v.to_string();
        let flag = wire.find("\"low_confidence\"").unwrap();
        let abst = wire.find("\"abstention\"").unwrap();
        let thr = wire.find("\"abstention_threshold\"").unwrap();
        assert!(flag < abst && abst < thr, "key order: {wire}");
    }

    #[test]
    fn at_or_above_threshold_passes() {
        let mut v = responses_with(serde_json::json!({"q": answer(Some(0.9.into()), Some(0.9))}));
        apply_confidence_gate(&mut v, Some(0.5));
        let a = &v["answers"]["q"];
        assert!(a.get("low_confidence").is_none());
        assert_eq!(a["abstention"], serde_json::json!("passed"));
        assert_eq!(a["abstention_threshold"], serde_json::json!(0.5));
    }

    #[test]
    fn missing_gate_confidence_is_unevaluated() {
        let mut v = responses_with(serde_json::json!({"q": answer(None, None)}));
        apply_confidence_gate(&mut v, Some(0.5));
        let a = &v["answers"]["q"];
        assert_eq!(a["abstention"], serde_json::json!("unevaluated"));
        assert_eq!(a["abstention_threshold"], serde_json::json!(0.5));
    }

    /// `min_confidence: 0.0` means "flag off" in laya (`mc == 0.0` skips the
    /// flag pass) but the abstention pass still reports every answer.
    #[test]
    fn zero_threshold_skips_flags_but_still_reports() {
        let mut v = responses_with(serde_json::json!({"q": answer(Some(0.1.into()), Some(0.1))}));
        apply_confidence_gate(&mut v, Some(0.0));
        let a = &v["answers"]["q"];
        assert!(a.get("low_confidence").is_none());
        assert_eq!(a["abstention"], serde_json::json!("passed"));
    }

    /// Without `answer_confidence` the gate falls back to `confidence`.
    #[test]
    fn gate_falls_back_to_confidence() {
        let mut v = responses_with(serde_json::json!({"q": answer(Some(0.4.into()), None)}));
        apply_confidence_gate(&mut v, Some(0.5));
        assert_eq!(
            v["answers"]["q"]["abstention"],
            serde_json::json!("abstained")
        );
    }

    /// The compared value is rounded to 4 decimals like Python's
    /// `round(conf, 4)` before the threshold comparison.
    #[test]
    fn gate_confidence_is_rounded_to_four_decimals() {
        let mut v = responses_with(serde_json::json!({"q": answer(None, Some(0.9000499))}));
        apply_confidence_gate(&mut v, Some(0.9));
        let a = &v["answers"]["q"];
        assert!(a.get("low_confidence").is_none(), "rounds to 0.9, passes");
        assert_eq!(a["abstention"], serde_json::json!("passed"));
    }
}
