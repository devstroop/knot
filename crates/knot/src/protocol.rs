//! Wire types mirroring the Jev/Laya `/v1/systemone` contract.
//!
//! Field names and order follow Laya's actual payloads (`agent.py` answer
//! builders, `Router.predict`): responses are `{model, answers, usage,
//! routing}`, `routing` is the `RouteDecision` dict (`model`, `repo`, `reason`,
//! `detection`, `workflow` — null when absent, never dropped), and maps keep
//! caller/insertion order (Laya dicts are ordered; serde's default BTreeMap
//! sorted them).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::lang::Analysis;
use crate::router::RouteDecision;

/// Answer payload `model` value: the name Laya reports on every result.
pub const AGENT_MODEL: &str = "laya-rl-agent";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

/// A single typed question as it appears on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub r#type: QuestionType,
    /// Laya accepts a string, dict, list or number and renders non-strings
    /// with `json.dumps` (agent.py `_to_internal`); null/empty containers
    /// become empty text and are rejected as caller errors by the engine.
    #[serde(default, deserialize_with = "de_instructions")]
    pub instructions: String,
    /// `choice`: label -> description map or a list of labels (normalized to a
    /// map by `InternalQuestion::from_wire`); `score`: ordered levels;
    /// `noul`: `{"false": ..., "true": ...}` criteria.
    #[serde(default)]
    pub criteria: serde_json::Value,
    #[serde(default)]
    pub labels: Option<serde_json::Value>,
    /// Caller option order: slot `s` in the prompt shows option `order[s]`.
    /// Validated as a permutation of `0..option_count`; probabilities are
    /// un-permuted back before answers are built (Laya `unpermute_probs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub option_order: Option<Vec<usize>>,
}

fn de_instructions<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => String::new(),
        serde_json::Value::Array(a) if a.is_empty() => String::new(),
        serde_json::Value::Object(o) if o.is_empty() => String::new(),
        // Python `json.dumps(ins, ensure_ascii=False)` (agent.py `_to_internal`)
        other => crate::pyjson::dumps(&other),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: serde_json::Value,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_max_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang_guess: Option<String>,
    #[serde(default)]
    pub min_confidence: Option<f64>,
}

/// Per-result token report. Laya always writes `state_tokens`,
/// `state_tokens_dropped`, `truncated` and `truncated_questions` once a forward
/// pass ran (even as `0`/`false`/`[]`), and omits them only on the
/// empty-questions short path (which reports just the two counters), so those
/// four are `Option`s: `Some` on the scored path, `None` when no forward ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_tokens_dropped: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated_questions: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub act_probability: f32,
}

/// Which decoded window produced a long-document answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowRef {
    pub index: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub count: usize,
}

/// Answer payload; `type` discriminates, mirroring Laya's per-type shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
        answer_confidence: f32,
        action: Action,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<WindowRef>,
    },
    Score {
        score: f32,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f32>,
        confidence: f32,
        answer_confidence: f32,
        action: Action,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<WindowRef>,
    },
    Noul {
        noul: f32,
        confidence: f32,
        answer_confidence: f32,
        action: Action,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window: Option<WindowRef>,
    },
}

impl Answer {
    pub fn answer_confidence(&self) -> f32 {
        match self {
            Answer::Choice {
                answer_confidence, ..
            } => *answer_confidence,
            Answer::Score {
                answer_confidence, ..
            } => *answer_confidence,
            Answer::Noul {
                answer_confidence, ..
            } => *answer_confidence,
        }
    }

    pub fn noul_prob(&self) -> Option<f32> {
        match self {
            Answer::Noul { noul, .. } => Some(*noul),
            _ => None,
        }
    }

    pub fn with_window(mut self, w: WindowRef) -> Self {
        match &mut self {
            Answer::Choice { window, .. } => *window = Some(w),
            Answer::Score { window, .. } => *window = Some(w),
            Answer::Noul { window, .. } => *window = Some(w),
        }
        self
    }
}

/// Laya's `RouteDecision` dict, serialized with all five keys in its order;
/// `detection`/`workflow` are `null` when absent rather than omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Routing {
    pub model: String,
    pub repo: String,
    pub reason: String,
    #[serde(default)]
    pub detection: Option<Analysis>,
    #[serde(default)]
    pub workflow: Option<String>,
}

impl From<&RouteDecision> for Routing {
    fn from(d: &RouteDecision) -> Self {
        Routing {
            model: d.model.to_string(),
            repo: d.repo.clone(),
            reason: d.reason.clone(),
            detection: d.detection.clone(),
            workflow: d.workflow.map(str::to_string),
        }
    }
}

/// Laya result shape: `{model, answers, usage, routing}` in that order
/// (`Agent.predict_batch` builds the first three, `Router.predict` appends
/// `routing`); `shortlist` is knot's own extension and is omitted when unused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    pub routing: Routing,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortlist: Option<serde_json::Value>,
}
