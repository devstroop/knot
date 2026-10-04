//! Checkpoint router — port of `laya/router.py`'s decision logic (M3).
//!
//! Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
//! explicit `lang` > `lang_guess` > detected script/language > default.
//! No inference runs here; `route`/`route_batch` decide which checkpoint a call
//! would take, and the LRU keeps at most `max_loaded` checkpoints resident.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{Error, Result};
use crate::lang::{Analysis, analyse};

/// Hub repo bundling all three checkpoints.
pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

pub const DEFAULT_MODELS: &[(&str, &str, Option<&str>)] = &[
    ("english", BUNDLE_REPO, None),
    ("multilingual", BUNDLE_REPO, Some("multilingual")),
    ("typed-decisions", BUNDLE_REPO, Some("typed-decisions")),
];

const ALIASES: &[(&str, &str)] = &[
    ("en", "english"),
    ("laya", "english"),
    ("default", "english"),
    ("multi", "multilingual"),
    ("ml", "multilingual"),
    ("laya-multilingual", "multilingual"),
    ("typed", "typed-decisions"),
    ("typed_decisions", "typed-decisions"),
    ("laya-typed-decisions", "typed-decisions"),
    ("decisions", "typed-decisions"),
];

const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

const LANGUAGE_AGNOSTIC_CODES: &[&str] = &[
    "c",
    "posix",
    "c.utf-8",
    "posix.utf-8",
    "und",
    "mul",
    "mis",
    "zxx",
    "art",
    "qaa",
];

const ENGLISH_SUBTAGS: &[&str] = &["en"];

/// Name of the typed-decisions workflow whose question ids exactly match, else None.
pub fn match_typed_decisions_workflow(
    questions: &HashMap<String, serde_json::Value>,
) -> Option<&'static str> {
    let ids: HashSet<&str> = questions.keys().map(|s| s.as_str()).collect();
    TYPED_DECISION_WORKFLOWS
        .iter()
        .find(|&(_, sig)| ids == sig.iter().copied().collect())
        .map(|&(wf, _)| wf)
}

pub fn normalise_name(name: &str) -> Result<&'static str> {
    let key = name.trim().to_lowercase();
    let key = ALIASES
        .iter()
        .find(|&&(a, _)| a == key)
        .map(|&(_, k)| k)
        .unwrap_or(&key);
    DEFAULT_MODELS
        .iter()
        .find(|&&(k, _, _)| k == key)
        .map(|&(k, _, _)| k)
        .ok_or_else(|| Error::InvalidRequest(format!("unknown model {name:?}")))
}

/// True/False for a language code, or None when the code identifies nothing.
pub fn english_from_code(value: &str) -> Option<bool> {
    let code = value.trim().to_lowercase();
    if code.is_empty() {
        return None;
    }
    let code = code.split('.').next().unwrap_or("");
    let primary = code.replace('_', "-");
    let primary = primary.split('-').next().unwrap_or("");
    if primary.is_empty() || LANGUAGE_AGNOSTIC_CODES.contains(&primary) {
        return None;
    }
    Some(ENGLISH_SUBTAGS.contains(&primary))
}

/// Human-readable checkpoint id for a routed model: `repo` or `repo/subfolder`
/// (laya `_repo_str(self.models[key])`).
pub fn repo_str(key: &str) -> String {
    DEFAULT_MODELS
        .iter()
        .find(|&&(k, _, _)| k == key)
        .map(|&(_, repo, sub)| match sub {
            Some(s) => format!("{repo}/{s}"),
            None => repo.to_string(),
        })
        .unwrap_or_else(|| BUNDLE_REPO.to_string())
}

#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub model: &'static str,
    /// Laya's `RouteDecision` carries `repo` as its second key.
    pub repo: String,
    pub reason: String,
    pub detection: Option<Analysis>,
    pub workflow: Option<&'static str>,
}

impl RouteDecision {
    fn new(
        model: &'static str,
        reason: String,
        detection: Option<Analysis>,
        workflow: Option<&'static str>,
    ) -> Self {
        RouteDecision {
            repo: repo_str(model),
            model,
            reason,
            detection,
            workflow,
        }
    }
}

pub struct Router {
    pub default: &'static str,
    pub max_loaded: usize,
    pub auto_task_detection: bool,
    pub lang_guess: Option<String>,
    pub(crate) loaded: HashMap<&'static str, ()>,
    pub(crate) order: VecDeque<&'static str>,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    pub fn new() -> Self {
        Self {
            default: "english",
            max_loaded: 2,
            auto_task_detection: false,
            lang_guess: None,
            loaded: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn loaded(&self) -> Vec<&'static str> {
        self.order.iter().copied().collect()
    }

    /// Mark a checkpoint as resident; evict LRU beyond max_loaded. Returns evicted names.
    pub fn touch(&mut self, name: &'static str) -> Vec<&'static str> {
        self.order.retain(|&n| n != name);
        self.order.push_back(name);
        self.loaded.insert(name, ());
        let mut evicted = Vec::new();
        while self.order.len() > self.max_loaded {
            if let Some(old) = self.order.pop_front() {
                self.loaded.remove(old);
                evicted.push(old);
            }
        }
        evicted
    }

    /// Decide which checkpoint to use, without loading or running anything.
    pub fn route(
        &self,
        state: &serde_json::Value,
        questions: Option<&HashMap<String, serde_json::Value>>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&str>,
    ) -> Result<RouteDecision> {
        if let Some(model) = model {
            let key = normalise_name(model)?;
            return Ok(RouteDecision::new(
                key,
                format!("explicit model={model:?}"),
                None,
                None,
            ));
        }

        if let Some(task) = task {
            let normalised = task.to_lowercase().replace('-', "_");
            let key = if normalised == "typed_decisions" {
                "typed-decisions"
            } else {
                normalise_name(task)?
            };
            return Ok(RouteDecision::new(
                key,
                format!("explicit task={task:?}"),
                None,
                None,
            ));
        }

        let workflow = match_typed_decisions_workflow(questions.unwrap_or(&HashMap::new()));
        if workflow.is_some() && self.auto_task_detection {
            return Ok(RouteDecision::new(
                "typed-decisions",
                format!("question ids match the {workflow:?} typed-decisions workflow"),
                None,
                workflow,
            ));
        }

        if let Some(lang) = lang
            && let Some(resolved) = english_from_code(lang)
        {
            let key = if resolved { "english" } else { "multilingual" };
            return Ok(RouteDecision::new(
                key,
                format!("explicit lang={lang:?}"),
                None,
                workflow,
            ));
        }

        for hint in [lang_guess, self.lang_guess.as_deref()] {
            if let Some(resolved) = hint.and_then(english_from_code) {
                let key = if resolved { "english" } else { "multilingual" };
                return Ok(RouteDecision::new(
                    key,
                    format!(
                        "caller identified this as {} text",
                        if resolved { "English" } else { "non-English" }
                    ),
                    None,
                    workflow,
                ));
            }
        }

        let det = analyse(state);
        let (key, reason) = if det.script == "unknown" {
            (
                self.default,
                format!(
                    "no letters detected in state; using default ({})",
                    self.default
                ),
            )
        } else if det.script != "latin" {
            (
                "multilingual",
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            let reason = if let Some(mixed) = &det.mixed_segment {
                format!(
                    "Latin script, mostly English, but a line or field reads as {:?} ({:?}); the English checkpoint cannot read it",
                    det.language,
                    crate::lang::trunc_str(mixed, 60)
                )
            } else if let Some(language) = &det.language {
                format!("Latin script but language looks like {language:?}, not English")
            } else {
                format!(
                    "Latin script, language not identified but {:.0}% non-English letters; not safe for the English checkpoint",
                    100.0 * det.diacritic_rate
                )
            };
            ("multilingual", reason)
        } else if det.language_undecided {
            (
                self.default,
                format!(
                    "Latin script, language not identified and no non-English letters; using default ({})",
                    self.default
                ),
            )
        } else {
            ("english", "English Latin text".into())
        };

        Ok(RouteDecision::new(key, reason, Some(det), workflow))
    }

    pub fn route_batch(
        &self,
        states: &[&serde_json::Value],
        questions: Option<&HashMap<String, serde_json::Value>>,
        model: Option<&str>,
        lang: Option<&str>,
    ) -> Result<Vec<RouteDecision>> {
        states
            .iter()
            .map(|s| self.route(s, questions, model, None, lang, None))
            .collect()
    }
}
