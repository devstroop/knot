//! The inference-side contract: what serving layers need from an engine.
//!
//! Owned by core (ADR-006 rule 6): adapters and servers depend on this
//! trait — `knot-serve` re-exports it — never the reverse. Gated with the
//! engine (`BatchOpts` lives there), so a no-runtime build stays clean.

use crate::Result;
use crate::engine::{BatchOpts, Engine};
use crate::protocol::{SystemOneRequest, SystemOneResponse};

/// Everything the server needs from the inference side; `Engine` implements
/// it and tests inject a stub.
pub trait Predictor: Send + Sync {
    fn predict(&self, req: &SystemOneRequest) -> Result<SystemOneResponse>;
    fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        opts: BatchOpts,
    ) -> Result<Vec<SystemOneResponse>>;
    fn loaded(&self) -> Vec<&'static str>;
    /// Device resident checkpoints compute on (SPEC §10) — the `device` /
    /// `checkpoint_devices` entries of `/health`. Defaults to `cpu`, the
    /// only device a stub can have.
    fn device(&self) -> &'static str {
        "cpu"
    }
    /// Artifact commit a resident checkpoint was loaded from — the
    /// `revisions` entry of `/health`. `None` (the default, what stubs
    /// report) means "unknown / local path", Laya's value for one.
    fn revision(&self, _name: &str) -> Option<String> {
        None
    }
    fn route(
        &self,
        state: &serde_json::Value,
        questions: Option<&std::collections::HashMap<String, serde_json::Value>>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&str>,
    ) -> Result<serde_json::Value>;
}

impl Predictor for Engine {
    fn predict(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        Engine::predict(self, req)
    }

    fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        opts: BatchOpts,
    ) -> Result<Vec<SystemOneResponse>> {
        Engine::predict_batch(self, states, template, opts)
    }

    fn loaded(&self) -> Vec<&'static str> {
        self.resident()
    }

    fn device(&self) -> &'static str {
        Engine::device(self).as_str()
    }

    fn revision(&self, name: &str) -> Option<String> {
        Engine::revision(self, name)
    }

    fn route(
        &self,
        state: &serde_json::Value,
        questions: Option<&std::collections::HashMap<String, serde_json::Value>>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&str>,
    ) -> Result<serde_json::Value> {
        let mut d = self
            .router
            .lock()
            .unwrap()
            .route(state, questions, model, task, lang, lang_guess)?;
        self.resolve_decision(&mut d)?;
        // Lay a's `RouteDecision` dict, in its key order, nulls included;
        // `fallback` (knot's #37 extension) appears only on a substitution.
        let mut out = serde_json::json!({
            "model": d.model,
            "repo": d.repo,
            "reason": d.reason,
            "detection": serde_json::to_value(&d.detection).unwrap_or(serde_json::Value::Null),
            "workflow": d.workflow,
        });
        if let Some(fb) = &d.fallback {
            out["fallback"] = serde_json::to_value(fb).unwrap_or(serde_json::Value::Null);
        }
        Ok(out)
    }
}
