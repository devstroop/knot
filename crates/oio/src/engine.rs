//! End-to-end inference: router → checkpoint → prompt → collate → ONNX → answers.
//!
//! Joins M1 (prompt), M2 (runtime), M3 (router) into the `predict` call shape
//! Laya's `Agent.predict` produces, with the same usage block semantics.

use std::collections::BTreeMap;
use std::path::Path;

use indexmap::IndexMap;

#[cfg(feature = "candle")]
use crate::candle_runtime::CandleRuntime;
use crate::error::{Error, Result};
use crate::prompt::{InternalQuestion, PromptBuilder, render_criterion, render_options};
use crate::protocol::{
    AGENT_MODEL, Action, Answer, Routing, SystemOneRequest, SystemOneResponse, Usage,
};
use crate::router::Router;
#[cfg(feature = "onnx")]
use crate::runtime::OnnxRuntime;
use crate::runtime::{Runtime, answer_confidence, confidence_from_probs, scaled_softmax};

/// A loaded checkpoint: tokenizer + ONNX session + budgets + calibration.
pub struct Checkpoint {
    pub name: &'static str,
    pub builder: PromptBuilder,
    pub runtime: std::sync::Arc<dyn Runtime>,
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Checkpoint {
    #[cfg(feature = "onnx")]
    pub fn load(name: &'static str, model_dir: &Path) -> Result<Self> {
        let runtime: std::sync::Arc<dyn Runtime> =
            std::sync::Arc::new(OnnxRuntime::load(model_dir)?);
        let tokenizer = model_dir.join("tokenizer/tokenizer.json");
        let tokenizer = if tokenizer.exists() {
            tokenizer
        } else {
            model_dir.join("tokenizer.json")
        };
        let builder = PromptBuilder::from_file(&tokenizer.to_string_lossy())?;
        let max_len = runtime
            .config()
            .get("max_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let head_max_len = runtime
            .config()
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(192) as usize;
        Ok(Self {
            name,
            builder,
            runtime,
            max_len,
            head_max_len,
        })
    }

    #[cfg(feature = "candle")]
    pub fn load_candle(name: &'static str, model_dir: &Path) -> Result<Self> {
        let runtime: std::sync::Arc<dyn Runtime> =
            std::sync::Arc::new(CandleRuntime::load(model_dir)?);
        let tokenizer = model_dir.join("tokenizer/tokenizer.json");
        let tokenizer = if tokenizer.exists() {
            tokenizer
        } else {
            model_dir.join("tokenizer.json")
        };
        let builder = PromptBuilder::from_file(&tokenizer.to_string_lossy())?;
        let max_len = runtime
            .config()
            .get("max_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let head_max_len = runtime
            .config()
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(192) as usize;
        Ok(Self {
            name,
            builder,
            runtime,
            max_len,
            head_max_len,
        })
    }
}

/// Which backend builds a `Checkpoint` (set by the constructor used).
#[derive(Clone, Copy)]
enum Loader {
    #[cfg(feature = "onnx")]
    Onnx,
    #[cfg(feature = "candle")]
    Candle,
}

impl Loader {
    fn load(self, name: &'static str, dir: &Path) -> Result<Checkpoint> {
        match self {
            #[cfg(feature = "onnx")]
            Loader::Onnx => Checkpoint::load(name, dir),
            #[cfg(feature = "candle")]
            Loader::Candle => Checkpoint::load_candle(name, dir),
        }
    }
}

pub struct Engine {
    pub router: std::sync::Mutex<Router>,
    /// Resident checkpoints. Entries can be evicted by the router LRU
    /// (PRD F6) and are reloaded on demand; each inference holds its own
    /// `Arc`, so a running prediction is never pulled out from under itself.
    checkpoints: std::sync::Mutex<BTreeMap<&'static str, std::sync::Arc<Checkpoint>>>,
    sources: Vec<(&'static str, std::path::PathBuf)>,
    loader: Loader,
}

/// One routed request: parsed questions, route decision, and the wire request
/// (state plus budget overrides).
struct Item {
    questions: IndexMap<String, InternalQuestion>,
    decision: crate::router::RouteDecision,
    req: SystemOneRequest,
}

/// Per-item decode accumulator for a collated group forward.
#[derive(Default)]
struct ItemAcc {
    answers: IndexMap<String, Answer>,
    dropped_max: usize,
    state_tokens: usize,
    truncated_questions: Vec<String>,
    collapsed: BTreeMap<String, serde_json::Value>,
    input_tokens: usize,
}

/// Round to 4 decimals like Laya's answer builder.
fn r4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

impl Engine {
    #[cfg(feature = "onnx")]
    pub fn load(router: Router, dirs: &[(&'static str, &Path)]) -> Result<Self> {
        Self::load_with(router, dirs, Loader::Onnx)
    }

    #[cfg(feature = "candle")]
    pub fn load_candle(router: Router, dirs: &[(&'static str, &Path)]) -> Result<Self> {
        Self::load_with(router, dirs, Loader::Candle)
    }

    fn load_with(router: Router, dirs: &[(&'static str, &Path)], loader: Loader) -> Result<Self> {
        let max_loaded = router.max_loaded;
        let sources: Vec<(&'static str, std::path::PathBuf)> =
            dirs.iter().map(|(n, p)| (*n, (*p).to_path_buf())).collect();
        let engine = Self {
            router: std::sync::Mutex::new(router),
            checkpoints: std::sync::Mutex::new(BTreeMap::new()),
            sources,
            loader,
        };
        // Eagerly bring up to `max_loaded` checkpoints resident (the LRU
        // bound PRD F6 promises); any further model loads on first use and
        // enters the LRU like every other access.
        for (name, dir) in dirs {
            if engine.checkpoints.lock().unwrap().len() >= max_loaded {
                break;
            }
            let ckpt = loader.load(name, dir)?;
            let mut cps = engine.checkpoints.lock().unwrap();
            cps.insert(*name, std::sync::Arc::new(ckpt));
            engine.router.lock().unwrap().touch(name);
        }
        Ok(engine)
    }

    /// Names of the checkpoints currently resident (LRU-bounded).
    pub fn resident(&self) -> Vec<&'static str> {
        self.checkpoints.lock().unwrap().keys().copied().collect()
    }

    /// The checkpoint for `name`, touching the LRU and (re)loading it if it
    /// is not resident. Returns an owned handle so inference keeps running
    /// even if a concurrent request evicts the model meanwhile.
    fn checkpoint(&self, name: &'static str) -> Result<std::sync::Arc<Checkpoint>> {
        {
            let mut cps = self.checkpoints.lock().unwrap();
            if let Some(c) = cps.get(name) {
                let c = std::sync::Arc::clone(c);
                let evicted = self.router.lock().unwrap().touch(name);
                for e in evicted {
                    cps.remove(e);
                }
                return Ok(c);
            }
        }
        let dir = self
            .sources
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, p)| p.clone())
            .ok_or_else(|| Error::Model(format!("checkpoint {name:?} has no source directory")))?;
        let loaded = self.loader.load(name, &dir)?;
        let mut cps = self.checkpoints.lock().unwrap();
        if let Some(c) = cps.get(name) {
            // another thread reloaded it while we were loading
            let c = std::sync::Arc::clone(c);
            let evicted = self.router.lock().unwrap().touch(name);
            for e in evicted {
                cps.remove(e);
            }
            return Ok(c);
        }
        let arc = std::sync::Arc::new(loaded);
        cps.insert(name, std::sync::Arc::clone(&arc));
        let evicted = self.router.lock().unwrap().touch(name);
        for e in evicted {
            cps.remove(e);
        }
        Ok(arc)
    }

    /// Parse and route one request into a collatable item.
    fn prepare(&self, req: SystemOneRequest) -> Result<Item> {
        let mut questions: IndexMap<String, InternalQuestion> = IndexMap::new();
        for (qid, q) in &req.questions {
            if qid.trim().is_empty() {
                return Err(Error::InvalidRequest(format!(
                    "question id must be a non-empty string, got {qid:?}"
                )));
            }
            questions.insert(qid.clone(), InternalQuestion::from_wire(qid, q)?);
        }

        let raw_questions: std::collections::HashMap<String, serde_json::Value> = req
            .questions
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap()))
            .collect();
        let decision = self.router.lock().unwrap().route(
            &req.state,
            Some(&raw_questions),
            req.model.as_deref().filter(|m| !m.starts_with("jev")),
            req.task.as_deref(),
            req.lang.as_deref(),
            req.lang_guess.as_deref(),
        )?;
        Ok(Item {
            questions,
            decision,
            req,
        })
    }

    /// Predict one state over a question set — the Laya `predict` shape.
    pub fn predict(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        let item = self.prepare(req.clone())?;
        if item.questions.is_empty() {
            // Laya short-circuits empty questions without tokenizing or running
            // a forward pass: empty answers, the two zero counters, no more.
            return Ok(Self::empty_response(&item));
        }
        let ckpt = self.checkpoint(item.decision.model)?;
        let mut out = self.run_group(&ckpt, &[&item])?;
        Ok(out.pop().expect("one response per item"))
    }

    /// The empty-questions result (laya `predict_batch`'s `if not ids` path).
    fn empty_response(item: &Item) -> SystemOneResponse {
        SystemOneResponse {
            model: AGENT_MODEL.into(),
            answers: IndexMap::new(),
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                state_tokens: None,
                state_tokens_dropped: None,
                truncated: None,
                truncated_questions: None,
                options: None,
                windows: None,
            },
            routing: Routing::from(&item.decision),
            shortlist: None,
        }
    }

    /// Rows for every item of one model, a single collated forward, then
    /// per-item answers/usage. `predict` is the n = 1 case; `predict_batch`
    /// groups all its states by routed model so N states cost one forward
    /// pass per model (Laya `predict_batch`'s collation property).
    fn run_group(&self, ckpt: &Checkpoint, items: &[&Item]) -> Result<Vec<SystemOneResponse>> {
        struct Row {
            item: usize,
            qid: String,
            ids: Vec<u32>,
            markers: Vec<usize>,
            qtype: i64,
            order: Option<Vec<usize>>,
            options_stats: crate::prompt::HeadStats,
            state_stats: crate::prompt::TruncationStats,
        }
        let mut rows: Vec<Row> = Vec::new();
        for (ii, item) in items.iter().enumerate() {
            let max_len = item.req.max_len.unwrap_or(ckpt.max_len);
            let head_max_len = item.req.head_max_len.unwrap_or(ckpt.head_max_len);
            let state_ids = ckpt.builder.encode_state(&item.req.state)?;
            for (qid, q) in &item.questions {
                let n_opts = render_options(q)?.len();
                let wire_order = item
                    .req
                    .questions
                    .get(qid)
                    .and_then(|w| w.option_order.as_deref());
                let order = match wire_order {
                    None => None,
                    Some(o) => {
                        if o.len() != n_opts {
                            return Err(Error::InvalidRequest(format!(
                                "question {qid:?}: option_order must have one index per option ({n_opts} expected, got {})",
                                o.len()
                            )));
                        }
                        let mut sorted = o.to_vec();
                        sorted.sort_unstable();
                        if sorted.iter().enumerate().any(|(i, &v)| v != i) {
                            return Err(Error::InvalidRequest(format!(
                                "question {qid:?}: option_order must be a permutation of 0..{n_opts}"
                            )));
                        }
                        Some(o.to_vec())
                    }
                };
                let (ids, markers, stats, state_stats) = ckpt.builder.build_sequence(
                    &state_ids,
                    q,
                    max_len,
                    head_max_len,
                    order.as_deref(),
                    false,
                )?;
                if markers.len() != n_opts {
                    return Err(Error::InvalidRequest(format!(
                        "question {qid:?}: only {} of its {} option markers fit in max_len={max_len} with head_max_len={head_max_len}",
                        markers.len(),
                        n_opts
                    )));
                }
                let qtype = match q.t.as_str() {
                    "choice" => 0,
                    "score" => 1,
                    _ => 2,
                };
                rows.push(Row {
                    item: ii,
                    qid: qid.clone(),
                    ids,
                    markers,
                    qtype,
                    order,
                    options_stats: stats,
                    state_stats,
                });
            }
        }

        // Collate: pad to max row length / max marker count (Laya `collate_items`).
        let n = rows.len();
        let lmax = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
        let pad_id = ckpt.builder.pad_id() as i64;
        let mut input_ids = vec![pad_id; n * lmax];
        let mut att = vec![0i64; n * lmax];
        let mut mpos = vec![0i64; n * kmax];
        let mut mmask = vec![false; n * kmax];
        let mut qtype = vec![0i64; n];
        for (i, r) in rows.iter().enumerate() {
            for (j, &id) in r.ids.iter().enumerate() {
                input_ids[i * lmax + j] = id as i64;
                att[i * lmax + j] = 1;
            }
            for (j, &m) in r.markers.iter().enumerate() {
                mpos[i * kmax + j] = m as i64;
                mmask[i * kmax + j] = true;
            }
            qtype[i] = r.qtype;
        }

        let (logits, act) = ckpt
            .runtime
            .forward(&input_ids, &att, &mpos, &mmask, &qtype, n, lmax, kmax)?;

        // Decode answers per row, accumulated per item.
        let mut acc: Vec<ItemAcc> = (0..items.len()).map(|_| ItemAcc::default()).collect();
        for (i, r) in rows.iter().enumerate() {
            let k = r.markers.len();
            let qtype_name = match r.qtype {
                0 => "choice",
                1 => "score",
                _ => "noul",
            };
            let t = ckpt
                .runtime
                .calibration()
                .for_question(qtype_name, r.qtype as usize, k);
            let p_slot = scaled_softmax(&logits[i][..k], t);
            // `build_sequence` put option `order[s]` in slot `s`, so the model
            // row comes back slot-indexed; downstream (keys, level index,
            // noul-true) all index by option — same as Laya's unpermute_probs.
            let p = match &r.order {
                Some(order) if order.len() == p_slot.len() => {
                    let mut out = vec![0.0f32; p_slot.len()];
                    for (slot, &opt) in order.iter().enumerate() {
                        out[opt] = p_slot[slot];
                    }
                    out
                }
                _ => p_slot,
            };
            let ans_conf = answer_confidence(&p);
            // Laya softmaxes the act head before exposing slot 0
            // (onnx_agent: act = softmax(act_logits); ext act_probability = act[0]).
            let act_p = scaled_softmax(&act[i], 1.0);
            let act_prob = r4(act_p[0]);

            let answer = match qtype_name {
                "choice" => {
                    let keys: Vec<String> = match &items[r.item].questions[&r.qid].crit {
                        Some(serde_json::Value::Object(m)) => m.keys().cloned().collect(),
                        _ => Vec::new(),
                    };
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let probs: IndexMap<String, f32> = keys
                        .iter()
                        .enumerate()
                        .map(|(j, kk)| (kk.clone(), r4(p[j])))
                        .collect();
                    Answer::Choice {
                        choice: keys.get(best).cloned().unwrap_or_default(),
                        probabilities: probs,
                        confidence: r4(confidence_from_probs(&p)),
                        answer_confidence: r4(ans_conf),
                        action: Action {
                            act_probability: act_prob,
                        },
                        window: None,
                    }
                }
                "score" => {
                    let crits: Vec<String> = match &items[r.item].questions[&r.qid].crit {
                        Some(serde_json::Value::Array(a)) => {
                            a.iter().map(render_criterion).collect()
                        }
                        _ => Vec::new(),
                    };
                    let exp: f32 = p.iter().enumerate().map(|(j, &v)| j as f32 * v).sum();
                    let legend = crits
                        .iter()
                        .enumerate()
                        .map(|(j, c)| (j.to_string(), c.clone()))
                        .collect::<IndexMap<String, String>>();
                    let probs: IndexMap<String, f32> = p
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| (j.to_string(), r4(v)))
                        .collect();
                    Answer::Score {
                        score: r4(exp),
                        legend,
                        probabilities: probs,
                        confidence: r4(confidence_from_probs(&p)),
                        answer_confidence: r4(ans_conf),
                        action: Action {
                            act_probability: act_prob,
                        },
                        window: None,
                    }
                }
                _ => Answer::Noul {
                    noul: r4(p[1]),
                    confidence: r4(p[1].max(1.0 - p[1])),
                    answer_confidence: r4(ans_conf),
                    action: Action {
                        act_probability: act_prob,
                    },
                    window: None,
                },
            };
            acc[r.item].answers.insert(r.qid.clone(), answer);

            let a = &mut acc[r.item];
            a.dropped_max = a.dropped_max.max(r.state_stats.state_tokens_dropped);
            a.state_tokens = a.state_tokens.max(r.state_stats.state_tokens);
            if r.state_stats.truncated {
                a.truncated_questions.push(r.qid.clone());
            }
            if r.options_stats.options_distinct < r.options_stats.options {
                a.collapsed.insert(
                    r.qid.clone(),
                    serde_json::json!({
                        "total": r.options_stats.options,
                        "distinct": r.options_stats.options_distinct,
                        "tokens_per_option": r.options_stats.tokens_per_option,
                    }),
                );
            }
            acc[r.item].input_tokens += r.ids.len();
        }

        acc.into_iter()
            .zip(items.iter())
            .map(|(a, item)| {
                let usage = Usage {
                    input_tokens: a.input_tokens,
                    output_tokens: 0,
                    state_tokens: Some(a.state_tokens),
                    state_tokens_dropped: Some(a.dropped_max),
                    truncated: Some(a.dropped_max > 0),
                    truncated_questions: Some(a.truncated_questions),
                    options: if a.collapsed.is_empty() {
                        None
                    } else {
                        Some(serde_json::to_value(&a.collapsed)?)
                    },
                    windows: None,
                };
                Ok(SystemOneResponse {
                    model: AGENT_MODEL.into(),
                    answers: a.answers,
                    usage,
                    routing: Routing::from(&item.decision),
                    shortlist: None,
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Run the same request shape over a list of states, Laya `predict_batch`'s
    /// per-item result shape (one `SystemOneResponse` per state). States are
    /// routed first, then grouped by checkpoint so each group is a single
    /// collated forward pass instead of one pass per state.
    pub fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
    ) -> Result<Vec<SystemOneResponse>> {
        let mut items: Vec<Item> = Vec::with_capacity(states.len());
        for state in states {
            items.push(self.prepare(SystemOneRequest {
                state: state.clone(),
                ..template.clone()
            })?);
        }
        let mut by_model: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
        let mut out: Vec<Option<SystemOneResponse>> = states.iter().map(|_| None).collect();
        for (i, item) in items.iter().enumerate() {
            if item.questions.is_empty() {
                out[i] = Some(Self::empty_response(item));
            } else {
                by_model.entry(item.decision.model).or_default().push(i);
            }
        }
        for (model, idxs) in by_model {
            let ckpt = self.checkpoint(model)?;
            let group: Vec<&Item> = idxs.iter().map(|&i| &items[i]).collect();
            let responses = self.run_group(&ckpt, &group)?;
            for (slot, resp) in idxs.into_iter().zip(responses) {
                out[slot] = Some(resp);
            }
        }
        Ok(out
            .into_iter()
            .map(|r| r.expect("one response per state"))
            .collect())
    }

    /// Laya `predict_long`: scan a long state in overlapping windows and
    /// aggregate per question — `noul` takes the strongest window, `choice` /
    /// `score` the most-confident one. Each deciding answer carries a
    /// `window` reference; `usage.windows` counts scored windows.
    pub fn predict_long(
        &self,
        req: &SystemOneRequest,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> Result<SystemOneResponse> {
        let mut questions: IndexMap<String, InternalQuestion> = IndexMap::new();
        for (qid, q) in &req.questions {
            if qid.trim().is_empty() {
                return Err(Error::InvalidRequest(format!(
                    "question id must be a non-empty string, got {qid:?}"
                )));
            }
            questions.insert(qid.clone(), InternalQuestion::from_wire(qid, q)?);
        }

        let raw_questions: std::collections::HashMap<String, serde_json::Value> = req
            .questions
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap()))
            .collect();
        let decision = self.router.lock().unwrap().route(
            &req.state,
            Some(&raw_questions),
            req.model.as_deref().filter(|m| !m.starts_with("jev")),
            req.task.as_deref(),
            req.lang.as_deref(),
            req.lang_guess.as_deref(),
        )?;
        let ckpt = self.checkpoint(decision.model)?;

        let max_len = req.max_len.unwrap_or(ckpt.max_len);
        let head_max_len = req.head_max_len.unwrap_or(ckpt.head_max_len);
        let qs: Vec<InternalQuestion> = questions.values().cloned().collect();
        let (budget, step, _room) =
            ckpt.builder
                .window_budget(&qs, max_len, head_max_len, window, stride)?;

        let state_ids = ckpt.builder.encode_state(&req.state)?;
        if state_ids.len() <= budget {
            let mut res = self.predict(req)?;
            res.usage.windows = Some(1);
            return Ok(res);
        }

        let mut windows: Vec<serde_json::Value> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        let mut i = 0usize;
        while i < state_ids.len() {
            let end = (i + budget).min(state_ids.len());
            windows.push(serde_json::Value::String(
                ckpt.builder.decode_ids(&state_ids[i..end]),
            ));
            starts.push(i);
            if end >= state_ids.len() {
                break;
            }
            i += step;
        }

        let results = self.predict_batch(
            &windows,
            SystemOneRequest {
                state: serde_json::Value::Null,
                ..req.clone()
            },
        )?;

        let mut answers: IndexMap<String, Answer> = IndexMap::new();
        for (qid, q) in &questions {
            let is_noul = q.t == "noul";
            let mut best = 0usize;
            let mut best_v = f32::MIN;
            for (j, r) in results.iter().enumerate() {
                let a = &r.answers[qid];
                let v = if is_noul {
                    a.noul_prob().unwrap_or(0.0)
                } else {
                    a.answer_confidence()
                };
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            let deciding =
                results[best].answers[qid]
                    .clone()
                    .with_window(crate::protocol::WindowRef {
                        index: best,
                        token_start: starts[best],
                        token_end: (starts[best] + budget).min(state_ids.len()),
                        count: results.len(),
                    });
            answers.insert(qid.clone(), deciding);
        }

        // Lay a's generic aggregation (`predict_long`): sum numeric fields
        // across windows — keys appear only if some window reported them —
        // merge `options`, then record the window count. `truncated` stays a
        // bool (OR) here; Laya's bool-is-int addition makes it a count, which
        // COMPAT.md documents as a deliberate difference.
        let mut usage = Usage {
            input_tokens: 0,
            output_tokens: 0,
            state_tokens: None,
            state_tokens_dropped: None,
            truncated: None,
            truncated_questions: None,
            options: None,
            windows: Some(results.len()),
        };
        let mut any_truncated = false;
        for r in &results {
            usage.input_tokens += r.usage.input_tokens;
            if let Some(v) = r.usage.state_tokens {
                usage.state_tokens = Some(usage.state_tokens.unwrap_or(0) + v);
            }
            if let Some(v) = r.usage.state_tokens_dropped {
                usage.state_tokens_dropped = Some(usage.state_tokens_dropped.unwrap_or(0) + v);
            }
            any_truncated |= r.usage.truncated.unwrap_or(false);
            if let Some(tq) = &r.usage.truncated_questions {
                usage.truncated_questions = Some(tq.clone());
            }
            match (&r.usage.options, &mut usage.options) {
                (Some(o), Some(prev)) => {
                    if let (Some(om), Some(pm)) = (o.as_object(), prev.as_object_mut()) {
                        for (k, v) in om {
                            pm.insert(k.clone(), v.clone());
                        }
                    }
                }
                (Some(_), None) => usage.options = r.usage.options.clone(),
                _ => {}
            }
        }
        // Laya sums the windows' bools into a count (0..N); oio reports the OR
        // as a bool — the documented divergence in COMPAT.md, but the key is
        // present either way because every window carries it.
        usage.truncated = Some(any_truncated);

        Ok(SystemOneResponse {
            model: AGENT_MODEL.into(),
            answers,
            usage,
            routing: Routing::from(&decision),
            shortlist: None,
        })
    }
}
