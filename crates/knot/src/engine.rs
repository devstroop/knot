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
use crate::runtime::{Device, Runtime, answer_confidence, confidence_from_probs, scaled_softmax};

/// A loaded checkpoint: tokenizer + ONNX session + budgets + calibration.
pub struct Checkpoint {
    pub name: &'static str,
    pub builder: PromptBuilder,
    pub runtime: std::sync::Arc<dyn Runtime>,
    pub max_len: usize,
    pub head_max_len: usize,
    /// Artifact commit the checkpoint was downloaded at (Laya's
    /// `loaded_revisions` value): the HF snapshot etag when the cache is
    /// present, else `None` for a plain local directory.
    pub revision: Option<String>,
}

/// Read the snapshot commit from a Hugging Face cache inside `model_dir`.
/// Every file's `.metadata` first line carries the same etag — the commit
/// the snapshot resolved to — so one file is enough; a directory without
/// the cache (weights copied by hand) reports `None`, which is exactly what
/// Laya's `loaded_revisions` returns for a local path.
fn snapshot_revision(model_dir: &Path) -> Option<String> {
    for rel in [
        "download/rl_agent_config.json.metadata",
        "download/model.safetensors.metadata",
    ] {
        let path = model_dir.join(".cache/huggingface").join(rel);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let first = text.lines().next().unwrap_or("").trim();
        if first.len() == 40 && first.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Some(first.to_string());
        }
    }
    None
}

impl Checkpoint {
    #[cfg(feature = "onnx")]
    pub fn load(name: &'static str, model_dir: &Path, device: Device) -> Result<Self> {
        crate::integrity::verify_sha256sums(model_dir)?;
        let runtime: std::sync::Arc<dyn Runtime> =
            std::sync::Arc::new(OnnxRuntime::load(model_dir, device)?);
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
            revision: snapshot_revision(model_dir),
        })
    }

    #[cfg(feature = "candle")]
    pub fn load_candle(name: &'static str, model_dir: &Path) -> Result<Self> {
        crate::integrity::verify_sha256sums(model_dir)?;
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
            revision: snapshot_revision(model_dir),
        })
    }
}

/// Which backend builds a `Checkpoint` (set by the constructor used).
#[derive(Clone, Copy)]
enum Loader {
    #[cfg(feature = "onnx")]
    Onnx(Device),
    #[cfg(feature = "candle")]
    Candle,
}

impl Loader {
    fn load(self, name: &'static str, dir: &Path) -> Result<Checkpoint> {
        match self {
            #[cfg(feature = "onnx")]
            Loader::Onnx(device) => Checkpoint::load(name, dir, device),
            #[cfg(feature = "candle")]
            Loader::Candle => Checkpoint::load_candle(name, dir),
        }
    }

    /// The device this loader puts checkpoints on (SPEC §10).
    fn device(self) -> Device {
        match self {
            #[cfg(feature = "onnx")]
            Loader::Onnx(device) => device,
            #[cfg(feature = "candle")]
            Loader::Candle => Device::Cpu,
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

/// Call controls for one `predict_batch` (Laya's `batch_size` /
/// `sort_by_length` kwargs). `Default` is Laya's own default: one forward
/// pass for the whole group, original order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchOpts {
    pub batch_size: Option<usize>,
    pub sort_by_length: bool,
}

/// Laya's chunk plan (`agent.py` `predict_batch` loop): the states of one
/// model group, split into forward-pass batches in execution order.
///
/// - `batch_size` bounds states per pass (`None` / `0` → all of them);
/// - `sort_by_length` stably sorts states by encoded length inside windows
///   of `8 × batch_size`, and only when `1 < batch_size < n` — otherwise
///   the flag has no effect, exactly as the docstring promises;
/// - membership, not output order: results are written back to their input
///   positions, so a caller always sees input order.
pub(crate) fn batch_chunks(n: usize, opts: BatchOpts, lens: &[usize]) -> Vec<Vec<usize>> {
    debug_assert_eq!(lens.len(), n);
    if n == 0 {
        return Vec::new();
    }
    let chunk = opts.batch_size.filter(|&c| c > 0).unwrap_or(n);
    let reorder = opts.sort_by_length && chunk > 1 && chunk < n;
    let window = if reorder { chunk * 8 } else { chunk };
    let mut out = Vec::new();
    let mut w_start = 0;
    while w_start < n {
        let w_end = (w_start + window).min(n);
        let mut order: Vec<usize> = (w_start..w_end).collect();
        if reorder {
            order.sort_by_key(|&i| lens[i]);
        }
        let mut c_start = 0;
        while c_start < order.len() {
            let c_end = (c_start + chunk).min(order.len());
            out.push(order[c_start..c_end].to_vec());
            c_start = c_end;
        }
        w_start = w_end;
    }
    out
}

/// Round to 4 decimals like Laya's answer builder.
fn r4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

impl Engine {
    #[cfg(feature = "onnx")]
    pub fn load(router: Router, dirs: &[(&'static str, &Path)]) -> Result<Self> {
        Self::load_with_device(router, dirs, Device::Cpu)
    }

    /// Load onto an explicit device (SPEC §10): `Device::Cuda` registers
    /// ort's CUDA execution provider and fails fast when it cannot come
    /// up — never a silent CPU fallback (PRD §4).
    #[cfg(feature = "onnx")]
    pub fn load_with_device(
        router: Router,
        dirs: &[(&'static str, &Path)],
        device: Device,
    ) -> Result<Self> {
        Self::load_with(router, dirs, Loader::Onnx(device))
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

    /// The device resident checkpoints compute on (SPEC §10) — `/health`
    /// reports it as `device` / `checkpoint_devices`.
    pub fn device(&self) -> Device {
        self.loader.device()
    }

    /// Names of the checkpoints currently resident (LRU-bounded).
    pub fn resident(&self) -> Vec<&'static str> {
        self.checkpoints.lock().unwrap().keys().copied().collect()
    }

    /// Artifact commit for a resident checkpoint, keyed by name for the
    /// `/health` payload; `None` when the checkpoint is not resident or its
    /// directory carries no HF snapshot metadata.
    pub fn revision(&self, name: &str) -> Option<String> {
        self.checkpoints
            .lock()
            .unwrap()
            .get(name)
            .and_then(|c| c.revision.clone())
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
        let mut out = self.run_group(&ckpt, &[&item], BatchOpts::default())?;
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
    fn run_group(
        &self,
        ckpt: &Checkpoint,
        items: &[&Item],
        opts: BatchOpts,
    ) -> Result<Vec<SystemOneResponse>> {
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

        // Per-item bookkeeping: which rows belong to each state, and the
        // post-truncation length laya sorts by (max row length per state).
        let mut item_rows: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
        let mut item_len: Vec<usize> = vec![0; items.len()];
        for (ri, r) in rows.iter().enumerate() {
            item_rows[r.item].push(ri);
            item_len[r.item] = item_len[r.item].max(r.ids.len());
        }
        let mut acc: Vec<ItemAcc> = (0..items.len()).map(|_| ItemAcc::default()).collect();

        // Laya's chunk plan: windows of 8 × chunk when reordering, a stable
        // length sort inside each window, forward passes of at most `chunk`
        // states (Laya `collate_items` per chunk). Collating per chunk is
        // what bounds padding; batch shapes may shift floats slightly — the
        // trade laya documents for `sort_by_length`.
        for group in batch_chunks(items.len(), opts, &item_len) {
            let chunk_row_indices: Vec<usize> = group
                .iter()
                .flat_map(|&i| item_rows[i].iter().copied())
                .collect();
            let chunk_rows: Vec<&Row> = chunk_row_indices.iter().map(|&ri| &rows[ri]).collect();

            // Collate: pad to this chunk's max row length / marker count.
            let n = chunk_rows.len();
            let lmax = chunk_rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
            let kmax = chunk_rows
                .iter()
                .map(|r| r.markers.len())
                .max()
                .unwrap_or(0);
            let pad_id = ckpt.builder.pad_id() as i64;
            let mut input_ids = vec![pad_id; n * lmax];
            let mut att = vec![0i64; n * lmax];
            let mut mpos = vec![0i64; n * kmax];
            let mut mmask = vec![false; n * kmax];
            let mut qtype = vec![0i64; n];
            for (i, r) in chunk_rows.iter().enumerate() {
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

            // Decode answers per row of this chunk, accumulated per item.
            for (i, r) in chunk_rows.iter().enumerate() {
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
        } // for group in batch_chunks

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
    /// routed first, then grouped by checkpoint; each group follows `opts`
    /// (`batch_size` cap, `sort_by_length` windowed reorder) and still comes
    /// back in input order.
    pub fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        opts: BatchOpts,
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
            let responses = self.run_group(&ckpt, &group, opts)?;
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
            BatchOpts::default(),
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
        // Laya sums the windows' bools into a count (0..N); knot reports the OR
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

#[cfg(test)]
mod revision_tests {
    use super::snapshot_revision;
    use std::fs;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("knot-revision-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(".cache/huggingface/download")).unwrap();
        dir
    }

    #[test]
    fn reads_snapshot_etag_from_hf_metadata() {
        let dir = tmp_dir("etag");
        fs::write(
            dir.join(".cache/huggingface/download/rl_agent_config.json.metadata"),
            "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851\n891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c\n1791049440.0834491\n",
        )
        .unwrap();
        assert_eq!(
            snapshot_revision(&dir).as_deref(),
            Some("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn falls_back_to_model_metadata_and_rejects_bad_values() {
        let dir = tmp_dir("fallback");
        fs::write(
            dir.join(".cache/huggingface/download/model.safetensors.metadata"),
            "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67\n",
        )
        .unwrap();
        assert_eq!(
            snapshot_revision(&dir).as_deref(),
            Some("e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")
        );
        // Not a commit SHA (too short) -> None, like a plain local path.
        fs::write(
            dir.join(".cache/huggingface/download/model.safetensors.metadata"),
            "55cf4c4e\n",
        )
        .unwrap();
        assert_eq!(snapshot_revision(&dir), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_cache_reports_none() {
        let dir = tmp_dir("nocache");
        assert_eq!(snapshot_revision(&dir), None);
        fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod batch_plan_tests {
    use super::{BatchOpts, batch_chunks};

    #[test]
    fn default_is_one_pass_for_the_whole_group() {
        assert_eq!(
            batch_chunks(4, BatchOpts::default(), &[10, 20, 30, 40]),
            vec![vec![0, 1, 2, 3]]
        );
    }

    #[test]
    fn batch_size_splits_in_input_order_without_sorting() {
        assert_eq!(
            batch_chunks(
                6,
                BatchOpts {
                    batch_size: Some(2),
                    sort_by_length: false
                },
                &[10, 30, 20, 40, 5, 15]
            ),
            vec![vec![0, 1], vec![2, 3], vec![4, 5]]
        );
    }

    #[test]
    fn sort_by_length_reorders_stably_inside_one_window() {
        // n=6, batch_size=2 -> window = 16 > 6: one window, sorted by length.
        let chunks = batch_chunks(
            6,
            BatchOpts {
                batch_size: Some(2),
                sort_by_length: true,
            },
            &[10, 30, 20, 40, 5, 15],
        );
        // execution order: lengths 5,10,15,20,30,40 -> items 4,0,5,2,1,3
        assert_eq!(chunks, vec![vec![4, 0], vec![5, 2], vec![1, 3]]);
        // membership covers every state exactly once (input order is
        // restored by the caller writing results to their positions)
        let mut seen: Vec<usize> = chunks.concat();
        seen.sort_unstable();
        assert_eq!(seen, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn sort_requires_explicit_batch_size_between_one_and_n() {
        let lens = [10, 30, 20, 40, 5, 15];
        // batch_size = 1: chunk > 1 fails -> no reorder
        let c = batch_chunks(
            6,
            BatchOpts {
                batch_size: Some(1),
                sort_by_length: true,
            },
            &lens,
        );
        assert_eq!(
            c,
            vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]
        );
        // batch_size = n: chunk < n fails -> no reorder
        let c = batch_chunks(
            6,
            BatchOpts {
                batch_size: Some(6),
                sort_by_length: true,
            },
            &lens,
        );
        assert_eq!(c, vec![vec![0, 1, 2, 3, 4, 5]]);
        // no batch_size: chunk = n -> no reorder
        let c = batch_chunks(
            6,
            BatchOpts {
                batch_size: None,
                sort_by_length: true,
            },
            &lens,
        );
        assert_eq!(c, vec![vec![0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn large_batch_size_collapses_to_one_chunk_and_zero_is_absent() {
        let lens = [1, 2];
        assert_eq!(
            batch_chunks(
                2,
                BatchOpts {
                    batch_size: Some(100),
                    sort_by_length: false
                },
                &lens
            ),
            vec![vec![0, 1]]
        );
        assert_eq!(
            batch_chunks(
                2,
                BatchOpts {
                    batch_size: Some(0),
                    sort_by_length: false
                },
                &lens
            ),
            vec![vec![0, 1]]
        );
        assert!(batch_chunks(0, BatchOpts::default(), &[]).is_empty());
    }

    #[test]
    fn sort_windows_are_eight_chunks() {
        // n=40, batch_size=2 -> window = 8 × 2 = 16 items: [0..16), [16..32),
        // [32..40). Ascending sort by length; lens decrease as i grows, so
        // each window orders high indices first.
        let lens: Vec<usize> = (0..40).map(|i| 100 - i).collect();
        let chunks = batch_chunks(
            40,
            BatchOpts {
                batch_size: Some(2),
                sort_by_length: true,
            },
            &lens,
        );
        assert_eq!(chunks.len(), 20);
        // window 1 = chunks 0..8: shortest in [0..16) is 15, then 14
        assert_eq!(chunks[0], vec![15, 14]);
        // window 2 = chunks 8..16: shortest in [16..32) is 31, then 30
        assert_eq!(chunks[8], vec![31, 30]);
        // window 3 = chunks 16..20: shortest in [32..40) is 39, then 38
        assert_eq!(chunks[16], vec![39, 38]);
        // windows never mix: last window starts at 32's region only
        let last: Vec<usize> = chunks[16..].concat();
        assert!(last.iter().all(|&i| (32..40).contains(&i)));
    }
}
