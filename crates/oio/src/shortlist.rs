//! Opt-in embedding shortlist for high-cardinality choice questions.
//!
//! Port of `laya/shortlist.py`: embed the state and each option with a
//! caller-supplied [`Embedder`], keep the top-`k` labels, and run one
//! `predict` on that reduced criteria set. Ranking is cosine similarity;
//! ties keep the earlier label; a zero vector scores 0.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{Error, Result};
use crate::prompt::{InternalQuestion, render_options, serialize_state};
use crate::protocol::{Question, QuestionType, SystemOneRequest, SystemOneResponse};

pub const DEFAULT_SHORTLIST_K: usize = 20;

/// Caller-supplied embedding backend: a list of texts → a (n, dim) matrix.
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Ranked shortlist metadata for one choice question.
#[derive(Debug, Clone)]
pub struct ShortlistMeta {
    pub labels: Vec<String>,
    /// Cosine scores in rank order; `None` when nothing was dropped.
    pub scores: Option<Vec<f32>>,
    pub k: usize,
    pub n: usize,
    pub passthrough: bool,
}

fn check_k(k: usize) -> Result<()> {
    if k == 0 {
        return Err(Error::InvalidRequest("k must be a positive integer".into()));
    }
    Ok(())
}

fn criteria_items(criteria: &serde_json::Value) -> Result<Vec<String>> {
    match criteria {
        serde_json::Value::Object(m) => Ok(m.keys().cloned().collect()),
        serde_json::Value::Array(a) => Ok(a
            .iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect()),
        other => Err(Error::InvalidRequest(format!(
            "choice criteria must be a dict or list, got {}",
            match other {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::String(_) => "string",
                _ => "unknown",
            }
        ))),
    }
}

fn option_texts(criteria: &serde_json::Value) -> Result<Vec<String>> {
    let q = InternalQuestion {
        t: "choice".into(),
        ins: String::new(),
        crit: Some(criteria.clone()),
        labels: None,
    };
    render_options(&q).map_err(|e| Error::InvalidRequest(e.to_string()))
}

fn query_text(state: &serde_json::Value, instructions: Option<&str>) -> String {
    let body = serialize_state(state);
    match instructions {
        None | Some("") => body,
        Some(ins) => format!("{ins}\n{body}"),
    }
}

fn subset_criteria(criteria: &serde_json::Value, labels: &[String]) -> serde_json::Value {
    match criteria {
        serde_json::Value::Object(m) => {
            let mut out = serde_json::Map::new();
            for l in labels {
                if let Some(v) = m.get(l) {
                    out.insert(l.clone(), v.clone());
                }
            }
            serde_json::Value::Object(out)
        }
        _ => serde_json::Value::Array(
            labels
                .iter()
                .map(|l| serde_json::Value::String(l.clone()))
                .collect(),
        ),
    }
}

fn cosine(query: &[f32], docs: &[Vec<f32>]) -> Vec<f32> {
    let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if qn == 0.0 || docs.is_empty() {
        return vec![0.0; docs.len()];
    }
    docs.iter()
        .map(|d| {
            let dn: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            if dn * qn == 0.0 {
                0.0
            } else {
                let dot: f32 = d.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                (dot / (dn * qn)).clamp(-1.0, 1.0)
            }
        })
        .collect()
}

fn rank(
    state: &serde_json::Value,
    criteria: &serde_json::Value,
    embed: &dyn Embedder,
    k: usize,
    instructions: Option<&str>,
) -> Result<ShortlistMeta> {
    check_k(k)?;
    let labels = criteria_items(criteria)?;
    let n = labels.len();
    if labels.len() != new_set(&labels).len() {
        return Err(Error::InvalidRequest("duplicated choice label".into()));
    }
    if k >= n {
        return Ok(ShortlistMeta {
            labels,
            scores: None,
            k,
            n,
            passthrough: true,
        });
    }
    let query = query_text(state, instructions);
    let mut texts = vec![query];
    texts.extend(option_texts(criteria)?);
    let matrix = embed.embed(&texts)?;
    if matrix.len() != texts.len() || matrix.iter().all(|r| r.is_empty()) {
        return Err(Error::Model(format!(
            "embed must return one row per text ({})",
            texts.len()
        )));
    }
    let sims = cosine(&matrix[0], &matrix[1..]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sims[b]
            .partial_cmp(&sims[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    Ok(ShortlistMeta {
        labels: order.iter().map(|&i| labels[i].clone()).collect(),
        scores: Some(order.iter().map(|&i| sims[i]).collect()),
        k,
        n,
        passthrough: false,
    })
}

fn new_set(labels: &[String]) -> std::collections::HashSet<&String> {
    labels.iter().collect()
}

/// `predict` with each choice question's criteria reduced to its top-`k`
/// labels. Non-choice questions pass through unchanged; the caller's request
/// is not mutated. Adds a top-level `shortlist` block to the response.
pub fn predict_shortlist(
    req: &SystemOneRequest,
    predict: &dyn Fn(&SystemOneRequest) -> Result<SystemOneResponse>,
    embed: &dyn Embedder,
    k: usize,
) -> Result<SystemOneResponse> {
    check_k(k)?;
    let state = &req.state;
    let questions = &req.questions;
    let mut reduced: indexmap::IndexMap<String, Question> = indexmap::IndexMap::new();
    let mut meta = serde_json::Map::new();
    for (qid, q) in questions {
        if q.r#type != QuestionType::Choice {
            reduced.insert(qid.clone(), q.clone());
            continue;
        }
        let criteria = match &q.criteria {
            serde_json::Value::Object(_) | serde_json::Value::Array(_) => &q.criteria,
            _ => {
                return Err(Error::InvalidRequest(format!(
                    "question {qid:?} is a choice but has no criteria"
                )));
            }
        };
        let m = rank(state, criteria, embed, k, Some(&q.instructions))?;
        meta.insert(
            qid.clone(),
            serde_json::json!({
                "labels": m.labels,
                "scores": m.scores,
                "k": m.k,
                "n": m.n,
                "passthrough": m.passthrough,
            }),
        );
        let mut updated = q.clone();
        if !m.passthrough {
            updated.criteria = subset_criteria(criteria, &m.labels);
        }
        reduced.insert(qid.clone(), updated);
    }
    let req = SystemOneRequest {
        state: state.clone(),
        questions: reduced,
        ..req.clone()
    };
    let mut res = predict(&req)?;
    res.shortlist = Some(serde_json::Value::Object(meta));
    Ok(res)
}

/// LRU-cached embedder wrapper mirroring Laya `cached_embed_fn`.
pub struct CachedEmbedder<E: Embedder> {
    inner: E,
    maxsize: usize,
    state: Mutex<CacheState>,
}

struct CacheState {
    rows: HashMap<String, Vec<f32>>,
    lru: std::collections::VecDeque<String>,
    hits: usize,
    misses: usize,
}

impl<E: Embedder> CachedEmbedder<E> {
    pub fn new(inner: E, maxsize: usize) -> Result<Self> {
        if maxsize == 0 {
            return Err(Error::InvalidRequest("maxsize must be positive".into()));
        }
        Ok(Self {
            inner,
            maxsize,
            state: Mutex::new(CacheState {
                rows: HashMap::new(),
                lru: std::collections::VecDeque::new(),
                hits: 0,
                misses: 0,
            }),
        })
    }

    pub fn cache_info(&self) -> (usize, usize, usize, usize) {
        let s = self.state.lock().unwrap();
        (s.rows.len(), self.maxsize, s.hits, s.misses)
    }

    pub fn cache_clear(&self) {
        let mut s = self.state.lock().unwrap();
        s.rows.clear();
        s.lru.clear();
        s.hits = 0;
        s.misses = 0;
    }
}

impl<E: Embedder> Embedder for CachedEmbedder<E> {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Rows for `texts` are cloned while the lock is held, never looked up
        // after the fact: an eviction (this call's or a concurrent one's) must
        // not be able to remove a row between computing the answer and
        // returning it. The old code ended in an unconditional index into the
        // cache, which panicked when one call had more distinct texts than
        // `maxsize` or when another thread evicted in between.
        let keys: Vec<String> = texts.to_vec();
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(keys.len());
        let mut missing: Vec<String> = Vec::new();
        {
            let mut s = self.state.lock().unwrap();
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for k in &keys {
                if let Some(row) = s.rows.get(k).cloned() {
                    s.lru.retain(|x| x != k);
                    s.lru.push_back(k.clone());
                    s.hits += 1;
                    out.push(Some(row));
                } else {
                    s.misses += 1;
                    out.push(None);
                    if seen.insert(k.as_str()) {
                        missing.push(k.clone());
                    }
                }
            }
        }
        if !missing.is_empty() {
            let raw = self.inner.embed(&missing)?;
            if raw.len() != missing.len() {
                return Err(Error::Model(format!(
                    "embed returned {} rows for {} texts",
                    raw.len(),
                    missing.len()
                )));
            }
            // One slot index per still-empty output position, grouped by key,
            // so duplicate texts all get filled from the single fetched row.
            let mut fill: std::collections::HashMap<&str, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, slot) in out.iter_mut().enumerate() {
                if slot.is_none() {
                    fill.entry(keys[i].as_str()).or_default().push(i);
                }
            }
            let mut s = self.state.lock().unwrap();
            for (k, row) in missing.iter().zip(raw) {
                let row: Vec<f32> = row
                    .into_iter()
                    .map(|v| if v.is_finite() { v } else { 0.0 })
                    .collect();
                if let Some(first) = s.rows.values().next()
                    && first.len() != row.len()
                {
                    return Err(Error::Model("embed dim mismatch with cache".into()));
                }
                if let Some(idxs) = fill.get(k.as_str()) {
                    for &i in idxs {
                        out[i] = Some(row.clone());
                    }
                }
                s.rows.insert(k.clone(), row);
                s.lru.retain(|x| x != k);
                s.lru.push_back(k.clone());
                while s.rows.len() > self.maxsize {
                    if let Some(old) = s.lru.pop_front() {
                        s.rows.remove(&old);
                    }
                }
            }
        }
        // Every slot is filled here: missing keys were filled above, hits in
        // the first lock; nothing can have evicted them in between.
        out.into_iter()
            .map(|slot| {
                slot.ok_or_else(|| Error::Model("cache row unavailable for request key".into()))
            })
            .collect()
    }
}
