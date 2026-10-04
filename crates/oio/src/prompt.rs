//! Prompt construction — port of `laya/common.py` (M1).
//!
//! Layout: `[CLS] <type> question: <instructions> [SEP] <opt0> <opt1> … [SEP] <state> [SEP]`
//! where each option is `[MASK]` + ≤48 tokens of rendered text, and option logits
//! are read at the `[MASK]` marker positions.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

const OPTION_MAX_TOKENS: usize = 48;

pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];

/// Internal question form, mirroring Laya's internal dict: `t`, `ins`, `crit`, `labels`.
#[derive(Debug, Clone)]
pub struct InternalQuestion {
    pub t: String,
    pub ins: String,
    pub crit: Option<serde_json::Value>,
    pub labels: Option<serde_json::Value>,
}

impl InternalQuestion {
    /// Port of Laya's `_check_question` + `_to_internal`: reject what cannot be
    /// answered, then normalize to the internal `t`/`ins`/`crit`/`labels` form.
    pub fn from_wire(qid: &str, q: &crate::protocol::Question) -> Result<Self> {
        let t = match q.r#type {
            crate::protocol::QuestionType::Choice => "choice",
            crate::protocol::QuestionType::Score => "score",
            crate::protocol::QuestionType::Noul => "noul",
        };
        // Laya 422s missing/None/blank instructions (agent.py `_validate`);
        // missing used to be a schema 400 and blank was accepted outright.
        if q.instructions.trim().is_empty() {
            return Err(Error::InvalidRequest(format!(
                "question {qid:?}: 'instructions' must not be empty"
            )));
        }
        let mut crit = q.criteria.clone();
        if t == "choice" {
            normalize_choice_labels(qid, &mut crit)?;
        } else if t == "noul"
            && let Some(obj) = crit.as_object_mut()
        {
            // `_to_internal`: `{str(k).lower(): v}` — a dict keyed `"True"` used
            // to pass the case-insensitive check and then render the defaults.
            let lowered: Vec<(String, serde_json::Value)> = obj
                .iter()
                .map(|(k, v)| (k.to_lowercase(), v.clone()))
                .collect();
            obj.clear();
            for (k, v) in lowered {
                obj.insert(k, v);
            }
        }
        Ok(Self {
            t: t.to_string(),
            ins: q.instructions.clone(),
            crit: Some(crit),
            labels: q.labels.clone(),
        })
    }
}

/// Python `str(label)` used as the option text and the answer key when a
/// choice carries a list of labels (Laya `_to_internal` `{c: None for c in crit}`).
fn choice_label_key(label: &serde_json::Value) -> String {
    crate::pyjson::py_str(label)
}

/// Python equality over scalars, so `[1, 1.0]` and `[True, 1]` collapse as
/// answer keys the way they do in a Python dict (laya `_check_question`).
fn py_eq(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::String(x), serde_json::Value::String(y)) => x == y,
        (serde_json::Value::Bool(x), serde_json::Value::Bool(y)) => x == y,
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            x.as_f64() == y.as_f64() && x.as_f64().is_some()
        }
        (serde_json::Value::Bool(x), serde_json::Value::Number(y))
        | (serde_json::Value::Number(y), serde_json::Value::Bool(x)) => {
            // Python: `True == 1`
            y.as_f64() == Some(if *x { 1.0 } else { 0.0 })
        }
        _ => false,
    }
}

/// Lay a accepts choice criteria as a dict *or* a list of labels and normalizes
/// the list to `{label: None}`, rejecting null/structured/duplicate labels
/// with 422s (`agent.py` `_check_question`).
fn normalize_choice_labels(qid: &str, crit: &mut serde_json::Value) -> Result<()> {
    let arr = match crit {
        serde_json::Value::Array(a) => a,
        _ => return Ok(()),
    };
    // Laya's order: structural checks over every label first, then the
    // duplicate pass — so a list that is both structural and duplicated
    // reports the structural problem first.
    for (i, label) in arr.iter().enumerate() {
        match label {
            serde_json::Value::Null => {
                return Err(Error::InvalidRequest(format!(
                    "question {qid:?}: choice label {i} is null; a label is rendered as option \
                     text and used as the answer key, so it must be a string, number or bool"
                )));
            }
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                return Err(Error::InvalidRequest(format!(
                    "question {qid:?}: choice label {i} is not a scalar; a label is rendered as \
                     option text and used as the answer key, so it must be a string, number or bool"
                )));
            }
            _ => {}
        }
    }
    let mut obj = serde_json::Map::new();
    for (i, label) in arr.iter().enumerate() {
        if let Some(first) = (0..i).find(|&j| py_eq(&arr[j], label)) {
            return Err(Error::InvalidRequest(format!(
                "question {qid:?}: choice label {i} repeats label {first}; the labels are the \
                 answer keys, so every option needs its own (1, 1.0 and True are one key)"
            )));
        }
        obj.insert(choice_label_key(label), serde_json::Value::Null);
    }
    *crit = serde_json::Value::Object(obj);
    Ok(())
}

/// Render a criterion value the same way Laya does: strings pass through,
/// structured values become Python `json.dumps` text (spaces after `,`/`:`,
/// raw UTF-8).
pub fn render_criterion(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => crate::pyjson::dumps(other),
    }
}

/// Lay a `serialize_state`: strings pass through, containers become Python
/// `json.dumps(state, ensure_ascii=False)`.
pub fn serialize_state(state: &serde_json::Value) -> String {
    match state {
        serde_json::Value::String(s) => s.clone(),
        other => crate::pyjson::dumps(other),
    }
}

fn resolve_noul_labels(labels: Option<&serde_json::Value>) -> Result<(String, String)> {
    let (f, t) = match labels {
        None => ("false".to_string(), "true".to_string()),
        Some(serde_json::Value::Object(m))
            if m.len() == 2 && m.contains_key("false") && m.contains_key("true") =>
        {
            let f = m["false"].as_str().unwrap_or("").trim().to_string();
            let t = m["true"].as_str().unwrap_or("").trim().to_string();
            (f, t)
        }
        _ => {
            return Err(Error::InvalidRequest(
                "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
                    .into(),
            ));
        }
    };
    if f.is_empty() || t.is_empty() || f == t {
        return Err(Error::InvalidRequest(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings".into(),
        ));
    }
    Ok((f, t))
}

/// One rendered option per criterion, in label order; noul semantic order is [false, true].
pub fn render_options(q: &InternalQuestion) -> Result<Vec<String>> {
    match q.t.as_str() {
        "choice" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_object())
                .ok_or_else(|| Error::InvalidRequest("choice needs criteria object".into()))?;
            // Laya rejects an empty criteria ("needs at least one criterion");
            // an empty option list used to reach the model (or a runtime shape
            // error) instead of a caller error.
            if crit.is_empty() {
                return Err(Error::InvalidRequest(
                    "choice needs at least one criterion".into(),
                ));
            }
            Ok(crit
                .iter()
                .map(|(k, v)| match v {
                    serde_json::Value::Null => k.clone(),
                    serde_json::Value::String(s) if s.is_empty() => k.clone(),
                    other => format!("{k}: {}", render_criterion(other)),
                })
                .collect())
        }
        "score" => {
            let crit = q
                .crit
                .as_ref()
                .and_then(|c| c.as_array())
                .ok_or_else(|| Error::InvalidRequest("score needs criteria list".into()))?;
            // SPEC: every level requires a description. Laya raises on an
            // empty list and on a null level; both used to be accepted here
            // and rendered as "level N: null" with 0 options.
            if crit.is_empty() {
                return Err(Error::InvalidRequest(
                    "score needs at least one level".into(),
                ));
            }
            if let Some(i) = crit.iter().position(|c| c.is_null()) {
                return Err(Error::InvalidRequest(format!(
                    "score level {i} is null; give every level a description, index 0 first"
                )));
            }
            Ok(crit
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect())
        }
        "noul" => {
            let (f_label, t_label) = resolve_noul_labels(q.labels.as_ref())?;
            // `render` reads only `crit.get("false")`/`crit.get("true")`, so
            // anything else was silently dropped and replaced with the
            // defaults — the same "quiet acceptance" Laya removed in #156.
            if let Some(c) = q.crit.as_ref().filter(|c| !c.is_null()) {
                let obj = c.as_object().ok_or_else(|| {
                    Error::InvalidRequest(
                        "noul question takes 'criteria' as a dict with optional                          'true'/'false' descriptions, or omits it"
                            .into(),
                    )
                })?;
                let bad: Vec<&String> = obj
                    .keys()
                    .filter(|k| !matches!(k.to_lowercase().as_str(), "true" | "false"))
                    .collect();
                if !bad.is_empty() {
                    return Err(Error::InvalidRequest(format!(
                        "noul question takes 'criteria' keyed only 'true'/'false' (either or                          both, and omitted is fine), got {bad:?}"
                    )));
                }
            }
            let crit = q.crit.as_ref().and_then(|c| c.as_object());
            let render = |key: &str, default: &str| {
                crit.and_then(|c| c.get(key))
                    .filter(|v| !matches!(v, serde_json::Value::Null))
                    .filter(|v| !matches!(v, serde_json::Value::String(s) if s.is_empty()))
                    .map(render_criterion)
                    .unwrap_or_else(|| default.to_string())
            };
            Ok(vec![
                format!(
                    "{f_label}: {}",
                    render("false", "no, the statement does not hold")
                ),
                format!("{t_label}: {}", render("true", "yes, the statement holds")),
            ])
        }
        other => Err(Error::InvalidRequest(format!(
            "unknown question type {other}"
        ))),
    }
}

#[derive(Debug, Clone, Default)]
pub struct HeadStats {
    pub options: usize,
    pub options_distinct: usize,
    pub tokens_per_option: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TruncationStats {
    pub state_tokens: usize,
    pub state_tokens_used: usize,
    pub state_tokens_dropped: usize,
    pub truncated: bool,
}

/// Tokenizer wrapper mirroring Laya's shared-tokenizer semantics: encoding a
/// question span is cached per call site; option spans truncate to 48.
pub struct PromptBuilder {
    tok: Tokenizer,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    mask_token: String,
    question_cache: Mutex<HashMap<String, Vec<u32>>>,
}

impl PromptBuilder {
    pub fn from_file(path: &str) -> Result<Self> {
        let tok = Tokenizer::from_file(path)
            .map_err(|e| Error::Model(format!("load tokenizer {path}: {e}")))?;
        let cls_id = Self::first_present(&tok, &["[CLS]", "<s>", "<bos>"])
            .ok_or_else(|| Error::Model("no CLS token found".into()))?;
        let sep_id = Self::first_present(&tok, &["[SEP]", "</s>", "<eos>"])
            .ok_or_else(|| Error::Model("no SEP token found".into()))?;
        let mask_id = Self::first_present(&tok, &["[MASK]", "<mask>"])
            .ok_or_else(|| Error::Model("no MASK token found".into()))?;
        let mask_token = tok
            .id_to_token(mask_id)
            .unwrap_or_else(|| "[MASK]".into())
            .trim_start_matches('▁')
            .to_string();
        Ok(Self {
            tok,
            cls_id,
            sep_id,
            mask_id,
            mask_token,
            question_cache: Mutex::new(HashMap::new()),
        })
    }

    fn first_present(tok: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        candidates.iter().find_map(|t| tok.token_to_id(t))
    }

    fn encode_ids(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Model(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    fn encode_question_text(&self, text: &str) -> Result<Vec<u32>> {
        // Question spans are reused across states in a request; cache them.
        let mut cache = self.question_cache.lock().unwrap();
        if let Some(ids) = cache.get(text) {
            return Ok(ids.clone());
        }
        let ids = self.encode_ids(text)?;
        cache.insert(text.to_string(), ids.clone());
        Ok(ids)
    }

    fn encode_option_text(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode_question_text(text)?;
        ids.truncate(OPTION_MAX_TOKENS);
        Ok(ids)
    }

    pub fn pad_id(&self) -> u32 {
        Self::first_present(&self.tok, &["[PAD]", "<pad>", "<pad>"]).unwrap_or(0)
    }

    pub fn cls_id(&self) -> u32 {
        self.cls_id
    }
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }
    pub fn mask_id(&self) -> u32 {
        self.mask_id
    }

    /// Decode window ids back to text so a state span can be re-tokenized as a
    /// normal state on the way to the model (Laya `predict_long`).
    pub fn decode_ids(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).unwrap_or_default()
    }

    pub fn encode_state(&self, state: &serde_json::Value) -> Result<Vec<u32>> {
        let text = serialize_state(state).replace(&self.mask_token, " ");
        self.encode_ids(&text)
    }

    /// `(ids, markers, stats)` for the question half of the sequence.
    pub fn build_head(
        &self,
        q: &InternalQuestion,
        head_max_len: usize,
        option_order: Option<&[usize]>,
    ) -> Result<(Vec<u32>, Vec<usize>, HeadStats)> {
        let opts = render_options(q)?;
        let order: Vec<usize> = option_order
            .map(|o| o.to_vec())
            .unwrap_or_else(|| (0..opts.len()).collect());
        let ins = q.ins.replace(&self.mask_token, " ");
        let mut head_ids = self.encode_question_text(&format!("{} question: {}", q.t, ins))?;

        let mut opt_ids: Vec<Vec<u32>> = Vec::new();
        for &i in &order {
            let opt_text = format!(" {}", opts[i].replace(&self.mask_token, " "));
            let mut token_ids = self.encode_option_text(&opt_text)?;
            token_ids.insert(0, self.mask_id);
            opt_ids.push(token_ids);
        }

        let mut opt_budget =
            head_max_len as isize - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        let mut per_option = None;
        if opt_budget < 16 {
            let per = (4usize).max((head_max_len.saturating_sub(16)) / opt_ids.len().max(1));
            per_option = Some(per);
            for o in &mut opt_ids {
                o.truncate(per);
                // re-prepend marker if it was cut off
                if o.is_empty() || o[0] != self.mask_id {
                    o.insert(0, self.mask_id);
                }
            }
            opt_budget =
                head_max_len as isize - opt_ids.iter().map(|o| o.len() as isize).sum::<isize>();
        }
        head_ids.truncate(8.max(opt_budget.max(0) as usize));

        let mut ids = vec![self.cls_id];
        ids.extend_from_slice(&head_ids);
        ids.push(self.sep_id);
        let mut markers = Vec::new();
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.push(self.sep_id);

        let distinct: HashSet<&Vec<u32>> = opt_ids.iter().collect();
        let stats = HeadStats {
            options: opt_ids.len(),
            options_distinct: distinct.len(),
            tokens_per_option: per_option,
        };
        Ok((ids, markers, stats))
    }

    /// Full sequence: head + clamped state + trailing SEP.
    pub fn build_sequence(
        &self,
        state_ids: &[u32],
        q: &InternalQuestion,
        max_len: usize,
        head_max_len: usize,
        option_order: Option<&[usize]>,
        truncate_left: bool,
    ) -> Result<(Vec<u32>, Vec<usize>, HeadStats, TruncationStats)> {
        let (ids, markers, stats) = self.build_head(q, head_max_len, option_order)?;
        let room = max_len.saturating_sub(ids.len() + 1);
        let st: &[u32] = if truncate_left {
            &state_ids[state_ids.len().saturating_sub(room)..]
        } else {
            &state_ids[..state_ids.len().min(room)]
        };
        let mut out = ids;
        out.extend_from_slice(st);
        out.push(self.sep_id);
        out.truncate(max_len);
        let markers: Vec<usize> = markers.into_iter().filter(|&m| m < max_len).collect();
        let truncation = TruncationStats {
            state_tokens: state_ids.len(),
            state_tokens_used: st.len(),
            state_tokens_dropped: state_ids.len() - st.len(),
            truncated: st.len() < state_ids.len(),
        };
        Ok((out, markers, stats, truncation))
    }

    /// Tokens left in `max_len` for the state once the question head is built.
    pub fn state_room(
        &self,
        q: &InternalQuestion,
        max_len: usize,
        head_max_len: usize,
    ) -> Result<usize> {
        let (head, _, _) = self.build_head(q, head_max_len, None)?;
        Ok(max_len.saturating_sub(head.len() + 1))
    }

    /// Window + stride for long-document scans (Laya `window_budget`).
    pub fn window_budget(
        &self,
        questions: &[InternalQuestion],
        max_len: usize,
        head_max_len: usize,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> Result<(usize, usize, usize)> {
        let rooms: Vec<usize> = questions
            .iter()
            .map(|q| self.state_room(q, max_len, head_max_len))
            .collect::<Result<_>>()?;
        let requested = window
            .filter(|&w| w > 0)
            .unwrap_or(64.max(max_len.saturating_sub(head_max_len + 8)));
        let room = if rooms.is_empty() {
            requested
        } else {
            *rooms.iter().min().unwrap()
        };
        if room == 0 {
            return Err(Error::InvalidRequest(
                "questions' options fill the whole sequence; no room for the state".into(),
            ));
        }
        let size = requested.min(room);
        if window.is_some_and(|w| w > room) {
            tracing::warn!("window {requested} clamped to state room {room}");
        }
        let stride = match stride {
            Some(s) if s > size => {
                return Err(Error::InvalidRequest(
                    "stride larger than window would skip tokens".into(),
                ));
            }
            Some(s) => s,
            None => (size / 2).max(1),
        };
        Ok((size, stride, room))
    }
}
