//! Language/script detection — Rust port of `laya/lang.py`.
//!
//! Routing only needs one decision: *is this English Latin text, or something the
//! English checkpoint cannot read?* Script detection is exact; the Latin-script
//! language guess is a stopword/diacritic heuristic and best-effort.

use std::collections::{BTreeMap, HashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use regex::Regex;

pub mod lang_data {
    include!("lang_data.rs");
}
use lang_data::*;

const NON_EN_DIACRITIC_RATE: f64 = 0.02;
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;
const NON_LATIN_FRACTION: f64 = 0.2;
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: usize = 10;

fn shared_words() -> HashSet<&'static str> {
    let mut union: BTreeMap<&str, usize> = BTreeMap::new();
    for &(_, words) in STOP_LANGS {
        for &w in words {
            *union.entry(w).or_insert(0) += 1;
        }
    }
    let mut shared: HashSet<&str> = union
        .into_iter()
        .filter_map(|(w, n)| (n > 1).then_some(w))
        .collect();
    shared.extend(NORDIC_OVERLAP_WORDS.iter().copied());
    shared
}

fn word_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\W\d_]+").unwrap())
}

fn identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\w-]*(?:[.@][\w-]+)+").unwrap())
}

fn code_line_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[=;{}\[\]]|\w\(").unwrap())
}

fn joined_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\W_][._/\\][^\W_]").unwrap())
}

fn letter_run_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\W\d_]{2,}").unwrap())
}

fn is_en_collision(w: &str) -> bool {
    EN_COLLISION_WORDS.contains(&w)
}

/// Iterate over the string leaves of a state (keys ignored: they are usually
/// English field names). Depth-capped like Laya.
pub fn iter_text(state: &serde_json::Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match state {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(items) => {
            for v in items {
                iter_text(v, depth + 1, out);
            }
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                iter_text(v, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Slice `s` to at most `max` bytes without splitting a multi-byte char.
/// (`&s[..n]` panics when `n` lands inside a char; Python's equivalent does not.)
pub fn trunc_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Flatten a state into the text used for detection.
pub fn state_text(state: &serde_json::Value, max_chars: usize) -> String {
    let mut leaves = Vec::new();
    iter_text(state, 0, &mut leaves);
    let mut parts = Vec::new();
    let mut budget = max_chars;
    for leaf in leaves {
        if budget == 0 {
            break;
        }
        if leaf.len() > budget {
            parts.push(trunc_str(&leaf, budget).to_string());
            break;
        }
        budget = budget.saturating_sub(leaf.len() + 1);
        parts.push(leaf);
    }
    parts.join(" ").chars().take(max_chars).collect()
}

fn script_of_char(ch: char) -> Option<&'static str> {
    let cp = ch as u32;
    if cp < 0x02B0
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
    {
        return None; // Latin-ish; caller counts Latin separately
    }
    for (name, ranges) in SCRIPT_RANGES {
        if ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi) {
            return Some(name);
        }
    }
    None
}

fn script_counts(text: &str) -> IndexMap<&'static str, usize> {
    // Insertion order matters: laya keeps the scripts in first-appearance
    // order with `latin` last (tie-break for `script_from_counts`, key order
    // for `script_profile`), which a BTreeMap's sort would destroy.
    let mut counts: IndexMap<&str, usize> = IndexMap::new();
    let mut latin = 0usize;
    for ch in text.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x02B0
            || (0x1E00..=0x1EFF).contains(&cp)
            || (0xFF21..=0xFF3A).contains(&cp)
            || (0xFF41..=0xFF5A).contains(&cp)
        {
            latin += 1;
            continue;
        }
        let mut claimed = false;
        for (name, ranges) in SCRIPT_RANGES {
            if ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi) {
                *counts.entry(name).or_insert(0) += 1;
                claimed = true;
                break;
            }
        }
        if !claimed {
            *counts.entry("other").or_insert(0) += 1;
        }
    }
    counts.insert("latin", latin);
    counts
}

fn script_from_counts(counts: &IndexMap<&'static str, usize>) -> &'static str {
    if counts.values().all(|&v| v == 0) {
        return "unknown";
    }
    // Named scripts win ties against Latin: max_by_key on counts but Latin last.
    let mut best: Option<(&str, usize)> = None;
    // iterate named scripts first, latin last for tie-break (first max wins)
    for (&name, &count) in counts.iter().filter(|(n, _)| **n != "latin") {
        if best.is_none_or(|(_, b)| count > b) {
            best = Some((name, count));
        }
    }
    let latin = counts.get("latin").copied().unwrap_or(0);
    if best.is_none_or(|(_, b)| latin > b) {
        return "latin";
    }
    best.map(|(n, _)| n).unwrap_or("latin")
}

fn profile_from_counts<'a>(counts: &IndexMap<&'a str, usize>) -> IndexMap<&'a str, f64> {
    let total: usize = counts.values().sum();
    if total == 0 {
        return IndexMap::new();
    }
    // Latin first, then first-appearance order (laya `_profile_from_counts`).
    let mut ordered = IndexMap::new();
    if let Some(&v) = counts.get("latin").filter(|&&v| v > 0) {
        ordered.insert("latin", v as f64 / total as f64);
    }
    for (&name, &v) in counts.iter() {
        if name != "latin" && v > 0 {
            ordered.insert(name, v as f64 / total as f64);
        }
    }
    ordered
}

/// Dominant script: 'latin', 'han', 'devanagari', ... or 'unknown'.
pub fn detect_script(text: &str) -> &'static str {
    script_from_counts(&script_counts(text))
}

/// Fraction of alphabetic characters belonging to each detected script.
pub fn script_profile(text: &str) -> IndexMap<&'static str, f64> {
    profile_from_counts(&script_counts(text))
}

fn non_latin_words(text: &str) -> Vec<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_script: Option<&str> = None;
    for ch in text.chars() {
        if is_combining_mark(ch) {
            continue;
        }
        match script_of_char(ch) {
            Some(s) if Some(s) == cur_script => cur.push(ch),
            Some(s) => {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
                cur.push(ch);
                cur_script = Some(s);
            }
            None => {
                if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
                cur_script = None;
            }
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs.into_iter()
        .filter(|w| w.len() >= 2 && !w.chars().next().unwrap().is_uppercase())
        .collect()
}

fn is_combining_mark(ch: char) -> bool {
    // Rough equivalent of unicodedata.combining != 0 for common marks (Mn/Me/Cf category proxy):
    // Rust has no direct API; conservatively treat the ranges Laya exercises.
    matches!(ch as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF |
        0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

fn english_rescued_by_words(words: &HashSet<String>, diac_rate: f64) -> bool {
    if diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    let shared = shared_words();
    let en_only: HashSet<&&str> = STOP_EN.iter().filter(|w| !shared.contains(*w)).collect();
    if words
        .iter()
        .filter(|w| en_only.contains(&w.as_str()))
        .count()
        < 2
    {
        return false;
    }
    words
        .iter()
        .filter(|w| w.chars().any(|c| NON_EN_DIACRITICS.contains(c)))
        .count()
        <= 1
}

#[derive(Debug, Clone, Default)]
pub struct LatinProfile {
    pub language: Option<String>,
    pub english_hits: usize,
    pub diacritic_rate: f64,
    pub looks_non_english: bool,
}

/// Evidence behind the Latin-script language guess.
pub fn latin_profile(text: &str) -> LatinProfile {
    let lowered_ident = identifier_re().replace_all(text, " ").replace('İ', "i");
    let words: Vec<String> = word_re()
        .find_iter(&lowered_ident.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    let lowered = text.to_lowercase();
    let diac = lowered
        .chars()
        .filter(|&c| NON_EN_DIACRITICS.contains(c))
        .count();
    let diac_rate = diac as f64 / lowered.len().max(1) as f64;
    let non_english = diac_rate >= NON_EN_DIACRITIC_RATE;

    let shared = shared_words();
    let word_set: HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
    let has_nordic = word_set.iter().any(|w| NORDIC_OVERLAP_WORDS.contains(w));
    let en_only_hit = STOP_EN
        .iter()
        .any(|w| !shared.contains(w) && word_set.contains(*w));
    let nordic_overlap = has_nordic && !en_only_hit;

    let short_swedish = !words.is_empty()
        && words.len() < 4
        && words.len() > 1
        && word_set.iter().any(|w| SHORT_SWEDISH_WORDS.contains(w));
    if short_swedish {
        return LatinProfile {
            language: Some("sv".into()),
            english_hits: 0,
            diacritic_rate: diac_rate,
            looks_non_english: non_english,
        };
    }
    if words.len() < 4 {
        return LatinProfile {
            language: None,
            english_hits: 0,
            diacritic_rate: diac_rate,
            looks_non_english: non_english || nordic_overlap,
        };
    }

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for w in &words {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let scores: BTreeMap<&str, usize> = STOP_LANGS
        .iter()
        .map(|&(lg, stop)| {
            let s = counts
                .iter()
                .filter(|(w, _)| stop.contains(w))
                .map(|(w, &n)| if is_en_collision(w) { 1 } else { n })
                .sum();
            (lg, s)
        })
        .collect();
    let en = *scores.get("en").unwrap_or(&0);

    let evidenced: BTreeMap<&str, usize> = scores
        .iter()
        .filter(|&(lg, _)| *lg != "en")
        .filter(|&(lg, _)| {
            let stop = STOP_LANGS
                .iter()
                .find(|&&(l, _)| l == *lg)
                .map(|&(_, s)| s)
                .unwrap();
            word_set
                .iter()
                .any(|w| stop.contains(w) && !shared.contains(*w))
        })
        .map(|(&lg, &s)| (lg, s))
        .collect();
    let (best_lg, best) = evidenced
        .iter()
        .max_by_key(|(_, s)| **s)
        .map(|(&lg, &s)| (Some(lg), s))
        .unwrap_or((None, 0));

    let word_hash: HashSet<String> = words.iter().cloned().collect();
    let mut lang = None;
    if let Some(blg) = best_lg
        && (best >= (2usize).max(en + 2)
            || (blg == "sv"
                && word_set.contains("inte")
                && word_set.contains("kan")
                && ["kan", "jag", "vi"].contains(&words[0].as_str())
                && en <= 1)
            || (non_english && best >= 2usize.max(en)))
    {
        lang = Some(blg.to_string());
    }
    if lang.is_none() && en > 0 && (!non_english || english_rescued_by_words(&word_hash, diac_rate))
    {
        lang = Some("en".into());
    }
    LatinProfile {
        looks_non_english: non_english || (lang.is_none() && nordic_overlap),
        language: lang,
        english_hits: en,
        diacritic_rate: diac_rate,
    }
}

/// Best-effort language code for Latin-script text, or None when undecided.
pub fn guess_latin_language(text: &str) -> Option<String> {
    latin_profile(text).language
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Analysis {
    pub script: String,
    pub script_profile: IndexMap<String, f64>,
    pub language: Option<String>,
    pub is_english: bool,
    pub language_undecided: bool,
    pub diacritic_rate: f64,
    pub non_latin_fraction: f64,
    pub mixed_segment: Option<String>,
}

fn analyse_text(text: &str) -> Analysis {
    let counts = script_counts(text);
    let prof = profile_from_counts(&counts);
    let script = script_from_counts(&counts);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        // laya rounds here (`round(1.0 - prof.get("latin", 0.0), 4)`) and the
        // rounded value is both what `analyse` reports on the wire (via
        // `routing.detection`) and what the thresholds compare against.
        let n = 1.0 - prof.get("latin").copied().unwrap_or(0.0);
        (n * 10_000.0).round() / 10_000.0
    };
    let n_non_latin =
        (non_latin * text.chars().filter(|c| c.is_alphabetic()).count() as f64).round() as usize;

    let mut script = script;
    if script == "latin"
        && !non_latin_words(text).is_empty()
        && (non_latin >= NON_LATIN_FRACTION
            || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS))
        && let Some((name, _)) = prof
            .iter()
            .filter(|(n, _)| **n != "latin")
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    {
        script = name;
    }
    if script == "unknown" {
        return Analysis {
            script: "unknown".into(),
            script_profile: prof.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
            mixed_segment: None,
        };
    }
    if script != "latin" {
        return Analysis {
            script: script.into(),
            script_profile: prof.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
            mixed_segment: None,
        };
    }
    let lp = latin_profile(text);
    let undecided = lp.language.is_none();
    let english = lp.language.as_deref() == Some("en") || (undecided && !lp.looks_non_english);
    Analysis {
        script: "latin".into(),
        script_profile: prof.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        language: lp.language,
        is_english: english,
        language_undecided: undecided,
        diacritic_rate: (lp.diacritic_rate * 10000.0).round() / 10000.0,
        non_latin_fraction: non_latin,
        mixed_segment: None,
    }
}

fn named_prose_language(segment: &str) -> Option<String> {
    if segment.trim().is_empty() || code_line_re().is_match(segment) {
        return None;
    }
    let joined_free: Vec<&str> = segment
        .split_whitespace()
        .filter(|tok| !joined_re().is_match(tok))
        .collect();
    let mut prose = joined_free.join(" ");
    if prose.chars().any(|c| c.is_lowercase()) {
        prose = letter_run_re()
            .replace_all(&prose, |caps: &regex::Captures| {
                let m = caps.get(0).unwrap().as_str();
                if m.chars().all(|c| !c.is_lowercase()) {
                    " ".to_string()
                } else {
                    m.to_string()
                }
            })
            .into_owned();
    }
    let tokens: Vec<String> = word_re()
        .find_iter(&prose)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    if tokens.len() < 4 {
        return None;
    }
    let lang = latin_profile(&prose).language;
    match lang.as_deref() {
        None | Some("en") => None,
        Some(lang) => {
            let stop = STOP_LANGS
                .iter()
                .find(|&&(l, _)| l == lang)
                .map(|&(_, s)| s)
                .unwrap();
            let hits = tokens
                .iter()
                .collect::<HashSet<_>>()
                .iter()
                .filter(|w| stop.contains(&w.as_str()))
                .count();
            (hits >= 2).then(|| lang.to_string())
        }
    }
}

fn non_english_segment(state: &serde_json::Value, max_chars: usize) -> Option<(String, String)> {
    let mut seen = 0;
    let mut leaves = Vec::new();
    iter_text(state, 0, &mut leaves);
    for leaf in leaves {
        for seg in leaf.split('\n') {
            if seen >= max_chars {
                return None;
            }
            let seg = trunc_str(seg, max_chars - seen);
            seen += seg.len();
            if let Some(lang) = named_prose_language(seg) {
                return Some((lang, seg.trim().to_string()));
            }
        }
    }
    None
}

fn leaf_non_english(leaf: &str) -> Option<Analysis> {
    let mut best_n = 0usize;
    let mut best: Option<Analysis> = None;
    for line in leaf.split('\n') {
        if line.len() < 7 {
            continue;
        }
        let sample = trunc_str(line, 4000);
        if sample.trim().is_empty() || code_line_re().is_match(sample) {
            continue;
        }
        let det = analyse_text(sample);
        if det.is_english {
            continue;
        }
        if det.language.as_deref().is_some_and(|l| l != "en") {
            if named_prose_language(sample).is_none() {
                continue;
            }
        } else if det.script != "latin" && det.script != "unknown" {
            if !(!non_latin_words(sample).is_empty()
                && sample.chars().filter(|c| c.is_alphabetic()).count() >= NON_LATIN_MIN_LETTERS)
            {
                continue;
            }
        } else if !(det.language_undecided
            && det.diacritic_rate >= NON_EN_DIACRITIC_RATE
            && word_re().find_iter(sample).count() >= 4)
        {
            continue;
        }
        let n_alpha = sample.chars().filter(|c| c.is_alphabetic()).count();
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    best
}

/// Full detection result for a state.
pub fn analyse(state: &serde_json::Value) -> Analysis {
    let mut result = analyse_text(&state_text(state, 4000));
    if result.script == "latin" && result.is_english {
        let mut leaves = Vec::new();
        iter_text(state, 0, &mut leaves);
        if (leaves.len() > 1 || leaves.iter().any(|l| l.contains('\n')))
            && let Some((lang, mixed)) = non_english_segment(state, 4000)
        {
            result = Analysis {
                language: Some(lang),
                is_english: false,
                language_undecided: false,
                mixed_segment: Some(mixed),
                ..result
            };
        }
    }
    if state.is_string() || state.is_null() || !result.is_english {
        return result;
    }
    let mut best_n = 0usize;
    let mut best: Option<Analysis> = None;
    let mut leaves = Vec::new();
    iter_text(state, 0, &mut leaves);
    for leaf in &leaves {
        if let Some(det) = leaf_non_english(leaf) {
            let n_alpha = trunc_str(leaf, 4000)
                .chars()
                .filter(|c| c.is_alphabetic())
                .count();
            if n_alpha > best_n {
                best_n = n_alpha;
                best = Some(det);
            }
        }
    }
    match best {
        Some(b) => Analysis {
            language: b.language,
            is_english: false,
            language_undecided: b.language_undecided,
            ..result
        },
        None => result,
    }
}

/// True when the English checkpoint can be expected to read this state.
pub fn is_english(state: &serde_json::Value) -> bool {
    analyse(state).is_english
}
