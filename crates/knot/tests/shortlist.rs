#![cfg(feature = "tokenizer")]
//! M6: embedding shortlist parity surface (labels/scores/k/n/passthrough).

use indexmap::IndexMap;

use knot::protocol::{
    Answer, Question, QuestionType, Routing, SystemOneRequest, SystemOneResponse, Usage,
};
use knot::shortlist::{CachedEmbedder, Embedder, predict_shortlist};

/// Toy bag-of-words embedder over a tiny vocab, so cosine similarity is
/// meaningful and deterministic.
struct BagEmbedder;

const VOCAB: [&str; 4] = ["fruit", "refund", "charge", "apple"];

impl Embedder for BagEmbedder {
    fn embed(&self, texts: &[String]) -> knot::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                VOCAB
                    .iter()
                    .map(|w| if t.contains(w) { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect())
    }
}

fn questions() -> IndexMap<String, Question> {
    IndexMap::from([(
        "topic".to_string(),
        Question {
            r#type: QuestionType::Choice,
            instructions: String::new(),
            criteria: serde_json::json!({
                "refund": "refund request",
                "fruit": "fruit delivered",
                "charge": "charge dispute",
                "weather": "rain",
            }),
            labels: None,
            option_order: None,
        },
    )])
}

fn canned(_res: SystemOneRequest) -> SystemOneResponse {
    SystemOneResponse {
        model: knot::protocol::AGENT_MODEL.into(),
        answers: IndexMap::from([(
            "topic".to_string(),
            Answer::Choice {
                choice: "refund".into(),
                probabilities: IndexMap::new(),
                confidence: 0.9,
                answer_confidence: 0.9,
                action: knot::protocol::Action {
                    act_probability: 0.1,
                },
                window: None,
            },
        )]),
        usage: Usage {
            input_tokens: 1,
            output_tokens: 0,
            state_tokens: Some(0),
            state_tokens_dropped: Some(0),
            truncated: Some(false),
            truncated_questions: Some(vec![]),
            options: None,
            windows: None,
        },
        routing: Routing {
            model: "english".into(),
            repo: "convaiinnovations/laya".into(),
            reason: "stub".into(),
            detection: None,
            workflow: None,
        },
        shortlist: None,
    }
}

#[test]
fn shortlist_reduces_and_reports_metadata() {
    let req = SystemOneRequest {
        state: serde_json::json!("refund please, I want to dispute the charge"),
        questions: questions(),
        model: None,
        max_len: None,
        head_max_len: None,
        task: None,
        lang: None,
        lang_guess: None,
        min_confidence: None,
    };
    let seen_labels = std::cell::RefCell::new(Vec::new());
    let res = predict_shortlist(
        &req,
        &|r| {
            if let serde_json::Value::Object(m) = &r.questions["topic"].criteria {
                *seen_labels.borrow_mut() = m.keys().cloned().collect();
            }
            Ok(canned(r.clone()))
        },
        &BagEmbedder,
        2,
    )
    .unwrap();
    assert!(seen_labels.borrow().contains(&"refund".to_string()));
    assert!(seen_labels.borrow().contains(&"charge".to_string()));
    let meta = &res.shortlist.as_ref().unwrap()["topic"];
    assert_eq!(meta["k"], 2);
    assert_eq!(meta["n"], 4);
    assert_eq!(meta["passthrough"], false);
    assert_eq!(meta["scores"].as_array().unwrap().len(), 2);
    assert_eq!(meta["labels"].as_array().unwrap().len(), 2);
}

#[test]
fn shortlist_passthrough_when_k_covers_all() {
    let req = SystemOneRequest {
        state: serde_json::json!("anything"),
        questions: questions(),
        model: None,
        max_len: None,
        head_max_len: None,
        task: None,
        lang: None,
        lang_guess: None,
        min_confidence: None,
    };
    let res = predict_shortlist(&req, &|r| Ok(canned(r.clone())), &BagEmbedder, 4).unwrap();
    let meta = &res.shortlist.as_ref().unwrap()["topic"];
    assert_eq!(meta["passthrough"], true);
    assert!(meta["scores"].is_null());
}

#[test]
fn cached_embedder_dedupes_and_evicts() {
    let caching = CachedEmbedder::new(BagEmbedder, 2).unwrap();
    let texts = vec!["fruit red apple".to_string(), "charge dispute".to_string()];
    let a = caching.embed(&texts).unwrap();
    let b = caching.embed(&texts).unwrap();
    assert_eq!(a, b);
    let (size, maxsize, hits, misses) = caching.cache_info();
    assert_eq!(size, 2);
    assert_eq!(maxsize, 2);
    assert_eq!(hits, 2);
    assert_eq!(misses, 2);
    // Third distinct text evicts the LRU entry.
    let _ = caching.embed(&["weather rain".to_string()]).unwrap();
    let (size, _, _, _) = caching.cache_info();
    assert_eq!(size, 2);
    caching.cache_clear();
    let (size, _, hits, misses) = caching.cache_info();
    assert_eq!((size, hits, misses), (0, 0, 0));
}

// Regression (review #2): a single call with more distinct texts than
// `maxsize` panicked with "no entry found for key" (eviction removed keys
// from the same call before the final by-index lookup), and a concurrent
// eviction could do the same. Rows are now cloned under the lock instead of
// looked up afterwards.
#[test]
fn cached_embedder_survives_oversized_and_concurrent_calls() {
    let caching = CachedEmbedder::new(BagEmbedder, 1).unwrap();
    let texts = vec![
        "fruit red apple".to_string(),
        "charge dispute".to_string(),
        "weather rain".to_string(),
    ];
    let out = caching.embed(&texts).unwrap();
    assert_eq!(out.len(), 3);
    let (size, maxsize, _, _) = caching.cache_info();
    assert_eq!(maxsize, 1);
    assert!(size <= maxsize);
    // deterministic rows regardless of cache hits
    let again = caching.embed(&texts).unwrap();
    assert_eq!(out, again);

    let caching = std::sync::Arc::new(CachedEmbedder::new(BagEmbedder, 2).unwrap());
    let mut handles = Vec::new();
    for t in 0..4 {
        let c = caching.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..25 {
                let texts = vec![
                    format!("t{t} first subject {i}"),
                    format!("t{t} second subject {i}"),
                    format!("t{t} third subject {i}"),
                ];
                let rows = c.embed(&texts).expect("embed");
                assert_eq!(rows.len(), 3);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
