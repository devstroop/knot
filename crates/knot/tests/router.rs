//! M3: router decision precedence + LRU.

use knot::router::{Router, english_from_code, match_typed_decisions_workflow, normalise_name};
use std::collections::HashMap;

fn state(text: &str) -> serde_json::Value {
    serde_json::json!(text)
}

#[test]
fn normalise_aliases() {
    assert_eq!(normalise_name("en").unwrap(), "english");
    assert_eq!(normalise_name("multi").unwrap(), "multilingual");
    assert_eq!(normalise_name("typed").unwrap(), "typed-decisions");
    assert!(normalise_name("nope").is_err());
}

#[test]
fn english_from_code_handles_env_forms() {
    assert_eq!(english_from_code("en_US.UTF-8"), Some(true));
    assert_eq!(english_from_code("en-US"), Some(true));
    assert_eq!(english_from_code("fr_FR"), Some(false));
    assert_eq!(english_from_code("C"), None);
    assert_eq!(english_from_code(""), None);
}

#[test]
fn explicit_model_wins() {
    let r = Router::new();
    let d = r
        .route(
            &state("hola mundo, esto es español"),
            None,
            Some("english"),
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(d.model, "english");
}

#[test]
fn explicit_lang_routes() {
    let r = Router::new();
    let d = r
        .route(&state("whatever"), None, None, None, Some("es_ES"), None)
        .unwrap();
    assert_eq!(d.model, "multilingual");
    let d = r
        .route(&state("whatever"), None, None, None, Some("en-US"), None)
        .unwrap();
    assert_eq!(d.model, "english");
}

#[test]
fn detection_routes_non_latin_and_english() {
    let r = Router::new();
    let d = r
        .route(
            &state("मुझसे मार्च में दो बार शुल्क लिया गया"),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(d.model, "multilingual");
    let d = r
        .route(
            &state("Hi, we were billed twice for March, please refund us today."),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(d.model, "english");
}

#[test]
fn lang_guess_short_circuits() {
    let r = Router::new();
    let d = r
        .route(
            &state("hello there, how are you doing today my friend"),
            None,
            None,
            None,
            None,
            Some("fr"),
        )
        .unwrap();
    assert_eq!(d.model, "multilingual");
}

#[test]
fn auto_task_detection_workflow_match() {
    let mut r = Router::new();
    r.auto_task_detection = true;
    let mut q: HashMap<String, serde_json::Value> = HashMap::new();
    for id in ["action", "needs_review", "outcome", "risk", "urgency"] {
        q.insert(id.into(), serde_json::json!({}));
    }
    let d = r
        .route(&state("anything"), Some(&q), None, None, None, None)
        .unwrap();
    assert_eq!(d.model, "typed-decisions");
    assert_eq!(
        match_typed_decisions_workflow(&q),
        Some("agent_trace_observability")
    );
}

#[test]
fn lru_evicts_beyond_max_loaded() {
    let mut r = Router::new();
    r.max_loaded = 2;
    assert!(r.touch("english").is_empty());
    assert!(r.touch("multilingual").is_empty());
    let evicted = r.touch("typed-decisions");
    assert_eq!(evicted, vec!["english"]);
    assert_eq!(r.loaded(), vec!["multilingual", "typed-decisions"]);
}
