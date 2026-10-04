//! M3: lang parity vs Laya's `analyse` on a frozen corpus of cases.

use oio::lang::{analyse, detect_script, guess_latin_language, state_text, trunc_str};

fn cases() -> Vec<serde_json::Value> {
    let path = format!(
        "{}/tests/fixtures/lang_cases.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn detect_script_values() {
    assert_eq!(detect_script("hello world"), "latin");
    assert_eq!(detect_script("मुझसे"), "devanagari");
    assert_eq!(detect_script("今日"), "han");
}

#[test]
fn analyse_matches_laya_corpus() {
    for case in cases() {
        let input = &case["input"];
        let det = analyse(input);
        assert_eq!(det.script, case["script"], "script mismatch for {input:?}");
        assert_eq!(
            det.is_english, case["is_english"],
            "is_english for {input:?}"
        );
        let expected_lang = case["language"].as_str();
        assert_eq!(
            det.language.as_deref(),
            expected_lang,
            "language for {input:?}"
        );
    }
}

#[test]
fn guess_latin_language_named_cases() {
    assert_eq!(
        guess_latin_language("kan inte logga in på mitt konto").as_deref(),
        Some("sv")
    );
    assert!(guess_latin_language("hello").is_none());
}

// Regression: byte-index slicing panicked on non-ASCII text (review #1).
// Python's str slicing never panics, so this is a porting defect only.
#[test]
fn trunc_str_never_splits_a_char() {
    assert_eq!(trunc_str("hello", 3), "hel");
    assert_eq!(trunc_str("héllo", 3), "hé"); // boundary(2) is mid-'é'
    assert_eq!(trunc_str("héllo", 2), "h");
    assert_eq!(trunc_str("汉", 4), "汉"); // shorter than max
    assert_eq!(trunc_str("汉汉汉", 4), "汉"); // was: panic at byte 4
    assert_eq!(trunc_str("é", 0), "");
    assert_eq!(trunc_str("ok", 99), "ok");
}

#[test]
fn state_text_handles_multibyte_and_exact_budget() {
    use serde_json::json;
    // leaf exactly filling the budget: was "attempt to subtract with overflow"
    let exact = json!({ "a": "x".repeat(100) });
    let _ = state_text(&exact, 100);
    let _ = state_text(&exact, 99);
    // CJK leaf over the budget: was "not a char boundary; inside '汉'"
    let cjk = json!({ "a": "汉".repeat(2000) });
    let t = state_text(&cjk, 4000);
    assert!(t.chars().count() <= 4000);
}

#[test]
fn analyse_survives_multibyte_leaves_past_budget() {
    use serde_json::json;
    // english leaf fills the detection budget, next leaf is multibyte
    let state = json!({
        "a": format!("{} école élève", "x".repeat(3999)),
        "b": "über naïve café"
    });
    let det = analyse(&state);
    let _ = det.script;
    let state2 = json!({ "a": "汉".repeat(2000) });
    let det2 = analyse(&state2);
    assert_eq!(det2.script, "han");
}
