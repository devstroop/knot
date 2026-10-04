#![cfg(feature = "tokenizer")]
//! M1: prompt assembly parity-of-structure tests (budgets, markers, stats).

use oio::prompt::{InternalQuestion, PromptBuilder, render_options};

fn builder() -> PromptBuilder {
    let path = format!(
        "{}/tests/fixtures/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    PromptBuilder::from_file(&path).expect("load bert tokenizer")
}

fn choice() -> InternalQuestion {
    InternalQuestion {
        t: "choice".into(),
        ins: "which department handles this?".into(),
        crit: Some(
            serde_json::json!({"billing": "invoices and refunds", "tech": "bugs and outages", "other": null}),
        ),
        labels: None,
    }
}

#[test]
fn render_options_choice() {
    let opts = render_options(&choice()).unwrap();
    assert!(opts.iter().any(|o| o.starts_with("billing:")));
    assert!(opts.iter().any(|o| o == "other"));
}

#[test]
fn render_options_score_and_noul() {
    let score = InternalQuestion {
        t: "score".into(),
        ins: "how urgent?".into(),
        crit: Some(serde_json::json!(["not urgent", "soon", "blocking"])),
        labels: None,
    };
    let opts = render_options(&score).unwrap();
    assert_eq!(opts[0], "level 0: not urgent");
    assert_eq!(opts[2], "level 2: blocking");

    let noul = InternalQuestion {
        t: "noul".into(),
        ins: "threaten to cancel?".into(),
        crit: None,
        labels: None,
    };
    let opts = render_options(&noul).unwrap();
    assert_eq!(opts.len(), 2);
    assert!(opts[0].starts_with("false:"));
    assert!(opts[1].starts_with("true:"));
}

#[test]
fn noul_custom_labels_validated() {
    let bad = InternalQuestion {
        t: "noul".into(),
        ins: "q?".into(),
        crit: None,
        labels: Some(serde_json::json!({"false": "same", "true": "same"})),
    };
    assert!(render_options(&bad).is_err());
}

#[test]
fn head_markers_align_with_options() {
    let b = builder();
    let q = choice();
    let (ids, markers, stats) = b.build_head(&q, 256, None).unwrap();
    assert_eq!(markers.len(), 3);
    assert_eq!(stats.options, 3);
    assert_eq!(stats.options_distinct, 3);
    assert!(stats.tokens_per_option.is_none());
    // first token CLS, last token SEP
    assert_eq!(ids[0], b.cls_id());
    let last = ids.len() - 1;
    assert_eq!(ids[last], b.sep_id());
    // markers point at MASK tokens
    for m in markers {
        assert_eq!(ids[m], b.mask_id());
    }
}

#[test]
fn many_options_share_budget_and_report_trim() {
    let b = builder();
    let mut map = serde_json::Map::new();
    for i in 0..40 {
        map.insert(
            format!("option_{i}"),
            serde_json::json!(format!("description number {i} for the option")),
        );
    }
    let q = InternalQuestion {
        t: "choice".into(),
        ins: "pick one".into(),
        crit: Some(serde_json::Value::Object(map)),
        labels: None,
    };
    let (_ids, _markers, stats) = b.build_head(&q, 192, None).unwrap();
    assert!(stats.tokens_per_option.is_some());
}

#[test]
fn sequence_reports_state_truncation() {
    let b = builder();
    let q = choice();
    let long_state: Vec<u32> = (0..2000).map(|_| 1000).collect();
    let (ids, _m, _s, trunc) = b
        .build_sequence(&long_state, &q, 512, 192, None, false)
        .unwrap();
    assert!(ids.len() <= 512);
    assert!(trunc.truncated);
    assert_eq!(trunc.state_tokens_dropped, 2000 - trunc.state_tokens_used);
}

#[test]
fn state_room_is_positive_and_shrinks_with_long_instructions() {
    let b = builder();
    let small = choice();
    let room_small = b.state_room(&small, 512, 192).unwrap();
    let big = InternalQuestion {
        ins: "word ".repeat(200),
        ..choice()
    };
    let room_big = b.state_room(&big, 512, 192).unwrap();
    assert!(room_big <= room_small);
}

#[test]
fn window_budget_clamps_to_smallest_room() {
    let b = builder();
    let qs = vec![choice(), choice()];
    let (size, stride, room) = b.window_budget(&qs, 512, 192, None, None).unwrap();
    assert!(size <= room);
    assert_eq!(stride, (size / 2).max(1));
    let (_, s2, _) = b.window_budget(&qs, 512, 192, Some(128), Some(40)).unwrap();
    assert_eq!(s2, 40);
    assert!(
        b.window_budget(&qs, 512, 192, Some(128), Some(10_000))
            .is_err()
    );
}

// Regression (review #3, #5): SPEC §3 and COMPAT.md promise these inputs are
// caller errors (422 over HTTP); they used to be accepted and rendered as
// "level N: null", 0 options, or silently-defaulted noul descriptions.
#[test]
fn invalid_criteria_rejected_as_caller_errors() {
    let mk = |t: &str, crit: serde_json::Value| InternalQuestion {
        t: t.into(),
        ins: "q?".into(),
        crit: Some(crit),
        labels: None,
    };

    let e = render_options(&mk("score", serde_json::json!([null, "blocking"]))).unwrap_err();
    assert!(e.to_string().contains("level 0 is null"), "{e}");

    let e = render_options(&mk("score", serde_json::json!([]))).unwrap_err();
    assert!(e.to_string().contains("at least one level"), "{e}");

    let e = render_options(&mk("choice", serde_json::json!({}))).unwrap_err();
    assert!(e.to_string().contains("at least one criterion"), "{e}");

    let e = render_options(&mk("noul", serde_json::json!({"maybe": "sort of"}))).unwrap_err();
    assert!(e.to_string().contains("true"), "{e}");

    let e = render_options(&mk("noul", serde_json::json!(["a", "b"]))).unwrap_err();
    assert!(e.to_string().contains("dict"), "{e}");

    // still accepted: noul criteria omitted / null, choice with a null value
    let noul = render_options(&mk("noul", serde_json::Value::Null)).unwrap();
    assert_eq!(noul.len(), 2);
    let choice = render_options(&mk("choice", serde_json::json!({"a": null}))).unwrap();
    assert_eq!(choice, vec!["a".to_string()]);
}

// Regression (review #11): Laya accepts dict/list/number instructions
// (json.dumps into the prompt text) and 422s missing/blank ones; oio used to
// 400 on non-strings and silently accept blanks.
#[test]
fn instructions_accept_json_values_and_reject_blank() {
    use oio::protocol::Question;
    let q: Question = serde_json::from_value(serde_json::json!({
        "type": "noul",
        "instructions": {"field": "churn", "k": 2},
    }))
    .unwrap();
    let iq = InternalQuestion::from_wire("q", &q).unwrap();
    assert_eq!(iq.ins, r#"{"field": "churn", "k": 2}"#);

    let q: Question =
        serde_json::from_value(serde_json::json!({"type": "noul", "instructions": 7})).unwrap();
    let iq = InternalQuestion::from_wire("q", &q).unwrap();
    assert_eq!(iq.ins, "7");

    for bad in [
        serde_json::json!({"type": "noul", "instructions": "   "}),
        serde_json::json!({"type": "noul", "instructions": null}),
        serde_json::json!({"type": "noul", "instructions": []}),
        serde_json::json!({"type": "noul", "instructions": {}}),
        serde_json::json!({"type": "noul"}),
    ] {
        let q: Question = serde_json::from_value(bad.clone()).unwrap();
        let err = InternalQuestion::from_wire("q", &q).unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "for {bad}: {err}"
        );
    }
}

// M2: choice criteria may arrive as a bare list of labels — laya normalizes
// it to `{str(label): None}` (`_to_internal`), and the answer keys, option
// text and probabilities all use that stringified label.
#[test]
fn choice_list_criteria_normalize_to_str_keys() {
    use oio::protocol::Question;
    let q: Question = serde_json::from_value(serde_json::json!({
        "type": "choice",
        "instructions": "pick one",
        "criteria": ["Yes", 7, true, 2.5],
    }))
    .unwrap();
    let iq = InternalQuestion::from_wire("q", &q).unwrap();
    let obj = iq
        .crit
        .as_ref()
        .unwrap()
        .as_object()
        .expect("list form becomes a dict");
    assert_eq!(
        obj.keys().collect::<Vec<_>>(),
        vec!["Yes", "7", "True", "2.5"],
        "Python str() of each label, insertion order kept"
    );
    assert!(obj.values().all(|v| v.is_null()));
    let opts = render_options(&iq).unwrap();
    assert_eq!(opts, vec!["Yes", "7", "True", "2.5"]);
}

#[test]
fn choice_list_criteria_reject_null_structured_and_duplicate_labels() {
    use oio::protocol::Question;
    let mk = |crit: serde_json::Value| {
        let q: Question = serde_json::from_value(serde_json::json!({
            "type": "choice",
            "instructions": "pick one",
            "criteria": crit,
        }))
        .unwrap();
        InternalQuestion::from_wire("q", &q)
            .unwrap_err()
            .to_string()
    };

    let e = mk(serde_json::json!([null]));
    assert!(e.contains("choice label 0 is null"), "{e}");
    let e = mk(serde_json::json!([["nested"]]));
    assert!(e.contains("choice label 0 is not a scalar"), "{e}");
    let e = mk(serde_json::json!(["a", "a"]));
    assert!(e.contains("repeats label 0"), "{e}");
    // Python dict keys: 1 == 1.0 and True == 1 are one key
    let e = mk(serde_json::json!([1, 1.0]));
    assert!(e.contains("repeats label 0"), "{e}");
    let e = mk(serde_json::json!([true, 1]));
    assert!(e.contains("repeats label 0"), "{e}");
    // structural checks run before the duplicate pass, as in laya
    let e = mk(serde_json::json!([null, "a", "a"]));
    assert!(e.contains("is null") && !e.contains("repeats"), "{e}");
}

// `_to_internal` lowercases noul criterion keys, so `{"True": ...}` reaches
// `render_options`' `crit.get("true")` while a non-boolean key still 422s.
#[test]
fn noul_criteria_keys_are_lowercased() {
    use oio::protocol::Question;
    let q: Question = serde_json::from_value(serde_json::json!({
        "type": "noul",
        "instructions": "check it",
        "criteria": {"True": "indeed"},
    }))
    .unwrap();
    let iq = InternalQuestion::from_wire("q", &q).unwrap();
    let obj = iq.crit.as_ref().unwrap().as_object().unwrap();
    assert_eq!(obj.keys().collect::<Vec<_>>(), vec!["true"]);
    let opts = render_options(&iq).unwrap();
    assert_eq!(
        opts,
        vec![
            "false: no, the statement does not hold".to_string(),
            "true: indeed".to_string()
        ]
    );
}
