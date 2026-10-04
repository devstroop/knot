#![cfg(feature = "onnx")]
//! Tier 2 / RESEARCH-COMPARE G3: Laya's `batch_size` and `sort_by_length`
//! batch call controls (`agent.py` `predict_batch`) are an execution concern
//! only — they pick chunking and a stable length sort inside the window, and
//! must not move a single answer: results are always written back to input
//! positions, so every response equals the default single-chunk run.
//!
//! Run: OIO_MODEL_DIR=/path/to/checkpoint cargo test -p oio --features onnx --test batch

use oio::engine::{BatchOpts, Engine};
use oio::protocol::{Answer, SystemOneRequest};
use oio::router::Router;

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/engine_english.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn batch_controls_change_execution_not_answers() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return;
    }
    let fix = fixture();
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": fix["state"],
        "questions": fix["questions"],
    }))
    .unwrap();
    let engine = Engine::load(Router::new(), &[("english", &dir)]).unwrap();

    // Mixed-length states: with sort_by_length the stable ascending reorder
    // (laya `agent.py`) diverges from input order, so a bookkeeping bug in
    // the write-back would show up as a changed answer here.
    let states = vec![
        serde_json::json!("I was charged twice for the same order and I want my money back"),
        serde_json::json!("ok"),
        serde_json::json!("thanks"),
        serde_json::json!(
            "The parcel arrived damaged, the box was crushed and the courier never asked for a signature"
        ),
    ];

    let baseline = engine
        .predict_batch(&states, req.clone(), BatchOpts::default())
        .unwrap();
    assert_eq!(baseline.len(), states.len());

    // batch_size=2 + sort → reorder is active (1 < 2 < 4), window = 16 → one
    // window, chunks of 2 in sorted order. batch_size=3, no sort → plain
    // chunking. batch_size=1 → sort is inert (laya: 1 < chunk fails).
    let runs = [
        (
            "sorted",
            BatchOpts {
                batch_size: Some(2),
                sort_by_length: true,
            },
        ),
        (
            "chunked",
            BatchOpts {
                batch_size: Some(3),
                sort_by_length: false,
            },
        ),
        (
            "one-per",
            BatchOpts {
                batch_size: Some(1),
                sort_by_length: true,
            },
        ),
    ];
    for (label, opts) in runs {
        let run = engine.predict_batch(&states, req.clone(), opts).unwrap();
        assert_eq!(run.len(), states.len(), "{label} result count");
        for (i, (b, r)) in baseline.iter().zip(run.iter()).enumerate() {
            assert_same_answers(&r.answers, &b.answers, &format!("{label} item {i}"));
            assert_eq!(
                r.usage.input_tokens, b.usage.input_tokens,
                "{label} item {i} input_tokens"
            );
        }
    }
}

fn assert_same_answers(
    got: &indexmap::IndexMap<String, Answer>,
    want: &indexmap::IndexMap<String, Answer>,
    item: &str,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{item}"
    );
    for (qid, w) in want {
        let g = &got[qid];
        match (g, w) {
            (
                Answer::Choice {
                    choice: gc,
                    probabilities: gp,
                    answer_confidence: gu,
                    ..
                },
                Answer::Choice {
                    choice: wc,
                    probabilities: wp,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert_eq!(gc, wc, "{item} question {qid}: choice");
                assert!(
                    (gu - wu).abs() < 1e-3,
                    "{item} question {qid}: confidence {gu} vs {wu}"
                );
                for (k, v) in wp {
                    assert!(
                        (gp[k] - v).abs() < 1e-3,
                        "{item} question {qid}: prob {k} {} vs {v}",
                        gp[k]
                    );
                }
            }
            (
                Answer::Score {
                    score: gs,
                    answer_confidence: gu,
                    ..
                },
                Answer::Score {
                    score: ws,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert!(
                    (gs - ws).abs() < 1e-3,
                    "{item} question {qid}: score {gs} vs {ws}"
                );
                assert!((gu - wu).abs() < 1e-3, "{item} question {qid}: confidence");
            }
            (
                Answer::Noul {
                    noul: gn,
                    answer_confidence: gu,
                    ..
                },
                Answer::Noul {
                    noul: wn,
                    answer_confidence: wu,
                    ..
                },
            ) => {
                assert!(
                    (gn - wn).abs() < 1e-3,
                    "{item} question {qid}: noul {gn} vs {wn}"
                );
                assert!((gu - wu).abs() < 1e-3, "{item} question {qid}: confidence");
            }
            (g, w) => panic!("{item} question {qid}: variant mismatch\n got {g:?}\nwant {w:?}"),
        }
    }
}
