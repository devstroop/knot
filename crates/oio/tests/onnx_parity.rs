//! M2: ONNXRuntime parity against the torch-exported fixture.
//!
//! Run with: OIO_MODEL_DIR=/path/to/checkpoint cargo test -p oio --features onnx --test onnx_parity
//! Skips when OIO_MODEL_DIR is unset or the export files are missing.

use oio::runtime::{Calibration, answer_confidence, confidence_from_probs, scaled_softmax};

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/parity_english.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn model_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var("OIO_MODEL_DIR").ok()?;
    let p = std::path::PathBuf::from(dir);
    (p.join("laya.onnx").exists() && p.join("rl_agent_config.json").exists()).then_some(p)
}

#[test]
fn calibration_lookup_matches_laya_buckets() {
    let cfg = serde_json::json!({
        "temperature": [1.6369, 1.2514, 1.9833],
        "temperature_by_options": {"choice:3-5": 1.7601, "noul:2": 1.9833, "choice:11+": 0.10058},
    });
    let cal = Calibration::from_config(&cfg);
    assert!((cal.for_question("choice", 0, 4) - 1.7601).abs() < 1e-9);
    assert!((cal.for_question("choice", 0, 20) - 0.5).abs() < 1e-9); // clamped 0.1006 -> 0.5
    assert!((cal.for_question("score", 1, 3) - 1.2514).abs() < 1e-9);
    assert!((cal.for_question("noul", 2, 2) - 1.9833).abs() < 1e-9);
}

#[test]
fn scaled_softmax_and_confidence() {
    let logits = vec![2.0f32, 0.5, -1.0];
    let p = scaled_softmax(&logits, 1.0);
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(p[0] > p[1] && p[1] > p[2]);
    let sharpened = scaled_softmax(&logits, 0.5);
    assert!(sharpened[0] > p[0]);
    let conf = confidence_from_probs(&p);
    assert!((0.0..=1.0).contains(&conf));
    assert!(answer_confidence(&p) >= p[0] - 1e-6);
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_forward_matches_torch_fixture() {
    let Some(dir) = model_dir() else {
        eprintln!("skip: OIO_MODEL_DIR not set or export missing");
        return;
    };
    use oio::runtime::OnnxRuntime;

    let fix = fixture();
    let collated = &fix["collated"];
    let flat = |key: &str| -> Vec<i64> {
        collated[key]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let batch = collated["input_ids"].as_array().unwrap().len();
    let seq = collated["input_ids"][0].as_array().unwrap().len();
    let nmarkers = collated["marker_pos"][0].as_array().unwrap().len();

    let mmask: Vec<bool> = collated["marker_mask"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>()
        })
        .collect();

    let rt = OnnxRuntime::load(&dir, oio::runtime::Device::Cpu).unwrap();
    let (logits, _act) = rt
        .forward(
            &flat("input_ids"),
            &flat("attention_mask"),
            &flat("marker_pos"),
            &mmask,
            &collated["qtype"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>(),
            batch,
            seq,
            nmarkers,
        )
        .unwrap();

    let expected: Vec<Vec<f32>> = fix["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();

    for (b, row) in logits.iter().enumerate() {
        for (m, &v) in row.iter().enumerate() {
            let e = expected[b][m];
            // masked markers sit at -1e4 in the torch reference; compare only real ones
            if e > -1e3 {
                assert!((v - e).abs() < 5e-3, "logits[{b}][{m}]: {v} vs {e}");
            }
        }
    }
}
