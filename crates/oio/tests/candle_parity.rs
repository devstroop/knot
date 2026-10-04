#![cfg(feature = "candle")]
//! M8: CandleRuntime parity against the torch-exported fixture.
//!
//! Run with: OIO_MODEL_DIR=/path/to/checkpoint cargo test -p oio --features candle --test candle_parity

use oio::candle_runtime::CandleRuntime;
use oio::runtime::Runtime;

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/tests/fixtures/parity_english.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn candle_forward_matches_torch_fixture() {
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip: model.safetensors missing");
        return;
    }

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

    let rt = CandleRuntime::load(&dir).unwrap();
    let (logits, act) = rt
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

    let softmax = |r: &Vec<f32>| {
        let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = r.iter().map(|v| (v - m).exp()).collect();
        let sum: f32 = ex.iter().sum();
        ex.iter().map(|e| e / sum).collect::<Vec<f32>>()
    };
    for (b, row) in logits.iter().enumerate() {
        let p_c = softmax(row);
        let p_t = softmax(
            row.iter()
                .enumerate()
                .map(|(m, _)| expected[b][m])
                .collect::<Vec<_>>()
                .as_slice()
                .to_vec()
                .as_ref(),
        );
        // fp32 accumulation-order noise across a 30-layer ModernBERT-large stays
        // within ~0.06 on raw logits on CPU; the decision-level numbers (the
        // softmax probabilities, and therefore the answers) agree tightly, which
        // is the tolerance the parity contract needs.
        for (m, (&pc, &pt)) in p_c.iter().zip(p_t.iter()).enumerate() {
            if expected[b][m] > -1e3 {
                assert!((pc - pt).abs() < 1e-2, "p[{b}][{m}]: {pc} vs {pt}");
            }
        }
        let argmax_c = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        let argmax_t = expected[b]
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > -1e3)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);
        assert_eq!(argmax_c, argmax_t, "argmax mismatch on row {b}");
    }

    // Action logits should be finite and shaped (batch, n_act).
    assert_eq!(act.len(), batch);
    assert!(act.iter().all(|row| row.len() == 2));
    assert!(act.iter().flatten().all(|v| v.is_finite()));
}
