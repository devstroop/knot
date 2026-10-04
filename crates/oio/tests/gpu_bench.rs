//! Device latency bench (PLAN: GPU timing — CUDA vs CPU on one machine).
//!
//! Loads the checkpoint onto the device `OIO_DEVICE` names (SPEC §10) and
//! prints one line with p50/p95/mean per predict. Two runs on a GPU box —
//! once with `cpu`, once with `cuda` — are the comparison;
//! `.github/workflows/gpu.yml` runs exactly that pair.
//!
//! ```text
//! OIO_MODEL_DIR=/path/to/checkpoint [OIO_DEVICE=cuda] cargo test \
//!     --features cuda --release --test gpu_bench -- --ignored --nocapture
//! ```
//!
//! Skips without `OIO_MODEL_DIR` (gate rules). `OIO_DEVICE=cuda` on a
//! build or GPU that cannot load fails loudly instead of skipping.

#![cfg(feature = "onnx")]

use std::path::PathBuf;
use std::time::Instant;

use oio::engine::Engine;
use oio::protocol::SystemOneRequest;
use oio::router::Router;
use oio::runtime::Device;
use serde_json::json;

/// One representative English choice request (eval fixture shape).
fn sample_request() -> SystemOneRequest {
    serde_json::from_value(json!({
        "state": "I was charged twice for the same order, please refund me.",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is the user asking for?",
                "criteria": {
                    "billing": "refunds or charges",
                    "technical": "bugs or errors",
                    "sales": "pricing or purchase"
                }
            }
        }
    }))
    .unwrap()
}

#[test]
#[ignore = "device latency bench: run with --ignored --nocapture under OIO_MODEL_DIR"]
fn device_latency_bench() {
    let device = Device::parse(std::env::var("OIO_DEVICE").ok().as_deref()).unwrap();
    let Ok(dir) = std::env::var("OIO_MODEL_DIR") else {
        eprintln!("skip: OIO_MODEL_DIR not set");
        return;
    };
    let dir = PathBuf::from(dir);
    if !dir.join("laya.onnx").exists() {
        eprintln!("skip: laya.onnx missing");
        return;
    }
    // A device that cannot load must fail here, not degrade (SPEC §10).
    let engine = Engine::load_with_device(Router::new(), &[("english", dir.as_path())], device)
        .unwrap_or_else(|e| panic!("load on device {}: {e}", device.as_str()));

    let request = sample_request();
    let rounds: usize = std::env::var("OIO_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    for _ in 0..3 {
        engine.predict(&request).unwrap();
    }

    let mut samples_ms: Vec<f64> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t0 = Instant::now();
        engine.predict(&request).unwrap();
        samples_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    let p = |q: f64| samples_ms[((n as f64) * q).floor().min(n as f64 - 1.0) as usize];
    let mean = samples_ms.iter().sum::<f64>() / n as f64;
    println!(
        "device={} rounds={n} p50={:.2}ms p95={:.2}ms mean={:.2}ms",
        device.as_str(),
        p(0.50),
        p(0.95),
        mean
    );
}
