#![cfg(all(feature = "onnx", feature = "candle"))]
//! Paired in-process benchmark for Knot's ONNX Runtime and Candle backends.
//!
//! Run in release mode with a checkpoint containing both laya.onnx and
//! model.safetensors:
//! KNOT_MODEL_DIR=/path/to/checkpoint cargo test -p knot \
//!   --features onnx,candle --release --test runtime_bench -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use knot::engine::{BatchOpts, Engine};
use knot::protocol::{SystemOneRequest, SystemOneResponse};
use knot::router::Router;
use knot::runtime::Device;
use serde_json::Value;

struct Sample {
    state: Value,
    request: SystemOneRequest,
}

fn samples() -> Vec<Sample> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_english.jsonl");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: Value = serde_json::from_str(line).unwrap();
            let request = serde_json::from_value(serde_json::json!({
                "state": row["state"],
                "questions": row["questions"],
                "model": "english",
            }))
            .unwrap();
            Sample {
                state: row["state"].clone(),
                request,
            }
        })
        .collect()
}

fn load_time(label: &str, load: impl FnOnce() -> Engine) -> (Engine, Duration) {
    let started = Instant::now();
    let engine = load();
    let elapsed = started.elapsed();
    eprintln!("{label} engine_load_ms={:.2}", elapsed.as_secs_f64() * 1e3);
    (engine, elapsed)
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

fn report_latency(label: &str, samples_ms: &[f64]) {
    let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let mut ordered = samples_ms.to_vec();
    let p50 = percentile(&mut ordered, 0.50);
    let p95 = percentile(&mut ordered, 0.95);
    eprintln!(
        "{label} n={} mean_ms={mean:.2} p50_ms={p50:.2} p95_ms={p95:.2}",
        samples_ms.len()
    );
}

#[derive(Default)]
struct OutputDelta {
    probability: f32,
    score: f32,
}

fn output_delta(a: &SystemOneResponse, b: &SystemOneResponse) -> OutputDelta {
    assert_eq!(a.answers.len(), b.answers.len());
    let mut delta = OutputDelta::default();
    for (qid, left) in &a.answers {
        let right = &b.answers[qid];
        let left = serde_json::to_value(left).unwrap();
        let right = serde_json::to_value(right).unwrap();
        for key in ["probabilities", "noul"] {
            if let (Some(l), Some(r)) = (left.get(key), right.get(key)) {
                if let (Some(l), Some(r)) = (l.as_object(), r.as_object()) {
                    for (name, lvalue) in l {
                        let rvalue = r[name].as_f64().unwrap() as f32;
                        delta.probability = delta
                            .probability
                            .max((lvalue.as_f64().unwrap() as f32 - rvalue).abs());
                    }
                } else if let (Some(l), Some(r)) = (l.as_f64(), r.as_f64()) {
                    delta.probability = delta.probability.max((l as f32 - r as f32).abs());
                }
            }
        }
        assert_eq!(left["type"], right["type"]);
        match left["type"].as_str().unwrap() {
            "choice" => assert_eq!(left["choice"], right["choice"], "{qid}"),
            "score" => {
                let score_delta = (left["score"].as_f64().unwrap()
                    - right["score"].as_f64().unwrap())
                .abs() as f32;
                assert!(score_delta.is_finite(), "{qid}: non-finite score delta");
                delta.score = delta.score.max(score_delta);
            }
            "noul" => assert_eq!(
                left["noul"].as_f64().unwrap() >= 0.5,
                right["noul"].as_f64().unwrap() >= 0.5,
                "{qid}"
            ),
            other => panic!("unexpected answer type {other}"),
        }
    }
    delta
}

fn run_paired_batch(
    onnx: &Engine,
    candle: &Engine,
    states: &[Value],
    template: &SystemOneRequest,
    batch_size: usize,
    rounds: usize,
) {
    let opts = BatchOpts {
        batch_size: Some(batch_size),
        sort_by_length: true,
    };
    let mut onnx_ms = Vec::with_capacity(rounds);
    let mut candle_ms = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let (first, second) = if round % 2 == 0 {
            (true, false)
        } else {
            (false, true)
        };
        for is_onnx in [first, second] {
            let engine = if is_onnx { onnx } else { candle };
            let started = Instant::now();
            let output = engine
                .predict_batch(states, template.clone(), opts)
                .unwrap();
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            assert_eq!(output.len(), states.len());
            if is_onnx {
                onnx_ms.push(elapsed);
            } else {
                candle_ms.push(elapsed);
            }
        }
    }
    let total_states = (states.len() * rounds) as f64;
    let onnx_seconds = onnx_ms.iter().sum::<f64>() / 1e3;
    let candle_seconds = candle_ms.iter().sum::<f64>() / 1e3;
    report_latency(&format!("onnx_batch{batch_size}"), &onnx_ms);
    report_latency(&format!("candle_batch{batch_size}"), &candle_ms);
    eprintln!(
        "batch{batch_size} states_per_second onnx={:.2} candle={:.2} candle_over_onnx_throughput={:.3}x",
        total_states / onnx_seconds,
        total_states / candle_seconds,
        onnx_seconds / candle_seconds
    );
}

fn model_dir() -> PathBuf {
    let dir = PathBuf::from(std::env::var("KNOT_MODEL_DIR").expect("set KNOT_MODEL_DIR"));
    for name in ["laya.onnx", "model.safetensors", "rl_agent_config.json"] {
        assert!(
            dir.join(name).is_file(),
            "missing {}",
            dir.join(name).display()
        );
    }
    dir
}

fn engine_onnx(dir: &Path) -> Engine {
    Engine::load_with_device(Router::new(), &[("english", dir)], Device::Cpu).unwrap()
}

fn engine_candle(dir: &Path) -> Engine {
    Engine::load_candle(Router::new(), &[("english", dir)]).unwrap()
}

fn print_host_settings() {
    let affinity = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "Cpus_allowed_list: unavailable".into());
    let cpu_quota = std::fs::read_to_string("/sys/fs/cgroup/cpu.max")
        .map(|value| format!("cgroup_cpu.max={}", value.trim()))
        .unwrap_or_else(|_| "cgroup_cpu.max=unavailable".into());
    eprintln!(
        "host_settings available_parallelism={:?} {} {} KNOT_ORT_INTRA_THREADS={:?} RAYON_NUM_THREADS={:?} OMP_NUM_THREADS={:?} MKL_NUM_THREADS={:?}",
        std::thread::available_parallelism().map(|n| n.get()),
        affinity,
        cpu_quota,
        std::env::var("KNOT_ORT_INTRA_THREADS").ok(),
        std::env::var("RAYON_NUM_THREADS").ok(),
        std::env::var("OMP_NUM_THREADS").ok(),
        std::env::var("MKL_NUM_THREADS").ok(),
    );
}

#[test]
#[ignore = "paired runtime benchmark; requires both model formats; run in release mode"]
fn benchmark_onnx_vs_candle() {
    let dir = model_dir();
    let samples = samples();
    assert!(!samples.is_empty());
    let rounds = std::env::var("KNOT_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20);
    assert!(rounds > 0, "KNOT_BENCH_ROUNDS must be positive");

    let mode = std::env::var("KNOT_BENCH_MODE").unwrap_or_else(|_| "all".into());
    assert!(
        ["all", "single", "batch"].contains(&mode.as_str()),
        "KNOT_BENCH_MODE must be all, single, or batch"
    );
    eprintln!(
        "runtime_bench checkpoint={} cases={} rounds={} mode={} device=cpu profile=release",
        dir.display(),
        samples.len(),
        rounds,
        mode
    );
    print_host_settings();
    let (onnx, _) = load_time("onnx", || engine_onnx(&dir));
    let (candle, _) = load_time("candle", || engine_candle(&dir));

    if mode == "all" || mode == "single" {
        let mut onnx_ms = Vec::with_capacity(samples.len() * rounds);
        let mut candle_ms = Vec::with_capacity(samples.len() * rounds);
        let mut max_delta = OutputDelta::default();

        for _ in 0..3 {
            for sample in &samples {
                let onnx_result = onnx.predict(&sample.request).unwrap();
                let candle_result = candle.predict(&sample.request).unwrap();
                let delta = output_delta(&onnx_result, &candle_result);
                max_delta.probability = max_delta.probability.max(delta.probability);
                max_delta.score = max_delta.score.max(delta.score);
            }
        }

        for round in 0..rounds {
            for sample in &samples {
                let (first_is_onnx, second_is_onnx) = if round % 2 == 0 {
                    (true, false)
                } else {
                    (false, true)
                };
                let mut results = [None, None];
                for is_onnx in [first_is_onnx, second_is_onnx] {
                    let (index, engine) = if is_onnx { (0, &onnx) } else { (1, &candle) };
                    let started = Instant::now();
                    results[index] = Some(engine.predict(&sample.request).unwrap());
                    let elapsed = started.elapsed().as_secs_f64() * 1e3;
                    if is_onnx {
                        onnx_ms.push(elapsed);
                    } else {
                        candle_ms.push(elapsed);
                    }
                }
                let delta =
                    output_delta(results[0].as_ref().unwrap(), results[1].as_ref().unwrap());
                max_delta.probability = max_delta.probability.max(delta.probability);
                max_delta.score = max_delta.score.max(delta.score);
            }
        }
        report_latency("onnx_single", &onnx_ms);
        report_latency("candle_single", &candle_ms);
        eprintln!(
            "single candle_over_onnx_latency={:.3}x max_probability_delta={:.8} max_score_delta={:.8}",
            candle_ms.iter().sum::<f64>() / onnx_ms.iter().sum::<f64>(),
            max_delta.probability,
            max_delta.score,
        );
    }

    if mode == "all" || mode == "batch" {
        let states: Vec<Value> = samples.iter().map(|s| s.state.clone()).collect();
        let template = samples[0].request.clone();
        for batch_size in [1, 2, 4] {
            run_paired_batch(
                &onnx,
                &candle,
                &states,
                &template,
                batch_size,
                rounds.min(10),
            );
        }
    }
}
