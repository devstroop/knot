//! Audit-adapter load test — the evidence run ADR-006 open questions Q3/Q5
//! defer to ("settle with numbers from the adapter's own load test").
//!
//! Ignored by default (like `runtime_bench`); run with a release build:
//!
//! ```sh
//! cargo test -p knot-nqlite --release --test load -- --ignored --nocapture
//! ```
//!
//! Env knobs (bindings are echoed in the report):
//! - `KNOT_LOAD_N`      total decisions in the paced phases (default 4000)
//! - `KNOT_LOAD_RPS`    comma-separated paced rates to probe (default 500,2000)
//! - `KNOT_LOAD_BURST`  unpaced burst size — the overflow diagnostic (default 8000)
//! - `KNOT_LOAD_CAP`    queue capacity (default 1024)
//!
//! What it settles:
//! - **Q5 (durability tier)**: the paced ladder finds the writer's sustained
//!   ceiling — below it the bounded queue absorbs every write (zero drops =
//!   evidence-grade), above it rows are *counted* drops (the documented
//!   operating envelope: never silent, never blocking, `written+dropped=n`
//!   and `durable count == written` asserted at every rate). ms/row comes
//!   from the writer's throughput.
//! - **Q3 (retention mechanics)**: bytes/row growth of the ledger file, and
//!   whether the boot-time `PRUNE HISTORY` actually binds history — measured
//!   as "old `AS OF` starts erroring (`HistoryPruned`) + rows intact", which
//!   is the retention contract, not just a file-size delta.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use knot::Predictor;
use knot::protocol::{SystemOneRequest, SystemOneResponse};
use knot_nqlite::{AuditConfig, AuditPredictor};

/// Returns immediately — pacing lives in the caller loop, so the writer's
/// ability to keep up is what gets measured, not inference latency.
#[derive(Clone)]
struct InstantStub;

impl Predictor for InstantStub {
    fn predict(&self, _req: &SystemOneRequest) -> knot::Result<SystemOneResponse> {
        Ok(SystemOneResponse {
            model: "load-stub".into(),
            answers: Default::default(),
            usage: serde_json::from_value(serde_json::json!({
                "input_tokens": 1, "output_tokens": 0
            }))
            .expect("usage json"),
            routing: serde_json::from_value(serde_json::json!({
                "model": "load", "repo": "load", "reason": "bench"
            }))
            .expect("routing json"),
            shortlist: None,
        })
    }

    fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        _opts: knot::engine::BatchOpts,
    ) -> knot::Result<Vec<SystemOneResponse>> {
        Ok(states
            .iter()
            .map(|_| self.predict(&template).unwrap())
            .collect())
    }

    fn loaded(&self) -> Vec<&'static str> {
        vec!["load-stub"]
    }

    fn route(
        &self,
        _state: &serde_json::Value,
        _questions: Option<&HashMap<String, serde_json::Value>>,
        _model: Option<&str>,
        _task: Option<&str>,
        _lang: Option<&str>,
        _lang_guess: Option<&str>,
    ) -> knot::Result<serde_json::Value> {
        Ok(serde_json::json!({"model": "load"}))
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn request(i: usize) -> SystemOneRequest {
    serde_json::from_value(serde_json::json!({
        // Distinct state per decision → one state node each, worst-case
        // provenance shape (2 RELATE targets per row).
        "state": {"seq": i, "kind": "load-probe"},
        "questions": {}
    }))
    .expect("request json")
}

fn file_size(path: &std::path::Path) -> u64 {
    // Sidecar WAL counts too — the ledger's on-disk footprint is both files.
    let main = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut wal = path.as_os_str().to_os_string();
    wal.push("-wal");
    let wal = std::fs::metadata(wal).map(|m| m.len()).unwrap_or(0);
    main + wal
}

fn decision_count(path: &std::path::Path) -> usize {
    let mut db = nqlite::Database::open(path).expect("reopen ledger");
    let plan = nql::parse("SELECT COUNT(*) FROM decision;").expect("parse");
    let res = db.execute(&plan).expect("count");
    match res[0].rows[0].record.body.get("count") {
        Some(nqlite::Value::Int(n)) => *n as usize,
        other => panic!("unexpected count cell: {other:?}"),
    }
}

/// One paced run: `n` decisions at `rps`, returning (drops, elapsed, count).
fn paced_run(
    dir: &std::path::Path,
    tag: &str,
    n: usize,
    rps: usize,
    cap: usize,
) -> (u64, Duration, u64) {
    let path = dir.join(format!("load-{tag}.nql"));
    let audit = AuditPredictor::new(
        InstantStub,
        AuditConfig {
            path: path.clone(),
            capacity: cap,
            excerpt_chars: 256,
        },
    )
    .expect("open ledger");
    let period = Duration::from_secs_f64(1.0 / rps as f64);
    let t0 = Instant::now();
    for i in 0..n {
        audit.predict(&request(i)).expect("predict");
        std::thread::sleep(period);
    }
    let elapsed = t0.elapsed();
    let (_written, dropped) = audit.stats();
    drop(audit); // join writer → everything accepted is flushed
    let count = decision_count(&path);
    (dropped, elapsed, count as u64)
}

#[test]
#[ignore = "load test: run with --release -- --ignored --nocapture (see header)"]
fn adapter_load_and_retention() {
    let n = env_usize("KNOT_LOAD_N", 4000);
    let burst = env_usize("KNOT_LOAD_BURST", 8000);
    let cap = env_usize("KNOT_LOAD_CAP", 1024);
    let rates: Vec<usize> = std::env::var("KNOT_LOAD_RPS")
        .unwrap_or_else(|_| "500,2000".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let dir = std::env::temp_dir().join(format!("knot-audit-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    println!("== knot-nqlite audit adapter load ==");
    println!("profile: release   cap: {cap}   n/rate: {n}   burst: {burst}");
    println!(
        "host: {} {:?} cores",
        std::env::consts::OS,
        std::thread::available_parallelism().map(|p| p.get())
    );

    // --- Q5a: paced sustain — find the writer's zero-loss envelope. -------
    let mut client_peak = 0.0f64;
    let mut durable_ceiling = 0.0f64;
    let mut sustainable = 0usize;
    let lowest = *rates.iter().min().expect("at least one rate");
    for &rps in &rates {
        let (drops, elapsed, count) = paced_run(&dir, &format!("r{rps}"), n, rps, cap);
        let secs = elapsed.as_secs_f64();
        let achieved = n as f64 / secs;
        let durable_rate = count as f64 / secs;
        client_peak = client_peak.max(achieved);
        durable_ceiling = durable_ceiling.max(durable_rate);
        println!(
            "paced {rps:>5} rps: achieved {achieved:>7.1} rows/s, drops {drops}, \
             durable count {count} (expect {n}), wall {secs:.2}s"
        );
        if drops == 0 {
            sustainable = sustainable.max(rps);
        }
        // Evidence-grade invariant at EVERY rate: no silent loss, no lost row.
        assert_eq!(count as usize, n - drops as usize, "durable == accepted");
        assert_eq!(
            count as usize + drops as usize,
            n,
            "accounting: durable + counted drops = calls ({rps} rps)"
        );
    }
    // Below the lowest probed rate the queue must be lossless (if it is not,
    // the floor itself is broken — fail loudly).
    let (floor_drops, _, floor_count) = paced_run(&dir, "floor", n, lowest, cap);
    println!(
        "floor {lowest} rps: drops {floor_drops}, durable {floor_count} — \
         zero-loss floor {}",
        if floor_drops == 0 { "holds" } else { "BROKEN" }
    );
    assert_eq!(floor_drops, 0, "zero-loss floor broken at {lowest} rps");
    assert_eq!(floor_count as usize, n);
    println!(
        "Q5 envelope: sustained zero-loss up to {sustainable} rps; above it rows are \
         counted drops (written+dropped == n) — capacity sizing is the operator's lever"
    );

    // --- Q5b: unpaced burst — where the counted incident takes over. -------
    // No sleep: the client outruns the writer; the queue (cap) absorbs a
    // while, then try_send reports Full. Nothing blocks, nothing is silent.
    let path = dir.join("load-burst.nql");
    let audit = AuditPredictor::new(
        InstantStub,
        AuditConfig {
            path: path.clone(),
            capacity: cap,
            excerpt_chars: 256,
        },
    )
    .expect("open ledger");
    let t0 = Instant::now();
    for i in 0..burst {
        audit.predict(&request(i)).expect("predict");
    }
    let burst_elapsed = t0.elapsed();
    let (written_burst, dropped_burst) = audit.stats();
    drop(audit);
    let count_burst = decision_count(&path);
    println!(
        "burst unpaced: {burst} calls in {:.2}s ({:.0} calls/s), accepted {written_burst}, \
         counted drops {dropped_burst} (cap {cap}), durable count {count_burst}",
        burst_elapsed.as_secs_f64(),
        burst as f64 / burst_elapsed.as_secs_f64(),
    );
    assert_eq!(
        count_burst as u64, written_burst,
        "durability: every accepted row lands (drops are counted, never lost silently)"
    );
    assert!(
        written_burst + dropped_burst == burst as u64,
        "every call is accounted: accepted + counted = total"
    );

    // --- Q3: growth + retention contract. ----------------------------------
    // Use the paced 2000 run's store (largest, fully durable).
    let keep = dir.join(format!("load-r{}.nql", rates.last().unwrap_or(&2000)));
    let size_before = file_size(&keep);
    let count_before = decision_count(&keep);
    let bytes_per_row = size_before as f64 / count_before as f64;

    // Boot again → the adapter's own prune-on-boot path (PRUNE HISTORY).
    let pruned = AuditPredictor::new(
        InstantStub,
        AuditConfig {
            path: keep.clone(),
            capacity: cap,
            excerpt_chars: 256,
        },
    )
    .expect("reopen for prune");
    drop(pruned);
    let size_after = file_size(&keep);
    let count_after = decision_count(&keep);
    println!(
        "retention: file+wal {} bytes for {count_before} rows ({bytes_per_row:.0} B/row); \
         after boot PRUNE: {size_after} bytes ({:+.1}%)",
        size_before,
        (size_after as f64 - size_before as f64) / size_before as f64 * 100.0,
    );
    assert_eq!(count_before, count_after, "PRUNE never deletes records");
    assert!(count_before > 0);

    // The retention contract is behavioural, not just bytes: after the prune,
    // replay below the snapshot horizon must fail loudly (`HistoryPruned`)
    // instead of reconstructing a pre-ledger world.
    let mut db = nqlite::Database::open(&keep).expect("reopen");
    let old = nql::parse("SELECT * FROM decision AS OF 1;").expect("parse");
    let pruned_err = db.execute(&old).expect_err("pre-snapshot AS OF must error");
    println!("retention contract: AS OF 1 → {pruned_err} (rows intact: {count_after})");

    std::fs::remove_dir_all(&dir).ok();
    println!(
        "verdict: writer durable ceiling ≈{durable_ceiling:.0} rows/s (client peak \
         {client_peak:.0} calls/s); paced drops=0 below ceiling (Q5 evidence-grade ✓), \
         burst drops counted not silent ✓, growth {bytes_per_row:.0} B/row + prune \
         contract ✓ (Q3: rows-forever + bounded history; max-age policy still an \
         operator choice)"
    );
}
