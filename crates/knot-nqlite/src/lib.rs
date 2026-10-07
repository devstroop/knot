//! knot ↔ nqlite — decision audit ledger (ADR-006 seam B).
//!
//! A `Predictor` decorator that appends one nqlite transaction **after** the
//! response returns: a `decision` row (hashes + length-limited excerpt, per
//! ADR-006 §4/privacy line) plus `state -[:decided]-> decision` and
//! `decision -[:used_checkpoint]-> checkpoint` provenance edges.
//!
//! Rules honored (ADR-006):
//! 1. **Single opener** — the writer thread owns the one `Database` handle.
//! 2. **Off the hot path** — bounded `sync_channel` + dedicated thread;
//!    overflow is a *counted* drop, never a block, never silent.
//! 3. **Retention** — `PRUNE HISTORY` on every open (prune-on-boot); a
//!    max-age/max-rows policy is open question 3 for the adapter's load test.
//! 7. **Votes/parity** — nothing here touches ranking; the feature is off
//!    unless `KNOT_AUDIT_DB` is set (open question 4: gate name/default).
//!
//! Enabled per deployment via `AuditConfig::from_env()` — no env, no adapter.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use knot::Predictor;
use knot::protocol::{SystemOneRequest, SystemOneResponse};
use nqlite::Database;
use sha2::{Digest, Sha256};

/// Deployment configuration — the adapter is constructed only when this
/// exists (default off).
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// nqlite store file (one per process — the #84 single-writer lock).
    pub path: PathBuf,
    /// Bounded write-queue capacity (counted overflow beyond it).
    pub capacity: usize,
    /// Max excerpt characters stored per decision (privacy/size line).
    pub excerpt_chars: usize,
}

impl AuditConfig {
    /// `KNOT_AUDIT_DB` (required to enable), `KNOT_AUDIT_CAPACITY` (1024),
    /// `KNOT_AUDIT_EXCERPT` (512). Returns `None` when disabled.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("KNOT_AUDIT_DB").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        Some(Self {
            path: PathBuf::from(path),
            capacity: env_usize("KNOT_AUDIT_CAPACITY", 1024),
            excerpt_chars: env_usize("KNOT_AUDIT_EXCERPT", 512),
        })
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
        .max(1)
}

/// `Predictor` decorator: responses pass through untouched; each successful
/// predict enqueues one NQL transaction for the writer thread.
pub struct AuditPredictor<P: Predictor> {
    inner: P,
    tx: Option<SyncSender<String>>,
    writer: Option<JoinHandle<()>>,
    excerpt_chars: usize,
    written: AtomicU64,
    dropped: AtomicU64,
}

impl<P: Predictor> AuditPredictor<P> {
    pub fn new(inner: P, cfg: AuditConfig) -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<String>(cfg.capacity);
        let path = cfg.path.clone();
        let writer = std::thread::Builder::new()
            .name("knot-audit-writer".into())
            .spawn(move || {
                let mut db = match Database::open(&path) {
                    Ok(db) => db,
                    Err(e) => {
                        tracing::error!(path = %path.display(), error = %e,
                            "audit ledger: open failed — ledger disabled for this process");
                        return;
                    }
                };
                // ADR-006 rule 3 / open Q3: prune-on-boot bounds the active
                // file (nqlite #95). Both statements are idempotent on a
                // re-opened store (verified against the engine).
                run(&mut db, "CREATE TABLE decision; CREATE TABLE state; CREATE TABLE checkpoint; PRUNE HISTORY;");
                while let Ok(program) = rx.recv() {
                    run(&mut db, &program);
                }
                // rx dropped → drain anything left, then the thread ends and
                // Drop's join() guarantees the flush before process exit.
                while let Ok(program) = rx.try_recv() {
                    run(&mut db, &program);
                }
            })
            .context("spawn audit writer")?;
        Ok(Self {
            inner,
            tx: Some(tx),
            writer: Some(writer),
            excerpt_chars: cfg.excerpt_chars,
            written: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        })
    }

    /// Decisions accepted into the queue / dropped as counted incidents
    /// (never silent). Drop joins the writer, so accepted == flushed by the
    /// time `stats()` can be observed after shutdown.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.written.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
        )
    }

    fn enqueue(&self, program: String) {
        let Some(tx) = &self.tx else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        match tx.try_send(program) {
            Ok(()) => {
                self.written.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl<P: Predictor> Drop for AuditPredictor<P> {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

impl<P: Predictor> Predictor for AuditPredictor<P> {
    fn predict(&self, req: &SystemOneRequest) -> knot::Result<SystemOneResponse> {
        let t0 = Instant::now();
        let resp = self.inner.predict(req)?;
        self.enqueue(build_program(
            "predict",
            req,
            &resp,
            t0.elapsed().as_millis() as u64,
            self.excerpt_chars,
            checkpoint_of(&self.inner),
        ));
        Ok(resp)
    }

    fn predict_batch(
        &self,
        states: &[serde_json::Value],
        template: SystemOneRequest,
        opts: knot::engine::BatchOpts,
    ) -> knot::Result<Vec<SystemOneResponse>> {
        let t0 = Instant::now();
        // Keep a copy for the audit row before the trait takes ownership.
        let mut combined: SystemOneRequest =
            serde_json::from_value(serde_json::to_value(&template).expect("request serializes"))
                .expect("request roundtrip");
        combined.state = serde_json::json!({ "batch": states });
        let out = self.inner.predict_batch(states, template, opts)?;
        // ADR-006 §4: batch = one parent row for the call (children later —
        // open Q1). The states themselves are hashed, never stored.
        let resp_stub = SystemOneResponse {
            model: format!("batch:{}", out.len()),
            ..empty_response()
        };
        self.enqueue(build_program(
            "predict_batch",
            &combined,
            &resp_stub,
            t0.elapsed().as_millis() as u64,
            self.excerpt_chars,
            checkpoint_of(&self.inner),
        ));
        Ok(out)
    }

    fn loaded(&self) -> Vec<&'static str> {
        self.inner.loaded()
    }

    fn device(&self) -> &'static str {
        self.inner.device()
    }

    fn revision(&self, name: &str) -> Option<String> {
        self.inner.revision(name)
    }

    fn route(
        &self,
        state: &serde_json::Value,
        questions: Option<&std::collections::HashMap<String, serde_json::Value>>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&str>,
    ) -> knot::Result<serde_json::Value> {
        // Routing is not a decision — out of the ledger (ADR-006 seam B).
        self.inner
            .route(state, questions, model, task, lang, lang_guess)
    }
}

fn checkpoint_of<P: Predictor>(inner: &P) -> String {
    inner
        .loaded()
        .first()
        .and_then(|name| inner.revision(name))
        .unwrap_or_else(|| "unknown".to_string())
}

/// `Usage`/`Routing` have no `Default` — build them the way the wire does.
fn empty_response() -> SystemOneResponse {
    SystemOneResponse {
        model: String::new(),
        answers: Default::default(),
        usage: serde_json::from_value(serde_json::json!({
            "input_tokens": 0, "output_tokens": 0
        }))
        .expect("usage json"),
        routing: serde_json::from_value(serde_json::json!({
            "model": "", "repo": "", "reason": ""
        }))
        .expect("routing json"),
        shortlist: None,
    }
}

fn run(db: &mut Database, program: &str) {
    match nql::parse(program) {
        Ok(plan) => {
            if let Err(e) = db.execute(&plan) {
                tracing::error!(error = %e, "audit ledger: write failed");
            }
        }
        Err(e) => tracing::error!(error = %e, "audit ledger: program parse failed"),
    }
}

/// One program = one transaction (topic-6 §3): decision row + provenance edges.
fn build_program(
    kind: &str,
    req: &SystemOneRequest,
    resp: &SystemOneResponse,
    latency_ms: u64,
    excerpt_chars: usize,
    checkpoint: String,
) -> String {
    let state_json = req.state.to_string();
    let state_hash = sha256_hex(&state_json);
    let prompt = serde_json::json!({
        "questions": req.questions,
        "model": req.model,
        "task": req.task,
        "lang": req.lang,
        "lang_guess": req.lang_guess,
    });
    let prompt_hash = sha256_hex(&prompt.to_string());
    let id = format!(
        "d{}",
        &sha256_hex(&format!(
            "{prompt_hash}:{state_hash}:{}:{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            std::process::id()
        ))[..16]
    );
    let answers = serde_json::to_string(&resp.answers).unwrap_or_default();
    let excerpt: String = state_json.chars().take(excerpt_chars).collect();
    let rev = sanitize(&checkpoint);

    let mut prog = format!(
        "INSERT INTO decision:{id} {{ \"kind\": \"{kind}\", \
\"prompt_hash\": \"sha256:{prompt_hash}\", \"state_hash\": \"sha256:{state_hash}\", \
\"state_chars\": {}, \"excerpt\": \"{}\", \"answers\": \"{}\", \
\"model\": \"{}\", \"route_model\": \"{}\", \
\"checkpoint\": \"{rev}\", \"latency_ms\": {latency_ms} }}",
        state_json.chars().count(),
        escape(&excerpt),
        escape(&answers),
        escape(&resp.model),
        escape(&resp.routing.model),
    );
    if let Some(task) = &req.task {
        prog.push_str(&format!(", \"task\": \"{}\"", escape(task)));
    }
    if let Some(lang) = &req.lang {
        prog.push_str(&format!(", \"lang\": \"{}\"", escape(lang)));
    }
    prog.push(';');
    // Endpoint records first: nqlite drops dangling edges silently (no
    // phantom nodes), so state/checkpoint must exist before the RELATEs.
    // Same-id INSERTs are upserts — repeated states stay one row.
    prog.push_str(&format!(
        " INSERT INTO state:s{} {{ \"hash\": \"sha256:{state_hash}\" }};",
        &state_hash[..16]
    ));
    prog.push_str(&format!(
        " INSERT INTO checkpoint:{rev} {{ \"revision\": \"{rev}\" }};"
    ));
    prog.push_str(&format!(
        " RELATE (state:s{}) -> :decided -> (decision:{id});",
        &state_hash[..16]
    ));
    prog.push_str(&format!(
        " RELATE (decision:{id}) -> :used_checkpoint -> (checkpoint:{rev});"
    ));
    prog
}

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Keep NQL string literals parse-safe (quotes/backslashes/control).
fn escape(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => "\\\\".to_string(),
            '"' => "\\\"".to_string(),
            '\n' => "\\n".to_string(),
            '\r' => "\\r".to_string(),
            '\t' => "\\t".to_string(),
            c if (c as u32) < 0x20 => ' '.to_string(),
            c => c.to_string(),
        })
        .collect()
}

/// RecordId-safe segment for checkpoint revisions. Two lexer constraints
/// (both hit in tests): names can't start with a digit (a digit-first hex id
/// lexes as a number) and can't contain `-` (it splits the token: `rev-1` →
/// name `rev` + integer `-1`).
fn sanitize(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned
        .chars()
        .next()
        .is_some_and(|c| !c.is_ascii_alphabetic())
    {
        format!("r{cleaned}")
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_is_counted_not_blocking() {
        let (tx, _rx) = mpsc::sync_channel::<String>(1);
        tx.send("a".into()).unwrap();
        // Channel full → try_send must report Full (the decorator counts it).
        assert!(matches!(
            tx.try_send("b".into()),
            Err(TrySendError::Full(_))
        ));
    }

    #[test]
    fn escaping_keeps_literals_parseable() {
        let nasty = "say \"hi\" \\ done\n\tline";
        let prog = format!("INSERT INTO t:x {{ \"s\": \"{}\" }};", escape(nasty));
        assert!(nql::parse(&prog).is_ok(), "must parse: {prog}");
    }

    #[test]
    fn program_parses_and_carries_provenance() {
        let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "state": {"doc": "I work on the ML team"},
            "questions": {}
        }))
        .expect("request json");
        let resp = SystemOneResponse {
            model: "english".into(),
            ..empty_response()
        };
        let prog = build_program("predict", &req, &resp, 12, 64, "rev-1".into());
        let plan = nql::parse(&prog).expect("program parses");
        // decision row + state row + checkpoint row + two RELATEs, one
        // transaction (endpoints must exist — nqlite drops dangling edges).
        assert_eq!(plan.len(), 5, "program: {prog}");
    }
}
