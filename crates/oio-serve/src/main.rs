use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use oio::engine::Engine;
use oio::router::{Router, normalise_name};
use oio_serve::{ServeConfig, build_app};

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => matches!(
            v.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.trim().parse().unwrap_or(default),
        Err(_) => default,
    }
}

/// `OIO_MODELS=name=/path,name2=/path2`, or `OIO_MODEL_DIR` as shorthand for a
/// single `english` checkpoint dir.
fn model_dirs() -> Result<Vec<(String, PathBuf)>> {
    if let Ok(spec) = std::env::var("OIO_MODELS") {
        let mut out = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let Some((name, path)) = part.split_once('=') else {
                bail!("OIO_MODELS entry {part:?} must be name=/path");
            };
            out.push((name.trim().to_string(), PathBuf::from(path.trim())));
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    if let Ok(dir) = std::env::var("OIO_MODEL_DIR") {
        return Ok(vec![("english".into(), PathBuf::from(dir))]);
    }
    bail!("set OIO_MODELS=name=/path[,...] or OIO_MODEL_DIR")
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let dirs = model_dirs()?;
    let mut router = Router::new();
    if let Ok(default) = std::env::var("OIO_DEFAULT_MODEL") {
        router.default = normalise_name(&default)
            .with_context(|| format!("invalid OIO_DEFAULT_MODEL {default:?}"))?;
    }
    router.max_loaded = env_usize("OIO_MAX_LOADED", router.max_loaded);
    router.auto_task_detection = env_bool("OIO_AUTO_TASK", false);

    let dirs: Vec<(&'static str, &std::path::Path)> = dirs
        .iter()
        .map(|(n, p)| {
            (
                Box::leak(n.clone().into_boxed_str()) as &'static str,
                p.as_path(),
            )
        })
        .collect();
    let engine = match std::env::var("OIO_RUNTIME").as_deref() {
        Ok("candle") => {
            tracing::info!("runtime: candle (native CPU)");
            Engine::load_candle(router, &dirs)?
        }
        _ => {
            tracing::info!("runtime: onnx");
            Engine::load(router, &dirs)?
        }
    };

    let config = ServeConfig {
        api_key: std::env::var("OIO_API_KEY").ok().filter(|k| !k.is_empty()),
        max_concurrent: env_usize("OIO_MAX_CONCURRENT", 16),
        max_token_budget: env_usize("OIO_MAX_TOKEN_BUDGET", 8192),
    };

    let engine: Arc<dyn oio_serve::Predictor> = Arc::new(engine);
    if std::env::args().any(|a| a == "--mcp") {
        tracing::info!("oio-mcp stdio server");
        oio_serve::mcp::run_stdio(engine, tokio::io::stdin(), tokio::io::stdout()).await?;
        return Ok(());
    }

    let app = build_app(engine, config);
    let host = std::env::var("OIO_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("OIO_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}");
    tracing::info!(%addr, "oio-serve listening");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
