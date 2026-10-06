use std::sync::Arc;

use anyhow::{Context, Result, bail};

use knot::engine::Engine;
use knot::router::{Router, normalise_name};
use knot_serve::model::{default_cache_root, resolve_model_dirs};
use knot_serve::{ServeConfig, build_app};

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

#[tokio::main]
async fn main() -> Result<()> {
    let mcp_mode = std::env::args().any(|a| a == "--mcp");
    let subscriber = tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    );
    if mcp_mode {
        subscriber.with_writer(std::io::stderr).init();
    } else {
        subscriber.init();
    }

    let (source, dirs) = resolve_model_dirs(
        std::env::var("KNOT_MODELS").as_deref().ok(),
        std::env::var("KNOT_MODEL_DIR").as_deref().ok(),
        Some(&default_cache_root()),
    )?;
    tracing::info!(
        %source,
        models = ?dirs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        "checkpoints resolved"
    );

    let mut router = Router::new();
    if let Ok(default) = std::env::var("KNOT_DEFAULT_MODEL") {
        router.default = normalise_name(&default)
            .with_context(|| format!("invalid KNOT_DEFAULT_MODEL {default:?}"))?;
    } else if !dirs.iter().any(|(n, _)| n == "english") {
        // SPEC §9: english is the fallback, but when it isn't loaded the
        // fallback becomes the first resolved directory (if it's a known name).
        if let Some(first) = dirs.first().and_then(|(n, _)| normalise_name(n).ok()) {
            tracing::info!(model = first, "fallback default: english not loaded");
            router.default = first;
        }
    }
    router.max_loaded = env_usize("KNOT_MAX_LOADED", router.max_loaded);
    router.auto_task_detection = env_bool("KNOT_AUTO_TASK", false);

    let dirs: Vec<(&'static str, &std::path::Path)> = dirs
        .iter()
        .map(|(n, p)| {
            (
                Box::leak(n.clone().into_boxed_str()) as &'static str,
                p.as_path(),
            )
        })
        .collect();
    let device = knot::runtime::Device::parse(std::env::var("KNOT_DEVICE").ok().as_deref())?;
    let engine = match std::env::var("KNOT_RUNTIME").as_deref() {
        Ok("candle") => {
            if device != knot::runtime::Device::Cpu {
                bail!(
                    "KNOT_DEVICE={} requires KNOT_RUNTIME=onnx — the candle runtime \
                     is native CPU (SPEC §10)",
                    device.as_str()
                );
            }
            tracing::info!("runtime: candle (native CPU)");
            Engine::load_candle(router, &dirs)?
        }
        _ => {
            tracing::info!(device = device.as_str(), "runtime: onnx");
            Engine::load_with_device(router, &dirs, device)?
        }
    };

    let config = ServeConfig {
        api_key: std::env::var("KNOT_API_KEY").ok().filter(|k| !k.is_empty()),
        max_concurrent: env_usize("KNOT_MAX_CONCURRENT", 16),
        max_token_budget: env_usize("KNOT_MAX_TOKEN_BUDGET", 8192),
    };

    let engine: Arc<dyn knot_serve::Predictor> = Arc::new(engine);
    if mcp_mode {
        tracing::info!("knot-mcp stdio server");
        knot_serve::mcp::run_stdio(engine, tokio::io::stdin(), tokio::io::stdout()).await?;
        return Ok(());
    }

    let app = build_app(engine, config);
    let host = std::env::var("KNOT_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port: u16 = std::env::var("KNOT_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}");
    tracing::info!(%addr, "knot listening");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
