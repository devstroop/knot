//! Checkpoint directory resolution (SPEC §9).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use oio::router::DEFAULT_MODELS;

/// Which source produced the checkpoint directories.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelSource {
    /// `OIO_MODELS=name=/path[,...]`.
    EnvModels,
    /// `OIO_MODEL_DIR=/path` (single `english` checkpoint).
    EnvModelDir,
    /// The cache root, scanned for known checkpoint names.
    Cache(PathBuf),
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSource::EnvModels => write!(f, "OIO_MODELS"),
            ModelSource::EnvModelDir => write!(f, "OIO_MODEL_DIR"),
            ModelSource::Cache(root) => write!(f, "cache {}", root.display()),
        }
    }
}

/// Resolve checkpoint directories, first source that yields at least one
/// directory wins (SPEC §9): `models` (`OIO_MODELS` value), then `model_dir`
/// (`OIO_MODEL_DIR` value), then a scan of `cache_root` for directories named
/// in [`DEFAULT_MODELS`]. Nothing resolves → error naming every source.
pub fn resolve_model_dirs(
    models: Option<&str>,
    model_dir: Option<&str>,
    cache_root: Option<&Path>,
) -> Result<(ModelSource, Vec<(String, PathBuf)>)> {
    if let Some(spec) = models {
        let mut out = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let Some((name, path)) = part.split_once('=') else {
                bail!("OIO_MODELS entry {part:?} must be name=/path");
            };
            let name = name.trim();
            let path = path.trim();
            if name.is_empty() || path.is_empty() {
                bail!("OIO_MODELS entry {part:?} must be name=/path");
            }
            out.push((name.to_string(), PathBuf::from(path)));
        }
        if !out.is_empty() {
            return Ok((ModelSource::EnvModels, out));
        }
    }
    if let Some(dir) = model_dir {
        let dir = dir.trim();
        if !dir.is_empty() {
            return Ok((
                ModelSource::EnvModelDir,
                vec![("english".to_string(), PathBuf::from(dir))],
            ));
        }
    }
    if let Some(root) = cache_root {
        let mut out = Vec::new();
        for &(name, _, _) in DEFAULT_MODELS {
            let dir = root.join(name);
            if dir.is_dir() {
                out.push((name.to_string(), dir));
            }
        }
        if !out.is_empty() {
            return Ok((ModelSource::Cache(root.to_path_buf()), out));
        }
    }
    let hint = match cache_root {
        Some(root) => format!("or populate the cache ({})", root.display()),
        None => "or populate the cache".to_string(),
    };
    bail!("no checkpoints: set OIO_MODELS=name=/path[,...], OIO_MODEL_DIR=/path, {hint}");
}

/// `$OIO_CACHE_DIR`, else `$XDG_CACHE_HOME/oio`, else `~/.cache/oio`
/// (SPEC §9, ADR-002).
pub fn default_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("OIO_CACHE_DIR") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        let dir = dir.trim();
        if !dir.is_empty() {
            return PathBuf::from(dir).join("oio");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("oio");
    }
    PathBuf::from(".cache").join("oio")
}

#[cfg(test)]
mod tests {
    use super::{ModelSource, default_cache_root, resolve_model_dirs};
    use std::path::{Path, PathBuf};

    fn tmp(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("oio-model-dirs-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn names(dirs: &[(String, PathBuf)]) -> Vec<&str> {
        dirs.iter().map(|(n, _)| n.as_str()).collect()
    }

    #[test]
    fn models_env_wins() {
        let cache = tmp("models-wins");
        let (source, dirs) =
            resolve_model_dirs(Some("a=/x, b=/y"), Some("/z"), Some(&cache)).unwrap();
        assert_eq!(source, ModelSource::EnvModels);
        assert_eq!(names(&dirs), ["a", "b"]);
        assert_eq!(dirs[0].1, PathBuf::from("/x"));
    }

    #[test]
    fn malformed_models_entry_errors() {
        let err = resolve_model_dirs(Some("nopath"), None, None).unwrap_err();
        assert!(err.to_string().contains("name=/path"), "{err}");
        let err = resolve_model_dirs(Some("=/x"), None, None).unwrap_err();
        assert!(err.to_string().contains("name=/path"), "{err}");
    }

    #[test]
    fn empty_sources_fall_through() {
        let cache = tmp("fall-through");
        std::fs::create_dir_all(cache.join("multilingual")).unwrap();
        let (source, dirs) = resolve_model_dirs(Some(""), Some("  "), Some(&cache)).unwrap();
        assert_eq!(source, ModelSource::Cache(cache.clone()));
        assert_eq!(names(&dirs), ["multilingual"]);
    }

    #[test]
    fn model_dir_shorthand_is_english() {
        let (source, dirs) = resolve_model_dirs(None, Some("/ckpt"), None).unwrap();
        assert_eq!(source, ModelSource::EnvModelDir);
        assert_eq!(names(&dirs), ["english"]);
        assert_eq!(dirs[0].1, PathBuf::from("/ckpt"));
    }

    #[test]
    fn cache_picks_known_names_in_table_order() {
        let cache = tmp("known-names");
        for d in ["english", "multilingual", "junk"] {
            std::fs::create_dir_all(cache.join(d)).unwrap();
        }
        let (source, dirs) = resolve_model_dirs(None, None, Some(&cache)).unwrap();
        assert_eq!(source, ModelSource::Cache(cache.clone()));
        assert_eq!(names(&dirs), ["english", "multilingual"]);
    }

    #[test]
    fn nothing_resolves_errors_with_every_source() {
        let cache = tmp("nothing");
        let err = resolve_model_dirs(None, None, Some(&cache)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("OIO_MODELS"), "{msg}");
        assert!(msg.contains("OIO_MODEL_DIR"), "{msg}");
        assert!(msg.contains(&cache.display().to_string()), "{msg}");

        let err = resolve_model_dirs(None, None, None).unwrap_err();
        assert!(err.to_string().contains("OIO_MODELS"), "{err}");
    }

    #[test]
    fn cache_root_env_order() {
        // The helper is env-dependent; only assert the shape it guarantees.
        let root: PathBuf = default_cache_root();
        assert!(!root.as_os_str().is_empty());
        let p = Path::new(&root);
        assert!(p.is_absolute() || p.starts_with(".cache"));
    }
}
