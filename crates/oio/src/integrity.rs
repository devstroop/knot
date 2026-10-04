//! `SHA256SUMS` verification at checkpoint load (SPEC §9, PRD N4).

use std::io::Read;
use std::path::{Component, Path};

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Verify a `SHA256SUMS` manifest (coreutils `sha256sum` format) beside a
/// checkpoint directory. Verify-if-present: a directory without the manifest
/// loads untouched. Every listed file must exist under `dir` and hash-match;
/// absolute paths and `..` components are rejected. Blank lines and `#`
/// comments are skipped.
pub fn verify_sha256sums(dir: &Path) -> Result<()> {
    let text = match std::fs::read_to_string(dir.join("SHA256SUMS")) {
        Ok(text) => text,
        Err(_) => return Ok(()),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((hex, rest)) = line.split_once(char::is_whitespace) else {
            return Err(Error::Model(format!(
                "SHA256SUMS: unparsable line {line:?}"
            )));
        };
        // coreutils: `<hex>␠␠<path>` (text) or `<hex>␠*<path>` (binary).
        let rest = rest.strip_prefix('*').unwrap_or(rest);
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::Model(format!("SHA256SUMS: bad digest for {rest:?}")));
        }
        if rest.is_empty() {
            return Err(Error::Model("SHA256SUMS: empty path".into()));
        }
        let rel = Path::new(rest);
        if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(Error::Model(format!(
                "SHA256SUMS: path escapes checkpoint directory: {rest:?}"
            )));
        }
        let path = dir.join(rel);
        let mut file = std::fs::File::open(&path)
            .map_err(|e| Error::Model(format!("SHA256SUMS: cannot read {rest:?}: {e}")))?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| Error::Model(format!("SHA256SUMS: read {rest:?}: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let got: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if !got.eq_ignore_ascii_case(hex) {
            return Err(Error::Model(format!(
                "SHA256SUMS mismatch for {rest:?}: expected {hex}, computed {got}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verify_sha256sums;
    use std::path::{Path, PathBuf};

    /// sha256("hello")
    const HELLO: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    fn tmp(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("oio-sha256sums-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, name: &str, body: &str, manifest: &str) {
        std::fs::write(dir.join(name), body).unwrap();
        std::fs::write(dir.join("SHA256SUMS"), manifest).unwrap();
    }

    fn check(dir: &Path) -> std::result::Result<(), String> {
        verify_sha256sums(dir).map_err(|e| e.to_string())
    }

    #[test]
    fn absent_manifest_is_ok() {
        let dir = tmp("absent");
        std::fs::write(dir.join("f.bin"), "data").unwrap();
        assert!(check(&dir).is_ok());
    }

    #[test]
    fn valid_manifest_verifies() {
        let dir = tmp("valid");
        write(&dir, "f.txt", "hello", &format!("{HELLO}  f.txt\n"));
        assert!(check(&dir).is_ok());
    }

    #[test]
    fn binary_marker_and_double_space_parse() {
        let dir = tmp("markers");
        write(&dir, "a.txt", "hello", &format!("{HELLO} *a.txt\n"));
        assert!(check(&dir).is_ok());
        write(&dir, "b.txt", "hello", &format!("{HELLO}  b.txt\n"));
        assert!(check(&dir).is_ok());
    }

    #[test]
    fn mismatch_names_the_file() {
        let dir = tmp("mismatch");
        write(&dir, "f.txt", "tampered", &format!("{HELLO}  f.txt\n"));
        let err = check(&dir).unwrap_err();
        assert!(err.contains("SHA256SUMS mismatch"), "{err}");
        assert!(err.contains("f.txt"), "{err}");
    }

    #[test]
    fn missing_listed_file_is_an_error() {
        let dir = tmp("missing");
        std::fs::write(dir.join("SHA256SUMS"), format!("{HELLO}  gone.txt\n")).unwrap();
        let err = check(&dir).unwrap_err();
        assert!(err.contains("gone.txt"), "{err}");
    }

    #[test]
    fn parent_and_absolute_paths_rejected() {
        let dir = tmp("escape");
        std::fs::write(dir.join("SHA256SUMS"), format!("{HELLO}  ../outside.txt\n")).unwrap();
        let err = check(&dir).unwrap_err();
        assert!(err.contains("escapes"), "{err}");

        std::fs::write(dir.join("SHA256SUMS"), format!("{HELLO}  /etc/passwd\n")).unwrap();
        let err = check(&dir).unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    fn garbage_digest_is_an_error() {
        let dir = tmp("garbage");
        std::fs::write(dir.join("SHA256SUMS"), "not-a-hash  f.txt\n").unwrap();
        let err = check(&dir).unwrap_err();
        assert!(err.contains("bad digest"), "{err}");
    }

    #[test]
    fn comments_and_blank_lines_skipped() {
        let dir = tmp("comments");
        write(
            &dir,
            "f.txt",
            "hello",
            &format!("# pinned at 55cf4c4\n\n{HELLO}  f.txt\n"),
        );
        assert!(check(&dir).is_ok());
    }

    #[test]
    fn empty_manifest_is_ok() {
        let dir = tmp("empty");
        std::fs::write(dir.join("SHA256SUMS"), "").unwrap();
        assert!(check(&dir).is_ok());
    }
}
