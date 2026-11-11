//! Records the compiler that built the benchmark, so reports name the rustc
//! that produced the measured code rather than whichever rustc is on PATH.
//! It also records the git commit and copies the argument crates' sources
//! into OUT_DIR, so a report names the prover code it measured.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Prover sources hashed into every report, relative to the repository root.
const ARGUMENT_SOURCES: &[&str] = &[
    "Cargo.lock",
    "crates/ram-argument/Cargo.toml",
    "crates/ram-argument/src",
    "crates/spice-argument/Cargo.toml",
    "crates/spice-argument/src",
];

fn collect(root: &Path, rel: &Path, out: &mut Vec<PathBuf>) {
    let path = root.join(rel);
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&path)
            .expect("read_dir")
            .map(|e| e.expect("dir entry").file_name())
            .collect();
        entries.sort();
        for name in entries {
            collect(root, &rel.join(name), out);
        }
    } else if path.is_file() {
        out.push(rel.to_path_buf());
    }
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.lines().collect::<Vec<_>>().join("; "))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=RAM_COMPARE_RUSTC_VERSION={version}");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\x1f', " ");
    println!("cargo:rustc-env=RAM_COMPARE_RUSTFLAGS={flags}");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let root = manifest.join("../..");
    let mut files = Vec::new();
    for rel in ARGUMENT_SOURCES {
        collect(&root, Path::new(rel), &mut files);
        println!("cargo:rerun-if-changed={}", root.join(rel).display());
    }
    // Layout: for each file, its path, a NUL, its length as 8 little-endian
    // bytes, then its contents. main.rs hashes each file the same way as
    // its own SOURCES.
    let mut blob = Vec::new();
    for rel in &files {
        let contents = std::fs::read(root.join(rel)).expect("read argument source");
        println!("cargo:rerun-if-changed={}", root.join(rel).display());
        blob.extend_from_slice(rel.to_string_lossy().as_bytes());
        blob.push(0);
        blob.extend_from_slice(&(contents.len() as u64).to_le_bytes());
        blob.extend_from_slice(&contents);
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out.join("argument-sources.bin"), blob).expect("write argument sources");

    // The commit the binary was built from, and whether the tracked tree
    // differed from it. Rerun when HEAD, the branch ref or the index moves.
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(status) => (!status.is_empty()).to_string(),
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=RAM_COMPARE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=RAM_COMPARE_GIT_DIRTY={dirty}");
    let mut watched = vec!["HEAD".to_string(), "index".to_string()];
    if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(head_ref);
    }
    for name in watched {
        if let Some(path) = git(&["rev-parse", "--git-path", &name]) {
            let path = PathBuf::from(path);
            let path = if path.is_absolute() {
                path
            } else {
                manifest.join(path)
            };
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
