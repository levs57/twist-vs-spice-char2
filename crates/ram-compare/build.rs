//! Records the compiler that built the benchmark, so reports name the rustc
//! that produced the measured code rather than whichever rustc is on PATH.

use std::process::Command;

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
}
