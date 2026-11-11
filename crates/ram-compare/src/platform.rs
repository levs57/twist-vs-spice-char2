//! Host description and load-average sampling for benchmark reports.
//!
//! Everything here uses the standard library, the C library's `getloadavg`
//! (present in both macOS libSystem and glibc) and short-lived `sysctl`/`uname`
//! calls. Nothing in this module runs inside a timed region.

use serde::Serialize;
use std::process::Command;

/// The 1/5/15-minute load averages, as reported by `getloadavg(3)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LoadAverage {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
}

#[cfg(unix)]
unsafe extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}

/// Current load average, or `None` where the platform does not provide one.
pub fn load_average() -> Option<LoadAverage> {
    #[cfg(unix)]
    {
        let mut values = [0f64; 3];
        // SAFETY: `values` has room for the three requested doubles.
        let filled = unsafe { getloadavg(values.as_mut_ptr(), 3) };
        (filled == 3).then_some(LoadAverage {
            one: values[0],
            five: values[1],
            fifteen: values[2],
        })
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Load averages taken immediately before and after one measured interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LoadWindow {
    pub start: Option<LoadAverage>,
    pub end: Option<LoadAverage>,
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn cpu_brand() -> Option<String> {
    if cfg!(target_os = "macos") {
        return command("sysctl", &["-n", "machdep.cpu.brand_string"]);
    }
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    // x86 reports `model name`; many aarch64 kernels report only the
    // implementer/part codes, which we keep verbatim.
    for key in ["model name", "Hardware", "CPU part"] {
        if let Some(line) = info.lines().find(|l| l.starts_with(key)) {
            return line.split_once(':').map(|(_, v)| v.trim().to_owned());
        }
    }
    None
}

fn sysctl_usize(name: &str) -> Option<usize> {
    command("sysctl", &["-n", name])?.parse().ok()
}

fn cores() -> serde_json::Value {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .ok();
    if cfg!(target_os = "macos") {
        serde_json::json!({
            "logical": logical,
            "physical": sysctl_usize("hw.physicalcpu"),
            // Apple silicon: perflevel0 is the performance cluster.
            "performance": sysctl_usize("hw.perflevel0.physicalcpu"),
            "efficiency": sysctl_usize("hw.perflevel1.physicalcpu"),
        })
    } else {
        serde_json::json!({ "logical": logical })
    }
}

fn os() -> serde_json::Value {
    let product = if cfg!(target_os = "macos") {
        command("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}"))
    } else {
        std::fs::read_to_string("/etc/os-release").ok().and_then(|text| {
            text.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                .map(|v| v.trim_matches('"').to_owned())
        })
    };
    serde_json::json!({
        "family": std::env::consts::OS,
        "release": product,
        "kernel": command("uname", &["-srm"]),
    })
}

/// Compile-time target features (what the kernels were built for) and, where
/// the standard library can tell, run-time support on this host.
fn native_features() -> serde_json::Value {
    let compiled = serde_json::json!({
        "avx2": cfg!(target_feature = "avx2"),
        "pclmulqdq": cfg!(target_feature = "pclmulqdq"),
        "vpclmulqdq": cfg!(target_feature = "vpclmulqdq"),
        "sha": cfg!(target_feature = "sha"),
        "avx512f": cfg!(target_feature = "avx512f"),
        "neon": cfg!(target_feature = "neon"),
        // On aarch64, Rust's `aes` feature includes PMULL/PMULL2.
        "aes": cfg!(target_feature = "aes"),
        "sha2": cfg!(target_feature = "sha2"),
        "sha3": cfg!(target_feature = "sha3"),
    });
    #[cfg(target_arch = "aarch64")]
    let detected = serde_json::json!({
        "neon": std::arch::is_aarch64_feature_detected!("neon"),
        "aes": std::arch::is_aarch64_feature_detected!("aes"),
        "pmull": std::arch::is_aarch64_feature_detected!("pmull"),
        "sha2": std::arch::is_aarch64_feature_detected!("sha2"),
        "sha3": std::arch::is_aarch64_feature_detected!("sha3"),
    });
    #[cfg(target_arch = "x86_64")]
    let detected = serde_json::json!({
        "avx2": std::arch::is_x86_feature_detected!("avx2"),
        "pclmulqdq": std::arch::is_x86_feature_detected!("pclmulqdq"),
        "vpclmulqdq": std::arch::is_x86_feature_detected!("vpclmulqdq"),
        "sha": std::arch::is_x86_feature_detected!("sha"),
        "avx512f": std::arch::is_x86_feature_detected!("avx512f"),
    });
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let detected = serde_json::Value::Null;
    serde_json::json!({ "compiled": compiled, "detected": detected })
}

/// Static description of the host and build, recorded once per report.
pub fn environment(executable_sha256: String, sources_sha256: String) -> serde_json::Value {
    serde_json::json!({
        "architecture": std::env::consts::ARCH,
        "cpu_brand": cpu_brand(),
        "cores": cores(),
        "os": os(),
        "rustc": env!("RAM_COMPARE_RUSTC_VERSION"),
        "rustflags": env!("RAM_COMPARE_RUSTFLAGS"),
        "executable_sha256": executable_sha256,
        "ram_compare_sources_sha256": sources_sha256,
        "native_features": native_features(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_average_is_finite_and_nonnegative_on_unix() {
        if cfg!(unix) {
            let load = load_average().expect("getloadavg");
            for v in [load.one, load.five, load.fifteen] {
                assert!(v.is_finite() && v >= 0.0);
            }
        }
    }

    #[test]
    fn environment_records_aarch64_and_x86_features_and_host_identity() {
        let env = environment("ab".into(), "cd".into());
        let compiled = &env["native_features"]["compiled"];
        for key in ["avx2", "pclmulqdq", "neon", "aes", "sha2", "sha3"] {
            assert!(compiled[key].is_boolean(), "{key}");
        }
        assert!(env["rustc"].as_str().unwrap().contains("rustc"));
        assert!(env["cores"]["logical"].as_u64().unwrap() >= 1);
        assert_eq!(env["os"]["family"], std::env::consts::OS);
        if cfg!(target_arch = "aarch64") {
            assert!(env["native_features"]["detected"]["pmull"].is_boolean());
        }
    }
}
