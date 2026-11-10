#!/usr/bin/env python3
"""Compare optimized RAM proof values against separately compiled pinned kernels.

Requires the Stage 1.1 sources/open-magnetar checkout and cached Cargo dependencies.
By default the reference uses the current RAM adapter in FlatLegacy mode,
original upstream prover kernels, and API-only dispatch/profiler shims. An
extracted reference also compares GKR against that saved backend. The new Flat
protocol is deliberately excluded: its messages differ by design. Generated
sources remain in /tmp; the original reference trees are never modified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def install_flat_api_shims(reference):
    """Bridge current adapter names without changing any original proof math."""
    onehot = reference / "crates/om-memcheck/src/onehot.rs"
    with onehot.open("a") as output:
        output.write('''

// Reference-only API shim: only FlatLegacy reaches the pinned prover/verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtualizationMode {
    #[default]
    FlatLegacy,
    GkrTree,
}
''')
    twist = reference / "crates/om-memcheck/src/twist.rs"
    with twist.open("a") as output:
        output.write('''

// Reference-only dispatch shims. The original functions above are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn twist_prove_with_virtualization(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: crate::onehot::VirtualizationMode,
) -> (TwistProof, TwistOutput) {
    assert_eq!(mode, crate::onehot::VirtualizationMode::FlatLegacy);
    twist_prove(params, m0, wit, claims, transcript, label, final_mem, inc_extra_flag)
}

#[allow(clippy::too_many_arguments)]
pub fn twist_verify_with_virtualization(
    params: &TwistParams,
    m0_eval: &dyn Fn(&[K]) -> K,
    claims: &TwistClaims,
    proof: &TwistProof,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: crate::onehot::VirtualizationMode,
) -> Result<TwistOutput, VerificationError> {
    assert_eq!(mode, crate::onehot::VirtualizationMode::FlatLegacy);
    twist_verify(params, m0_eval, claims, proof, transcript, label, final_mem, inc_extra_flag)
}
''')


def rename_extracted_flat_api(reference):
    """Rename the saved Flat API, retaining every original dispatch body."""
    directory = reference / "crates/om-memcheck/src"
    onehot = directory / "onehot.rs"
    text = onehot.read_text()
    text, count = re.subn(
        r"(pub enum VirtualizationMode\s*\{\s*#\[default\]\s*)Flat,",
        r"\1FlatLegacy,", text,
    )
    if count != 1:
        raise SystemExit("Extracted reference must contain the pre-change Flat/GkrTree enum")
    onehot.write_text(text)
    for path in directory.rglob("*.rs"):
        text = path.read_text()
        renamed = re.sub(r"\bVirtualizationMode::Flat\b", "VirtualizationMode::FlatLegacy", text)
        if renamed != text:
            path.write_text(renamed)


def install_shared_address_prover_shim(reference):
    """Ignore a prover-only source-sharing hint in the unchanged reference math."""
    twist = reference / "crates/om-memcheck/src/twist.rs"
    with twist.open("a") as output:
        output.write('''

// Reference-only source-sharing API shim: retain the original prover math.
#[allow(clippy::too_many_arguments)]
pub fn twist_prove_with_virtualization_and_shared_address(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: crate::onehot::VirtualizationMode,
    _shared_address: bool,
) -> (TwistProof, TwistOutput) {
    twist_prove_with_virtualization(
        params, m0, wit, claims, transcript, label, final_mem, inc_extra_flag, mode,
    )
}
''')


def reference_adapter_source(text):
    """Remove only adapter branches for the unsupported new Flat protocol.

    FlatLegacy retains the original v2 domain/config bytes; GkrTree retains v3.
    The copied reference kernels have neither the new functions nor proof field.
    Fail closed if later adapter changes need a new compatibility shim.
    """
    substitutions = [
        (r"(?m)^\s*VirtualizationMode::Flat => b\"ram-argument/v4\",\n", ""),
        (r"(?m)^\s*VirtualizationMode::Flat => bytes\.extend_from_slice\(&2u64\.to_le_bytes\(\)\),\n", ""),
        (r"\bvirtualization: VirtualizationMode::Flat,", "virtualization: VirtualizationMode::FlatLegacy,"),
    ]
    for pattern, replacement in substitutions:
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise SystemExit(f"Current adapter compatibility shim matched {count} times: {pattern}")
    if re.search(r"\bVirtualizationMode::Flat\b|\bvirt_flat\b", text):
        raise SystemExit("Reference adapter still refers to the unsupported new Flat protocol")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "sources/open-magnetar")
    parser.add_argument("--extracted-reference", type=Path,
                        help="Previously extracted vendor directory (with UPSTREAM.json); compare FlatLegacy and GKR exactly")
    args = parser.parse_args()
    vendor = ROOT / "vendor/open-magnetar"
    metadata = json.loads((vendor / "UPSTREAM.json").read_text())
    source = (args.extracted_reference or args.reference).resolve()
    reference_metadata = metadata
    expected_hashes = metadata["baseline_sha256"]
    if args.extracted_reference:
        reference_metadata = json.loads((source / "UPSTREAM.json").read_text())
        expected_hashes = {**reference_metadata["baseline_sha256"],
                           **reference_metadata["local_sha256"]}
    for name, expected in expected_hashes.items():
        if not name.startswith("crates/"):
            continue
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Reference differs from pinned upstream: {path}")

    with tempfile.TemporaryDirectory(prefix="memarg-sumcheck-equivalence-") as temporary:
        work = Path(temporary)
        reference = work / "reference"
        reference.mkdir()
        shutil.copy2(vendor / "Cargo.toml", reference / "Cargo.toml")
        for package in (vendor / "crates").iterdir():
            if package.is_dir():
                shutil.copytree(source / "crates" / package.name, reference / "crates" / package.name)
        # Adapter timing has no cryptographic effect. Keep this shim available
        # without borrowing any of the modified sumcheck or memory kernels.
        if not args.extracted_reference:
            for name in ["Cargo.toml", "src/lib.rs", "src/profiling.rs"]:
                target = reference / "crates/om-transcript" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vendor / "crates/om-transcript" / name, target)
            install_flat_api_shims(reference)
        else:
            rename_extracted_flat_api(reference)
        install_shared_address_prover_shim(reference)

        for manifest in reference.rglob("Cargo.toml"):
            text = manifest.read_text()
            text = re.sub(r'(?m)^name = "(om-[^"]+)"$', r'name = "\1-reference"', text)
            text = re.sub(r'(?m)^(om-[\w-]+) = \{ path = ',
                          lambda match: f'{match[1]} = {{ package = "{match[1]}-reference", path = ', text)
            manifest.write_text(text)

        adapter = work / "adapter"
        shutil.copytree(ROOT / "crates/ram-argument/src", adapter / "src")
        adapter_source = adapter / "src/lib.rs"
        adapter_source.write_text(reference_adapter_source(adapter_source.read_text()))
        manifest = (ROOT / "crates/ram-argument/Cargo.toml").read_text()
        # Integration-test payload helpers are not needed by the reference
        # library and their relative paths do not belong in this temporary tree.
        manifest = re.sub(r"(?ms)^\[dev-dependencies\]\n.*?(?=^\[|\Z)", "", manifest)
        manifest = manifest.replace('name = "ram-argument"', 'name = "ram-argument-reference"')
        manifest = re.sub(r'(?m)^(om-[\w-]+) = \{ path = "[^"]+" \}',
                          lambda match: f'{match[1]} = {{ package = "{match[1]}-reference", path = "{reference / "crates" / match[1]}" }}', manifest)
        (adapter / "Cargo.toml").write_text(manifest)
        probe = work / "probe"
        (probe / "src").mkdir(parents=True)
        (probe / "Cargo.toml").write_text(f'''[package]
name = "memarg-optimization-equivalence"
version = "0.0.0"
edition = "2024"
[features]
extracted-reference = []
[dependencies]
ram-argument = {{ path = "{ROOT / 'crates/ram-argument'}" }}
ram-argument-reference = {{ path = "{adapter}" }}
rayon = "1"
[profile.release]
lto = "thin"
codegen-units = 1
''')
        shutil.copy2(ROOT / "audit/optimization-equivalence.rs", probe / "src/main.rs")
        environment = os.environ.copy()
        environment.setdefault("RUSTFLAGS", "-C target-cpu=native")
        environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        print(f"Reference kernels: {reference_metadata.get('local_patch_sha256') if args.extracted_reference else metadata['revision']}", flush=True)
        command = ["cargo", "run", "--offline", "--release", "--manifest-path", str(probe / "Cargo.toml")]
        if args.extracted_reference:
            command += ["--features", "extracted-reference"]
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
