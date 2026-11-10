#!/usr/bin/env python3
"""Verify the pinned BitZ reference extraction and the recorded GF128 port.

This is an offline, read-only check. No donor checkout, network, or Rust build
is required. The production fingerprint includes Cargo.toml, LICENSE, and all
src files; Cargo.lock, provenance/docs, and target outputs are excluded.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "vendor/bitz-logup"
REVISION = "cdfb9b73b5857f4dc83761866e475186c6a75ab3"


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid port path: {name}")
    return path


def source_fingerprint(hashes):
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(hashes[name]))
    return digest.hexdigest()


def verify_port(port, metadata, reference):
    if metadata["revision"] != REVISION or reference["revision"] != REVISION:
        raise ValueError("unexpected BitZ revision")
    if metadata["reference_manifest_sha256"] != reference["file_manifest_sha256"]:
        raise ValueError("BitZ reference manifest fingerprint differs")
    if metadata["license"] != "MIT":
        raise ValueError("unexpected BitZ port license")

    expected = metadata["local_sha256"]
    if set(expected) != {"Cargo.toml", "LICENSE"} | {
        p.relative_to(port).as_posix() for p in (port / "src").rglob("*") if p.is_file()
    }:
        raise ValueError("port hashes must cover exactly Cargo.toml, LICENSE, and every src file")
    actual = {}
    for name, digest in expected.items():
        path = port / safe_relative(name)
        if not path.is_file():
            raise ValueError(f"missing BitZ port file: {name}")
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != digest:
            raise ValueError(f"BitZ port source hash differs: {name}")
    fingerprint = source_fingerprint(actual)
    if fingerprint != metadata["local_source_sha256"]:
        raise ValueError("BitZ port aggregate source fingerprint differs")

    # A new build.rs, extra target, or helper source must be recorded too.
    # Only the explicitly excluded metadata/lockfile/build outputs may differ.
    allowed = set(expected) | {"Cargo.lock", "UPSTREAM.json", "LOCAL_CHANGES.md"}
    for directory, dirs, files in os.walk(port):
        if Path(directory) == port:
            dirs[:] = [name for name in dirs if name != "target"]
        for filename in files:
            name = (Path(directory) / filename).relative_to(port).as_posix()
            if name not in allowed:
                raise ValueError(f"unrecorded BitZ port file: {name}")

    donor_root = ROOT / "sources/bitz-reference"
    for donor in metadata["donor_sources"]:
        name = donor["path"]
        if donor["sha256"] != reference["files"][name]["sha256"]:
            raise ValueError(f"donor file hash differs from reference: {name}")
        lines = (donor_root / safe_relative(name)).read_text().splitlines()
        for function in donor["function_ranges"]:
            start, end = function["lines"]
            if not (1 <= start <= end <= len(lines)):
                raise ValueError(f"invalid donor line range: {name}: {start}-{end}")
            if function["symbol"] not in "\n".join(lines[start - 1:end]):
                raise ValueError(f"donor symbol missing from recorded range: {name}: {function['symbol']}")
    if (port / "LICENSE").read_bytes() != (donor_root / "LICENSE").read_bytes():
        raise ValueError("port MIT license differs from the pinned donor notice")
    return fingerprint


def main():
    # Validate the byte-exact extraction before trusting any donor anchors.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/extract_bitz_reference.py"), "--check"],
        check=True,
    )
    metadata = json.loads((PORT / "UPSTREAM.json").read_text())
    reference = json.loads((ROOT / "audit/bitz-reference.json").read_text())
    fingerprint = verify_port(PORT, metadata, reference)
    print(f"Verified {len(metadata['local_sha256'])} BitZ port production/source files")
    print(f"BitZ port source: {fingerprint}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
