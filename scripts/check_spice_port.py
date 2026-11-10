#!/usr/bin/env python3
"""Offline provenance checks for Stage 1.4, its GKR port and Boolean ordering."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(hashes):
    result = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        result.update(name.encode() + b"\0")
        result.update(bytes.fromhex(value))
    return result.hexdigest()


def verify_hashes(base, hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid source path: {name}")
        if hashlib.sha256((base / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash differs: {base / path}")


def main():
    port = ROOT / "vendor/lean-grandproduct"
    metadata = json.loads((port / "UPSTREAM.json").read_text())
    if metadata["revision"] != "e5d75f858e91161921c0afe3d9f56300e9a3a03c":
        raise ValueError("unexpected leanVM revision")
    verify_hashes(port / "reference", metadata["source_sha256"])
    adapter = json.loads((ROOT / "audit/stage14-implementation.json").read_text())
    verify_hashes(ROOT, adapter["files_sha256"])
    if fingerprint(adapter["files_sha256"]) != adapter["source_sha256"]:
        raise ValueError("Stage 1.4 aggregate fingerprint differs")
    om = json.loads((ROOT / "vendor/open-magnetar/UPSTREAM.json").read_text())
    if adapter["open_magnetar_patch_sha256"] != om["local_patch_sha256"]:
        raise ValueError("Stage 1.4 / open-magnetar fingerprints differ")
    flock_root = ROOT / "vendor/flock-boolean"
    flock = json.loads((flock_root / "UPSTREAM.json").read_text())
    if flock["revision"] != "daa0cf426563b01f9e744f6265ebae37c97afcb7":
        raise ValueError("unexpected Flock revision")
    flock_hashes = {**flock["source_sha256"], **flock["local_sha256"]}
    verify_hashes(flock_root, flock_hashes)
    if fingerprint(flock_hashes) != adapter["flock_boolean_source_sha256"]:
        raise ValueError("Stage 1.4 / Flock fingerprints differ")
    expected_crates = {name for name in flock["source_sha256"] if name.startswith("crates/")}
    actual_crates = {p.relative_to(flock_root).as_posix() for p in (flock_root / "crates").rglob("*") if p.is_file()}
    if expected_crates != actual_crates:
        raise ValueError("Flock package source set differs")
    print(f"Verified {len(metadata['source_sha256'])} pinned leanVM donor files")
    print(f"Verified {len(flock['source_sha256'])} byte-identical Flock donor files")
    print(f"Verified {len(adapter['files_sha256'])} Stage 1.4 implementation files")
    print(f"Stage 1.4 source: {adapter['source_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
