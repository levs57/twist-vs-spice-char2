#!/usr/bin/env python3
"""Verify current typed-Spice adapter and unchanged donor sources offline."""

import json

from check_spice_port import ROOT, fingerprint, verify_hashes


def main():
    metadata = json.loads((ROOT / "audit/stage15-implementation.json").read_text())
    verify_hashes(ROOT, metadata["files_sha256"])
    if fingerprint(metadata["files_sha256"]) != metadata["source_sha256"]:
        raise ValueError("Stage 1.5 aggregate source fingerprint differs")
    om_root = ROOT / "vendor/open-magnetar"
    om = json.loads((om_root / "UPSTREAM.json").read_text())
    verify_hashes(om_root, {**om["baseline_sha256"], **om["local_sha256"]})
    if metadata["open_magnetar_patch_sha256"] != om["local_patch_sha256"]:
        raise ValueError("Stage 1.5 / OM fingerprints differ")
    flock_root = ROOT / "vendor/flock-boolean"
    flock = json.loads((flock_root / "UPSTREAM.json").read_text())
    hashes = {**flock["source_sha256"], **flock["local_sha256"]}
    verify_hashes(flock_root, hashes)
    if fingerprint(hashes) != metadata["flock_boolean_source_sha256"]:
        raise ValueError("Stage 1.5 / Flock fingerprints differ")
    print(f"Verified {len(metadata['files_sha256'])} Stage 1.5 source files")
    print(f"Stage 1.5 source: {metadata['source_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
