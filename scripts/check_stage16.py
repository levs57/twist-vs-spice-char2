#!/usr/bin/env python3
"""Check that audit/stage16-implementation.json matches the current tree.

Unlike the Stage 1.5 check, which only re-hashes the files a manifest
lists, this recomputes the manifest from the tree with
scripts/write_manifest.py and requires the same file set, the same hash for
every file and the same aggregate fingerprint. It also re-verifies the
vendored open-magnetar and Flock sources against their own manifests.
"""

import json

from check_spice_port import fingerprint, verify_hashes
from write_manifest import MANIFEST, ROOT, collect


def main():
    stored = json.loads(MANIFEST.read_text())
    current = collect()
    old, new = stored["files_sha256"], current["files_sha256"]
    problems = [f"missing from manifest: {n}" for n in sorted(set(new) - set(old))]
    problems += [f"listed but absent: {n}" for n in sorted(set(old) - set(new))]
    problems += [f"hash differs: {n}" for n in sorted(set(old) & set(new)) if old[n] != new[n]]
    if fingerprint(old) != stored["source_sha256"]:
        problems.append("aggregate source fingerprint does not match the listed hashes")
    for key in ("implementation", "open_magnetar_patch_sha256", "flock_boolean_source_sha256"):
        if stored.get(key) != current[key]:
            problems.append(f"{key} differs")
    om_root = ROOT / "vendor/open-magnetar"
    om = json.loads((om_root / "UPSTREAM.json").read_text())
    verify_hashes(om_root, {**om["baseline_sha256"], **om["local_sha256"]})
    flock_root = ROOT / "vendor/flock-boolean"
    flock = json.loads((flock_root / "UPSTREAM.json").read_text())
    verify_hashes(flock_root, {**flock["source_sha256"], **flock["local_sha256"]})
    if problems:
        raise ValueError(
            "Stage 1.6 manifest does not match the tree; run scripts/write_manifest.py:\n  "
            + "\n  ".join(problems)
        )
    print(f"Verified {len(old)} Stage 1.6 source files against the current tree")
    print(f"Stage 1.6 source: {stored['source_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
