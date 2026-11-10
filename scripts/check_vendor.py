#!/usr/bin/env python3
"""Check pinned upstream hashes and the recorded instrumentation/optimization patch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "vendor" / "open-magnetar"
metadata = json.loads((root / "UPSTREAM.json").read_text())
allowed = set(metadata["permitted_modified_files"])
modified = []
errors = []
if not allowed.issubset(metadata["baseline_sha256"]):
    errors.append("Allowed modifications must name pinned upstream files")
for name, expected in metadata["baseline_sha256"].items():
    path = root / name
    if not path.is_file():
        errors.append(f"Missing upstream file: {name}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        if name in allowed:
            modified.append(name)
        else:
            errors.append(f"Unexpected upstream modification: {name}")
expected_files = set(metadata["baseline_sha256"]) | set(metadata["added_files"])
for name in metadata["added_files"]:
    if not (root / name).is_file():
        errors.append(f"Missing added file: {name}")
local_hashes = metadata.get("local_sha256", {})
if local_hashes and set(local_hashes) != allowed | set(metadata["added_files"]):
    errors.append("Local hashes must cover every permitted modification and added file")
for name, expected in local_hashes.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(f"Local patch hash mismatch: {name}")
if local_hashes:
    digest = hashlib.sha256()
    for name in sorted(local_hashes):
        digest.update(name.encode() + b"\0")
        digest.update(bytes.fromhex(local_hashes[name]))
    if digest.hexdigest() != metadata.get("local_patch_sha256"):
        errors.append("Local patch aggregate hash mismatch")
for path in (root / "crates").rglob("*"):
    if path.is_file() and str(path.relative_to(root)) not in expected_files:
        errors.append(f"Unexpected added file: {path.relative_to(root)}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(metadata['baseline_sha256'])} source/vector/license files at {metadata['revision']}")
for name in modified:
    print(f"Recorded local change: {name}")
if local_hashes:
    print(f"Local patch: {metadata['local_patch_sha256']}")
