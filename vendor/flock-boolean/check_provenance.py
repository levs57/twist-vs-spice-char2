#!/usr/bin/env python3
"""Validate this extraction without the original checkout or network access."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "UPSTREAM.json").read_text())
    expected = {**metadata["source_sha256"], **metadata["local_sha256"]}
    for name, digest in expected.items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"Hash mismatch: {name}: {actual} != {digest}")
    actual_crate_files = {
        path.relative_to(root).as_posix()
        for path in (root / "crates").rglob("*")
        if path.is_file()
    }
    expected_crate_files = {
        name for name in metadata["source_sha256"] if name.startswith("crates/")
    }
    if actual_crate_files != expected_crate_files:
        raise SystemExit("Copied package file set differs from the pinned manifest")
    print(
        f"Validated {len(metadata['source_sha256'])} byte-identical upstream files "
        f"and {len(metadata['local_sha256'])} local build files at "
        f"{metadata['revision']}."
    )


if __name__ == "__main__":
    main()
