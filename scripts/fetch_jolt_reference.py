#!/usr/bin/env python3
"""Fetch or verify only the pinned Jolt files selected for the virtualization audit."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def verify(data, expected):
    if len(data) != expected["bytes"]:
        raise ValueError("source length mismatch")
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("SHA256 mismatch")
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    if hashlib.sha1(blob).hexdigest() != expected["git_blob_sha"]:
        raise ValueError("Git blob hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "sources/jolt-reference")
    parser.add_argument("--check", action="store_true", help="verify existing files without network access")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    manifest = json.loads((ROOT / "audit/jolt-reference.json").read_text())
    destination = args.destination.resolve()

    def process(item):
        name, expected = item
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid source path: {name}")
        path = destination / relative
        if path.is_file():
            try:
                verify(path.read_bytes(), expected)
                return
            except ValueError:
                if args.check:
                    raise
        if args.check:
            raise FileNotFoundError(f"missing pinned source: {path}")
        url = f'https://raw.githubusercontent.com/a16z/jolt/{manifest["revision"]}/{name}'
        request = urllib.request.Request(url, headers={"User-Agent": "memarg-virtualization-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        verify(data, expected)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for job in as_completed([pool.submit(process, item) for item in manifest["files"].items()]):
            job.result()
    if not args.check:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "REFERENCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f'Verified {len(manifest["files"])} Jolt files ({manifest["source_bytes"]:,} bytes) at {manifest["revision"]}')


if __name__ == "__main__":
    main()
