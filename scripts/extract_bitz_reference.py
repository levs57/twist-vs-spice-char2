#!/usr/bin/env python3
"""Extract or verify the pinned local BitZ LogUp source subset, without building it.

Extraction reads a clean local checkout. --check needs only the extracted copy
and this repository's manifest, and performs no Git, network, or build commands.
All writes are confined to this workspace. The donor checkout is never modified.
"""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REVISION = "cdfb9b73b5857f4dc83761866e475186c6a75ab3"
MANIFEST = ROOT / "audit/bitz-reference.json"
DEFAULT_SOURCE = Path("/home/user/Programming/flock/BitZ")
COMPLETE_SOURCE_CRATES = {"gkr", "poly", "field", "transcript", "common", "prover", "verifier"}
EXTRA_PATHS = {
    "Cargo.toml", "Cargo.lock", "LICENSE", "README.md", "rust-toolchain.toml",
    "crates/gkr/proptest-regressions/lib.txt",
    "crates/tests/Cargo.toml", "crates/tests/src/lib.rs",
    "crates/tests/tests/prove.rs",
    "crates/tests/examples/logup_cut_compare.rs",
    "crates/tests/examples/support/mod.rs",
}


def git(source, *arguments):
    return subprocess.check_output(["git", "-C", str(source), *arguments])


def selected(name):
    if name in EXTRA_PATHS:
        return True
    parts = Path(name).parts
    return (
        len(parts) >= 3
        and parts[0] == "crates"
        and parts[1] in COMPLETE_SOURCE_CRATES
        and (parts[2] == "Cargo.toml" or parts[2] in {"src", "tests"})
    )


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid reference path: {name}")
    return path


def file_record(data):
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha": hashlib.sha1(blob).hexdigest(),
    }


def verify(data, expected, name):
    if file_record(data) != expected:
        raise ValueError(f"pinned BitZ source differs: {name}")


def file_manifest_hash(files):
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def inspect_source(source):
    head = git(source, "rev-parse", "HEAD").decode().strip()
    if head != REVISION:
        raise ValueError(f"donor HEAD is {head}, expected {REVISION}")
    status = git(source, "status", "--porcelain=v1", "--untracked-files=all").decode()
    if status:
        raise ValueError("donor checkout has tracked or untracked changes; extraction requires a clean source")
    return head, status


def new_manifest(source, head, status):
    names = git(source, "ls-tree", "-r", "--name-only", REVISION).decode().splitlines()
    names = sorted(name for name in names if selected(name))
    if not EXTRA_PATHS.issubset(names):
        raise ValueError("the selected source revision lacks an expected reference file")
    files = {}
    for name in names:
        data = (source / safe_relative(name)).read_bytes()
        committed = git(source, "show", f"{REVISION}:{name}")
        if data != committed:
            raise ValueError(f"working source does not match its pinned Git object: {name}")
        files[name] = file_record(data)
    return {
        "repository": "https://github.com/worldfnd/BitZ",
        "revision": REVISION,
        "commit_date": git(source, "show", "-s", "--format=%cI", REVISION).decode().strip(),
        "source_date": date.today().isoformat(),
        "source_checkout": str(source),
        "source_head": head,
        "source_clean": not bool(status),
        "source_status_porcelain": status.splitlines(),
        "license": "MIT",
        "license_notice": "LICENSE",
        "copyright": "Copyright (c) 2026 World Foundation",
        "workspace_declared_license": "MIT OR Apache-2.0",
        "license_note": "The retained upstream LICENSE grants MIT permission; the subset is reused under that grant.",
        "selection_note": (
            "Exact selected local files only: optimized GKR/LogUp, polynomial/field/transcript/common "
            "sources and tests, prover/verifier integration, selected end-to-end tests, and Cargo metadata. "
            "No donor edits, history, build artifacts, full PoC copy, or external dependency sources. "
            "This is a reference subset, not a complete standalone donor workspace."
        ),
        "files": files,
        "file_manifest_sha256": file_manifest_hash(files),
        "source_bytes": sum(record["bytes"] for record in files.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="clean local worldfnd/BitZ checkout at the pinned revision")
    parser.add_argument("--destination", type=Path, default=ROOT / "sources/bitz-reference")
    parser.add_argument("--check", action="store_true",
                        help="verify the extracted copy offline, without reading the donor")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not args.check and not destination.is_relative_to(ROOT):
        parser.error("extraction destination must be inside this writable workspace")
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text())
    elif args.check:
        raise SystemExit(f"missing manifest: {MANIFEST}")
    else:
        source = args.source.resolve()
        head, status = inspect_source(source)
        manifest = new_manifest(source, head, status)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["revision"] != REVISION:
        raise ValueError("unexpected reference revision in manifest")
    if file_manifest_hash(manifest["files"]) != manifest["file_manifest_sha256"]:
        raise ValueError("source file-manifest hash differs")
    if not args.check:
        source = args.source.resolve()
        inspect_source(source)
    for name, expected in manifest["files"].items():
        target = destination / safe_relative(name)
        if args.check:
            verify(target.read_bytes(), expected, name)
        else:
            data = (source / safe_relative(name)).read_bytes()
            verify(data, expected, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    marker = destination / "REFERENCE.json"
    if args.check:
        if json.loads(marker.read_text()) != manifest:
            raise ValueError("extracted REFERENCE.json differs from the audit manifest")
    else:
        marker.write_text(json.dumps(manifest, indent=2) + "\n")
        inspect_source(source)
    print(f'Verified {len(manifest["files"])} BitZ files ({manifest["source_bytes"]:,} bytes) at {REVISION}')


if __name__ == "__main__":
    main()
