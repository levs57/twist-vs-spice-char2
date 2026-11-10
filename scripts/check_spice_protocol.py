#!/usr/bin/env python3
"""Run the audit-only Spice field and protocol counterexamples offline.

Both native and default compiler targets run by default. Only the local om-field
dependency is built; no donor/workspace implementation is modified or fetched.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native", action="store_true", help="only target-cpu=native")
    mode.add_argument("--portable", action="store_true", help="only the default compiler target")
    parser.add_argument("--output", type=Path, default=ROOT / "audit/logs/stage14-protocol-probe.log")
    args = parser.parse_args()
    probe = ROOT / "audit/spice-protocol-probe.rs"
    log = ["Audit-only Spice protocol probe; production code unchanged.\n",
           f"probe_sha256={hashlib.sha256(probe.read_bytes()).hexdigest()}\n"]
    versions = subprocess.run(["rustc", "--version"], text=True, capture_output=True, check=True)
    log.append(versions.stdout)
    status = 0
    with tempfile.TemporaryDirectory(prefix="memarg-spice-protocol-") as temporary:
        manifest = Path(temporary) / "Cargo.toml"
        quote = lambda value: json.dumps(str(value))
        manifest.write_text(f'''[package]
name = "memarg-spice-protocol-probe"
version = "0.0.0"
edition = "2024"
[[bin]]
name = "memarg-spice-protocol-probe"
path = {quote(probe)}
[dependencies]
om-field = {{ path = {quote(ROOT / "vendor/open-magnetar/crates/om-field")} }}
''')
        base = os.environ.copy()
        base.setdefault("CARGO_BUILD_JOBS", "4")
        base.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        lock = subprocess.run(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(manifest)],
                              env=base, text=True, capture_output=True)
        log.extend([lock.stderr, lock.stdout])
        status = lock.returncode
        modes = ["default"] if args.portable else ["native"] if args.native else ["native", "default"]
        if status == 0:
            for target in modes:
                environment = base.copy()
                environment["RUSTFLAGS"] = "-C target-cpu=native" if target == "native" else ""
                command = ["cargo", "run", "--release", "--locked", "--offline", "--manifest-path", str(manifest)]
                result = subprocess.run(command, env=environment, text=True, capture_output=True)
                log.extend([f"\nTarget={target} RUSTFLAGS={environment['RUSTFLAGS']!r}\n",
                            result.stderr, result.stdout])
                if result.returncode:
                    status = result.returncode
                    break
    text = "".join(log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")
    print(f"Log: {args.output}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
