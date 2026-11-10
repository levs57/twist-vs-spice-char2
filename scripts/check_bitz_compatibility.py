#!/usr/bin/env python3
"""Run the small BitZ/OM field and variable-order conformance oracle offline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable", action="store_true", help="omit target-cpu=native")
    parser.add_argument("--output", type=Path, help="captured compiler/probe log")
    args = parser.parse_args()
    subprocess.run(["python3", str(ROOT / "scripts/extract_bitz_reference.py"), "--check"], check=True)
    output = args.output or ROOT / "audit/logs" / (
        "stage13-bitz-compatibility-portable.log" if args.portable else "stage13-bitz-compatibility-native.log"
    )
    with tempfile.TemporaryDirectory(prefix="memarg-bitz-compatibility-") as temporary:
        work = Path(temporary)
        manifest = work / "Cargo.toml"
        # JSON string syntax is valid for these TOML paths and preserves spaces.
        quote = lambda value: json.dumps(str(value))
        manifest.write_text(f'''[package]
name = "memarg-bitz-compatibility"
version = "0.0.0"
edition = "2024"
[[bin]]
name = "memarg-bitz-compatibility"
path = {quote(ROOT / "audit/bitz-compatibility.rs")}
[dependencies]
om-field = {{ path = {quote(ROOT / "vendor/open-magnetar/crates/om-field")} }}
om-sumcheck = {{ path = {quote(ROOT / "vendor/open-magnetar/crates/om-sumcheck")} }}
''')
        environment = os.environ.copy()
        environment["RUSTFLAGS"] = "" if args.portable else "-C target-cpu=native"
        environment.setdefault("CARGO_BUILD_JOBS", "4")
        environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        command = ["cargo", "run", "--release", "--offline", "--manifest-path", str(manifest)]
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        log = f"RUSTFLAGS={environment['RUSTFLAGS']!r}\n" + result.stderr + result.stdout
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(log)
        print(log, end="")
        print(f"Log: {output}")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
