#!/usr/bin/env python3
"""Run matched native commitment/hash/NTT microbenchmarks against a vendor tree."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, default=ROOT / "vendor/open-magnetar")
    args = parser.parse_args()
    vendor = args.vendor.resolve()
    sha4 = "hash_prefixed_four_into" in (vendor / "crates/om-transcript/src/hashing.rs").read_text()
    with tempfile.TemporaryDirectory(prefix="memarg-commit-benchmark-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        shutil.copy2(ROOT / "audit/commitment-benchmark.rs", work / "src/main.rs")
        dependencies = "\n".join(
            f'{name} = {{ path = {json.dumps(str(vendor / "crates" / name))} }}'
            for name in ["om-field", "om-transcript", "om-pcs"]
        )
        (work / "Cargo.toml").write_text(f'''[package]
name = "memarg-commit-benchmark"
version = "0.0.0"
edition = "2024"
[features]
sha4 = []
[dependencies]
{dependencies}
rayon = "1"
[profile.release]
lto = "thin"
codegen-units = 1
''')
        environment = os.environ.copy()
        environment.setdefault("RUSTFLAGS", "-C target-cpu=native")
        environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
        command = ["cargo", "run", "--offline", "--release", "--manifest-path", str(work / "Cargo.toml")]
        if sha4:
            command += ["--features", "sha4"]
        print(f"Vendor: {vendor}; four-stream SHA: {sha4}; 2 warmups, median of 7", flush=True)
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
