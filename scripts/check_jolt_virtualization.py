#!/usr/bin/env python3
"""Run the small research-only GF128 probe using cached local dependencies."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=root / "target")
    parser.add_argument("--log", type=Path, default=root / "audit/logs/jolt-virtualization-probe.log")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="memarg-jolt-virt-") as directory:
        work = Path(directory)
        (work / "src").mkdir()
        shutil.copyfile(root / "audit/jolt-virtualization-probe.rs", work / "src/main.rs")
        crates = root / "vendor/open-magnetar/crates"
        (work / "Cargo.toml").write_text(
            '[package]\nname = "jolt-virtualization-probe"\nversion = "0.0.0"\nedition = "2024"\n'
            '[dependencies]\n'
            f'om-field = {{ path = "{crates / "om-field"}" }}\n'
            f'om-sumcheck = {{ path = "{crates / "om-sumcheck"}" }}\n'
        )
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(args.target_dir.resolve())
        env["CARGO_BUILD_JOBS"] = "4"
        env["RUSTFLAGS"] = env.get("RUSTFLAGS", "-C target-cpu=native")
        command = ["cargo", "run", "--offline", "--release", "--manifest-path", str(work / "Cargo.toml")]
        result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(result.stdout)
        print(result.stdout, end="")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
