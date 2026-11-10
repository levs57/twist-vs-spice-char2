#!/usr/bin/env python3
"""Run the controlled Shout ablation through public production APIs, offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sources():
    selected = {ROOT / "Cargo.toml", ROOT / "Cargo.lock"}
    for directory in [ROOT / "crates/lookup-argument", ROOT / "crates/ram-payload",
                      ROOT / "vendor/open-magnetar", ROOT / "vendor/bitz-logup"]:
        for parent, dirs, files in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in {"target", ".git"}]
            for name in files:
                if name.endswith(".rs") or name == "Cargo.toml" or name.startswith("LICENSE"):
                    selected.add(Path(parent) / name)
    selected.update((ROOT / "payloads/open-magnetar").glob("*.elf"))
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(selected)}
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode() + b"\0")
        digest.update(bytes.fromhex(value))
    return hashes, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-lookups", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage13-shout-structure-structural.json")
    parser.add_argument("--log", type=Path, default=ROOT / "audit/logs/stage13-shout-structure-structural.log")
    args = parser.parse_args()
    before, fingerprint = sources()
    with tempfile.TemporaryDirectory(prefix="memarg-shout-structure-") as directory:
        work = Path(directory)
        (work / "src").mkdir()
        shutil.copyfile(ROOT / "audit/shout-structure-probe.rs", work / "src/main.rs")
        crates = ROOT / "vendor/open-magnetar/crates"
        manifest = ('[package]\nname="shout-structure-probe"\nversion="0.0.0"\nedition="2024"\n'
                    '[dependencies]\n'
                    f'lookup-argument={{path="{ROOT / "crates/lookup-argument"}"}}\n'
                    f'ram-payload={{path="{ROOT / "crates/ram-payload"}"}}\n'
                    f'om-transcript={{path="{crates / "om-transcript"}",features=["profiling"]}}\n'
                    f'om-memcheck={{path="{crates / "om-memcheck"}"}}\n'
                    f'om-sumcheck={{path="{crates / "om-sumcheck"}"}}\n'
                    'rayon="1"\nserde_json="1"\n'
                    '[profile.release]\nlto="thin"\ncodegen-units=1\n')
        (work / "Cargo.toml").write_text(manifest)
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(ROOT / "target")
        env["CARGO_BUILD_JOBS"] = "4"
        env["RUSTFLAGS"] = "-C target-cpu=native"
        env["SHOUT_PROBE_SOURCE_SHA256"] = fingerprint
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = ["cargo", "run", "--offline", "--release", "--manifest-path", str(work / "Cargo.toml"), "--",
                   str(args.log_lookups), str(args.threads), str(args.warmup), str(args.repeat), str(args.output.resolve())]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("RUSTFLAGS=-C target-cpu=native\nProduction source: " + fingerprint + "\n" + result.stdout)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit(result.returncode)
        cargo_lock = (work / "Cargo.lock").read_text()
    after, after_fingerprint = sources()
    if before != after:
        raise SystemExit("production sources changed during the probe")
    report = json.loads(args.output.read_text())
    report["provenance"].update({"production_source_sha256_after": after_fingerprint,
        "production_sources_unchanged": True, "production_files_sha256": before,
        "probe_source_sha256": hashlib.sha256((ROOT / "audit/shout-structure-probe.rs").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rustflags": env["RUSTFLAGS"], "release_profile": {"lto": "thin", "codegen_units": 1},
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "temporary_cargo_lock": cargo_lock, "temporary_cargo_lock_sha256": hashlib.sha256(cargo_lock.encode()).hexdigest(),
        "temporary_cargo_manifest": manifest})
    assert report["verified_proofs"] == 4 * (args.warmup + args.repeat)
    assert report["cases"][0]["query_values_sha256"] == report["cases"][1]["query_values_sha256"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        print(case["name"], "U=", case["structure"]["unique_onehot_components"], case["median"])
    print("Verified", report["verified_proofs"], "proofs; production sources unchanged:", fingerprint)


if __name__ == "__main__":
    main()
