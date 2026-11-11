#!/usr/bin/env python3
"""Run ram-compare over a grid of memory sizes, trace lengths and thread counts.

Every point of the grid (address bits A, log trace length N, threads T) runs
one ram-compare invocation with the arguments after `--` and writes
`a{A}-n{N}-t{T}.json` and `.log` to the output directory. The script records
`uptime` before and after each point and rewrites `index.json` after every
point, so an interrupted sweep keeps what it finished and `--skip-existing`
resumes it. A failed point is recorded and the sweep continues; the script
exits non-zero at the end if any point failed.

Points run with N in the outer loop, then A, then T, so the cheap points
finish first. `--analyze` (also run at the end of every sweep) writes
`summary.json` and `summary.md`: one row per point and case with the median
prover and verifier time, prover ns per trace row, proof bytes and composed
soundness bits (the last three need schema-5 reports).

Example (the final-benchmark grid; run it in the background):
  scripts/sweep.py --binary target/release/ram-compare \\
      --address-bits 12,16,20 --log-cycles 16,18,20 --threads 1,8,16 \\
      --out-dir results/sweep \\
      -- --trace-format operations --methods twist,spice-typed \\
         --patterns random,locality --warmup 1 --repeat 5

Re-analyse an existing directory without running anything:
  scripts/sweep.py --analyze results/sweep
"""

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_stats import case_key, case_medians, key_label, load_report, uptime  # noqa: E402

RESERVED = {"--address-bits", "--log-cycles", "--threads", "--output"}


def integers(text):
    values = [int(v) for v in text.split(",") if v]
    if not values or len(set(values)) != len(values) or min(values) < 1:
        raise argparse.ArgumentTypeError(f"expected distinct positive integers, got {text!r}")
    return values


def points(address_bits, log_cycles, threads):
    """Grid points in run order: N outermost, then A, then T."""
    return [
        {"address_bits": a, "log_cycles": n, "threads": t}
        for n, a, t in itertools.product(log_cycles, address_bits, threads)
    ]


def stem(point):
    return f"a{point['address_bits']}-n{point['log_cycles']}-t{point['threads']}"


def run_sweep(args, extra):
    for flag in extra:
        if flag.split("=")[0] in RESERVED:
            raise SystemExit(f"{flag} is set by sweep.py; do not pass it after --")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    finished = {entry["json"] for entry in index if entry["exit_code"] == 0}
    grid = points(args.address_bits, args.log_cycles, args.threads)
    failures = 0
    for number, point in enumerate(grid, 1):
        name = stem(point)
        if args.skip_existing and f"{name}.json" in finished and (out / f"{name}.json").exists():
            print(f"[{number}/{len(grid)}] {name}: already done, skipped", flush=True)
            continue
        command = [
            args.binary,
            *extra,
            "--address-bits",
            str(point["address_bits"]),
            "--log-cycles",
            str(point["log_cycles"]),
            "--threads",
            str(point["threads"]),
            "--output",
            str(out / f"{name}.json"),
        ]
        before = uptime()
        print(f"[{number}/{len(grid)}] {name}: {before.get('raw')}", flush=True)
        start = time.monotonic()
        with open(out / f"{name}.log", "w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        entry = {
            **point,
            "command": command,
            "exit_code": result.returncode,
            "wall_seconds": round(time.monotonic() - start, 3),
            "uptime_before": before,
            "uptime_after": uptime(),
            "json": f"{name}.json",
            "log": f"{name}.log",
        }
        index = [e for e in index if e["json"] != entry["json"]] + [entry]
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        if result.returncode != 0:
            failures += 1
            print(f"  failed with exit code {result.returncode}; see {out / name}.log", flush=True)
    return failures


def analyze(out_dir):
    out = Path(out_dir)
    index = json.loads((out / "index.json").read_text())
    rows = []
    for entry in sorted(index, key=lambda e: (e["log_cycles"], e["address_bits"], e["threads"])):
        path = out / entry["json"]
        if entry["exit_code"] != 0 or not path.exists():
            continue
        report = load_report(path)
        medians = case_medians(report)
        for case in report["cases"]:
            key = case_key(case)
            throughput = case.get("throughput") or {}
            rows.append(
                {
                    "address_bits": entry["address_bits"],
                    "log_cycles": entry["log_cycles"],
                    "threads": entry["threads"],
                    "pattern": key[0],
                    "method": key[1],
                    "chunk_bits": key[2],
                    "prover_ms": medians[key]["prover_ms"],
                    "verifier_ms": medians[key]["verifier_ms"],
                    "prover_ns_per_operation": throughput.get("prover_ns_per_operation"),
                    "prover_ns_per_source_memory_instruction": throughput.get(
                        "prover_ns_per_source_memory_instruction"
                    ),
                    "proof_bytes": (case.get("proof_size") or {}).get("total_bytes"),
                    "populated_bits": (case.get("area") or {}).get("populated_bits"),
                    "committed_bits": (case.get("area") or {}).get("committed_bits"),
                    "soundness_bits": (case.get("soundness") or {}).get("bits"),
                    "load1_before": (entry["uptime_before"].get("load") or [None])[0],
                }
            )
    failed = [e["json"] for e in index if e["exit_code"] != 0]
    summary = {"points": len(index), "failed": failed, "rows": rows}

    def number(value, digits):
        return "" if value is None else f"{value:.{digits}f}"

    def log2(value):
        return "" if not value else f"{math.log2(value):.2f}"

    lines = [
        f"Points: {len(index)} ({len(failed)} failed).",
        "",
        "| A | N | Threads | Case | Prover (ms) | Verifier (ms) | Prover ns/row "
        "| log2 populated | log2 committed | Proof (B) | Soundness (bits) | Load before |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        label = key_label((r["pattern"], r["method"], r["chunk_bits"]))
        lines.append(
            f"| {r['address_bits']} | {r['log_cycles']} | {r['threads']} | {label} "
            f"| {r['prover_ms']:.2f} | {r['verifier_ms']:.2f} "
            f"| {number(r['prover_ns_per_operation'], 1)} "
            f"| {log2(r['populated_bits'])} | {log2(r['committed_bits'])} "
            f"| {'' if r['proof_bytes'] is None else r['proof_bytes']} "
            f"| {number(r['soundness_bits'], 2)} | {number(r['load1_before'], 2)} |"
        )
    if failed:
        lines += ["", "Failed points: " + ", ".join(failed) + "."]
    markdown = "\n".join(lines) + "\n"
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown)
    return summary, markdown


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", help="ram-compare binary")
    parser.add_argument("--address-bits", type=integers, default=[16])
    parser.add_argument("--log-cycles", type=integers, default=[16, 18, 20])
    parser.add_argument("--threads", type=integers, default=[1, 8, 16])
    parser.add_argument("--out-dir", default="results/sweep")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip points that index.json records as finished",
    )
    parser.add_argument("--analyze", metavar="DIR", help="only summarise an existing sweep directory")
    args = parser.parse_args(argv)
    if args.analyze:
        _, markdown = analyze(args.analyze)
        print(markdown, end="")
        return 0
    if not args.binary:
        parser.error("--binary is required unless --analyze is given")
    failures = run_sweep(args, extra)
    _, markdown = analyze(args.out_dir)
    print(markdown, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
