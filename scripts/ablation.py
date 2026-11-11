#!/usr/bin/env python3
"""Interleave several ram-compare binaries on the same Twist v2 cases.

Each round runs every binary once, in alternating order, with the given
ram-compare arguments, and keeps each run's JSON report. The summary gives,
per workload and binary, the median prover time, the median of each
selected stage and the 1-minute load average before each run. Use it for
ablations where each binary adds one prover change.

usage: scripts/ablation.py --rounds R --out-dir DIR NAME=BINARY ... -- ARGS
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

STAGES = ["twist.cycle.rounds", "twist.read", "pcs.commit", "pcs.ligerito_open"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("binaries", nargs="+", help="NAME=PATH, in ablation order")
    argv = sys.argv[1:]
    if "--" not in argv:
        ap.error("pass ram-compare arguments after --")
    split = argv.index("--")
    opts = ap.parse_args(argv[:split])
    rc_args = argv[split + 1 :]
    bins = [b.split("=", 1) for b in opts.binaries]
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    samples = defaultdict(lambda: defaultdict(list))
    loads = defaultdict(list)
    for r in range(opts.rounds):
        order = bins if r % 2 == 0 else bins[::-1]
        for name, path in order:
            load = os.getloadavg()[0]
            loads[name].append(load)
            out = opts.out_dir / f"{name}-round{r}.json"
            subprocess.run([path, *rc_args, "--output", str(out)], check=True, capture_output=True)
            for case in json.loads(out.read_text())["cases"]:
                if case["method"] != "twist":
                    continue
                key = case["pattern"]
                for s in case["samples"]:
                    samples[key][name, "prover"].append(s["prover_ms"])
                    for stage in STAGES:
                        samples[key][name, stage].append(s["stages_ms"].get(stage, float("nan")))
            print(f"round {r} {name}: load(1m) {load:.1f}", file=sys.stderr)
    names = [n for n, _ in bins]
    first = names[0]
    lines = [f"rounds={opts.rounds} binaries={' '.join(f'{n}={p}' for n, p in bins)}", f"args: {' '.join(rc_args)}"]
    for name in names:
        lines.append(f"load(1m) before {name} runs: min {min(loads[name]):.1f} median {statistics.median(loads[name]):.1f} max {max(loads[name]):.1f}")
    for key in sorted(samples):
        lines.append("")
        lines.append(f"== {key} (median ms over all samples; ratio to {first})")
        lines.append(f"{'':22s}" + "".join(f"{n:>16s}" for n in names))
        for metric in ["prover", *STAGES]:
            base = statistics.median(samples[key][first, metric])
            cells = []
            for n in names:
                m = statistics.median(samples[key][n, metric])
                cells.append(f"{m:8.2f} ({m / base:4.2f})")
            lines.append(f"{metric:22s}" + "".join(f"{c:>16s}" for c in cells))
    text = "\n".join(lines) + "\n"
    (opts.out_dir / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
