#!/usr/bin/env python3
"""Alternate two ram-compare binaries round by round and compare them.

Each round runs the baseline and the branch binary once each, on the same
arguments, and the order flips every round so neither binary always runs
first. Every invocation uses its own small --warmup/--repeat (default 1/1),
writes its own JSON and log, and has `uptime` recorded before and after.

The summary gives, for every case present in both binaries' reports
(workload, method, Twist chunk size), the median over rounds of each
binary's per-invocation median prover and verifier time, and the
branch/baseline ratio of those medians with a paired 95% bootstrap interval
(rounds resampled with replacement, 10,000 resamples, fixed seed). It reads
both the schema-4 and schema-5 ram-compare JSON.

Example:
  scripts/interleave.py --baseline /path/base/ram-compare \\
      --branch target/release/ram-compare --rounds 10 --out-dir results/il \\
      -- --trace-format operations --methods twist,spice-typed \\
         --patterns random --address-bits 16 --log-cycles 20 --threads 8

Re-analyse an existing directory without running anything:
  scripts/interleave.py --analyze results/il
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_stats import (  # noqa: E402
    RESAMPLES,
    SEED,
    case_medians,
    key_label,
    load_report,
    median,
    paired_ratio,
    uptime,
)

LABELS = ("baseline", "branch")
RESERVED = {"--output", "--csv", "--warmup", "--repeat"}


def run_rounds(args, extra):
    for flag in extra:
        if flag in RESERVED:
            raise SystemExit(f"{flag} is set by interleave.py; do not pass it after --")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    binaries = {"baseline": args.baseline, "branch": args.branch}
    invocations = []
    for round_index in range(args.rounds):
        order = LABELS if round_index % 2 == 0 else LABELS[::-1]
        for position, label in enumerate(order):
            stem = f"round{round_index:03d}-{label}"
            command = [
                binaries[label],
                *extra,
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--output",
                str(out / f"{stem}.json"),
            ]
            before = uptime()
            print(f"[{round_index + 1}/{args.rounds}] {label}: {before.get('raw')}", flush=True)
            with open(out / f"{stem}.log", "w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            after = uptime()
            invocations.append(
                {
                    "round": round_index,
                    "position": position,
                    "label": label,
                    "binary": binaries[label],
                    "command": command,
                    "exit_code": result.returncode,
                    "uptime_before": before,
                    "uptime_after": after,
                    "json": f"{stem}.json",
                }
            )
            if result.returncode != 0:
                raise SystemExit(f"{label} failed in round {round_index}; see {out / stem}.log")
    (out / "invocations.json").write_text(json.dumps(invocations, indent=2) + "\n")
    return invocations


def analyze(out_dir, resamples=RESAMPLES, seed=SEED):
    out = Path(out_dir)
    invocations = json.loads((out / "invocations.json").read_text())
    per_round = {}
    for inv in invocations:
        report = load_report(out / inv["json"])
        per_round.setdefault(inv["round"], {})[inv["label"]] = case_medians(report)
    rounds = sorted(r for r, runs in per_round.items() if set(runs) == set(LABELS))
    keys = None
    for r in rounds:
        present = set(per_round[r]["baseline"]) & set(per_round[r]["branch"])
        keys = present if keys is None else keys & present
    rows = []
    for key in sorted(keys or (), key=lambda k: tuple(str(v) for v in k)):
        row = {
            "pattern": key[0],
            "method": key[1],
            "chunk_bits": key[2],
            "rounds": len(rounds),
        }
        for metric in ("prover_ms", "verifier_ms"):
            base = [per_round[r]["baseline"][key][metric] for r in rounds]
            branch = [per_round[r]["branch"][key][metric] for r in rounds]
            row[metric] = {
                "baseline_median": median(base),
                "branch_median": median(branch),
                "baseline_by_round": base,
                "branch_by_round": branch,
                "branch_over_baseline": paired_ratio(branch, base, resamples, seed),
            }
        rows.append((key, row))
    loads = [
        inv["uptime_before"]["load"][0]
        for inv in invocations
        if inv["uptime_before"].get("load")
    ]
    summary = {
        "rounds": len(rounds),
        "resamples": resamples,
        "seed": seed,
        "load1_range": [min(loads), max(loads)] if loads else None,
        "cases": [row for _, row in rows],
    }
    lines = [
        f"Rounds: {len(rounds)}; bootstrap: {resamples} paired resamples, seed {seed:#x}.",
        "",
        "| Case | Baseline prover (ms) | Branch prover (ms) | Branch/baseline prover [95% CI] "
        "| Baseline verifier (ms) | Branch verifier (ms) | Branch/baseline verifier [95% CI] |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]

    def cell(entry):
        r = entry["branch_over_baseline"]
        return f"{r['ratio']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"

    for key, row in rows:
        p, v = row["prover_ms"], row["verifier_ms"]
        lines.append(
            f"| {key_label(key)} | {p['baseline_median']:.2f} | {p['branch_median']:.2f} | {cell(p)} "
            f"| {v['baseline_median']:.2f} | {v['branch_median']:.2f} | {cell(v)} |"
        )
    if loads:
        lines += ["", f"One-minute load before invocations: {min(loads):.2f} to {max(loads):.2f}."]
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
    parser.add_argument("--baseline", help="baseline ram-compare binary")
    parser.add_argument("--branch", help="branch ram-compare binary")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out-dir", default="results/interleave")
    parser.add_argument("--analyze", metavar="DIR", help="only summarise an existing run directory")
    args = parser.parse_args(argv)
    if args.analyze:
        _, markdown = analyze(args.analyze)
    else:
        if not args.baseline or not args.branch:
            parser.error("--baseline and --branch are required unless --analyze is given")
        if args.rounds < 1:
            parser.error("--rounds must be positive")
        run_rounds(args, extra)
        _, markdown = analyze(args.out_dir)
    print(markdown, end="")


if __name__ == "__main__":
    main()
