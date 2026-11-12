"""Validate and summarize the direct-pushforward Twist experiment."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_stats import median, paired_ratio


def load(path):
    return json.loads((ROOT / path).read_text())


manifest = load("audit/stage16-implementation.json")
files = ["twist-logup-smoke", "twist-logup-a16-n20", "twist-logup-a12-n20", "twist-logup-a20-n18"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, help="optionally verify the original measured executable")
args = parser.parse_args()
exe = load(f"results/{files[0]}.json")["environment"]["executable_sha256"]
assert re.fullmatch(r"[0-9a-f]{64}", exe)
if args.binary is not None:
    assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == exe
rows = []
proofs = 0
measured = 0
for name in files:
    report = load(f"results/{name}.json")
    assert report["environment"]["executable_sha256"] == exe
    assert report["provenance"]["implementation_manifest"]["source_sha256"] == manifest["source_sha256"]
    assert report["environment"]["rustflags"] == "-C target-cpu=native"
    o = report["options"]
    a, n = o["address_bits"], o["log_cycles"]
    N, T = 1 << a, 1 << n
    assert report["verified_proofs"] == len(report["cases"]) * (o["warmup"] + o["repeat"])
    proofs += report["verified_proofs"]
    measured += len(report["cases"]) * o["repeat"]
    grouped = {}
    for c in report["cases"]:
        grouped.setdefault(c["pattern"], {})[c["method"]] = c
        area = c["area"]
        assert area["cycles"] == T and area["memory_cells"] == N
        assert area["populated_bits"] == sum(v["bits"] for v in area["components"])
        assert area["padding_bits"] == area["committed_bits"] - area["populated_bits"]
        private = o["final_memory"] == "private"
        domains = area.get("commitments", [])
        if c["method"] == "twist-logup-star":
            assert [v["name"] for v in domains] == ["main", "pushforward"]
            main_bits = (a + 64) * T + (64 * N if private else 0)
            assert [v["populated_bits"] for v in domains] == [main_bits, 128 * N]
            assert [v["ell"] for v in domains] == [max(14, (main_bits - 1).bit_length()), max(14, a + 7)]
            assert domains[0]["mosaic_claims"] == a + 2 + private
            assert domains[1]["mosaic_claims"] == 2
            assert area["committed_bits"] == sum(1 << v["ell"] for v in domains)
            assert area["populated_bits"] == main_bits + 128 * N
            terms = {v["name"]: v for v in c["soundness"]["terms"]}
            assert terms["twist2.logup.pushforward.rational"]["probability"] == (N - 1) * 2.0 ** -128
            for domain in ("main", "aux"):
                assert f"{domain}.ringswitch.rho" in terms
                assert any(v.startswith(f"{domain}.ligerito.") for v in terms)
            assert c["chunk_bits"] is None
        else:
            chunk = c["chunk_bits"]
            assert area["populated_bits"] == (a // chunk * (1 << chunk) + 64) * T + (64 * N if private else 0)
        for s in c["samples"]:
            stages = s["stages_ms"]
            get = lambda key: stages.get(key, 0.0)
            assert abs(s["commit_ms"] - get("pcs.commit") - get("logup.aux.commit")) < 1e-7
            assert abs(s["opening_ms"] - sum(get(k) for k in ("pcs.open", "pcs.ringswitch", "pcs.ligerito_open", "logup.aux.open"))) < 1e-7
            assert abs(sum(stages.values()) + s["unattributed_ms"] - s["prover_ms"]) < 1e-7
            assert s["unattributed_ms"] >= -1e-7
            adjusted = (get("pcs.commit") * domains[0]["populated_fraction"]
                        + get("logup.aux.commit") * domains[1]["populated_fraction"]) if domains else s["commit_ms"] * area["populated_fraction"]
            p = s["padding_adjusted_estimate"]
            assert abs(p["commit_ms"] - adjusted) < 1e-7
            assert abs(p["prover_ms"] - s["prover_ms"] + s["commit_ms"] - adjusted) < 1e-7
            assert s["proof_size"] == c["proof_size"]
        soundness = c["soundness"]
        assert abs(sum(t["probability"] for t in soundness["terms"]) - soundness["total_probability"]) < 1e-45
        assert abs(-math.log2(soundness["total_probability"]) - soundness["bits"]) < 1e-10
    for pattern, pair in grouped.items():
        baseline, branch = pair["twist"], pair["twist-logup-star"]
        for key in ("trace_sha256", "operation_trace_sha256", "metadata", "trace_format", "final_memory"):
            assert baseline[key] == branch[key], (pattern, key)
        by_round = lambda case, key: [s[key] for s in sorted(case["samples"], key=lambda s: s["iteration"])]
        row = {"report": f"results/{name}.json", "pattern": pattern, "address_bits": a, "log_cycles": n,
               "read_order": o["read_order"], "final_memory": o["final_memory"], "rounds": o["repeat"]}
        for key in ("prover_ms", "verifier_ms", "commit_ms", "opening_ms"):
            before, after = by_round(baseline, key), by_round(branch, key)
            row[key] = {"twist": median(before), "logup": median(after), "logup_over_twist": paired_ratio(after, before)}
        for label, c in (("twist", baseline), ("logup", branch)):
            def phase_total(sample):
                return sum(value for key, value in sample["stages_ms"].items()
                           if key.startswith("twist.cycle") or
                           (key.startswith("logup.") and key not in ("logup.aux.commit", "logup.aux.open")))
            row[label] = {
                "cycle_and_address_iop_ms": median([phase_total(s) for s in c["samples"]]),
                "read_ms": c["median"]["stages_ms"]["twist.read"],
                "mosaic_ms": c["median"]["mosaic_ms"],
                "area": c["area"], "soundness_bits": c["soundness"]["bits"],
                "proof_size": c["proof_size"],
            }
        rows.append(row)

with (ROOT / "results/twist-logup-a16-n20.csv").open() as f:
    assert len(list(csv.DictReader(f))) == 120
tests = {}
for mode in ("default", "native"):
    path = f"audit/logs/twist-logup-workspace-{mode}.log"
    text = (ROOT / path).read_text()
    count = sum(int(n) for n in re.findall(r"test result: ok\. (\d+) passed", text))
    assert count == 208 and "test result: FAILED" not in text
    tests[mode] = {"passed": count, "log": path}

out = {
    "source_sha256": manifest["source_sha256"], "executable_sha256": exe,
    "verified_proofs": proofs, "measured_proofs": measured, "tests": tests,
    "checks": ["matched operation and RMW traces, metadata and public/private boundaries",
               "dense address and auxiliary commitment areas match independent formulas",
               "each PCS and RingSwitch uses its actual domain",
               "rational error equals (N-1)/2^128",
               "per-proof exclusive accounting closes and both commitments/openings charged",
               "padding adjustment independently weights the two commitment times",
               "all reports name frozen source and executable"],
    "cases": rows,
}
destination = ROOT / "results/twist-logup-summary.json"
destination.write_text(json.dumps(out, indent=2) + "\n")
print(f"Validated {proofs} proofs ({measured} measured), {len(rows)} pairs, both 208-test builds.")
for r in rows:
    p = r["prover_ms"];ratio=p["logup_over_twist"]
    print(r["address_bits"], r["log_cycles"], r["pattern"],
          f"{p['twist']:.3f} -> {p['logup']:.3f} ms; ratio {ratio['ratio']:.3f} [{ratio['ci95_low']:.3f}, {ratio['ci95_high']:.3f}];",
          f"cycle/address {r['twist']['cycle_and_address_iop_ms']:.3f} -> {r['logup']['cycle_and_address_iop_ms']:.3f}")
print(destination)
