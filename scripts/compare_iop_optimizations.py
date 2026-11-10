#!/usr/bin/env python3
"""Validate matched before/after RAM runs and retain raw median comparisons."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["payload", "guest_program", "guest_rows", "layout", "chunk_bits", "virtualization"]
SAME = ["source_payload_metadata", "work_normalization", "proof_summary", "virtualization_stats", "address_relabeling"]
OPTIONS = ["threads", "log_cycles", "address_bits", "seed", "profile", "log_rate", "memory_model", "reads", "guest_keys", "guest_queries", "guest_size", "guest_operations", "guest_fill", "repeat", "warmup"]
METRICS = ["prover_ms", "commitment_ms", "everything_else_ms", "virtualization_ms", "verifier_ms"]


def near(left, right):
    assert math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7), (left, right)


def read(path):
    report = json.loads(path.read_text())
    assert report["schema"] in ["memarg-stage-1.2/v5", "memarg-stage-1.2/v6"]
    for case in report["cases"]:
        assert len(case["samples"]) == report["options"]["repeat"]
        for sample in case["samples"]:
            assert all(math.isfinite(sample[metric]) for metric in METRICS)
            near(sample["commitment_ms"], sample["stages_ms"]["pcs.commit"])
            near(sample["everything_else_ms"], sample["prover_ms"] - sample["commitment_ms"])
            near(sample["virtualization_ms"], sum(v for k, v in sample["stages_ms"].items() if k == "twist.virtualization" or k.startswith("virt.")))
            near(sample["prover_ms"], sample["unattributed_ms"] + sum(sample["stages_ms"].values()))
    csv_path = path.with_suffix(".csv")
    with csv_path.open(newline="") as source:
        assert len(list(csv.DictReader(source))) == sum(len(c["samples"]) for c in report["cases"])
    return report


def metric(before, after):
    old, new = statistics.median(before), statistics.median(after)
    return {"before_ms": old, "after_ms": new, "speedup": old / new if new else None,
            "saving_pct": 100 * (old - new) / old if old else None}


def compare(before_path, after_path):
    before, after = read(before_path), read(after_path)
    # This comparison requires the same protocol and proof layout. V6 changes
    # the meaning of Flat; retained V5 measurements describe legacy Flat.
    assert before["schema"] == after["schema"]
    assert before["field"] == after["field"]
    assert before["upstream_revision"] == after["upstream_revision"]
    for key in OPTIONS:
        assert before["options"].get(key) == after["options"].get(key), key
    for key in ["architecture", "os", "cpu_model", "compiled_pclmulqdq", "compiled_avx2"]:
        assert before["environment"][key] == after["environment"][key], key
    key = lambda case: tuple(case[k] for k in KEYS)
    old_cases = {key(c): c for c in before["cases"]}
    new_cases = {key(c): c for c in after["cases"]}
    assert len(old_cases) == len(before["cases"])
    assert len(new_cases) == len(after["cases"])
    assert old_cases.keys() == new_cases.keys()
    cases = []
    for identity, old in old_cases.items():
        new = new_cases[identity]
        for name in SAME:
            assert old[name] == new[name], (identity, name)
        entry = dict(zip(KEYS, identity))
        entry["address_components"] = old["address_components"]
        entry["metrics"] = {m: metric([s[m] for s in old["samples"]], [s[m] for s in new["samples"]]) for m in METRICS}
        stages = set.intersection(*(set(s["stages_ms"]) for s in old["samples"] + new["samples"]))
        entry["stages"] = {m: metric([s["stages_ms"][m] for s in old["samples"]], [s["stages_ms"][m] for s in new["samples"]]) for m in sorted(stages)}
        cases.append(entry)
    return {"before": str(before_path.relative_to(ROOT)), "after": str(after_path.relative_to(ROOT)),
            "before_patch": before["vendor_patch_sha256"], "after_patch": after["vendor_patch_sha256"],
            "cases": cases, "verified_proofs_including_warmups": len(cases) * 2 * (before["options"]["repeat"] + before["options"]["warmup"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="stage12-full")
    args = parser.parse_args()
    results = ROOT / "results"
    pairs = [compare(results / f"{args.prefix}-before{suffix}-a16-n18.json", results / f"{args.prefix}-after{suffix}-a16-n18.json") for suffix in ["", "-reversed"]]
    # Retained measurements belong to their recorded source checkpoint even
    # when the live vendor tree has subsequently changed.
    provenance = json.loads((results / f"{args.prefix}-after-a16-n18.json").read_text())
    assert all(pair["after_patch"] == provenance["vendor_patch_sha256"] for pair in pairs)
    rows = [row for pair in pairs for row in pair["cases"]]
    summary = {}
    for name in METRICS:
        values = [row["metrics"][name] for row in rows]
        summary[name] = {"speedup_range": [min(v["speedup"] for v in values), max(v["speedup"] for v in values)],
                         "geometric_mean_speedup": statistics.geometric_mean(v["speedup"] for v in values),
                         "saving_pct_range": [min(v["saving_pct"] for v in values), max(v["saving_pct"] for v in values)]}
    report = {"schema": "memarg-full-iop-comparison/v1", "implementation": provenance["implementation"],
              "patch_sha256": provenance["vendor_patch_sha256"], "summary": summary, "pairs": pairs,
              "verified_proofs_including_warmups": sum(p["verified_proofs_including_warmups"] for p in pairs)}
    (results / f"{args.prefix}-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"Validated {len(rows)} matched before/after pairs; {report['verified_proofs_including_warmups']} benchmark proofs verified."]
    for name, values in summary.items():
        lo, hi = values["speedup_range"]
        lines.append(f"{name}: {lo:.3f}–{hi:.3f}x; geometric mean {values['geometric_mean_speedup']:.3f}x")
    (results / f"{args.prefix}-comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
