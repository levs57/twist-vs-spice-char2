"""Shared statistics and report readers for the benchmark driver scripts.

The bootstrap matches crates/ram-compare/src/stats.rs: SplitMix64 with the
same seed, rejection sampling for indices, 10,000 resamples, and linear
interpolation between order statistics for the 2.5% and 97.5% percentiles.
"""

import json
import re
import subprocess
from pathlib import Path

RESAMPLES = 10_000
SEED = 0x5EED_7715_7A1C_E5E5
MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, bound):
        zone = MASK - (MASK % bound)
        while True:
            value = self.next()
            if value < zone:
                return value % bound


def median(values):
    ordered = sorted(values)
    m = len(ordered) // 2
    return ordered[m] if len(ordered) % 2 else (ordered[m - 1] + ordered[m]) / 2


def quantile(ordered, q):
    position = q * (len(ordered) - 1)
    low, high = int(position // 1), int(-(-position // 1))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_ratio(numerator, denominator, resamples=RESAMPLES, seed=SEED):
    """Ratio of medians with a paired percentile-bootstrap 95% interval.

    numerator[i] and denominator[i] belong to round i; resampling draws the
    same rounds for both, preserving the pairing.
    """
    rounds = len(numerator)
    if rounds == 0 or rounds != len(denominator):
        return None
    rng = SplitMix64(seed)
    ratios = []
    for _ in range(resamples):
        picks = [rng.below(rounds) for _ in range(rounds)]
        ratios.append(
            median([numerator[i] for i in picks]) / median([denominator[i] for i in picks])
        )
    ratios.sort()
    return {
        "ratio": median(numerator) / median(denominator),
        "ci95_low": quantile(ratios, 0.025),
        "ci95_high": quantile(ratios, 0.975),
        "rounds": rounds,
        "resamples": resamples,
        "seed": seed,
    }


def case_key(case):
    """Identify a case across runs: workload, method and Twist chunk size."""
    return (case["pattern"], case["method"], case.get("chunk_bits"))


def key_label(key):
    pattern, method, chunk = key
    return f"{pattern} {method}" + (f" (chunk={chunk})" if chunk is not None else "")


def case_medians(report):
    """Per-case median prover/verifier ms from a schema-4 or schema-5 report.

    Both schemas flatten the median times into case["median"]; older or
    partial files fall back to the median over case["samples"].
    """
    version = report.get("schema_version")
    if version not in (4, 5):
        raise ValueError(f"unsupported ram-compare schema_version {version}")
    out = {}
    for case in report["cases"]:
        med = case.get("median", {})
        values = {}
        for metric in ("prover_ms", "verifier_ms"):
            if metric in med:
                values[metric] = med[metric]
            else:
                values[metric] = median([s[metric] for s in case["samples"]])
        values["implementation"] = case_implementation(case, version)
        values["trace_sha256"] = case.get("trace_sha256")
        out[case_key(case)] = values
    return out


def case_implementation(case, version):
    """The prover a case's method name ran.

    Current reports record it. Schema-4 reports do not, and their `twist` is
    the prover that schema-5 binaries call `twist-legacy`.
    """
    if "implementation" in case:
        return case["implementation"]
    if version == 4 and case["method"] == "twist":
        return "ram_argument BinaryTwist flat (schema-4 twist)"
    return f"{case['method']} (implementation not recorded)"


def load_report(path):
    return json.loads(Path(path).read_text())


LOAD_RE = re.compile(r"load averages?:\s*([\d.]+),?\s+([\d.]+),?\s+([\d.]+)")


def uptime():
    """Raw `uptime` output plus the parsed 1/5/15-minute load averages."""
    try:
        text = subprocess.run(
            ["uptime"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"raw": None, "error": str(error), "load": None}
    match = LOAD_RE.search(text)
    load = [float(v) for v in match.groups()] if match else None
    return {"raw": text, "load": load}
