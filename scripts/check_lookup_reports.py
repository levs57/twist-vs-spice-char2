#!/usr/bin/env python3
"""Validate recorded lookup benchmark accounting and paired execution orders.

This is read-only. It checks JSON, all recorded samples, optional CSV, source
identities, and comparison arithmetic; it does not regenerate private traces or
reverify proof objects (the benchmark verifies every proof before recording it).
Archived reports need not match today's sources. --check-current additionally
checks their embedded provenance against this workspace's current source bytes.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
HASH = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
MEASUREMENTS = (
    "prove_ms", "verify_ms", "main_commit_ms", "auxiliary_commit_ms",
    "main_open_ms", "auxiliary_open_ms", "commit_ms", "opening_ms", "remaining_ms",
)
PHASE_MEASUREMENTS = ("prep_ms", "argument_ms", "virtualization_ms")
AREAS = (
    "lookups", "table_rows", "core_bits", "argument_bits", "main_committed_bits",
    "main_padding_bits", "auxiliary_bits", "auxiliary_committed_bits",
    "auxiliary_padding_bits", "total_committed_bits", "main_claims", "auxiliary_claims",
)
ORDERED_OPTIONS = {"methods", "patterns", "chunk_bits", "virtualization", "guest_program", "guest_rows"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{name}: expected integer >= {minimum}")
    return value


def number(value, name, minimum=0):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum, f"{name}: invalid finite number")
    return value


def near(actual, expected, name):
    require(isinstance(actual, (int, float)) and math.isfinite(actual), f"{name}: invalid number")
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7),
            f"{name}: {actual!r} != {expected!r}")


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            f"invalid provenance source path: {name}")
    return path


def hash_map(hashes, name):
    require(isinstance(hashes, dict) and hashes, f"{name}: missing source inventory")
    for path, digest in hashes.items():
        safe_relative(path)
        require(isinstance(digest, str) and HASH.fullmatch(digest), f"{name}: invalid SHA256")


def fingerprint(hashes):
    result = hashlib.sha256()
    for name, digest in sorted(hashes.items()):
        result.update(name.encode() + b"\0")
        result.update(bytes.fromhex(digest))
    return result.hexdigest()


def current_files(base, hashes):
    for name, expected in hashes.items():
        path = base / safe_relative(name)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"current source differs: {path}")


def provenance(report, check_current):
    metadata = report["provenance"]
    om = metadata["open_magnetar"]
    bitz = metadata["bitz_logup"]
    for name, entry in [("OM", om), ("BitZ", bitz)]:
        require(REVISION.fullmatch(entry["revision"]), f"{name}: invalid revision")
        hash_map(entry["local_sha256"], name)
    require(set(om["local_sha256"]) == set(om["permitted_modified_files"]) | set(om["added_files"]),
            "OM local source inventory differs from permitted/added files")
    require(fingerprint(om["local_sha256"]) == om["local_patch_sha256"], "OM patch aggregate differs")
    require(fingerprint(bitz["local_sha256"]) == bitz["local_source_sha256"], "BitZ source aggregate differs")
    hash_map(om["baseline_sha256"], "OM baseline")
    require(set(om["permitted_modified_files"]).issubset(om["baseline_sha256"]), "OM allowed source lacks baseline")
    if "adapter" in metadata:
        adapter = metadata["adapter"]
        hash_map(adapter["files_sha256"], "lookup adapter")
        require(fingerprint(adapter["files_sha256"]) == adapter["source_sha256"], "adapter aggregate differs")
        require(adapter["implementation"] == report["implementation"], "adapter implementation differs")
        require(adapter["open_magnetar_patch_sha256"] == om["local_patch_sha256"], "adapter/OM identity differs")
        require(adapter["bitz_logup_source_sha256"] == bitz["local_source_sha256"], "adapter/BitZ identity differs")
        if check_current:
            current_files(ROOT, adapter["files_sha256"])
            require(adapter == json.loads((ROOT / "audit/stage13-implementation.json").read_text()),
                    "embedded adapter manifest differs from current manifest")
    if check_current:
        for entry, folder in [(om, ROOT / "vendor/open-magnetar"), (bitz, ROOT / "vendor/bitz-logup")]:
            require(entry == json.loads((folder / "UPSTREAM.json").read_text()),
                    f"embedded provenance differs from current {folder}")
            hashes = {**entry.get("baseline_sha256", {}), **entry["local_sha256"]}
            current_files(folder, hashes)
    require(HASH.fullmatch(report["environment"]["executable_sha256"]), "invalid executable SHA256")


def expected_patterns(options):
    result = []
    for name in options["patterns"]:
        if name == "guest":
            result.extend(f"guest/{program}/{rows}" for program in options["guest_program"] for rows in options["guest_rows"])
        else:
            require(name in {"random", "local"}, f"unknown pattern: {name}")
            result.append(name)
    require(result and len(result) == len(set(result)), "duplicate/empty pattern selection")
    return result


def case_key(case):
    return case["pattern"], case["method"], case["chunk_bits"], case["virtualization"]


def expected_keys(options):
    keys = []
    for pattern in expected_patterns(options):
        for method in options["methods"]:
            if method == "logup":
                keys.append((pattern, "logup-star", None, None))
            else:
                require(method == "shout", f"unknown method: {method}")
                keys.extend((pattern, "shout", c, v) for c in options["chunk_bits"] for v in options["virtualization"])
    require(len(keys) == len(set(keys)), "duplicate configuration selection")
    return keys


def padded(bits):
    return max(1 << 14, 1 << (bits - 1).bit_length())


def check_area(case, options):
    area = case["areas"]
    for key in AREAS:
        integer(area[key], f"area.{key}")
    a = options["table_bits"]
    t, n = 1 << options["log_lookups"], 1 << a
    core = t * (64 + a)
    if case["method"] == "shout":
        c = integer(case["chunk_bits"], "chunk_bits", 1)
        require(a % c == 0, "Shout chunk width does not divide address width")
        require(case["virtualization"] in {"flat", "gkr", "legacy-flat"}, "unknown virtualization")
        d = a // c
        require(case["virtualization"] != "gkr" or d & (d - 1) == 0, "non-power-of-two GKR")
        argument, auxiliary, aux_commit = d * (1 << c) * t, 0, 0
        main_claims, auxiliary_claims = a + 1 + 3 * d, 0
        require(case["logup_extra_bound"] is None, "Shout has a Logup bound")
    else:
        require(case["method"] == "logup-star" and case["chunk_bits"] is None
                and case["virtualization"] is None, "noncanonical Logup configuration")
        argument, auxiliary = 0, 128 * n
        aux_commit = padded(auxiliary)
        main_claims, auxiliary_claims = a + 1, 2
        bound = case["logup_extra_bound"]
        require(bound["conservative_degree_numerator"] == t + n and bound["field_bits"] == 128,
                "Logup conservative degree/field bound differs")
        near(bound["negative_log2_bound"], 128 - math.log2(t + n), "Logup bound bits")
        require(math.isclose(bound["probability_bound"], (t + n) / 2**128, rel_tol=1e-12, abs_tol=0),
                "Logup probability bound differs")
        require(isinstance(bound["scope"], str) and bound["scope"], "missing bound scope")
    expected = dict(lookups=t, table_rows=n, core_bits=core, argument_bits=argument,
                    main_committed_bits=padded(core + argument), auxiliary_bits=auxiliary,
                    auxiliary_committed_bits=aux_commit, main_claims=main_claims,
                    auxiliary_claims=auxiliary_claims)
    expected["main_padding_bits"] = expected["main_committed_bits"] - core - argument
    expected["auxiliary_padding_bits"] = aux_commit - auxiliary
    expected["total_committed_bits"] = expected["main_committed_bits"] + aux_commit
    require(area == expected, f"area/claim formulas differ for {case_key(case)}: {area} != {expected}")


def check_sample(sample, method):
    for name in MEASUREMENTS:
        number(sample[name], name)
    require(sample["prove_ms"] > 0 and sample["verify_ms"] > 0, "nonpositive proof/verify duration")
    near(sample["commit_ms"], sample["main_commit_ms"] + sample["auxiliary_commit_ms"], "commit components")
    near(sample["opening_ms"], sample["main_open_ms"] + sample["auxiliary_open_ms"], "opening components")
    near(sample["prove_ms"], sample["commit_ms"] + sample["opening_ms"] + sample["remaining_ms"], "inclusive prover accounting")
    stages = sample["exclusive_stage_ms"]
    require(isinstance(stages, dict) and stages, "missing exclusive stages")
    for name, value in stages.items():
        require(isinstance(name, str) and name, "invalid stage name")
        number(value, name)
    mandatory = {"lookup.statement", "lookup.pack", "lookup.main.commit", "lookup.main.open",
                 "lookup.mosaic", "pcs.ringswitch", "pcs.ligerito_open"}
    if method == "logup-star":
        mandatory |= {"lookup.aux.commit", "lookup.aux.open", "lookup.logup.rational", "lookup.logup.table_product"}
    else:
        mandatory |= {"shout.onehot", "shout.read", "shout.virtualization"}
        near(sample["auxiliary_commit_ms"], 0, "Shout auxiliary commit")
        near(sample["auxiliary_open_ms"], 0, "Shout auxiliary open")
    require(mandatory.issubset(stages), f"missing phase metrics: {mandatory - stages.keys()}")
    phases = {
        "prep_ms": sum(value for name, value in stages.items()
                       if name in {"lookup.witness", "lookup.pack", "lookup.statement", "lookup.claims"}),
        "argument_ms": sum(value for name, value in stages.items()
                           if name.startswith(("shout.", "virt.", "lookup.logup"))),
        "virtualization_ms": sum(value for name, value in stages.items()
                                 if name == "shout.virtualization" or name.startswith("virt.")),
    }
    # These fields were added without changing schema-v1. Older reports retain
    # their recorded stages and are valid without the derived phase subtotals.
    for name, expected in phases.items():
        if name in sample:
            number(sample[name], name)
            near(sample[name], expected, name)
    total = sum(stages.values())
    near(sample["exclusive_stages_total_ms"], total, "exclusive stage total")
    near(sample["unattributed_prove_ms"], sample["prove_ms"] - total, "unattributed time")
    require(sample["unattributed_prove_ms"] >= -1e-7, "exclusive stages exceed inclusive prover time")


def check_cases(report):
    options = report["options"]
    a, log_t = integer(options["table_bits"], "table_bits", 1), integer(options["log_lookups"], "log_lookups")
    require(a <= 31 and log_t <= 31, "invalid table/lookup dimensions")
    integer(options["threads"], "threads", 1)
    repeat, warmup = integer(options["repeat"], "repeat", 1), integer(options["warmup"], "warmup")
    require(options["profile"] in {"provable-120", "conjectured-100"}, "unknown PCS profile")
    minimum = 2 if options["profile"] == "provable-120" else 1
    require(minimum <= integer(options["log_rate"], "log_rate") <= 8, "invalid PCS rate")
    expected = expected_keys(options)
    require([case_key(c) for c in report["cases"]] == expected, "case order/coverage differs from ordered options")
    by_pattern = {}
    for case in report["cases"]:
        require(HASH.fullmatch(case["trace_sha256"]), "invalid trace SHA256")
        identity = (case["trace_sha256"], case["source_metadata"])
        require(case["pattern"] not in by_pattern or by_pattern[case["pattern"]] == identity,
                "method/chunk cases do not share the same input hash and metadata")
        by_pattern[case["pattern"]] = identity
        check_area(case, options)
        samples = case["samples"]
        require(len(samples) == repeat and [s["iteration"] for s in samples] == list(range(repeat)),
                "raw sample count/iterations differ from repeat")
        for sample in samples:
            check_sample(sample, case["method"])
        medians = case["median"]
        for name in MEASUREMENTS:
            near(medians[name], statistics.median(s[name] for s in samples), f"median.{name}")
        for name in (*PHASE_MEASUREMENTS, "unattributed_prove_ms"):
            if name in medians or any(name in s for s in samples) and name in PHASE_MEASUREMENTS:
                require(name in medians and all(name in s for s in samples),
                        f"inconsistent optional phase field: {name}")
                near(medians[name], statistics.median(s[name] for s in samples), f"median.{name}")
        stage_keys = set().union(*(s["exclusive_stage_ms"] for s in samples))
        require(set(medians["exclusive_stage_ms"]) == stage_keys, "median stage keys differ")
        for name in stage_keys:
            near(medians["exclusive_stage_ms"][name], statistics.median(s["exclusive_stage_ms"].get(name, 0) for s in samples), f"median stage {name}")
    # Warmups have no raw timing samples. Their declared count is retained and
    # used for the expected verification count, not invented as measured rows.
    measured = len(expected) * repeat
    verified = len(expected) * (repeat + warmup)
    if "verified_proofs" in report:
        require(report["verified_proofs"] == verified, "reported verification count differs")
    return measured, verified


def check_pairs(report):
    cases = report["cases"]
    expected = {}
    for s, shout in enumerate(cases):
        if shout["method"] != "shout":
            continue
        matches = [i for i, case in enumerate(cases) if case["method"] == "logup-star" and case["pattern"] == shout["pattern"]]
        if not matches:
            continue
        require(len(matches) == 1, "duplicate Logup case in one pattern")
        l = matches[0]
        logup = cases[l]
        require(shout["trace_sha256"] == logup["trace_sha256"] and shout["areas"]["core_bits"] == logup["areas"]["core_bits"],
                "paired inputs/core areas differ")
        expected[(s, l)] = {
            "logup_prover_speedup": shout["median"]["prove_ms"] / logup["median"]["prove_ms"],
            "logup_verifier_speedup": shout["median"]["verify_ms"] / logup["median"]["verify_ms"],
            "shout_to_logup_commit_ratio": shout["median"]["commit_ms"] / logup["median"]["commit_ms"],
            "shout_to_logup_opening_ratio": shout["median"]["opening_ms"] / logup["median"]["opening_ms"],
            "shout_to_logup_committed_area_ratio": shout["areas"]["total_committed_bits"] / logup["areas"]["total_committed_bits"],
        }
    seen = set()
    for pair in report["comparisons"]:
        key = pair["shout_case"], pair["logup_case"]
        require(key in expected and key not in seen, "unknown or duplicate paired comparison")
        seen.add(key)
        require(pair["trace_sha256"] == cases[key[0]]["trace_sha256"], "paired hash differs")
        for name, ratio in expected[key].items():
            near(pair[name], ratio, name)
    require(seen == set(expected), "missing paired comparisons")


def csv_path(report_path, value):
    path = Path(value)
    candidates = [path] if path.is_absolute() else [ROOT / path, report_path.parent / path, report_path.parent / path.name]
    return next((p for p in candidates if p.is_file()), candidates[0])


def check_csv(report, path):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        names, rows = reader.fieldnames, list(reader)
    require(names and len(names) == len(set(names)), "missing or duplicate CSV columns")
    stages = set().union(*(sample["exclusive_stage_ms"] for case in report["cases"] for sample in case["samples"]))
    require({name.removeprefix("exclusive:") for name in names if name.startswith("exclusive:")} == stages,
            "CSV exclusive stage columns differ")
    expected = {(i, sample["iteration"]): (case, sample) for i, case in enumerate(report["cases"]) for sample in case["samples"]}
    require(len(rows) == len(expected), "CSV sample count differs")
    seen = set()
    for row in rows:
        require(None not in row and all(v is not None for v in row.values()), "CSV row width differs")
        key = int(row["case"]), int(row["iteration"])
        require(key in expected and key not in seen, "CSV unknown/duplicate sample")
        seen.add(key)
        case, sample = expected[key]
        for name in ["method", "pattern", "trace_sha256"]:
            require(row[name] == case[name], f"CSV {name} differs")
        for name in ["chunk_bits", "virtualization"]:
            require(row[name] == ("" if case[name] is None else str(case[name])), f"CSV {name} differs")
        for name in AREAS:
            require(int(row[name]) == case["areas"][name], f"CSV area {name} differs")
        for name in MEASUREMENTS:
            near(float(row[name]), sample[name], f"CSV {name}")
        for name in (*PHASE_MEASUREMENTS, "exclusive_stages_total_ms", "unattributed_prove_ms"):
            if name in row:
                require(name in sample, f"CSV {name} lacks JSON sample field")
                near(float(row[name]), sample[name], f"CSV {name}")
        # Earlier schema-v1 smoke CSV lacked these global columns. Check every
        # one that is present; current CSV emits them all.
        for name in ["table_bits", "log_lookups", "threads", "seed", "profile", "log_rate"]:
            if name in row:
                require(row[name] == str(report["options"][name]), f"CSV option {name} differs")
        for name in stages:
            near(float(row[f"exclusive:{name}"]), sample["exclusive_stage_ms"].get(name, 0), f"CSV stage {name}")
        bits = row["logup_extra_bound_bits"]
        if case["logup_extra_bound"] is None:
            require(bits == "", "CSV Shout bound must be empty")
        else:
            near(float(bits), case["logup_extra_bound"]["negative_log2_bound"], "CSV Logup bound")


def validate(path, check_current=False, csv_override=None, no_csv=False):
    report = json.loads(path.read_text())
    require(report["schema_version"] == 1 and report["implementation"] == "gf128-indexed-rom-lookup-v1", "unknown lookup report schema")
    provenance(report, check_current)
    measured, verified = check_cases(report)
    check_pairs(report)
    csv_name = csv_override or report["options"].get("csv")
    if csv_name and not no_csv:
        check_csv(report, csv_path(path, csv_name))
    print(f"PASS {path}: {len(report['cases'])} cases, {measured} raw samples, {verified} declared verified proofs including warmups")
    return report


def compare_order(first, second):
    def fixed_options(options):
        result = copy.deepcopy(options)
        result.pop("output", None)
        result.pop("csv", None)
        for name in ORDERED_OPTIONS:
            if name in result:
                result[name] = sorted(result[name])
        return result
    require(fixed_options(first["options"]) == fixed_options(second["options"]), "fixed options differ between execution orders")
    require(first["provenance"] == second["provenance"], "source provenance differs between execution orders")
    require(first["environment"] == second["environment"], "executable/hardware environment differs between execution orders")
    def fixed_cases(report):
        result = {}
        for case in report["cases"]:
            fixed = copy.deepcopy(case)
            fixed.pop("samples")
            fixed.pop("median")
            require(case_key(case) not in result, "duplicate case key")
            result[case_key(case)] = fixed
        return result
    require(fixed_cases(first) == fixed_cases(second), "inputs, source metadata, areas or bounds differ between execution orders")
    changed = [case_key(c) for c in first["cases"]] != [case_key(c) for c in second["cases"]]
    print(f"PASS paired execution orders: identical fixed options, source identities, inputs and areas; order_changed={changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--compare-order", type=Path, help="second report to compare with the single positional report")
    parser.add_argument("--csv", type=Path, help="override CSV path for one positional report")
    parser.add_argument("--no-csv", action="store_true", help="explicitly skip CSV verification")
    parser.add_argument("--check-current", action="store_true", help="also verify embedded provenance against current source files")
    args = parser.parse_args()
    if (args.compare_order or args.csv) and len(args.reports) != 1:
        parser.error("--compare-order and --csv require one positional report")
    if args.csv and args.no_csv:
        parser.error("--csv and --no-csv cannot be combined")
    try:
        reports = [validate(p.resolve(), args.check_current, args.csv, args.no_csv) for p in args.reports]
        if args.compare_order:
            other = validate(args.compare_order.resolve(), args.check_current, no_csv=args.no_csv)
            compare_order(reports[0], other)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        raise SystemExit(f"lookup report validation failed: {error}") from error


if __name__ == "__main__":
    main()
