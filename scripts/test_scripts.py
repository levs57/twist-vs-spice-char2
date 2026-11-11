#!/usr/bin/env python3
"""Self-tests for the benchmark driver scripts. Run: python3 scripts/test_scripts.py"""

import itertools
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import bench_stats  # noqa: E402
import check_stage16  # noqa: E402
import interleave  # noqa: E402
import sweep  # noqa: E402
import write_manifest  # noqa: E402

FAKE = """#!{python}
import json, sys
args = sys.argv[1:]
out = args[args.index("--output") + 1]
order = "{order_log}"
with open(order, "a") as f:
    f.write("{label}\\n")
scale = {scale}
report = {{
    "schema_version": {schema},
    "cases": [
        {{"pattern": "random/operations", "method": "twist", "chunk_bits": 4,
          "median": {{"prover_ms": 100.0 * scale, "verifier_ms": 10.0}},
          "samples": [{{"prover_ms": 100.0 * scale, "verifier_ms": 10.0}}]}},
        {{"pattern": "random/operations", "method": "spice-typed", "chunk_bits": None,
          "median": {{"prover_ms": 40.0 * scale, "verifier_ms": 5.0}},
          "samples": [{{"prover_ms": 40.0 * scale, "verifier_ms": 5.0}}]}}
    ]
}}
json.dump(report, open(out, "w"))
"""


def fake_binary(directory, label, scale, schema, order_log):
    path = Path(directory) / f"fake-{label}"
    path.write_text(
        FAKE.format(
            python=sys.executable,
            order_log=order_log,
            label=label,
            scale=scale,
            schema=schema,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class BootstrapTests(unittest.TestCase):
    def test_matches_rust_stats(self):
        # crates/ram-compare/src/stats.rs asserts the same three numbers.
        num = [100.0, 104.0, 98.0, 120.0, 101.0, 99.0, 103.0, 97.0, 110.0, 102.0]
        den = [50.0, 49.0, 52.0, 51.0, 48.0, 50.0, 53.0, 47.0, 50.0, 49.0]
        r = bench_stats.paired_ratio(num, den)
        self.assertEqual(r["ratio"], 2.03)
        self.assertAlmostEqual(r["ci95_low"], 1.97, places=12)
        self.assertAlmostEqual(r["ci95_high"], 2.122448979591837, places=12)

    def test_schema_check(self):
        with self.assertRaises(ValueError):
            bench_stats.case_medians({"schema_version": 3, "cases": []})


class InterleaveTests(unittest.TestCase):
    def test_alternates_order_and_reads_schema_4_and_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = Path(tmp) / "order.log"
            base = fake_binary(tmp, "baseline", 1.0, 4, order)
            branch = fake_binary(tmp, "branch", 0.5, 5, order)
            out = Path(tmp) / "run"
            interleave.main(
                ["--baseline", base, "--branch", branch, "--rounds", "4",
                 "--out-dir", str(out), "--", "--methods", "twist,spice-typed"]
            )
            self.assertEqual(
                order.read_text().split(),
                ["baseline", "branch", "branch", "baseline"] * 2,
            )
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["rounds"], 4)
            self.assertEqual(len(summary["cases"]), 2)
            for case in summary["cases"]:
                r = case["prover_ms"]["branch_over_baseline"]
                self.assertEqual((r["ratio"], r["ci95_low"], r["ci95_high"]), (0.5, 0.5, 0.5))
                self.assertEqual(case["verifier_ms"]["branch_over_baseline"]["ratio"], 1.0)
            invocations = json.loads((out / "invocations.json").read_text())
            self.assertEqual(len(invocations), 8)
            self.assertTrue(all("raw" in inv["uptime_before"] for inv in invocations))
            self.assertIn("--warmup", invocations[0]["command"])
            self.assertIn("| random/operations twist (chunk=4) |", (out / "summary.md").read_text())
            # Re-analysis of the directory is deterministic.
            first = (out / "summary.json").read_text()
            interleave.main(["--analyze", str(out)])
            self.assertEqual((out / "summary.json").read_text(), first)

    def test_rejects_reserved_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                interleave.main(
                    ["--baseline", "x", "--branch", "y", "--out-dir", tmp, "--", "--repeat", "3"]
                )


SWEEP_FAKE = """#!{python}
import json, sys
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
a, n, t = int(value("--address-bits")), int(value("--log-cycles")), int(value("--threads"))
with open("{calls}", "a") as f:
    f.write(f"{{a}} {{n}} {{t}}\\n")
if a == {fail_at}:
    sys.exit(3)
case = {{"pattern": "random/operations", "method": "spice-typed", "chunk_bits": None,
         "median": {{"prover_ms": float(2 ** n) / t, "verifier_ms": 1.0}},
         "throughput": {{"prover_ns_per_operation": 1e6 / t}},
         "proof_size": {{"total_bytes": 1000 + n}}, "soundness": {{"bits": 110.5}},
         "samples": []}}
json.dump({{"schema_version": 5, "cases": [case]}}, open(value("--output"), "w"))
"""


class SweepTests(unittest.TestCase):
    def fake(self, tmp, fail_at=0):
        path = Path(tmp) / "fake-sweep"
        calls = Path(tmp) / "calls.log"
        path.write_text(SWEEP_FAKE.format(python=sys.executable, calls=calls, fail_at=fail_at))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path), calls

    def test_writes_one_json_per_point_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, calls = self.fake(tmp)
            out = Path(tmp) / "sweep"
            args = ["--binary", binary, "--address-bits", "12,16", "--log-cycles", "4,6",
                    "--threads", "1,8", "--out-dir", str(out), "--", "--methods", "spice-typed"]
            self.assertEqual(sweep.main(args), 0)
            ran = calls.read_text().split("\n")[:-1]
            # N outermost, then A, then T.
            self.assertEqual(ran[:4], ["12 4 1", "12 4 8", "16 4 1", "16 4 8"])
            self.assertEqual(len(ran), 8)
            for a, n, t in itertools.product((12, 16), (4, 6), (1, 8)):
                self.assertTrue((out / f"a{a}-n{n}-t{t}.json").exists())
            index = json.loads((out / "index.json").read_text())
            self.assertEqual(len(index), 8)
            self.assertTrue(all("raw" in e["uptime_before"] and "raw" in e["uptime_after"] for e in index))
            self.assertIn("--methods", index[0]["command"])
            summary = json.loads((out / "summary.json").read_text())
            row = next(r for r in summary["rows"] if (r["address_bits"], r["log_cycles"], r["threads"]) == (16, 6, 8))
            self.assertEqual((row["prover_ms"], row["proof_bytes"], row["soundness_bits"]), (8.0, 1006, 110.5))
            self.assertIn("| 16 | 6 | 8 | random/operations spice-typed | 8.00 |", (out / "summary.md").read_text())
            # Resuming runs nothing that already finished.
            self.assertEqual(sweep.main(args[:-3] + ["--skip-existing"]), 0)
            self.assertEqual(len(calls.read_text().split("\n")[:-1]), 8)

    def test_records_failures_and_rejects_reserved_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, _ = self.fake(tmp, fail_at=16)
            out = Path(tmp) / "sweep"
            code = sweep.main(["--binary", binary, "--address-bits", "12,16", "--log-cycles", "4",
                               "--threads", "1", "--out-dir", str(out)])
            self.assertEqual(code, 1)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["failed"], ["a16-n4-t1.json"])
            self.assertEqual(len(summary["rows"]), 1)
            with self.assertRaises(SystemExit):
                sweep.main(["--binary", binary, "--out-dir", str(out), "--", "--threads", "4"])
            with self.assertRaises(SystemExit):
                sweep.main(["--binary", binary, "--threads", "1,1"])


class ManifestTests(unittest.TestCase):
    def test_collect_is_deterministic_and_covers_harness_sources(self):
        first, second = write_manifest.collect(), write_manifest.collect()
        self.assertEqual(first, second)
        files = first["files_sha256"]
        for name in ("crates/ram-compare/src/main.rs", "scripts/write_manifest.py", "Cargo.lock"):
            self.assertIn(name, files)
        self.assertNotIn("audit/stage16-implementation.json", files)
        self.assertEqual(first["source_sha256"], write_manifest.fingerprint(files))

    def test_check_rejects_a_stale_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = write_manifest.collect()
            name = "crates/ram-compare/src/main.rs"
            stale["files_sha256"][name] = "00" * 32
            stale["source_sha256"] = write_manifest.fingerprint(stale["files_sha256"])
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(stale))
            original = check_stage16.MANIFEST
            check_stage16.MANIFEST = path
            try:
                with self.assertRaisesRegex(ValueError, "hash differs: " + name):
                    check_stage16.main()
            finally:
                check_stage16.MANIFEST = original


if __name__ == "__main__":
    unittest.main(verbosity=1)
