# Benchmark methodology

This file records how `ram-compare` measures the two RAM arguments and why each choice was made. It covers the harness items of issue #1 (§2, §3, §4, §6 and §8).

## Composed soundness

Each case reports a composed soundness error: the union bound over every PCS term and every IOP term, in `cases[].soundness`. The console prints the composed bits next to every timing, and `soundness.terms` lists each term with its probability, its bits and the verifier code it comes from.

**PCS terms.** The harness calls `om_pcs::ligerito::soundness_bits` on the parameters the provers actually use: `make_params(ell - 7, profile, log_rate)`, where `2^ell` is the committed bit count reported by the prover. Every level contributes four terms (sum-check, β-batching, proximity gap including its repetitions, queries), and all of them are summed.

**IOP terms.** The argument crates do not yet expose `soundness_terms`, so `crates/ram-compare/src/soundness.rs` contains a local fallback derived by reading the verifiers. A sum-check with `r` rounds of degree `d` contributes `r·d/2^128`; a powers-of-γ batch over `m` claims contributes `(m−1)/2^128`; a Schwartz–Zippel check of a multilinear identity at a random point in `K^v` contributes `v/2^128`. Fiat–Shamir and hash-collision terms are out of scope. The untyped `spice` and `spice-ripple` methods get only their grand-product fingerprint term and are marked `partial`.

At `n = 20`, `a = 16`, chunk 4, `provable-120`, log rate 2:

| Method | PCS | IOP | Composed | Dominant term |
| --- | --- | --- | --- | --- |
| `twist` | 2^-117.63 | 650/2^128 (2^-118.66) | 2^-117.06 | Ligerito query and batch terms, about 2^-120 per level |
| `spice-typed` | 2^-117.64 | 1,115,011/2^128 (2^-107.91) | 2^-107.91 | Grand-product fingerprint (T+K)/2^128 = 1,114,112/2^128 |

These agree with the issue's estimates (about 2^-117 and 2^-107.9). The non-fingerprint IOP sums are somewhat lower than the issue's (650 against about 670 for Twist, 899 against about 1030 for typed Spice). The likely reasons: the GKR round messages carry four coefficients with the eq factor divided out, so the enforced degree is 4, not 5; only the 13 random coordinates of Flock's zero-check point count, because 7 are fixed constants; and Twist's input identities share one random point, so their `w + n` term is counted once. None of these differences moves the composed totals.

**Caveat.** The Flock fixed coordinates cost nothing only if the 128 values `eq(c, x)` form an F2-basis of GF(2^128). The in-repo test checks a weaker property. This was argued on paper, not checked in code.

## Interleaved measurement and ratio intervals

The machine is shared and its load swings widely, so absolute times drift by tens of percent between runs (§8). Ratios measured close together in time are what count.

**Within one run.** For each workload, `ram-compare` first runs every configuration's warmups, then `--repeat` measured rounds. Round `i` runs every configuration once, back to back. The order is forward on even rounds and reversed on odd rounds, so no configuration always runs first or always follows the same neighbour. Each sample records its round (`iteration`) and its position within the round (`position`). The final benchmarks should use `--repeat 10` or more.

**Ratio intervals.** For every pair of configurations on the same trace, `paired_ratios` reports the ratio of median prover times and of median verifier times, with a 95% percentile bootstrap interval. Each resample draws rounds with replacement and uses the same rounds for both configurations, so a slow round counts against both, which is the point of interleaving. The harness uses 10,000 resamples and a fixed SplitMix64 seed, so the interval is reproducible from the samples. The Twist-versus-Spice `comparisons` also carry these intervals. The percentiles interpolate linearly between order statistics. With few rounds the interval is coarse, because a median over five values can take only a few distinct values.

**Across binaries.** `scripts/interleave.py` alternates two `ram-compare` binaries (for example the upstream baseline and a branch) for N rounds on the same arguments, flipping their order every round. Each invocation runs `--warmup 1 --repeat 1` by default and writes its own JSON and log, and the script records `uptime` before and after each one. The summary takes each invocation's per-case median, then reports per case the median over rounds for each binary and the branch/baseline ratio with the same paired bootstrap. `scripts/bench_stats.py` implements the bootstrap with the same random stream as the Rust code, and both test suites check one shared example. The script reads schema-4 and schema-5 reports.

## Result schema and provenance

Reports use schema 5. Relative to schema 4 they add `samples[].position`, `samples[].verifier_stages_ms`, `samples[].proof_size`, `samples[].load`, `cases[].load`, `cases[].proof_size`, `cases[].soundness`, `median.verifier_stages_ms`, `paired_ratios`, and confidence intervals on `comparisons`. All schema-4 fields keep their meaning, except that `samples[].iteration` is now the measured round.

The report's `provenance.implementation_manifest` embeds `audit/stage16-implementation.json`, and `environment.ram_compare_sources_sha256` hashes the harness sources compiled into the binary. The issue (§8) found that the Stage 1.4 and 1.5 manifests listed hashes from a later tree than the results they accompanied. To prevent that:

- `scripts/write_manifest.py` regenerates the manifest from the working tree: every file under the four comparison crates, `scripts/` and `payloads/open-magnetar/`, plus the workspace manifests, this document and the vendored-source manifests. Running it twice writes the same bytes, and `captured_on` changes only when a hash changes.
- `scripts/check_stage16.py` recomputes the manifest and fails unless the file set, every hash and the aggregate fingerprint match the tree.

Regenerate the manifest after the last source change and before the final build, so the binary embeds the manifest of the tree it was built from. `scripts/check_stage15.py` still checks the Stage 1.5 tree (632855e) and fails on the current one by design.
