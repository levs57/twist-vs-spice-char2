# Benchmark methodology

This file records how `ram-compare` measures the two RAM arguments and why each choice was made. It covers the harness items of issue #1 (§2, §3, §4, §6 and §8).

## Composed soundness

Each case reports a composed soundness error: the union bound over every PCS term and every IOP term, in `cases[].soundness`. The console prints the composed bits next to every timing, and `soundness.terms` lists each term with its probability, its bits and the verifier code it comes from.

**PCS terms.** The harness calls `om_pcs::ligerito::soundness_bits` on the parameters the provers actually use: `make_params(ell - 7, profile, log_rate)`, where `2^ell` is the committed bit count reported by the prover. Every level contributes four terms (sum-check, β-batching, proximity gap including its repetitions, queries), and all of them are summed.

**IOP terms.** For `twist` (Twist v2) the terms come from `ram_argument::twist2::soundness_terms`, which `docs/soundness-twist.md` derives; the harness adds the ring switch. For `spice-typed` and `spice-typed-2fp` they come from `spice_argument::operations::soundness_terms`, plus the ring switch. For `twist-legacy`, `crates/ram-compare/src/soundness.rs` contains a local fallback derived by reading the verifiers. A sum-check with `r` rounds of degree `d` contributes `r·d/2^128`; a powers-of-γ batch over `m` claims contributes `(m−1)/2^128`; a Schwartz–Zippel check of a multilinear identity at a random point in `K^v` contributes `v/2^128`. Fiat–Shamir and hash-collision terms are out of scope. The untyped `spice` and `spice-ripple` methods get only their grand-product fingerprint term and are marked `partial`.

At `n = 20`, `a = 16`, chunk 4, `provable-120`, log rate 2:

| Method | PCS | IOP | Composed | Dominant term |
| --- | --- | --- | --- | --- |
| `twist` (v2) | 2^-117.63 | 403/2^128 (2^-119.35) | 2^-117.25 | Ligerito query and batch terms, about 2^-120 per level |
| `twist-legacy` | 2^-117.63 | 650/2^128 (2^-118.66) | 2^-117.06 | Ligerito query and batch terms, about 2^-120 per level |
| `spice-typed` | 2^-117.64 | 1,114,914/2^128 (2^-107.91) | 2^-107.91 | Grand-product fingerprint (T+K)/2^128 = 1,114,112/2^128 |
| `spice-typed-2fp` | 2^-117.64 | 843/2^128 (2^-118.28) | 2^-116.92 | Ligerito query and batch terms; the fingerprint term becomes ((T+K)/2^128)^2 and the second GKR adds 41/2^128 |

The `twist-legacy` and `spice-typed` rows agree with the issue's estimates (about 2^-117 and 2^-107.9). The non-fingerprint IOP sums are somewhat lower than the issue's (650 against about 670 for the old Twist, 802 against about 1030 for typed Spice). The likely reasons: the GKR round messages carry four coefficients with the eq factor divided out, so the enforced degree is 4, not 5; the seven fixed inner coordinates of Flock's zero-check point add nothing, so its term is 6 + 20 − 7 = 19; and Twist's input identities share one random point, so their `w + n` term is counted once. None of these differences moves the composed totals.

The local derivation of typed Spice at 632855e, kept as `spice_typed_iop`, gave 899. Grouped Mosaic chunks cut the claim count from 126 to 19, which lowers the Mosaic batch term by 107; the crate's other terms add 10.

The Flock fixed coordinates cost nothing only if the 128 values `eq(c, x)` form an F2-basis of GF(2^128). The test `fixed_inner_equality_weights_form_an_f2_basis` in `crates/spice-argument/src/order_flock_tests.rs` checks this by elimination.

## Interleaved measurement and ratio intervals

The machine is shared and its load swings widely, so absolute times drift by tens of percent between runs (§8). Ratios measured close together in time are what count. The harness does not pin threads to cores or fix the CPU frequency, and macOS offers no portable way to do either. On a machine with performance and efficiency cores, a thread count above the number of performance cores puts some threads on efficiency cores; `environment.cores` records both counts next to `options.threads`.

**Within one run.** For each workload, `ram-compare` first runs every configuration's warmups, then `--repeat` measured rounds. Round `i` runs every configuration once, back to back. The order is forward on even rounds and reversed on odd rounds, so no configuration always runs first or always follows the same neighbour. Each sample records its round (`iteration`) and its position within the round (`position`). The final benchmarks should use `--repeat 10` or more.

**Ratio intervals.** For every pair of configurations on the same trace, `paired_ratios` reports the ratio of median prover times and of median verifier times, with a 95% percentile bootstrap interval. Each resample draws rounds with replacement and uses the same rounds for both configurations, so a slow round counts against both, which is the point of interleaving. The harness uses 10,000 resamples and a fixed SplitMix64 seed, so the interval is reproducible from the samples. The Twist-versus-Spice `comparisons` also carry these intervals. The percentiles interpolate linearly between order statistics. With few rounds the interval is coarse, because a median over five values can take only a few distinct values.

**Across binaries.** `scripts/interleave.py` alternates two `ram-compare` binaries (for example the upstream baseline and a branch) for N rounds on the same arguments, flipping their order every round. Each invocation runs `--warmup 1 --repeat 1` by default and writes its own JSON and log, and the script records `uptime` before and after each one. The summary takes each invocation's per-case median, then reports per case the median over rounds for each binary and the branch/baseline ratio with the same paired bootstrap. Cases pair by workload, method name and chunk size. The summary lists every case where the two binaries ran different provers under one method name (from `cases[].implementation`; a schema-4 `twist` is the prover now called `twist-legacy`), and it does not pair a case whose trace hashes differ. With `--repeat 1` each invocation runs its methods in forward order only, so within-invocation ratios between methods from these files have fixed positions; the branch/baseline ratios are unaffected. `scripts/bench_stats.py` implements the bootstrap with the same random stream as the Rust code, and both test suites check one shared example. The script reads schema-4 and schema-5 reports.

**Ablations.** `scripts/ablation.py` alternates any number of `ram-compare` binaries, reversing their order every round, and keeps each run's JSON. For each Twist v2 case it reports, per binary, the median over all samples of the prover time and of selected stages, with the ratio to the first binary, and the 1-minute load average before each binary's runs. It reports no intervals; use `interleave.py` for an interval on one pair of binaries.

**Building binaries to compare.** Build each binary in its own checkout with its own target directory, or run `cargo clean --release` before each build. When two checkouts of this workspace share one `CARGO_TARGET_DIR`, a build in one checkout can leave artifacts that a later build in the other treats as up to date. I hit this once: a binary built at one commit linked the ram-argument code of an earlier commit while it recorded the later commit in `environment.git_commit`. Every binary behind `results/m4max/round2` was built after `cargo clean --release`.

**Sweeps.** `scripts/sweep.py` runs one `ram-compare` invocation per point of a grid of `--address-bits`, `--log-cycles` and `--threads` values, with the remaining arguments passed through after `--`. It writes `a{A}-n{N}-t{T}.json` and `.log` per point, records `uptime` before and after each point in `index.json`, and summarises every point and case in `summary.json` and `summary.md`. Points run with N outermost so the cheap ones finish first; the script rewrites `index.json` after every point, a failed point does not stop the sweep, and `--skip-existing` resumes an interrupted one. Each point is its own process, so interleaving and ratio intervals apply within a point, not across points. The thread counts 1, 8 and 16 are the default.

## Workloads

**Why a synthetic locality trace.** The issue asks for one `2^20`-operation trace that does not repeat and has realistic locality. None of the three included guests can produce one. Their input buffer holds `2^12` words, so a guest run ends long before `2^20` memory operations, and the harness then repeats the template. Measured at `--address-bits 16 --log-cycles 18 --guest-rows memory` with the operations format:

| Guest and parameters | Operations before repeating | Distinct words in 2^18 operations | Reads |
| --- | --- | --- | --- |
| `erc20-mixed-block`, size 1024, 1000 transfers | 55,155 | 5,885 | 80.1% |
| `pointer-chase`, size 4000, 30,000 steps | 240,062 | 4,006 | 99.99% |
| `binary-search`, 3,800 keys, 255 queries | 30,109 | 1,427 | 92.4% |

All three have their median reuse distance in bucket 0 (the previous operation), largely because each narrow store becomes a read and a write of the same word. Pointer chase with 100,000 steps stops with `MaxCyclesExceeded`, because `guest_options` in `ram-compare` fixes `max_cycles` at `2^20` VM cycles, and even then it would be almost read-only. Writing a new guest program and widening the input ABI belongs to the payload owners, so this branch adds a generator instead.

**The `locality` pattern.** `ram_payload::locality::locality_operations(log_operations, address_bits, seed)` produces a seeded trace that mixes three streams, the usual ingredients of synthetic memory-trace models. The module documentation gives every parameter. In short:

- **Stack, 45% of operations, 65% reads.** The top 1/16 of memory is a downward-growing stack of frames of 2 to 16 words, which may fill half of it. Each stack access calls with probability 1/48, or otherwise returns with probability 1/48 when more than one frame exists, and touches the current frame at a geometric offset (p = 1/4), so a few slots are hot.
- **Sequential scans, 30% of operations, 2/3 reads.** Half of memory holds arrays. Four concurrent loops each scan `2^4` to `2^10` words from a random base, with stride 1 (90%) or 8 (10%). A new loop is read-only (60%), write-only (20%) or an in-place update (20%) that reads and then writes each word.
- **Heap, 25% of operations, 75% reads.** The other 7/16 of memory holds 8-word objects. Objects are drawn from a Zipf(s = 1) distribution over a seeded permutation, and fields from a geometric distribution (p = 1/2). Every `2^16` operations the hot set rotates.

The shares put the hot, short-reuse stack traffic first, as in compiled code, where most memory operations are spills, locals and return addresses. The 68% read fraction matches the usual 2:1 load-to-store ratio of integer workloads. Zipf with s = 1 is the standard model of object popularity, and the phase rotation keeps the trace from settling into one fixed working set. The parameters are judgement calls, not fitted to a measured program, so the pattern is a plausible locality profile rather than a model of any particular application.

At `--log-cycles 20 --address-bits 16` with seed 1 the trace had 68.2% reads and touched 55,062 of 65,536 words in one run on this branch; no test pins these figures, and the `locality` cases in `results/m4max/final/` record the histogram. Its median reuse distance fell in bucket 7 (128 to 255 operations), while the median for `random` fell in bucket 15. The reuse histogram is bimodal: one mode below 16 operations from stack slots and in-place updates, and one near `2^16` from scans and the Zipf tail.

**Reuse-distance histogram.** Every case's `metadata.reuse_distance` records, for the trace that was proved, the number of first accesses and `buckets[b]`, the number of accesses whose previous access to the same word was between `2^b` and `2^(b+1) - 1` operations earlier. `median_bucket` is the bucket holding the median repeated access. The histogram is computed over rows, so for the `rmw` format it counts each read-modify-write row once.

**Prover time per source memory instruction.** Per-row prover time is not comparable across trace formats or guest row policies, because a row can be a dummy (a non-memory VM instruction under `--guest-rows all`) or half of a narrow store (the operations format splits a byte, half-word or word store into a read and a full-word write). Every case therefore reports `throughput`: `operations` (T), `prover_ns_per_operation`, and for guest traces `source_memory_instructions` and `prover_ns_per_source_memory_instruction`, all from the median prover time. Source memory instructions are the guest loads and stores among the proved rows, before narrow-store expansion:

- operations format: `memory_operations - narrow_store_write_operations`, so a narrow store counts once and dummy rows do not count;
- `rmw` format: `guest.memory_access_cycles`, the VM rows whose instruction accesses memory.

If the trace ends between the two halves of a narrow store, that store counts once. Synthetic traces (`random`, `locality`) have no source instructions, and the field is null. The CSV carries the same three columns per sample, computed from that sample's prover time.

## Result schema and provenance

Reports use schema 5. Relative to schema 4 they add `cases[].implementation`, `cases[].final_memory`, `environment.argument_sources_sha256`, `environment.git_commit`, `environment.git_dirty`, `samples[].position`, `samples[].verifier_stages_ms`, `samples[].proof_size`, `samples[].load`, `cases[].load`, `cases[].proof_size`, `cases[].soundness`, `median.verifier_stages_ms`, `cases[].throughput`, `metadata.reuse_distance`, `paired_ratios`, and confidence intervals on `comparisons`. All schema-4 fields keep their meaning, except that `samples[].iteration` is now the measured round.

The report's `provenance.implementation_manifest` embeds `audit/stage16-implementation.json`, `environment.ram_compare_sources_sha256` hashes the harness sources compiled into the binary, `environment.argument_sources_sha256` hashes the `ram-argument` and `spice-argument` sources and `Cargo.lock` as the build script found them, and `environment.git_commit` and `environment.git_dirty` record the commit and whether tracked files differed from it at build time. The issue (§8) found that the Stage 1.4 and 1.5 manifests listed hashes from a later tree than the results they accompanied. To prevent that:

- `scripts/write_manifest.py` regenerates the manifest from the working tree: every file under the four comparison crates, `scripts/` and `payloads/open-magnetar/`, plus the workspace manifests, this document and the vendored-source manifests. Running it twice writes the same bytes, and `captured_on` changes only when a hash changes.
- `scripts/check_stage16.py` recomputes the manifest and fails unless the file set, every hash and the aggregate fingerprint match the tree.

Regenerate the manifest after the last source change and before the final build, so the binary embeds the manifest of the tree it was built from. `scripts/check_stage15.py` still checks the Stage 1.5 tree (632855e) and fails on the current one by design.
