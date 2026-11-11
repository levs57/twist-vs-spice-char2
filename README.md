# Twist versus Spice

This repository compares the current optimized implementations of two RAM arguments over GF(2^128):

- **Twist:** Twist v2 (`ram_argument::twist2`, method `twist`): a read check with no write term, then one sum-check over the cycles that proves the value, final-memory, virtualization and one-hot claims together. It binds address variables first and merges loads between stores by default. `twist-legacy` runs the previous BinaryTwist prover with Flat virtualization and packed OneHot checks. [docs/soundness-twist.md](docs/soundness-twist.md) explains why the changes are sound.
- **Spice:** single-value read/write records (`spice-typed`), a batched masked-value reduction, leanVM-derived grand-product kernels and Flock's packed Boolean ordering argument.

Both use the same Mosaic, RingSwitch and Ligerito commitment stack. Each benchmark case uses identical ordered word operations and public initial/final memory. Every warmup and measured proof is independently verified.

## Run the comparison

Use a recent Rust toolchain with Cargo (tested with Rust 1.96). Python 3 is needed only for source checks. The proof kernels and guest ELFs are included; running the benchmark requires no upstream repository checkout or RISC-V compiler. Cargo downloads ordinary registry dependencies on the first build.

From the repository root:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run --locked --release -p ram-compare -- \
  --trace-format operations --methods twist,spice-typed \
  --patterns random,guest --threads 8 \
  --address-bits 16 --log-cycles 20 --chunk-bits 4 \
  --guest-program erc20-mixed-block,pointer-chase --guest-rows memory \
  --guest-fill repeat --warmup 1 --repeat 5 \
  --output results/comparison.json --csv results/comparison.csv
```

This runs both methods on random operations, an ERC20 mixed-block pattern and a pointer-chase pattern: six cases, each with one warmup and five measured proofs. The output directories are created automatically. Build time is excluded from benchmark timings.

| Parameter | Meaning |
| --- | --- |
| `--threads 8` | Eight Rayon worker threads. Cases and repetitions run sequentially. |
| `--address-bits 16` | Memory domain of `2^16` 64-bit words. |
| `--log-cycles 20` | Exactly `2^20` normalized RAM operations per trace. |
| `--chunk-bits 4` | Four-bit address chunks for Twist; must divide `address-bits`. |
| `--warmup 1 --repeat 5` | One verified warmup, then five verified samples per case. |
| `--patterns random,guest` | Run both random and guest-derived workloads. |
| `--guest-rows memory` | Keep actual memory accesses, excluding non-memory dummy rows. |
| `--guest-fill repeat` | Repeat the guest access pattern against continuous memory until the requested operation count is reached. |
| `--seed 1` | Deterministic random seed; default is `1`. |

For a smaller run, change `--log-cycles 20` to `18`. To check sensitivity to measurement order, repeat the command with `--methods spice-typed,twist` and different output filenames. `--patterns random` or `--patterns guest` selects one workload family, and `--patterns locality` adds a seeded synthetic trace with stack, scan and Zipf-heap locality that does not repeat (see [docs/methodology.md](docs/methodology.md#workloads)); `--guest-program` accepts a single program as well as a comma-separated list.

The comparison explicitly selects `--trace-format operations` and both methods, so it does not depend on executable defaults. Native CPU compilation enables the local machine's optimized kernels; rebuild when moving the binary to a different CPU.

## Workload and timing semantics

Random traces use independent uniform addresses and an unbiased read/write coin, with random 64-bit write values. Guest traces come from checked executions of the included ERC20 and pointer-chase programs. Full-word stores become writes; narrow stores become a read followed by a merged full-word write. Consequently, `log-cycles` counts normalized RAM operations rather than original instructions or VM cycles.

Guest patterns repeat without resetting memory. Reads observe the current state, and stores retain their original masks and source operands. The final repetition may be a prefix. This gives a valid ordered RAM trace with exactly a power-of-two operation count and no halted tail.

Twist receives the same operation sequence in its read-before-write representation: a read writes back its old value, and a write obtains the old value internally before replacing it. This adds no operation rows. Read/write kinds for Spice come from the source operation, including stores that leave a value unchanged.

The console prints per-case medians and individual phases. JSON contains every measured sample, workload metadata, trace hashes, commitment components, source/binary identity and paired comparisons. CSV provides one row per measured sample.

- **Prover:** complete measured prover time, including witness construction, packing, argument and commitment/opening work. Payload generation and result destruction are excluded.
- **Commitment:** the actual `pcs.commit` time for the padded commitment domain.
- **Opening:** RingSwitch plus Ligerito and the opening wrapper. Mosaic batching is reported separately.
- **IOP + preparation:** prover time excluding commitment and opening; it includes Mosaic, trace preparation and all RAM argument phases.
- **Populated/padded bits:** actual component data versus the power-of-two commitment allocation. This commitment padding is separate from trace rows.

Padding-adjusted estimates are reported separately from measured values and apply only to commitment time. Let `r = populated_bits / committed_bits`. Estimated commitment time is `commit_ms * r`, and estimated total prover time is `prover_ms - commit_ms * (1-r)`. Openings, Mosaic, ordering, all other sumchecks, packing, verification and unattributed time retain their measured costs. The trace already contains exactly `2^n` operations.

These are linear commitment-cost estimates, not measured timings of an unpadded implementation; fixed setup and logarithmic effects are approximated. JSON schema 5 keeps actual timings in the existing sample/median fields and puts estimates under `padding_adjusted_estimate`, with `prover_ms`, `commit_ms` and `removed_padding_ms`. CSV uses the `padding_adjusted_estimate:` prefix. Each adjusted sample is computed before its median, so independently reported medians need not add up exactly.

At `address-bits=16, log-cycles=20`, Twist has 24 MiB populated and 32 MiB padded commitment input; Spice has about 15.55 MiB populated and 16 MiB padded input. These are witness footprints, not proof sizes or total process memory.

Both methods use GF(2^128), the inherited `provable-120` PCS profile and `log-rate=2` by default. Common field/PCS parameters do not establish equal composed security: Spice's grand-product fingerprint error grows with the trace and memory sizes. Every case therefore reports composed soundness bits (PCS plus IOP terms), and `spice-typed-2fp` runs typed Spice with two independent fingerprints; [docs/methodology.md](docs/methodology.md) gives the numbers.

The proved statement is RAM consistency with public initial and final memory. It does not prove execution of the guest CPU program.

## Check the implementation

```sh
cargo test --locked --release --workspace
RUSTFLAGS='-C target-cpu=native' cargo test --locked --release --workspace
python3 scripts/check_vendor.py
python3 scripts/check_spice_port.py
python3 scripts/check_stage16.py
python3 scripts/test_scripts.py
```

`scripts/sweep.py` runs the same arguments over a grid of address bits, trace lengths and thread counts and writes one JSON per point; `scripts/interleave.py` alternates two binaries round by round. Both are described in [docs/methodology.md](docs/methodology.md). `scripts/check_stage16.py` requires `audit/stage16-implementation.json` to match the current tree exactly. After changing any source, run `python3 scripts/write_manifest.py` and rebuild, because each report embeds the manifest. `scripts/check_stage15.py` checks the earlier Stage 1.5 tree. [docs/methodology.md](docs/methodology.md) explains the measurement method, the schema and the soundness accounting.

The source checks use the included hash manifests and retained upstream licenses. The benchmark embeds provenance from [open-magnetar](vendor/open-magnetar/UPSTREAM.json), [lean-grandproduct](vendor/lean-grandproduct/UPSTREAM.json), [Flock](vendor/flock-boolean/UPSTREAM.json) and the [comparison implementation](audit/stage16-implementation.json).
