# Twist with direct LogUp* address openings

`twist-logup-star` replaces the address encoding and its authentication in current optimized Twist v2. It keeps the existing read-check prover/verifier, virtual read values, increment representation, final-memory policy, load merging, and CPU interface. The default `twist` method is unchanged.

Run a matched comparison:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run --locked --release -p ram-compare -- \
  --trace-format operations --methods twist,twist-logup-star \
  --patterns random,guest --threads 8 \
  --address-bits 16 --log-cycles 20 --chunk-bits 4 \
  --guest-program erc20-mixed-block,pointer-chase \
  --guest-rows memory --guest-fill repeat --warmup 2 --repeat 20 \
  --output results/twist-logup-comparison.json \
  --csv results/twist-logup-comparison.csv
```

`--threads` sets the Rayon worker count, `--address-bits a` selects `2^a` memory words, and `--log-cycles n` selects exactly `2^n` operations. `--chunk-bits` applies to the onehot Twist baseline. The LogUp* method runs once per workload even when several chunk sizes are requested. Both methods support `--read-order address|cycle` and `--final-memory public|private`.

The reported memory words are **64 bits** (`word_log_bits=6`). The `u32` address storage type does not set the word width.

## Reduction

Write `T=2^n`, `N=2^a`, `f(j)=Eq(r_k,addr[j])`, and let `r_k,r_j` be the address and cycle points from the unchanged read check. Current Twist fuses value checking, final-memory checking, virtualization, and its onehot mirror test in one cycle sumcheck. There is consequently no isolated virtualization function to swap in that implementation.

The variant commits `a` Boolean address planes and Inc before the CPU point. It uses a cubic cycle sumcheck for

```text
sum_j (LT(j,r_j) + gamma) * f(j) * Inc(j)
      + delta * Eq(r_j,j) * f(j).
```

Its terminal claims are `Inc(z)` and `f(z)`. The increment claim uses the existing word opening. For the address claim, form

```text
M[k] = sum_{j: addr[j]=k} Eq(z,j).
```

Then `f(z) = M(r_k)` is directly an MLE evaluation. Commit M, sample the denominator challenge tau, and use the existing BitZ weighted fraction-tree reduction to prove

```text
sum_j Eq(z,j)/(tau + enc(addr[j]))
    = sum_{k<N} M[k]/(tau + enc(k)).
```

Authenticate the source denominator through the main committed address-bit planes. Authenticate M at both `r_k` and the fraction tree's right leaf through the auxiliary Mosaic/PCS opening. The source numerator is the public `Eq(z,.)`; the right denominator is the public address-index MLE. The auxiliary commitment precedes tau. No table inner-product sumcheck is needed.

This is the direct onehot-matrix opening construction in [Wiese, *Twist and Shout via logup\**, §4/Figure 1](https://www.powdr.org/papers/twist_shout_logup_star.pdf). The rational reduction comes from the existing extracted BitZ implementation of [LogUp\*, §4](https://levs57.github.io/logup_.pdf). Address labels use distinct polynomial-basis elements `K::from_u128(k)` in GF128.

The new cubic kernel uses coefficient statistics, split Eq/LT weights, paired CLMUL, deferred reductions, zero-increment skipping, fused folding/next-round accumulation, and reused buffers. Exact messages and transcript continuation match an independent generic sumcheck oracle. The existing fraction-tree kernels are reused without modification.

## Soundness

For rangechecked addresses, the rational identity's error is at most **`(N-1)/2^128`**, independent of T. It is not `1/N`.

Let `Delta[k]` be the difference between the true and claimed pushforward. Repeated accesses combine into one residue per address. Multiplying `sum_k Delta[k]/(tau+enc(k))` by `product_{k<N}(tau+enc(k))` gives a nonzero polynomial of degree at most N−1 whenever Delta is nonzero. Evaluation at a pole identifies its residue, establishing nonzeroness. Thus there are at most N−1 bad challenges. Binary address commitments provide the required range restriction. Table-pole challenges reject; they are an honest completeness abort, not false acceptance.

The full composed error also includes the read and cubic sumchecks, final-memory identity, fraction-tree reductions, both Mosaic reductions, and both RingSwitch/Ligerito instances. Fraction-tree terms still depend on `log T`. The harness reports the actual composition using each separately padded commitment domain. At a16/n20, measured configurations report approximately **111.92 composed bits** for LogUp* Twist and **117.28 bits** for current Twist. Matching field and PCS settings does not establish equal security.

## Measured comparison

Native x86_64 build, AVX2/VPCLMUL, i9-13980HX, eight worker threads pinned to logical CPUs 0–7. The main run uses public final memory and address-first read order, two warmups and 20 measured rounds. Method order reverses each round, both methods prove identical operation/RMW traces, and every proof and returned CPU claim is checked. CPU frequency was not fixed. These figures characterize this implementation and machine.

For a16/n20/c4, total prover medians in milliseconds:

| Payload | Twist | LogUp* Twist | LogUp*/Twist, paired 95% interval |
| --- | ---: | ---: | --- |
| Random | 209.707 | 231.883 | 1.106 [1.047, 1.132] |
| ERC20 mixed block | 123.661 | 134.765 | 1.090 [1.012, 1.137] |
| Pointer chase | 116.205 | 136.646 | 1.176 [1.146, 1.229] |

The new cycle sumcheck is much cheaper: random payload `twist.cycle` is about 7.5 ms, versus about 51.3 ms for the baseline cycle check including its child spans. However, the new variant also spends about 30.8 ms constructing fraction witnesses and 16.7 ms proving their rational relation, plus pushforward preparation and source-address claims. The per-proof median of the complete cycle/address preparation and IOP is **51.3 → 61.7 ms** for random, **46.1 → 52.0 ms** for ERC20, and **41.5 → 56.6 ms** for pointer chase. Auxiliary commitment/opening and Mosaic work is excluded from those IOP figures and remains charged in the total.

Random-payload total commitment time is **12.2 → 13.3 ms**, and total PCS opening time is **28.1 → 34.2 ms**. Verifier medians are **12.2 → 20.3 ms**. The additional PCS is material even though its input is smaller than the main commitment. Fraction-witness construction is the main remaining implementation target that could change the prover comparison.

Commitment pricing at the same parameters:

| Input | Twist | LogUp* Twist |
| --- | ---: | ---: |
| Address bits per access | 64 (four 16-bit onehot chunks) | 16 |
| Increment bits per access | 64 | 64 |
| Main populated input | 16 MiB | 10 MiB |
| Main padded input | 16 MiB | 16 MiB |
| Auxiliary populated/padded input | 0 | 1 MiB / 1 MiB |
| Total populated / padded input | 16 / 16 MiB | 11 / 17 MiB |

The fourfold address reduction saves logical witness area, but does not reduce the main commitment's power-of-two allocation at this shape. The auxiliary table costs 128N bits. `area.commitments` preserves both actual domains; aggregate committed bits need not be a power of two. Commitment-only padding estimates multiply each domain's measured commitment time by its own populated/padded fraction. Openings and sumchecks retain their full measured costs.

The main and auxiliary openings are separate proofs, each with its own Mosaic, RingSwitch, and Ligerito work. Profiling uses exclusive spans: `logup.aux.open` shows only wrapper overhead, while its expensive children are combined with the main opening under `pcs.ringswitch` and `pcs.ligerito_open`. `opening_ms` charges both full openings. Existing reports cannot separate the two child totals.

The auxiliary M table already contains full GF128 values, so direct field-PCS authentication could remove its bit-packing/RingSwitch adapter. Its two evaluation claims would still require a batching reduction or native multipoint opening. The main address/Inc commitment continues to use bit packing. This experiment retains the shared bit-PCS stack throughout.

Additional random-payload runs use one warmup and ten measured rounds:

| Shape | Twist prover (ms) | LogUp* prover (ms) | Ratio, paired 95% interval |
| --- | ---: | ---: | --- |
| a12/n20/c4 | 173.122 | 205.341 | 1.186 [1.126, 1.258] |
| a20/n18/c4 | 211.134 | 313.103 | 1.483 [1.385, 1.548] |

The a20/n18 case has a 16 MiB auxiliary input. Total padded commitment input grows from 8 MiB to 20 MiB, and the table-side fraction tree also grows. A smaller cycle-first/private-final run verifies that interface as well; it is a functional smoke check, not a stable timing result.

## Validation and artifacts

Both default and native release workspace suites pass **208 tests**. Ten new protocol tests cover independent scalar CPU claims and memory replay, both read orders, public/private boundaries, narrow/full values, supplied/derived reads, malformed/tampered proof boundaries, exact cubic messages against generic sumcheck through the parallel path, a histogram/MLE oracle, and a sharp N−1-root rational-error construction. Harness tests cover per-domain commitment charging and padding normalization. Existing Twist/Spice tests and soundness goldens remain unchanged. Nine Python driver tests and all source/provenance checks pass.

The campaign verifies **180 proofs**, including 164 measured proofs. The [validator](../audit/validate-twist-logup.py) checks input equality, independent area formulas, complete exclusive timing accounting, both commitment/opening costs, per-domain padding estimates, composed soundness, and matching source/executable identities.

Run `python3 audit/validate-twist-logup.py` to validate the retained reports and their common source/executable identity. Optionally supply `--binary /path/to/original/ram-compare` to check the original measured executable too. A rebuild after committing can have a different executable hash because the binary embeds Git metadata, even when its implementation source fingerprint is unchanged.

- [Main a16/n20 JSON](../results/twist-logup-a16-n20.json) and [CSV](../results/twist-logup-a16-n20.csv).
- [a12/n20](../results/twist-logup-a12-n20.json), [a20/n18](../results/twist-logup-a20-n18.json), and [cycle-first/private-final smoke](../results/twist-logup-smoke.json).
- [Validated summary and paired intervals](../results/twist-logup-summary.json).
- Source: [LogUp* adapter](../crates/ram-argument/src/twist2/logup.rs), [independent tests](../crates/ram-argument/src/twist2/logup_tests.rs), and [soundness composition](../crates/ram-compare/src/soundness.rs).

The final implementation fingerprint is `696db9798fc245d52a6d59fc9b9c3eecbe05c0fd9243dd2164b890b7a2cee94c`; every report embeds it. BitZ production sources and its manifest are now included in the comparison provenance. No vendored kernel was modified for this experiment. Build/test/benchmark logs are under `audit/logs/twist-logup-*`.
