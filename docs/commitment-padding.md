# Commitment padding in Twist v2

**Status.** This note describes Twist v2 with a committed read value, the version benchmarked in `results/m4max/final-committed-rv` on the `like-for-like-results` branch. Twist v2 now keeps the read value virtual and commits only the address components and `Inc`: `4 · 2^24 + 2^26 = 2^27` bits at `n = 20, a = 16, c = 4`, with no padding at all. Padding still matters for private final memory (`2^27 + 2^22` bits, padded to `2^28`, for both methods) and for `a > 16` at `c = 4` (Twist v2 needs `2^{n+8}`). The analysis below of what the zero region costs and what a PCS would need to skip it still applies to those cases.

At `n = 20, a = 16, c = 4`, Twist v2 commits `192 · 2^20 = 3 · 2^26` bits per trace: four address components of `2^24` bits, `Inc` and `RV` of `2^26` bits each. Mosaic and the PCS pad this to `2^28` bits. Typed Spice commits about `2^26.96` bits, padded to `2^27`. This note records what removing Twist v2's padding would take and what it would save. I did not implement it: the vendored PCS cannot skip the padding through its public API, and the one layout change that the public API allows saves about 1 ms and changes the soundness accounting.

## Where the padding comes from

`om_mosaic::compute_layout` (`vendor/open-magnetar/crates/om-mosaic/src/mosaic.rs`, lines 87–122) places the components back to back in decreasing size and sets `ell` to the next power of two above their total. For Twist v2 the order is `Inc` and `RV` at offsets 0 and `2^26`, then the four address components from `2^27`. The zero region is the top quarter of the `2^28` hypercube. Mosaic lifts every claim into this one `ell`-variable hypercube, and the PCS requires a power-of-two table:

- `ligerito::commit_stripped` asserts `table.len() == 1 << p.n` (`om-pcs/src/ligerito.rs`, line 615);
- the ring switch asserts `bits.len() == 1 << ell` (`om-pcs/src/ring_switch.rs`, line 174).

## What the zero region costs

The commitment RS-encodes every row of the Ligerito matrix and hashes every column, and the zero region sits in the column variables, so each row is a quarter zeros and no row is entirely zero. The NTT does not skip zero inputs. The ring switch's `a_table` and product sum-check and the first Ligerito sum-check also run over the full table. Only the ring switch's partial evaluations (`partial_evals_packed`) and Mosaic's bit phase walk the populated bits alone.

The example `crates/ram-argument/examples/pcs_sizes.rs` times `BitPCS::commit_packed`, `open_packed` and `verify` in isolation (best of 4, load 6–9, 8 threads):

| bits | commit | open | verify | Ligerito proof | PCS soundness |
|---|---|---|---|---|---|
| `2^26` | 9.7 ms | 15.8 ms | 4.70 ms | 266,720 B | 117.639 bits |
| `2^27` | 15.3 ms | 23.4 ms | 4.81 ms | 316,992 B | 117.637 bits |
| `2^28`, top quarter zero | 26.8 ms | 38.5 ms | 5.04 ms | 372,480 B | 117.635 bits |
| `2^28`, full | 26.8 ms | 39.4 ms | 4.84 ms | 365,600 B | 117.635 bits |

The zero quarter costs as much as populated data.

## Options

**Two PCS instances (public API, not implemented).** Mosaic could keep one batch at `ell = 28`, and the prover could commit `G[0, 2^27)` and `G[2^27, 2^27 + 2^26)` as two separate Ligerito instances. The verifier would check `v_G = (1 + r_27) v_A + r_27 (1 + r_26) v_B` at Mosaic's output point, which is exact. From the table, commit drops from 26.8 ms to 25.0 ms and open rises from 38.5 ms to 39.2 ms, a net saving of about 1 ms. The PCS proof grows by about 211 kB and the PCS verifier time doubles. The soundness accounting also changes: the composed bound adds a second Ligerito term and a second ring-switch term, about one bit worse. Because it changes the soundness accounting for a negligible saving, I did not implement it.

**Skipping the zero region (needs vendor changes).** The saving needs a PCS that commits a non-power-of-two first dimension, or one that knows which rows are zero:

1. permute the variables of `G` so that the zero quarter becomes whole Ligerito rows, which this repository can do;
2. change `commit_stripped`, `ntt::encode_coset_col_major` and the column hashing to skip those rows, and have the verifier treat them as zero;
3. change `rs_prove_from_s`, `a_table` and the first Ligerito sum-check to skip the zero region.

The soundness terms would not change, because the committed polynomial and the opening point are the same.

**Estimated saving.** Interpolating the table linearly between `2^27` and `2^28` gives about 21 ms commit and 31 ms open at `3 · 2^26` bits, against 26.8 ms and 38.5 ms padded: about 13 ms in isolation. This is an estimate, not a measurement of an unpadded PCS. In the full prover the PCS stages run slower than in isolation (`pcs.commit` is about 40 ms in the benchmark runs), so the saving there is probably 13–17 ms, about 8–10% of Twist v2's prover time at these parameters. The harness's `padding_adjusted_estimate` models only the commit part of this saving.
