# Soundness notes for Twist v2

These notes cover `crates/ram-argument/src/twist2`, which implements items 1–8 of §1 of the review issue. `twist2::soundness_terms(&Statement)` returns every term below as a probability. The field is `K = GF(2^128)`. A trace has `T = 2^n` cycles over `2^a` cells of `2^ℓ`-bit words, and addresses are split into `d = a / c` components of `c` bits. The PCS (Ligerito and the ring switch) and Fiat–Shamir terms are not included; the harness adds the PCS and ring-switch terms.

## What the verifier checks

The statement is the configuration, `n`, the initial memory `M0` and, when final memory is public, the final memory `Mf`. The prover commits to three kinds of bit tables in one PCS commitment:

- `A_i(k_i, j)` for each component `i`: one row of `2^c` bits per cycle, meant to be the unit vector of the cycle's address chunk;
- `Inc(o, j)`: bit `o` of the XOR difference that cycle `j` writes;
- `RV(o, j)`: bit `o` of the value cycle `j` reads.

When final memory is private, `Mf` is committed as a fourth table, `ram/final`.

Let `ra(k, j) = Π_i A_i(k_i, j)` and let `V(o, k, j) = M0(o, k) + Σ_{j' < j} ra(k, j') Inc(o, j')` be bit `o` of cell `k` before cycle `j`. After the commitment, the verifier draws `r_o ∈ K^ℓ` and `r_c ∈ (K \ {0})^n` and runs:

1. **Read check** (`read.rs`). One sum-check proves `RV(r_o, r_c) = Σ_{k, j} eq(r_c, j) ra(k, j) V(r_o, k, j)`, and ends with claims `ra(r_k, r_j)` and `V(r_o, r_k, r_j)` at a random point. The cycle rounds use the split-eq form: the prover sends a quadratic `q̃` and the verifier multiplies by the known eq factor.
2. **Cycle check** (`cycle.rs`). After drawing `γ`, `δ ≠ 0` and `ε_i ≠ 0`, one sum-check over the cycles proves the sum `(V − M0) + γ (Mf − M0) + δ ra + Σ_i ε_i π(r_k^i)` of four kinds of identity, all at `(r_o, r_k, r_j)`:
   - the value claim `V(r_o, r_k, r_j) = M0(r_o, r_k) + Σ_j LT(j, r_j) ra(r_k, j) Inc(r_o, j)`;
   - the final-memory claim `Mf(r_o, r_k) = M0(r_o, r_k) + Σ_j ra(r_k, j) Inc(r_o, j)`;
   - the virtualization `ra(r_k, r_j) = Σ_j eq(r_j, j) Π_i A_i(r_k^i, j)`;
   - the mirror test `Σ_j eq(r_j, j) A_i(r_k^i, j) A_i(1 + r_k^i, j) = π(r_k^i)` for each `i`, where `π(r) = Π_h r_h (1 + r_h)`.

   It ends with claims on `A_i(r_k^i, z)`, `A_i(1 + r_k^i, z)` and `Inc(r_o, z)`.
3. **Mosaic** proves all `2d + 2` claims (one more with private final memory) against the commitment: `RV(r_o, r_c)`, the `2d` component claims, `Inc(r_o, z)` and, if private, `Mf(r_o, r_k)`.

The verifier computes `M0(r_o, r_k)` and, when it is public, `Mf(r_o, r_k)` itself, from the words in the statement (`util::word_mle`).

## The changes and why each one is sound

**No write term (item 3).** Each cycle reads `RV` from its address and writes `RV ⊕ Inc` to the same address. The public statement contains only `M0` and `Mf`, so the written value appears in no check other than through `Inc`. The old protocol proved `wv = rv + Inc` about a `wv` that the prover built as exactly `rv + Inc` and that no later check used. Dropping that term changes no statement. The read check alone now proves that each read returns the current memory value.

**Exact one-hot rows from the mirror test (item 5).** The old protocol proved `Σ_k ra(k, j) = 1` and the address bits through the `Φ` term of the read-write sum-check. Twist v2 has no `Φ` and no address-bit claims. It relies on this fact instead:

> Over a field of characteristic 2, a vector `a ∈ K^{2^c}` satisfies `ã(r) ã(1 + r) = π(r)` as a polynomial identity in `r` if and only if `a` is a unit vector.

*Proof.* Induct on `c`. For `c = 0`, the identity is `a^2 = 1`, so `a = 1` in characteristic 2. For `c ≥ 1`, write `ã(r) = (1 + x) ã_0(r') + x ã_1(r')` with `x = r_c`. Since `1 + (1 + x) = x`, the product is

`(1 + x)^2 ã_0(r') ã_1(1 + r') + x^2 ã_1(r') ã_0(1 + r') + x (1 + x) [ã_0(r') ã_0(1 + r') + ã_1(r') ã_1(1 + r')]`.

The right side is `x (1 + x) π(r')`. The polynomials `(1 + x)^2`, `x^2` and `x (1 + x)` are linearly independent, so `ã_0(r') ã_1(1 + r') = 0`. The polynomial ring has no zero divisors, so `ã_0 = 0` or `ã_1 = 0`. The remaining half satisfies the identity in `c − 1` variables, so by induction it is a unit vector. ∎

The proof does not need the entries to be bits. Because `A_i` is committed bitwise, its rows are bit vectors anyway. If some row `j` of some `A_i` is not a unit vector, then `A_i(r, j) A_i(1 + r, j) − π(r)` is a nonzero polynomial of degree at most `2c` in `r`. The `eq(r_j, ·)`-weighted sum over rows is then a nonzero polynomial of degree at most `2c + n` in `(r_k^i, r_j)`. Both points are drawn after the commitment, so the sum vanishes with probability at most `(2c + n) / |K|`. This is the `mirror.point` term. The unit test `mirror_identity_accepts_one_hot_and_rejects_others` checks two-hot, zero and three-hot rows, and `cycle_check_rejects_two_hot_component` checks that the whole cycle check rejects a two-hot component. I also checked the identity exhaustively for every bit row with `c ≤ 3` over `GF(2^8)`, with a throwaway script that is not in the repository.

Once every `A_i` row is a unit vector, every `ra(·, j)` is the unit vector of cycle `j`'s address, which is all the read check and the cycle check need.

**Value and final memory at the same point (item 4).** The final-memory claim reuses `r_k` from the read check instead of a fresh point. `Mf − M0 − Σ_j ra Inc` is a polynomial fixed by the commitment and the statement, and `r_k` is drawn after both, so a false final memory survives with probability at most `(ℓ + a) / |K|` (`final.point`). The order of the read check (cycle-first or address-first) does not matter for this argument, because both `r_k` and `r_j` are fresh transcript challenges drawn after the commitment.

**Virtualization without claim reduction (item 6).** The cycle check proves `ra(r_k, r_j) = Σ_j eq(r_j, j) Π_i A_i(r_k^i, j)` directly. Since `Π_i A_i` is multilinear in `k`, this equals the MLE of `ra` at `(r_k, r_j)` exactly, so it adds no term beyond the cycle sum-check and the batching.

**Batching.** The four kinds of identity are combined with coefficients `(1, γ, δ, ε_1, …, ε_d)`, all drawn after the claims are fixed. A false identity makes the combined claim false except with probability `1 / (|K| − 1)` (`cycle.batch`). The prover folds `ε_i / δ` into the mirror column, which is an implementation detail and does not change the claim.

**Read check without `Φ` (items 1 and 2).** The summand is `eq(r_c, j) ra(k, j) V(r_o, k, j)`: degree 3 in each cycle variable and 2 in each address variable. Both binding orders produce the same proof shape and the same verifier. The order is absorbed into the statement transcript, so a proof made for one order does not verify for the other (tested in `rejects_modified_proof_and_statement`). Load merging (item 8) changes only how the address-first prover computes its messages. The test `load_merging_does_not_change_the_proof` checks that the proof is byte-identical with and without it.

**Private final memory (§5).** `Mf` is committed as `ram/final`. The prover sends `Mf(r_o, r_k)` before the cycle check's challenges, the verifier uses it in the cycle check's claim, and Mosaic checks it against the commitment. A wrong value either makes the cycle check's claim false or the Mosaic claim false.

**Nonzero `r_c`.** The cycle rounds of the read check divide by `r_c[i]`, so `r_c` is drawn from `K \ {0}`. The Schwartz–Zippel bound over `K \ {0}` is `deg / (|K| − 1)`. The terms below use `|K|` and ignore this factor of `1 + 2^-128`.

## Terms at n = 20, a = 16, c = 4, ℓ = 6

Numerators over `2^128`. The Mosaic `ell` is 28 at these parameters.

| term | formula | value | code |
|---|---|---|---|
| `read.point` | `ℓ + n` | 26 | `mod.rs` `post_commit_points` |
| `read.sumcheck` | `3n + 2a` | 92 | `read.rs` `verify` |
| `final.point` | `ℓ + a` | 22 | `mod.rs` `verify_inner` (`word_mle` at `r_k`) |
| `cycle.batch` | 1 | 1 | `cycle.rs` `challenges` |
| `cycle.sumcheck` | `(d + 2) n` | 120 | `cycle.rs` `verify` |
| `mirror.point` | `2c + n` | 28 | `cycle.rs` `mirror_rhs` |
| `mosaic.batch` | claims − 1 | 9 | `om_mosaic::batch_verify` |
| `mosaic.sumcheck` | `2 ell` | 56 | `om_mosaic::batch_verify` |
| **total** | | **354** | |

The harness adds the ring switch (`7 + 2(ell − 7) = 49`) for an IOP total of 403, against 650 for the old Twist at the same parameters. With PCS error `2^-117.63`, Twist v2 composes to `2^-117.25`, against `2^-117.06` for the old Twist. The unit test `soundness_terms_at_n20_a16` checks the total.

## What these notes do not cover

- Fiat–Shamir. Every term above is an interactive bound. The transcript absorbs every prover message before the next challenge, and the statement (configuration, order, layout, initial memory and public final memory) before the first.
- The PCS. Its terms come from the Ligerito `provable-120` profile, which the harness composes.
- Read ports other than one shared read/write address per cycle. `prove` rejects such traces.
