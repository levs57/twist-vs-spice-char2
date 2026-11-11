# Soundness notes for Twist v2

These notes cover `crates/ram-argument/src/twist2`, which implements items 1–8 of §1 of the review issue and, since the third round of changes, keeps the read value virtual and hands the CPU word-level claims, as Jolt does. `twist2::soundness_terms(&Statement)` returns every term below as a probability. The field is `K = GF(2^128)`. A trace has `T = 2^n` cycles over `2^a` cells of `2^ℓ`-bit words (`ℓ = 6`), and addresses are split into `d = a / c` components of `c` bits. The PCS (Ligerito and the ring switch) and Fiat–Shamir terms are not included; the harness adds the PCS and ring-switch terms.

## What the verifier checks

The statement is the configuration, `n`, the initial memory `M0` and, when final memory is public, the final memory `Mf`. The prover commits to two kinds of bit tables in one PCS commitment:

- `A_i(k_i, j)` for each component `i` (`ram/ra0/{i+1}`): one row of `2^c` bits per cycle, meant to be the unit vector of the cycle's address chunk;
- `Inc(o, j)` (`ram/inc`): bit `o` of the XOR difference that cycle `j` writes.

When final memory is private, `Mf` is committed as a third table, `ram/final`. The read value is not committed. At `n = 20, a = 16, c = 4` the tables hold `4 · 16 + 64 = 128` bits per cycle, exactly `2^27`.

Words enter the field as `enc(x) = Σ_o x_o X^o` (`word::enc`, `K::from_u128`). `enc` is F2-linear and injective. For a column of words `x_j`, `W[x](r) = Σ_j eq(r, j) enc(x_j)` is its word MLE. A bit table `B` holding the bits of the words opens at the fixed point `p_ℓ` of `word::encoding_point` as `B̃(p_ℓ ‖ r) = W[x](r) / φ_ℓ`, so Mosaic proves word-level claims about bit tables.

Let `ra(k, j) = Π_i A_i(k_i, j)`, `Inc_w(j) = enc(Inc_j)` and `V(k, j) = W[M0](k) + Σ_{j' < j} ra(k, j') Inc_w(j')`, the word in cell `k` before cycle `j`. The transcript runs:

1. **CPU point and claims** (`cpu_point`, `cpu_claims`). After the commitment, the verifier draws `r_c ∈ (K \ {0})^n`. The prover sends `v_addr`, `v_rv` and `v_wv`, meant to be `W[addr](r_c)`, `W[rv](r_c)` and `W[wv](r_c)`, where `wv_j = rv_j ⊕ Inc_j` is the word after cycle `j`. The verifier then draws `γ_a`.
2. **Read check with the address fold** (`read.rs`). One sum-check proves `v_rv + γ_a v_addr = Σ_{k, j} eq(r_c, j) ra(k, j) V'(k, j)`, where `V'(k, j) = V(k, j) + γ_a ẽnc(k)` and `ẽnc(k) = Σ_b k_b X^b`. `V'` is the memory table started from `M0'(k) = enc(M0[k]) + γ_a enc(k)` with the same increments, because `γ_a enc(k)` does not depend on `j`. The check ends with claims `ra(r_k, r_j)` and `V'(r_k, r_j)`, and the verifier sets `V(r_k, r_j) = V'(r_k, r_j) + γ_a ẽnc(r_k)` (`word::enc_mle`). The cycle rounds use the split-eq form: the prover sends a quadratic `q̃` and the verifier multiplies by the known eq factor.
3. **Cycle check** (`cycle.rs`). After drawing `γ`, `δ ≠ 0` and `ε_i ≠ 0`, one sum-check over the cycles proves the sum `(V − W[M0]) + γ (W[Mf] − W[M0]) + δ ra + Σ_i ε_i π(r_k^i)` of four kinds of identity, all at `(r_k, r_j)`:
   - the value claim `V(r_k, r_j) = W[M0](r_k) + Σ_j LT(j, r_j) ra(r_k, j) Inc_w(j)`;
   - the final-memory claim `W[Mf](r_k) = W[M0](r_k) + Σ_j ra(r_k, j) Inc_w(j)`;
   - the virtualization `ra(r_k, r_j) = Σ_j eq(r_j, j) Π_i A_i(r_k^i, j)`;
   - the mirror test `Σ_j eq(r_j, j) A_i(r_k^i, j) A_i(1 + r_k^i, j) = π(r_k^i)` for each `i`, where `π(r) = Π_h r_h (1 + r_h)`.

   It ends with claims on `A_i(r_k^i, z)`, `A_i(1 + r_k^i, z)` and `Inc_w(z)`.
4. **Mosaic** proves all `2d + 2` claims (one more with private final memory) against the commitment: the `2d` component claims, `Ĩnc(p_ℓ ‖ z) = Inc_w(z) / φ_ℓ`, the write claim `Ĩnc(p_ℓ ‖ r_c) = (v_wv + v_rv) / φ_ℓ` and, if private, `M̃f(p_ℓ ‖ r_k) = W[Mf](r_k) / φ_ℓ`.

The verifier computes `W[M0](r_k)` and, when it is public, `W[Mf](r_k)` itself, from the words in the statement (`word::mle`). An accepting verifier returns `Interface { point: r_c, claims }`.

## What Twist v2 proves

Suppose the verifier accepts with probability above the sum of the terms below plus the PCS terms. Then the committed tables decode to `addr ∈ [2^a]^T`, where each `A_i(·, j)` is the unit vector of chunk `i` of `addr_j`, and to `Inc ∈ ({0,1}^64)^T`. Define `M_0 = M0` and `M_{j+1} = M_j` with `M_{j+1}[addr_j] = M_j[addr_j] ⊕ Inc_j`, and let `rv_j = M_j[addr_j]` and `wv_j = rv_j ⊕ Inc_j`. Then `M_T = Mf`, `v_addr = W[addr](r_c)`, `v_rv = W[rv](r_c)` and `v_wv = W[wv](r_c)`.

The statement does not distinguish loads from stores: any cycle may change its cell.

## The CPU interface and the load constraint

A RAM argument inside a zkVM serves a CPU proof. Jolt commits only `RamInc` and the one-hot `RamRa` chunks. The read value, the write value and the address are virtual: the CPU's constraint system consumes them at a random cycle point, and the RAM argument proves them there. Twist v2 now has the same interface. The harness plays the CPU: `crates/ram-compare/src/interface.rs` recomputes `W[addr]`, `W[rv]` and `W[wv]` from the trace at the returned point and fails the run on a mismatch. A real CPU proof would compare its own columns at `r_c` with the three claims, which by Schwartz–Zippel makes the CPU's columns equal the RAM's except with probability `3n / |K|`. Typed Spice pays the same term, so it stays out of both crates' terms and out of the comparison (`docs/methodology.md`).

**The write claim.** The previous protocol proved nothing about `wv`, which was sound only because nothing consumed `wv`. Once the CPU consumes it, Twist must prove it again. Mosaic proves `Ĩnc(p_ℓ ‖ r_c) = (v_wv + v_rv) / φ_ℓ`. Because `enc` is F2-linear, this gives `v_wv = v_rv + W[Inc](r_c) = W[rv ⊕ Inc](r_c) = W[wv](r_c)`. Upstream `BinaryTwist` does the same: it opens `Inc` at the read point with value `wv + rv`. The claim adds no IOP term beyond one more Mosaic claim, and it replaces the old `RV(r_o, r_c)` claim, so the claim count is unchanged.

**The load constraint lives in the CPU.** Jolt enforces `RamReadEqRamWriteIfLoad` (a load leaves memory unchanged) in its R1CS, not in the RAM argument. Here the CPU stand-in owns it for both methods: for Twist, `Load_j ⇒ rv_j = wv_j` (that is, `Inc_j = 0`), and for typed Spice, `is_write_j = Store_j`. With these constraints both statements give the same semantics: a load returns the current word and changes nothing, and a store replaces the word. Inside Twist, a load flag would need one more committed bit per cycle, which pushes `2^27` to `2^28` at the shipped point.

Two tests show where the constraint is enforced. In `rejects_tampered_inc_on_load_with_honest_claims`, the prover sets `Inc_j = δ` on a load and cancels it at the cell's next access, so the final memory still matches. With the honest claims the RAM verifier rejects. In `tampered_inc_on_load_with_consistent_claims_fails_the_stand_in`, the claims follow the tampered witness, the RAM verifier accepts, and the stand-in check against the honest trace rejects. `relabelled_load_address_fails_the_stand_in` moves a load to another cell holding the same word: the RAM verifier accepts and the stand-in rejects on the address.

## The changes and why each one is sound

**Address fold and read batch.** Once every row of `ra` is a unit vector, `Σ_k ra(k, j) V'(k, j) = enc(rv_j) + γ_a enc(addr_j)`, so the read check's true sum is `W[rv](r_c) + γ_a W[addr](r_c)`. The prover fixes `v_rv` and `v_addr` before `γ_a` is drawn. If either is wrong, `v_rv + γ_a v_addr` equals the true sum for at most one `γ_a` (`read.batch`). The fold costs the prover at most `2^n + 2^a` field additions. The convention for `ẽnc(r_k)` must match the one used for `W[M0](r_k)`; the tests `enc_mle_matches_bruteforce` and `cpu_claims_match_trace` (both read orders, both final-memory modes) pin it.

**Exact one-hot rows from the mirror test (item 5).** The old protocol proved `Σ_k ra(k, j) = 1` and the address bits through the `Φ` term of the read-write sum-check. Twist v2 has no `Φ` and no address-bit claims. It relies on this fact instead:

> Over a field of characteristic 2, a vector `a ∈ K^{2^c}` satisfies `ã(r) ã(1 + r) = π(r)` as a polynomial identity in `r` if and only if `a` is a unit vector.

*Proof.* Induct on `c`. For `c = 0`, the identity is `a^2 = 1`, so `a = 1` in characteristic 2. For `c ≥ 1`, write `ã(r) = (1 + x) ã_0(r') + x ã_1(r')` with `x = r_c`. Since `1 + (1 + x) = x`, the product is

`(1 + x)^2 ã_0(r') ã_1(1 + r') + x^2 ã_1(r') ã_0(1 + r') + x (1 + x) [ã_0(r') ã_0(1 + r') + ã_1(r') ã_1(1 + r')]`.

The right side is `x (1 + x) π(r')`. The polynomials `(1 + x)^2`, `x^2` and `x (1 + x)` are linearly independent, so `ã_0(r') ã_1(1 + r') = 0`. The polynomial ring has no zero divisors, so `ã_0 = 0` or `ã_1 = 0`. The remaining half satisfies the identity in `c − 1` variables, so by induction it is a unit vector. ∎

The proof does not need the entries to be bits. Because `A_i` is committed bitwise, its rows are bit vectors anyway. If some row `j` of some `A_i` is not a unit vector, then `A_i(r, j) A_i(1 + r, j) − π(r)` is a nonzero polynomial of degree at most `2c` in `r`. The `eq(r_j, ·)`-weighted sum over rows is then a nonzero polynomial of degree at most `2c + n` in `(r_k^i, r_j)`. Both points are drawn after the commitment, so the sum vanishes with probability at most `(2c + n) / |K|`. This is the `mirror.point` term. The unit test `mirror_identity_accepts_one_hot_and_rejects_others` checks two-hot, zero and three-hot rows, and `cycle_check_rejects_two_hot_component` checks that the whole cycle check rejects a two-hot component. I also checked the identity exhaustively for every bit row with `c ≤ 3` over `GF(2^8)`, with a throwaway script that is not in the repository.

Once every `A_i` row is a unit vector, every `ra(·, j)` is the unit vector of cycle `j`'s address, which is all the read check and the cycle check need.

**Value and final memory at the same point (item 4).** The final-memory claim reuses `r_k` from the read check instead of a fresh point. `W[Mf](k) − W[M0](k) − Σ_j ra(k, j) Inc_w(j)` is a multilinear polynomial in `k` fixed by the commitment and the statement, and `r_k` is drawn after both, so a nonzero one vanishes at `r_k` with probability at most `a / |K|` (`final.point`). If it is zero on `{0,1}^a`, the injectivity and F2-linearity of `enc`, with `ra ∈ {0,1}`, give `Mf[k] = M0[k] ⊕ (⊕_{j: addr_j = k} Inc_j)`. The term was `ℓ + a` while the memory claims were bit-level at `(r_o, r_k)`. The order of the read check (cycle-first or address-first) does not matter for this argument, because both `r_k` and `r_j` are fresh transcript challenges drawn after the commitment.

**Virtualization without claim reduction (item 6).** The cycle check proves `ra(r_k, r_j) = Σ_j eq(r_j, j) Π_i A_i(r_k^i, j)` directly. Since `Π_i A_i` is multilinear in `k`, this equals the MLE of `ra` at `(r_k, r_j)` exactly, so it adds no term beyond the cycle sum-check and the batching.

**Batching.** The four kinds of identity are combined with coefficients `(1, γ, δ, ε_1, …, ε_d)`, all drawn after the claims are fixed. A false identity makes the combined claim false except with probability `1 / (|K| − 1)` (`cycle.batch`). The prover folds `ε_i / δ` into the mirror column, which is an implementation detail and does not change the claim.

**Read check without `Φ` (items 1 and 2).** The summand is `eq(r_c, j) ra(k, j) V'(k, j)`: degree 3 in each cycle variable and 2 in each address variable. Both binding orders produce the same proof shape and the same verifier. The order is absorbed into the statement transcript, so a proof made for one order does not verify for the other (tested in `rejects_modified_proof_and_statement`). Load merging (item 8) changes only how the address-first prover computes its messages. The test `load_merging_does_not_change_the_proof` checks that the proof is byte-identical with and without it.

**Private final memory (§5).** `Mf` is committed as `ram/final`. The prover sends `W[Mf](r_k)` before the cycle check's challenges, the verifier uses it in the cycle check's claim, and Mosaic checks it against the commitment at `(p_ℓ ‖ r_k)`. A wrong value either makes the cycle check's claim false or the Mosaic claim false. Jolt does not commit final memory (`RamValFinal` is virtual and checked by `RamOutputCheck`). These notes keep `Mf` committed in both methods so the comparison stays symmetric.

**Cycle prover (second round of changes).** The cycle check's prover (`cycle_prover.rs`) changed in four ways, and none of them changes the protocol. The verifier, the transcript and the proof are the same, and `soundness_terms` is unchanged.

- *Split eq.* The prover no longer stores `δ eq(r_j, ·)`. In round `i` it uses `eq(r_j, (z_{<i}, X, m)) = eq(r_{<i}, z_{<i}) · eq(r_i, X) · eq(r_{>i}, m)`, accumulates the eq-weighted part (degree `d` in `X`) over split tables for `eq(r_{>i}, ·)` and multiplies by the known linear factor once per round.
- *Split LT.* The prover no longer stores `LT(·, r_j) + γ`. With the low variables least significant, `LT((z_{<i}, X, m), r_j) = LT(m, r_{>i}) + eq(m, r_{>i}) · LT((z_{<i}, X), r_{≤i})`, and the last factor is a linear polynomial in `X` that the prover tracks with one running scalar.
- *Round-0 tables.* Before binding, `A_i(r_k^i, j)` and `A_i(1 + r_k^i, j)` take one of `2^c` values, so on a pair of rows they are fixed by the pair's two chunk values. For `c ≤ 5` the prover reads round 0 from per-component tables over all `2^{2c}` pairs.
- *Inc skipping.* When `Inc` is zero on both rows of a pair, the `Inc` term vanishes at every node and the prover skips it. With `inc_e = enc(Inc)`, `inc_e = 0` exactly when `Inc = 0`. A tampered "load" with `Inc ≠ 0` counts as a store, which is only a prover-side choice.

These are identities about the same round polynomials, so an honest prover sends exactly the messages it sent before. The test `cycle_prover_matches_the_reference_prover` checks that the new prover's messages and evaluations equal those of the reference row prover in `cycle.rs` (which keeps both columns), across shapes that cover the sequential and parallel paths, chunks with and without round-0 tables, and traces with about half loads. The test `rejects_tampered_cycle_messages_from_the_split_prover` changes each value of each round message and each final evaluation of a proof from the new prover and checks that the verifier rejects.

**Nonzero `r_c`.** The cycle rounds of the read check divide by `r_c[i]`, so `r_c` is drawn from `K \ {0}`. Typed Spice draws its CPU point the same way. The Schwartz–Zippel bound over `K \ {0}` is `deg / (|K| − 1)`. The terms below use `|K|` and ignore this factor of `1 + 2^-128`.

**No `read.point` term.** It bounded the error of testing the committed `RV` against the read check at `(r_o, r_c)`. There is no committed `RV` now, and `r_o` is gone: every word table is opened at the fixed point `p_ℓ`.

## Terms at n = 20, a = 16, c = 4, ℓ = 6

Numerators over `2^128`. The Mosaic `ell` is 27 at these parameters with public final memory.

| term | formula | value | code |
|---|---|---|---|
| `read.batch` | 1 | 1 | `mod.rs`, `γ_a` (`ram/address_batch`) |
| `read.sumcheck` | `3n + 2a` | 92 | `read.rs` `verify` |
| `final.point` | `a` | 16 | `mod.rs` `verify_inner` (`W[M0]`, `W[Mf]` at `r_k`) |
| `cycle.batch` | 1 | 1 | `cycle.rs` `challenges` |
| `cycle.sumcheck` | `(d + 2) n` | 120 | `cycle.rs` `verify` |
| `mirror.point` | `2c + n` | 28 | `cycle.rs` `mirror_rhs` |
| `mosaic.batch` | claims − 1 = `2d + 1` | 9 | `om_mosaic::batch_verify` |
| `mosaic.sumcheck` | `2 ell` | 54 | `om_mosaic::batch_verify` |
| **total** | | **321** | |

The harness adds the ring switch (`7 + 2(ell − 7) = 47`) for an IOP total of 368, against 403 for Twist v2 with a committed read value and 650 for the old Twist at the same parameters. With PCS error `2^-117.64` at `ell = 27`, Twist v2 composes to `2^-117.28`, against `2^-117.25` with a committed read value and `2^-117.06` for the old Twist. The unit test `soundness_terms_at_n20_a16` checks the total (`267 + 2 ell`), and `composed_bits_at_the_shipped_parameters` in `crates/ram-compare/src/soundness.rs` checks 368 and `2^-117.28`.

With private final memory, `ell = 28`, `mosaic.batch = 10`, `mosaic.sumcheck = 56` and the ring switch is 49. The crate total is 324 and the IOP total 373.

## Implementation notes

These choices in the code paths change neither the protocol nor a soundness term.

- The address shift `γ_a enc(k)` is applied in `mod.rs` when the read witness is built, not inside `read.rs`. `ReadWitness` has no `address_batch` field.
- The cycle-first read order needs the shifted per-cycle values `enc(rv_j) + γ_a enc(addr_j)`. The address-first order does not use them, so the prover builds them only for cycle-first.
- `word::claim_at` returns the point and value of a word claim, and each crate wraps them in its own claim type.
- The test hook that edits the CPU claims changes only the claims the prover sends and absorbs. The prover's own Mosaic claims keep the computed values, so the vendored Mosaic prover does not panic, and the verifier is what rejects. Typed Spice has the same hook for its CPU evaluations.

## What these notes do not cover

- Fiat–Shamir. Every term above is an interactive bound. The transcript absorbs every prover message before the next challenge, and the statement (configuration, order, layout, initial memory and public final memory, under the label `ram-argument/twist-v2/virtual-read`) before the first.
- The CPU-side Schwartz–Zippel term `3n / |K|`, which both methods share.
- The PCS. Its terms come from the Ligerito `provable-120` profile, which the harness composes.
- Read ports other than one shared read/write address per cycle. `prove` rejects such traces.
