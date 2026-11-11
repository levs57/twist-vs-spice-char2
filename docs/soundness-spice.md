# Soundness notes for the typed Spice changes

These notes cover the protocol changes on the `spice-work` branch. `operations::soundness_terms(&Statement, &Options)` returns every term below as a probability, with a comment at each term naming the code it bounds. The field is `K = GF(2^128)`. `T = 2^n` is the number of operations and `M = 2^a` the number of cells. The PCS and Fiat–Shamir terms are not included; the harness adds them.

## Terms at n = 20, a = 16

Numerators over 2^128:

| term | spice-typed | spice-typed-2fp | code |
|---|---|---|---|
| fingerprint | (2^20 + 2^16), i.e. 2^-107.91 | its square, 2^-215.8 | `fingerprints`, `check_products` |
| GKR batching / rounds / child points | 11 / 400 / 21 | 11 / 440 / 22 | `lean_grandproduct::verify` |
| masked reduction batching / rounds | 1 / 60 | 1 / 60 | `masked_reduction::verify_sumcheck` |
| Flock zerocheck / skip / rounds | 19 / 127 / 40 | 19 / 127 / 40 | `order_flock::verify_zerocheck_with` |
| ρ chunk combination | 4 | 4 | `finish_claims`, `group_rho` |
| Mosaic batching / rounds | 21 / 54 | 21 / 54 | `om_mosaic::batch_verify` |
| everything but the fingerprint | 758, i.e. 2^-118.4 | 799, i.e. 2^-118.4 | |
| total | 2^-107.91 | 2^-118.4 | |

These totals exclude the PCS and the ring switch. `ram-compare` adds both and reports 2^-107.91 composed with one fingerprint and 2^-116.92 with two.

With private final memory the Mosaic terms are 22 / 56 (one more claim, one more variable) and the total is unchanged at 2^-107.91. The issue's estimate of about 2^-107.9 for typed Spice is correct. The fingerprint term dominates it by a factor of about 1,400.

The unit test `soundness_terms_match_the_issue_estimate` checks these numbers. The non-fingerprint terms come from the code as follows:

- **GKR.** Each layer costs one batching challenge for the two trees, a degree-4 round per bound coordinate (the normalized quartic, so the equality factor adds nothing), and a child point of 1 or 2 coordinates.
- **Masked reduction.** Three claims are batched with independent weights, followed by `l` cubic rounds.
- **Flock.** The term is `6 + 20 − 7 = 19` random equality coordinates. The seven fixed inner coordinates add nothing, because their 128 equality weights are F_2-independent (tested by `fixed_inner_equality_weights_form_an_f2_basis`), so a Boolean gate error never cancels over them. The skip polynomial has degree below 128, and there are 20 quadratic rounds.
- **Mosaic.** Batching the `c` claims with powers of one challenge costs `c − 1`. The sum-check adds two per committed variable. At `a = 16` there are 22 claims, three of them for the CPU interface below.

## CPU interface claims

Twist v2 hands the CPU word-level claims at a random cycle point (`docs/soundness-twist.md`), so typed Spice does the same, and the comparison charges both for the interface. After the commitment, the verifier draws `r_c ∈ (K \ {0})^n` (`spice/typed/cpu_point`). The prover sends the `a` address planes at `r_c`, `W[value](r_c)` and `Σ_j eq(r_c, j) is_write_j`, and the transcript absorbs them (`spice/typed/cpu_claims`) before the fingerprints are drawn. The verifier returns `Interface { point, address, value, is_write }` with `address = combine_bits(planes)`, which is exact.

Three Mosaic claims bind the values to the commitment: the address planes at `r_c` through the ρ-combined chunk claim (one chunk at `a = 16`, two at `a = 12` or `a = 20`), `word_claim("input/value", r_c, value)`, and `is_write` as plane 0 of `input/flags` at `r_c ‖ 0`. The values are fixed before every later challenge, so a wrong one makes a Mosaic claim false, which adds only to `mosaic.batch`. The ρ term is unchanged because the largest chunk is still 16 planes. The test `mosaic_rejects_wrong_cpu_evaluations_sent_before_absorption` sends a wrong value, flag or address plane through a prover hook, and the verifier rejects at Mosaic; `rejects_tampered_cpu_evaluations` edits each value of a finished proof.

**Cost.** Mosaic claims go from 19 to 22, `mosaic.batch` from 18 to 21, and the non-fingerprint totals from 755 / 796 to 758 / 799. The composed figures are unchanged at two decimals. The proof grows by `(a + 2) · 16` bytes, 288 at `a = 16`.

**The load constraint.** The CPU must check `is_write_j = Store_j`. Twist leaves the matching check (`Load_j ⇒ Inc_j = 0`) to the CPU as well, so neither RAM argument proves it. The CPU-side Schwartz–Zippel term `3n / |K|` is the same for both methods and stays out of both.

## Two fingerprints

**Claim.** If the two multisets of each fingerprint set differ, the verifier accepts with probability at most `((T + M) / |K|)^2`.

Each set `i` draws `(γ_i, α_i, β_i, δ_i)` after the commitment, from the transcript. Each set multiplies `T + M` affine forms `γ + ts + α·addr + β·val + δ·flag` per side. Two different multisets of tuples give two different products of linear polynomials, by unique factorization. Their difference is a nonzero polynomial of total degree `T + M`, so Schwartz–Zippel bounds one set's collision by `(T + M)/|K|`. The two sets use independent challenges and the verifier checks each set's equality separately, so the bounds multiply.

The verifier gets the per-set products from the GKR's first-layer children, not from the roots. With `log_f = 1` each tree is `[set 0 | set 1]`. The top layer's children are the per-set products when that layer is binary, and pairs of halves when it is radix-4. The GKR binds these children like any other layer's, so a prover that sends wrong per-set products is caught by the GKR terms above. `check_products` rejects equal overall products with unequal per-set products (unit test `per_set_products_come_from_the_first_layer`).

**Why not an extension field.** An extension of degree 2 over `K` would give the same `(T + M)/|K|^2`, but it would need extension arithmetic in the GKR, the leaves and the leaf binding. Two sets cost one more GKR variable and reuse every kernel. The prover measured 7% slower at n=20.

## ρ-combined chunk claims

**Claim.** If any per-plane value in the evaluation message is wrong, the Mosaic claims are false except with probability `k_max/|K|`, where `k_max` is the largest chunk's log size (4 here).

A chunk of `2^k` planes is one component of log size `plane_log + k`, with the plane index in the high coordinates. Its MLE at `point ∥ ρ[..k]` equals `Σ_i eq(ρ[..k], i) · v_i(point)` exactly. The verifier forms this sum from the claimed `v_i`. If some `v_i` is wrong, the difference between the claimed and true sums is a nonzero multilinear polynomial in `ρ[..k]` of degree `k`. The prover fixes every `v_i` before ρ is drawn, because `absorb_evaluations` runs before `group_rho`. A bad event needs every wrong chunk to vanish at ρ, so its probability is at most that of one wrong chunk, `k/|K| ≤ k_max/|K|`.

The fused evaluation message (item 3) is sound for the same reason. The points are fixed by earlier challenges, so the true value of each committed polynomial at its point is determined. The prover sends a claimed value, the transcript absorbs it before any later challenge, and the verifier uses it both in its deferred terminal checks (`check_leaves`, `Deferred::check`, `check_sources`) and in a Mosaic claim. A wrong value either fails a deferred check deterministically or makes a Mosaic claim false.

## Batching the two sum-checks

This was not done, so it adds no term. `docs/spice-changes.md` §4 explains why.

## Private final memory

**Claim.** With `private_final_memory`, an accepting proof shows that a committed vector `F ∈ {0,1}^{64·M}` exists such that the operations are consistent with initial memory `I`, and that `F` is the final memory. The error is the same as above, plus one Mosaic claim and one Mosaic variable.

The final words are part of the single commitment, which the transcript absorbs before the fingerprints are drawn, so `F` is fixed before any challenge. The memory-side left tuple still uses `final_k`. In `check_leaves`, its value coordinate at the leaf point is the claimed `final_value`, and the Mosaic claim `word_claim("boundary/final", leaf[..a], final_value)` binds it to `F̃(leaf[..a])`. The ρ term does not change, because this is a direct claim. If `final_value` is wrong, the Mosaic claim is false. If it is right, the leaf check is the same exact identity it was with public final memory, with `F` in place of the statement's vector, and the fingerprint argument then proves `F` is the final memory. The statement transcript records the mode (`spice/final-committed` and the options word), so a proof for one mode never verifies in the other.
