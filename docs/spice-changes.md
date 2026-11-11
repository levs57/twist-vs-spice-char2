# Typed Spice changes for issue #1

This file records the Spice-side changes for levs57/twist-vs-spice-char2#1 (§2, §3, §5 and §7), the design choices behind them and the measurements. The soundness argument for each protocol change is in [soundness-spice.md](soundness-spice.md).

All code is in `crates/spice-argument/src/operations.rs` unless stated otherwise. The `ram-compare` edits are limited to exposing the new methods.

## 1. Two independent fingerprints (§3)

`Options { fingerprints: 2 }` (`Options::TWO_FINGERPRINTS`, method `spice-typed-2fp`) runs the permutation check with two independent challenge sets `(gamma, alpha, beta, delta)`. Both sets are drawn after the commitment is absorbed. `spice-typed` keeps one set.

**How the two checks share one GKR.** `lean_grandproduct` proves exactly two trees (left and right) and cannot be extended without editing `vendor/`. The 2fp variant therefore doubles each tree instead of adding two more: each tree holds `[set 0 | set 1]`, so its size is `2^(l+2)` instead of `2^(l+1)`. The batched GKR then proves all four products (left and right, for each set) at once.

- The per-set products are read from the first GKR layer. If the first layer is binary, its two children per tree are the two sets' products. If it is radix-four, its four children are the cycle and memory halves of each set, and the verifier multiplies pairs. `fingerprint_products` implements both cases. The GKR binds these children exactly as it binds every other layer: a wrong child vector changes the next layer's claim and is caught with the usual GKR error.
- The verifier checks `left == right` per set. Checking only the roots would be unsound for this purpose: equal root products do not imply equal per-set products, and the error would then be `2(T+K)/2^128` rather than its square.
- All leaves share one GKR point. The top coordinate selects the set, so both sets' leaf values are linear in the same input evaluations. The 2fp variant adds no Mosaic claims and no committed bits.
- The verifier, not the proof, chooses the number of sets (`verify_with(.., &Options)`), and the transcript absorbs the option. A one-set proof does not verify as 2fp.

**Why this and not an extension-field grand product.** An extension field (for example GF(2^256) as a quadratic extension of GF(2^128)) would also make the fingerprint term about `(T+K)/2^256`. It would need a new GKR kernel in the extension field (each multiplication about three base multiplications), extension-valued leaves, and an adapter that authenticates base-field Mosaic claims against extension-field challenge points. None of these exist in the repository (the README already notes the missing "extension opening adapter"). Two base-field fingerprints reuse every existing kernel, keep one leaf point, and cost one extra GKR level plus twice the leaf work. The resulting fingerprint term, about `((T+K)/2^128)^2`, is far below the remaining GKR and PCS terms, so the extension field would buy nothing measurable.

**Cost.** At n=20, a=16 on random operations (8 threads, load 26–27, one interleaved run): `spice.permutation.gkr` 5.1 → 9.5 ms and `spice.permutation.leaves` 2.3 → 4.8 ms. The issue estimated about 11 ms; the measured increase is about 7 ms.

## 2. Fewer Mosaic claims for the bitplane groups (§7)

Typed Spice used to send 126 Mosaic claims at n=20, a=16: every address, predecessor, carry and boundary bitplane was its own component and was claimed separately, some at two points. It now sends 19.

**What changed.** The committed bits are the same bitplanes, grouped differently for Mosaic:

- Each group (address, predecessor, carry, boundary predecessor, and the flag pair `is_write`/`next_is_write`) is split into aligned power-of-two chunks, largest first (`plane_chunks`: 21 = 16 + 4 + 1, 20 = 16 + 4). A chunk of `2^k` planes of log size `m` is one Mosaic component of log size `m + k`. Plane `i` sits at offset `i * 2^m`, so the plane index is the high `k` coordinates (`place_planes`).
- The prover still sends every per-plane value, as before. They are needed by the leaf check and the Flock terminal check.
- After all per-plane values are absorbed, the verifier draws `rho` (5 coordinates, `group_rho`). For each chunk and point, one claim replaces the `2^k` per-plane claims: the chunk component at `point || rho[..k]` equals `sum_i eq(rho[..k], i) * value_i` (`finish_claims`).

**Why not commit word-encoded groups.** The issue suggests committing each group as one word-encoded component, like `input/value`. That would round each group up to a power-of-two width (21 → 32 planes), adding 11 + 11 + 12 bits per row and crossing the 128-bit boundary. Chunking the existing planes keeps the committed area exactly the same.

**Claim count at n=20, a=16 (b = 21 predecessor bits).**

| Point | Before | After |
|---|---|---|
| GKR leaf point: address, predecessor, flags, boundary, initial flag | 16 + 21 + 2 + 21 + 1 = 61 | 1 + 3 + 1 + 3 + 1 = 9 |
| Masked-reduction point: value, flags, initial flag | 4 | 3 |
| Flock A/B point: predecessor, carries | 21 + 20 = 41 | 3 + 2 = 5 |
| Flock C point: carries | 20 | 2 |
| Total | 126 | 19 |

The committed area does not change: 130,416,640 populated bits (124.375 per row) in a 2^27-bit commitment, before and after. `chunks_are_aligned_and_keep_the_primary_commitment_size` checks this at n=20, a=16.

**Cost.** The Mosaic prover's work is proportional to the claimed component area, not the claim count, so `mosaic.batch` changes little (29.9 → 25.3 ms in one noisy pair at load 27). The verifier evaluates 19 lifted equality polynomials instead of 126.

## 3. One pass over the bitplanes for all evaluation claims (§7)

`spice.claims` (bitplanes at the GKR leaf point) and `spice.order.claims` (bitplanes at Flock's two points) used to be separate full passes, each building full `2^n` equality tables. The masked-reduction flags at its own point were a third pass (`spice.masked.claims`).

**Protocol reorder.** The two passes could not share a table because the leaf-point values were sent, and absorbed, before the Flock zerocheck chose its points. Now the prover sends every committed-plane evaluation in one message (`Evaluations`) after the GKR, the masked-reduction sum-check and the Flock zerocheck have all fixed their points:

1. Commit; draw fingerprints; GKR (leaf point).
2. Send the masked-value claims at the leaf point (`Proof::masked`); masked-reduction sum-check (mask point). Only the sum-check runs here (`masked_reduction::prove_sumcheck`).
3. Flock zerocheck (A/B and C points). Only the zerocheck runs here (`order_flock::prove_zerocheck`).
4. Send `Evaluations`: address, predecessor and flags at the leaf point; boundary predecessor and initial flag at its memory prefix; value, flags and initial flag at the mask point; predecessor and carries at the A/B point; carries at the C point.
5. The verifier runs the deferred checks (`check_evaluations`: leaf binding, `masked_reduction::Deferred::check`, `order_flock::check_sources`), draws `rho` and runs Mosaic.

Sending a claimed evaluation of a committed polynomial later does not help a cheating prover: the true value is fixed once the point is fixed, and every value is absorbed before `rho` and the Mosaic challenges that check it. The soundness note gives the argument.

**Kernel.** `planes::evaluate` evaluates any set of committed bitplanes at several points in one parallel pass. It splits each point's equality table into a low table over 10 coordinates and a high table over the rest, builds sixteen 16-entry subset-sum tables per point for each 64-row word, and costs one lookup per plane per four rows. No `2^n` equality table is built. `planes::masked_word_evals` computes both masked-value claims with one word evaluation per row instead of two passes over two masked copies of the values.

**Measured** at n=20, a=16, random operations, interleaved (load 49–51): `spice.claims` + `spice.order.claims` + `spice.masked.claims` went from 4.8 + 3.8 + 0.5 = 9.2 ms to 2.0 + 0.8 = 2.8 ms (`spice.order.claims` no longer exists), and the prover from 91.6 / 91.1 ms to 84.8 / 84.1 ms. `spice.claims` now holds the fused pass and `spice.masked.claims` the masked-value claims.

## 4. Batching the masked reduction with the ordering zerocheck (§7): not done

The two sum-checks are not batched. Batching needs a round-by-round prover for both, so that one challenge per round can bind both and the messages can be combined, and neither exists outside `vendor/`:

- The masked reduction calls `om_memcheck::eq_factor::prove_with_initial_claim`, which runs all `l = max(n, a)` rounds internally and draws its own challenges.
- The ordering argument calls `flock_core::zerocheck::prove_packed_padded`, which runs the univariate-skip round over the 6 gate variables and then `max(n, 7)` multilinear rounds internally, drawing every challenge (including its own equality point) through the `Challenger`.

The variable counts would allow it when `n >= a` and `n >= 7` (20 cycle rounds each at the primary parameters), and the different equality points are no obstacle in a batched sum-check. What it needs is a round-level API in both vendor crates: `round_message(&self) -> poly` and `bind(&mut self, r)` for the eq-factored product prover, and the same for Flock's multilinear phase after the skip round, plus matching verifier steps. With that, a combined sum-check over the cycle variables would send one degree-3 message per round with a batching coefficient.

**Expected gain is small.** Batching does not reduce prover work: each term's per-round work is unchanged. It would save about 20 round messages (about 1 KB of proof and 20 transcript squeezes) and make the mask point equal to Flock's A/B point, which removes one point from the fused evaluation pass (about 0.2 ms; that pass now costs about 2 ms in total). I estimate under 1% of the prover. I judged that not worth forking two vendor crates.

Reimplementing the ordering argument in this crate as an eq-factored sum-check would allow batching, but that is the ripple ordering (`order.rs`), which is slower than Flock's packed Boolean kernel. That would be a net loss.

## 5. Private final memory (§5, Spice half)

`Options::private_final_memory` commits final memory instead of reading it from the statement. It is off by default and has no ram-compare method yet, because the harness's area formula and its final-memory check assume public final memory. The example `cargo run --release -p spice-argument --example typed_options -- 20 16 5` measures it.

**Construction.** The prover places the final words as component `boundary/final` (one 64-bit word per cell, word-encoded like `input/value`, log size `a + 6`) in the same commitment as every other column, so the words are fixed before the fingerprints are drawn. The memory-side left tuple `(last_k, k, final_k, 0)` is unchanged. Its value coordinate at the GKR leaf point is now `Evaluations::final_value`, which the prover sends and the transcript absorbs with the other evaluations. `check_leaves` uses it where it used `word_eval(final_memory, leaf[..a])`, and one extra Mosaic claim, `word_claim("boundary/final", leaf[..a], final_value)`, binds it to the committed words. The verifier then ignores `Statement::final_memory`: `validate_memory` skips its shape check, and `transcript_with` absorbs an empty `spice/final-committed` entry in its place, so public and private proofs do not verify under each other's options. Initial memory stays public.

**Why this and not the cheaper form.** The cheapest private variant, matching Twist's `None` final state, drops the final value from the tuple altogether (left `(last_k, k, 0, 1)` on the memory side with matching changes on the right), so nothing is committed. That variant proves only that the reads are consistent, not what memory holds at the end. The issue asks for final memory committed and bound, which a later proof can open, so I implemented that.

**Cost at n=20, a=16.** Populated bits go from 130,416,640 to 134,610,944 (+2^22), which crosses 2^27, so the padded commitment doubles from 2^27 to 2^28 bits. There is one more Mosaic claim (19 to 20). Measured interleaved on a random workload (load 37–40): prove 92.7 to 118.9 ms, verify 6.18 to 5.92 ms. The verifier saves the 2^16-cell `word_eval` of final memory but checks a larger commitment. The doubling comes from the threshold, not from the 4M bits: at n=20, a=16 the primary layout has only 3.8M bits of slack.
