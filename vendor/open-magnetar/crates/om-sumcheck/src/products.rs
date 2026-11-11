//! Explicit product relations. Equality factors stay structured until the
//! requested terminal tables are needed; no relation is inferred from samples.

use std::borrow::Cow;

use om_field::{K, U256, clmul_pairs, fold_pairs, mul_pairs};
use om_transcript::Transcript;
use rayon::prelude::*;

use crate::mle::{PAR_THRESHOLD, eq_table};
use crate::sumcheck::{SumcheckProof, interpolate_eval, sumcheck_init};

/// Carryless dot product with one field reduction after the XOR reduction.
pub fn product_sum(a: &[K], b: &[K]) -> K {
    assert_eq!(a.len(), b.len(), "product sum shape mismatch");
    let block = |(a, b): (&[K], &[K])| {
        let mut sum = U256::ZERO;
        let mut a_pairs = a.chunks_exact(2);
        let mut b_pairs = b.chunks_exact(2);
        for (a, b) in a_pairs.by_ref().zip(b_pairs.by_ref()) {
            let [p0, p1] = clmul_pairs([a[0], a[1]], [b[0], b[1]]);
            sum += p0 + p1;
        }
        if let (Some(&a), Some(&b)) = (a_pairs.remainder().first(), b_pairs.remainder().first()) {
            sum += a.clmul(b);
        }
        sum
    };
    let sum = if a.len() >= PAR_THRESHOLD {
        a.par_chunks(2048)
            .zip(b.par_chunks(2048))
            .map(block)
            .reduce(|| U256::ZERO, |a, b| a + b)
    } else {
        block((a, b))
    };
    sum.reduce()
}

fn label(base: &[u8], suffix: &[u8]) -> Vec<u8> {
    [base, suffix].concat()
}

fn message(s0: K, sd: K, scale: K, point: K) -> [K; 3] {
    let [p0, pd] = mul_pairs([scale, scale], [s0, sd]);
    let g0 = (K::ONE + point) * p0;
    let g1 = point * (p0 + pd);
    [g0, g1, g0 + (g0 + g1 + pd).mul_x() + pd.mul_x().mul_x()]
}

/// Compute the two coefficient dots against a split suffix equality table.
/// With `bind`, produce the folded F and its *next* message in one traversal.
/// Low equality factors fit L1; high factors are applied once per block, after
/// reducing its two carryless dot accumulators.
fn dots(input: &[K], bind: Option<K>, suffix: &[K], out: &mut Vec<K>, need_s0: bool) -> (K, K) {
    let low_bits = suffix.len().min(10);
    let lo = eq_table(&suffix[..low_bits]);
    let hi = eq_table(&suffix[low_bits..]);
    let stride = if bind.is_some() { 4 } else { 2 };
    let block = stride * lo.len();
    let eval = |source: &[K], mut dst: Option<&mut [K]>, h: K| {
        let mut u0 = U256::ZERO;
        let mut ud = U256::ZERO;
        let mut row = |i: usize| {
            let row = &source[stride * i..stride * (i + 1)];
            let (a, b) = if let Some(r) = bind {
                let [x, y] = mul_pairs([r, r], [row[0] + row[1], row[2] + row[3]]);
                (row[0] + x, row[2] + y)
            } else {
                (row[0], row[1])
            };
            if let Some(ref mut dst) = dst {
                dst[2 * i] = a;
                dst[2 * i + 1] = b;
            }
            (a, b)
        };
        // Pair independent rows so the single-channel rounds still use both
        // native carryless lanes. Only the first round needs the H(0) dot to
        // check the caller's initial claim against the actual table.
        for (i, e) in lo.chunks_exact(2).enumerate() {
            let (a0, b0) = row(2 * i);
            let (a1, b1) = row(2 * i + 1);
            if need_s0 {
                let [p0, p1] = clmul_pairs([a0, a1], [e[0], e[1]]);
                u0 += p0 + p1;
            }
            let [p0, p1] = clmul_pairs([a0 + b0, a1 + b1], [e[0], e[1]]);
            ud += p0 + p1;
        }
        if lo.len() == 1 {
            let (a, b) = row(0);
            if need_s0 {
                u0 += a.clmul(lo[0]);
            }
            ud += (a + b).clmul(lo[0]);
        }
        (
            if need_s0 { h * u0.reduce() } else { K::ZERO },
            h * ud.reduce(),
        )
    };
    let add = |(a, b): (K, K), (c, d): (K, K)| (a + c, b + d);
    if bind.is_some() {
        out.resize(input.len() / 2, K::ZERO);
        if input.len() >= PAR_THRESHOLD {
            input
                .par_chunks(block)
                .zip(out.par_chunks_mut(block / 2))
                .zip(hi.par_iter())
                .map(|((src, dst), &h)| eval(src, Some(dst), h))
                .reduce(|| (K::ZERO, K::ZERO), add)
        } else {
            input
                .chunks(block)
                .zip(out.chunks_mut(block / 2))
                .zip(&hi)
                .map(|((src, dst), &h)| eval(src, Some(dst), h))
                .fold((K::ZERO, K::ZERO), add)
        }
    } else if input.len() >= PAR_THRESHOLD {
        input
            .par_chunks(block)
            .zip(hi.par_iter())
            .map(|(src, &h)| eval(src, None, h))
            .reduce(|| (K::ZERO, K::ZERO), add)
    } else {
        input
            .chunks(block)
            .zip(&hi)
            .map(|(src, &h)| eval(src, None, h))
            .fold((K::ZERO, K::ZERO), add)
    }
}

fn bind_into(input: &[K], output: &mut Vec<K>, r: K) {
    output.resize(input.len() / 2, K::ZERO);
    if input.len() >= PAR_THRESHOLD {
        const CHUNK: usize = 2048;
        output
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let start = 2 * i * CHUNK;
                fold_pairs(&input[start..start + 2 * dst.len()], r, dst);
            });
    } else {
        fold_pairs(input, r, output);
    }
}

/// The same degree-two partial-round protocol as `sumcheck_prove_rounds_mul2`
/// for `[f, eq_table(point)]`. It retains equality as a scalar and split suffix
/// factors and materializes its bound table only on return. C3 low-first order,
/// messages, transcript initialization, and returned column order are unchanged.
pub fn sumcheck_prove_rounds_eq_mul2(
    claim: K,
    f: Vec<K>,
    point: &[K],
    rounds: usize,
    transcript: &mut Transcript,
    base: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<Vec<K>>) {
    prove_rounds_eq_mul2(claim, Cow::Owned(f), point, rounds, transcript, base)
}

/// Borrow the initial F table instead of cloning it for the first fold. Every
/// nonempty fold produces its owned half-size buffer directly; subsequent
/// rounds use the same fused dot/fold kernels and reusable owned buffers.
/// With zero rounds the return contract requires a copy of the unbound table.
pub fn sumcheck_prove_rounds_eq_mul2_borrowed(
    claim: K,
    f: &[K],
    point: &[K],
    rounds: usize,
    transcript: &mut Transcript,
    base: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<Vec<K>>) {
    prove_rounds_eq_mul2(claim, Cow::Borrowed(f), point, rounds, transcript, base)
}

fn prove_rounds_eq_mul2(
    claim: K,
    mut f: Cow<'_, [K]>,
    point: &[K],
    rounds: usize,
    transcript: &mut Transcript,
    base: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<Vec<K>>) {
    assert_eq!(
        f.len(),
        1usize << point.len(),
        "equality/table shape mismatch"
    );
    assert!(rounds <= point.len(), "more rounds than table variables");
    sumcheck_init(transcript, base, claim, rounds, 2);
    let msg_label = label(base, b"/msg");
    let chal_label = label(base, b"/chal");
    let mut msgs = Vec::with_capacity(rounds);
    let mut challenges = Vec::with_capacity(rounds);
    let mut scratch = Vec::new();
    let mut scale = K::ONE;
    let mut current = claim;
    let mut ev = [K::ZERO; 3];
    let (mut s0, mut sd) = (K::ZERO, K::ZERO);
    if rounds != 0 {
        (s0, sd) = dots(&f, None, &point[1..], &mut scratch, true);
        ev = message(s0, sd, scale, point[0]);
    }
    for round in 0..rounds {
        assert_eq!(
            ev[0] + ev[1],
            current,
            "claim inconsistent with equality product"
        );
        let msg = vec![ev[0], ev[2]];
        transcript.absorb_ks(&msg_label, &msg);
        let r = transcript.challenge_k(&chal_label);
        msgs.push(msg);
        challenges.push(r);
        current = interpolate_eval(&ev, r);
        // Keep the normalized cofactor claim separately from the equality
        // prefix. It remains available even when that prefix becomes zero.
        let normalized = s0 + r * sd;
        scale *= K::ONE + point[round] + r;
        if round + 1 == rounds {
            bind_into(&f, &mut scratch, r);
        } else {
            (_, sd) = dots(&f, Some(r), &point[round + 2..], &mut scratch, false);
            s0 = normalized + point[round + 1] * sd;
            ev = message(s0, sd, scale, point[round + 1]);
        }
        let old = std::mem::replace(&mut f, Cow::Owned(std::mem::take(&mut scratch)));
        if let Cow::Owned(old) = old {
            scratch = old;
        }
    }
    let mut b = eq_table(&point[rounds..]);
    if b.len() >= PAR_THRESHOLD {
        b.par_iter_mut().for_each(|x| *x *= scale);
    } else {
        b.iter_mut().for_each(|x| *x *= scale);
    }
    (
        SumcheckProof { round_msgs: msgs },
        challenges,
        vec![f.into_owned(), b],
    )
}

fn weighted_evals(
    tables: &[Vec<K>],
    weights: &[K],
    bind: Option<K>,
    bound: &mut Vec<Vec<K>>,
    claim: Option<K>,
) -> [K; 3] {
    let groups = weights.len();
    let size = tables[0].len();
    let output_size = size / 2;
    let pairs = if bind.is_some() { size / 4 } else { size / 2 };
    if bind.is_some() {
        bound.resize_with(tables.len(), Vec::new);
        for output in bound.iter_mut() {
            output.resize(output_size, K::ZERO);
        }
    }
    let process = |start: usize, count: usize, mut columns: Option<Vec<&mut [K]>>| {
        let mut sums = [U256::ZERO; 3];
        for row in 0..count {
            let source = if bind.is_some() {
                4 * (start + row)
            } else {
                2 * (start + row)
            };
            for (g, &weight) in weights.iter().enumerate() {
                let a = &tables[g];
                let b = &tables[groups + g];
                let (a0, a1, b0, b1) = if let Some(r) = bind {
                    let [ax, ay] = mul_pairs(
                        [r, r],
                        [a[source] + a[source + 1], a[source + 2] + a[source + 3]],
                    );
                    let [bx, by] = mul_pairs(
                        [r, r],
                        [b[source] + b[source + 1], b[source + 2] + b[source + 3]],
                    );
                    let (a0, a1) = (a[source] + ax, a[source + 2] + ay);
                    let (b0, b1) = (b[source] + bx, b[source + 2] + by);
                    let dst = columns.as_mut().expect("binding outputs");
                    dst[g][2 * row] = a0;
                    dst[g][2 * row + 1] = a1;
                    dst[groups + g][2 * row] = b0;
                    dst[groups + g][2 * row + 1] = b1;
                    (a0, a1, b0, b1)
                } else {
                    (a[source], a[source + 1], b[source], b[source + 1])
                };
                if weight.is_zero() {
                    continue;
                }
                // Scale a line before the unreduced product, not each finite
                // point product. Keep original columns for terminal claims.
                let [wa0, wa1] = mul_pairs([weight, weight], [a0, a1]);
                let [p0, pd] = clmul_pairs([wa0, wa0 + wa1], [b0, b0 + b1]);
                sums[0] += p0;
                if claim.is_none() {
                    sums[1] += wa1.clmul(b1);
                }
                sums[2] += pd;
            }
        }
        sums
    };
    let add = |mut a: [U256; 3], b: [U256; 3]| {
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
        a
    };
    let sums = if bind.is_some() {
        if size >= PAR_THRESHOLD {
            const CHUNK: usize = 1024;
            let mut pieces: Vec<Vec<&mut [K]>> = (0..pairs.div_ceil(CHUNK))
                .map(|_| Vec::with_capacity(tables.len()))
                .collect();
            for column in bound {
                for (piece, part) in pieces.iter_mut().zip(column.chunks_mut(2 * CHUNK)) {
                    piece.push(part);
                }
            }
            pieces
                .into_par_iter()
                .enumerate()
                .map(|(i, columns)| process(i * CHUNK, columns[0].len() / 2, Some(columns)))
                .reduce(|| [U256::ZERO; 3], add)
        } else {
            process(
                0,
                pairs,
                Some(bound.iter_mut().map(Vec::as_mut_slice).collect()),
            )
        }
    } else if size >= PAR_THRESHOLD {
        (0..pairs)
            .into_par_iter()
            .with_min_len(1024)
            .fold(
                || [U256::ZERO; 3],
                |acc, row| add(acc, process(row, 1, None)),
            )
            .reduce(|| [U256::ZERO; 3], add)
    } else {
        process(0, pairs, None)
    };
    let g0 = sums[0].reduce();
    let g1 = claim.map_or_else(|| sums[1].reduce(), |claim| claim + g0);
    let gd = sums[2].reduce();
    [g0, g1, g0 + (g0 + g1 + gd).mul_x() + gd.mul_x().mul_x()]
}

/// Prove the actual sum of `Σ_g weights[g] A_g B_g`, with columns ordered
/// `[A_0,..,A_g,B_0,..,B_g]`. This explicit relation preserves unscaled terminal
/// MLEs (also when a weight is zero) and obtains the honest initial claim from
/// its first message pass instead of scanning the tables separately.
pub fn sumcheck_prove_weighted_mul2(
    mut tables: Vec<Vec<K>>,
    weights: &[K],
    transcript: &mut Transcript,
    base: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<K>) {
    assert!(
        !weights.is_empty(),
        "weighted product needs at least one group"
    );
    assert_eq!(
        tables.len(),
        2 * weights.len(),
        "weighted product width mismatch"
    );
    let size = tables[0].len();
    assert!(
        size.is_power_of_two() && tables.iter().all(|t| t.len() == size),
        "tables must share a power-of-two length"
    );
    let rounds = size.ilog2() as usize;
    let mut scratch = Vec::new();
    let mut ev = if rounds == 0 {
        [K::ZERO; 3]
    } else {
        weighted_evals(&tables, weights, None, &mut scratch, None)
    };
    let mut current = if rounds == 0 {
        weights.iter().enumerate().fold(K::ZERO, |s, (g, &w)| {
            s + w * tables[g][0] * tables[weights.len() + g][0]
        })
    } else {
        ev[0] + ev[1]
    };
    sumcheck_init(transcript, base, current, rounds, 2);
    let msg_label = label(base, b"/msg");
    let chal_label = label(base, b"/chal");
    let mut msgs = Vec::with_capacity(rounds);
    let mut challenges = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let msg = vec![ev[0], ev[2]];
        transcript.absorb_ks(&msg_label, &msg);
        let r = transcript.challenge_k(&chal_label);
        msgs.push(msg);
        challenges.push(r);
        current = interpolate_eval(&ev, r);
        if round + 1 == rounds {
            scratch.resize_with(tables.len(), Vec::new);
            for (source, output) in tables.iter().zip(&mut scratch) {
                bind_into(source, output, r);
            }
        } else {
            ev = weighted_evals(&tables, weights, Some(r), &mut scratch, Some(current));
        }
        std::mem::swap(&mut tables, &mut scratch);
    }
    (
        SumcheckProof { round_msgs: msgs },
        challenges,
        tables.iter().map(|t| t[0]).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bind_low, sumcheck_round_evals_mul2};

    fn random(state: &mut u64) -> K {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        K::from_u128(u128::from(*state) | (u128::from(state.rotate_left(23)) << 64))
    }

    #[test]
    fn borrowed_eq_rounds_match_owned_proof_tables_and_transcript() {
        let mut seed = 0x93a1_7345_7788;
        for n in [0usize, 1, 3, 14] {
            let f: Vec<K> = (0..1usize << n).map(|_| random(&mut seed)).collect();
            for kind in 0..3 {
                let point: Vec<K> = (0..n)
                    .map(|i| match kind {
                        0 => K::ZERO,
                        1 => K::from_bit(i % 2 == 0),
                        _ => random(&mut seed),
                    })
                    .collect();
                let claim = product_sum(&f, &eq_table(&point));
                for rounds in [0, n.min(1), n / 2, n] {
                    let mut owned_tr = Transcript::new_default(b"borrow-eq");
                    let mut borrowed_tr = owned_tr.clone();
                    let owned = sumcheck_prove_rounds_eq_mul2(
                        claim,
                        f.clone(),
                        &point,
                        rounds,
                        &mut owned_tr,
                        b"sc",
                    );
                    let borrowed = sumcheck_prove_rounds_eq_mul2_borrowed(
                        claim,
                        &f,
                        &point,
                        rounds,
                        &mut borrowed_tr,
                        b"sc",
                    );
                    assert_eq!(owned, borrowed, "n={n} rounds={rounds} kind={kind}");
                    assert_eq!(
                        owned_tr.challenge_k(b"after"),
                        borrowed_tr.challenge_k(b"after")
                    );
                }
            }
        }
    }

    #[test]
    fn borrowed_product_and_first_fold_match_owned_kernels() {
        use crate::{
            bind_low_and_round_evals_mul2_into, bind_low_and_round_evals_mul2_slices_into,
            sumcheck_prove_mul2, sumcheck_prove_mul2_borrowed_first,
            sumcheck_prove_mul2_generated_first,
        };
        let mut seed = 0xabcd_7823_2288;
        for n in [0usize, 1, 2, 14] {
            let first: Vec<K> = (0..1usize << n).map(|_| random(&mut seed)).collect();
            let second: Vec<K> = (0..1usize << n)
                .map(|i| {
                    if i % 3 == 0 {
                        K::ZERO
                    } else {
                        random(&mut seed)
                    }
                })
                .collect();
            let claim = product_sum(&first, &second);
            let mut old_tr = Transcript::new_default(b"borrow-product");
            let mut new_tr = old_tr.clone();
            let mut generated_tr = old_tr.clone();
            let old = sumcheck_prove_mul2(
                claim,
                vec![first.clone(), second.clone()],
                &mut old_tr,
                b"sc",
            );
            let new = sumcheck_prove_mul2_borrowed_first(
                claim,
                &first,
                second.clone(),
                &mut new_tr,
                b"sc",
            );
            // This is the same little-endian word-pair view RingSwitch uses,
            // rather than a closure indexing the materialized field table.
            let words: Vec<u64> = first
                .iter()
                .flat_map(|x| [x.value() as u64, (x.value() >> 64) as u64])
                .collect();
            let generated = sumcheck_prove_mul2_generated_first(
                claim,
                first.len(),
                |i| K::from_u128(u128::from(words[2 * i]) | (u128::from(words[2 * i + 1]) << 64)),
                second.clone(),
                &mut generated_tr,
                b"sc",
            );
            assert_eq!(old, new, "n={n}");
            assert_eq!(old, generated, "generated n={n}");
            let after = old_tr.challenge_k(b"after");
            assert_eq!(after, new_tr.challenge_k(b"after"));
            assert_eq!(after, generated_tr.challenge_k(b"after"));
            if n > 0 {
                for r in [K::ZERO, K::ONE, random(&mut seed)] {
                    let next = product_sum(&bind_low(&first, r), &bind_low(&second, r));
                    for known in [None, Some(next)] {
                        let (mut old, mut new) = (Vec::new(), Vec::new());
                        let old_ev = bind_low_and_round_evals_mul2_into(
                            &[first.clone(), second.clone()],
                            &mut old,
                            r,
                            known,
                        );
                        let new_ev = bind_low_and_round_evals_mul2_slices_into(
                            &first, &second, &mut new, r, known,
                        );
                        assert_eq!((old_ev, old), (new_ev, new), "n={n} r={r:?}");
                    }
                }
            }
        }
        let first = [K::ONE, K::X];
        assert!(
            std::panic::catch_unwind(|| sumcheck_prove_mul2_borrowed_first(
                K::ZERO,
                &first,
                vec![K::ONE; 2],
                &mut Transcript::new_default(b"false"),
                b"sc"
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| sumcheck_prove_mul2_generated_first(
                K::ZERO,
                first.len(),
                |i| first[i],
                vec![K::ONE; 2],
                &mut Transcript::new_default(b"false"),
                b"sc"
            ))
            .is_err()
        );
    }

    #[test]
    fn equality_cofactor_recovery_survives_zero_prefixes() {
        for n in [3usize, 14] {
            let point: Vec<K> = (0..n)
                .map(|i| match i % 3 {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => K::from_u128(0x1fed3 + i as u128),
                })
                .collect();
            let mut state = 0x6a09_e667_f3bc_c909u64;
            let mut f: Vec<K> = (0..1usize << n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    K::from_u128(u128::from(state) | (u128::from(state.rotate_left(23)) << 64))
                })
                .collect();
            let mut b = eq_table(&point);
            let mut scratch = Vec::new();
            let mut scale = K::ONE;
            let (mut s0, mut sd) = dots(&f, None, &point[1..], &mut scratch, true);
            for round in 0..n {
                assert_eq!(
                    message(s0, sd, scale, point[round]),
                    sumcheck_round_evals_mul2(&f, &b)
                );
                // Force a zero equality prefix immediately, then continue
                // recovering the unscaled cofactor without any inversion.
                let r = if round == 0 {
                    K::ONE + point[round]
                } else {
                    K::from_u128(0xbadcafe + round as u128)
                };
                let normalized = s0 + r * sd;
                scale *= K::ONE + point[round] + r;
                let expected = bind_low(&f, r);
                b = bind_low(&b, r);
                if round + 1 < n {
                    (_, sd) = dots(&f, Some(r), &point[round + 2..], &mut scratch, false);
                    s0 = normalized + point[round + 1] * sd;
                    let (actual_s0, actual_sd) =
                        dots(&expected, None, &point[round + 2..], &mut Vec::new(), true);
                    assert_eq!((s0, sd), (actual_s0, actual_sd));
                    assert_eq!(scratch, expected);
                    std::mem::swap(&mut f, &mut scratch);
                }
            }
            assert_eq!(scale, K::ZERO);
        }
    }
}
