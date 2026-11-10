use super::*;
use crate::order::build_carries;
use flock_core::{
    field::PHI_8_TABLE,
    zerocheck::univariate_skip_optimized::{medium_challenges_ghash, small_challenges_ghash},
};
use om_sumcheck::mle_eval;

fn field(i: usize) -> K {
    K::from_u128((i as u128 + 3).wrapping_mul(0x9e3779b97f4a7c15_6a09e667f3bcc909))
}

fn inputs(n: usize, a: usize, pattern: usize) -> (Vec<u32>, Vec<u32>) {
    let base = 1u64 << (width(n, a).unwrap() - 1);
    let previous = (0..1usize << n)
        .map(|j| match pattern {
            0 => 0,
            1 => (base + j as u64 - 1) as u32,
            _ => ((j as u64 * 13 + 7) % (base + j as u64)) as u32,
        })
        .collect::<Vec<_>>();
    let carry = build_carries(&previous, n, a).unwrap();
    (previous, carry)
}

fn eval_bits(words: &[u32], point: &[K], width: usize) -> Vec<K> {
    (0..width)
        .map(|b| {
            mle_eval(
                &words
                    .iter()
                    .map(|&word| K::from_bit(word >> b & 1 != 0))
                    .collect::<Vec<_>>(),
                point,
            )
        })
        .collect()
}

/// Independent Boolean gate construction, with no native packed carry formula.
fn dense_rows(previous: &[u32], carry: &[u32], n: usize, a: usize) -> [Vec<u8>; 3] {
    let w = width(n, a).unwrap();
    let cycles = 1usize << n.max(MIN_CYCLE_BITS);
    let mut out = std::array::from_fn(|_| vec![0; cycles * 8]);
    for j in 0..previous.len() {
        let time = (1u64 << (w - 1)) + j as u64;
        for b in 0..w {
            let c = if b == 0 { 1 } else { (carry[j] >> (b - 1)) & 1 };
            let next = if b + 1 == w { 0 } else { (carry[j] >> b) & 1 };
            let bits = [
                ((previous[j] >> b) & 1) ^ c,
                1 ^ ((time >> b) as u32 & 1) ^ c,
                c ^ next,
            ];
            for k in 0..3 {
                out[k][8 * j + b / 8] |= (bits[k] as u8) << (b % 8);
            }
        }
    }
    out
}

fn direct_terminal(rows: &[u8], weights: &[F128], point: &[K]) -> K {
    let values: Vec<K> = rows
        .chunks_exact(8)
        .map(|row| {
            (0..64).fold(K::ZERO, |sum, b| {
                if row[b / 8] >> (b % 8) & 1 != 0 {
                    sum + from_flock(weights[b])
                } else {
                    sum
                }
            })
        })
        .collect();
    mle_eval(&values, point)
}

#[test]
fn all_friendly_cycle_eq_weights_have_full_binary_rank() {
    let point: Vec<K> = small_challenges_ghash()
        .into_iter()
        .chain(medium_challenges_ghash())
        .map(from_flock)
        .collect();
    assert_eq!(point.len(), 7);
    let mut pivots = [0u128; 128];
    let mut rank = 0;
    for value in eq_table(&point) {
        let mut word = value.value();
        while word != 0 {
            let bit = 127 - word.leading_zeros() as usize;
            if pivots[bit] == 0 {
                pivots[bit] = word;
                rank += 1;
                break;
            }
            word ^= pivots[bit];
        }
    }
    assert_eq!(rank, 128);
}

#[test]
fn word_packing_and_flock_proof_match_dense_boolean_oracle() {
    for (n, a) in [(0, 0), (0, 31), (1, 0), (1, 4), (7, 4), (8, 5)] {
        for pattern in 0..3 {
            let (previous, carries) = inputs(n, a, pattern);
            let w = width(n, a).unwrap();
            let (packed, _) = packed_rows(&previous, &carries, n, a).unwrap();
            let dense = dense_rows(&previous, &carries, n, a);
            assert_eq!(packed, dense);
            let mut tr = Transcript::new_default(b"flock-order-dense");
            let mut reference = tr.clone();
            let (proof, claims) = prove(&previous, &carries, n, a, &mut tr, b"order").unwrap();
            let padded_n = setup(n, a, &mut reference, b"order").unwrap();
            let mut ch = OmChallenger { tr: reference };
            let (zc, terminal) = prove_packed_padded(
                &dense[0],
                &dense[1],
                &dense[2],
                K_SKIP + padded_n,
                &PaddingSpec::dense(K_SKIP + padded_n),
                &mut ch,
            );
            assert_eq!(proof.zerocheck, zc);
            let weights = lagrange_weights_naive(K_SKIP, terminal.z);
            let all_ab: Vec<K> = terminal
                .mlv_challenges
                .iter()
                .copied()
                .map(from_flock)
                .collect();
            let all_c: Vec<K> = terminal.r_rest.iter().copied().map(from_flock).collect();
            assert_eq!(
                direct_terminal(&dense[0], &weights, &all_ab),
                from_flock(terminal.a_eval)
            );
            assert_eq!(
                direct_terminal(&dense[1], &weights, &all_ab),
                from_flock(terminal.b_eval)
            );
            assert_eq!(
                direct_terminal(&dense[2], &weights, &all_c),
                from_flock(terminal.c_eval)
            );
            assert_eq!(proof.previous_ab, eval_bits(&previous, &claims.point_ab, w));
            assert_eq!(
                proof.carries_ab,
                eval_bits(&carries, &claims.point_ab, w - 1)
            );
            assert_eq!(proof.carries_c, eval_bits(&carries, &claims.point_c, w - 1));
            absorb_sources(&mut ch.tr, b"order", &proof);
            assert_eq!(tr, ch.tr);
            let mut verifier = Transcript::new_default(b"flock-order-dense");
            assert_eq!(
                verify(n, a, &proof, &mut verifier, b"order").unwrap(),
                claims
            );
            assert_eq!(verifier, tr);
            if n > 0 {
                assert_ne!(claims.point_ab, claims.point_c);
            }
        }
    }
}

#[test]
fn source_recombination_handles_phi_nodes_boolean_cycles_and_padding() {
    for (n, a) in [(0, 0), (0, 31), (1, 0), (1, 4), (3, 2), (7, 5)] {
        let w = width(n, a).unwrap();
        let (previous, carries) = inputs(n, a, 2);
        let dense = dense_rows(&previous, &carries, n, a);
        for kind in 0..3 {
            let all_ab: Vec<K> = (0..n.max(MIN_CYCLE_BITS))
                .map(|b| match kind {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => field(b + 13),
                })
                .collect();
            let all_c: Vec<K> = all_ab.iter().rev().copied().collect();
            for z in [F128::ZERO, F128::ONE, PHI_8_TABLE[17], to_flock(field(67))] {
                let weights = lagrange_weights_naive(K_SKIP, z);
                let p = eval_bits(&previous, &all_ab[..n], w);
                let cab = eval_bits(&carries, &all_ab[..n], w - 1);
                let cc = eval_bits(&carries, &all_c[..n], w - 1);
                let claim = ZerocheckClaim {
                    z,
                    mlv_challenges: all_ab.iter().copied().map(to_flock).collect(),
                    r_rest: all_c.iter().copied().map(to_flock).collect(),
                    a_eval: to_flock(direct_terminal(&dense[0], &weights, &all_ab)),
                    b_eval: to_flock(direct_terminal(&dense[1], &weights, &all_ab)),
                    c_eval: to_flock(direct_terminal(&dense[2], &weights, &all_c)),
                };
                let expected = check_terminals(n, w, &claim, &p, &cab, &cc).unwrap();
                assert_eq!(expected.point_ab, all_ab[..n]);
                assert_eq!(expected.point_c, all_c[..n]);
            }
        }
    }
}

#[test]
fn malformed_words_proof_shapes_and_every_payload_mutation_are_detected() {
    let (n, a) = (7, 4);
    let w = width(n, a).unwrap();
    let (previous, carries) = inputs(n, a, 2);
    let t = || Transcript::new_default(b"flock-order-tamper");
    let (proof, _) = prove(&previous, &carries, n, a, &mut t(), b"order").unwrap();
    assert!(prove(&[], &[], n, a, &mut t(), b"order").is_err());
    assert!(prove(&previous, &carries[..1], n, a, &mut t(), b"order").is_err());
    let mut bad_carries = carries.clone();
    bad_carries[0] ^= 1;
    assert!(prove(&previous, &bad_carries, n, a, &mut t(), b"order").is_err());
    let mut overflow = carries.clone();
    overflow[0] |= 1 << (w - 1);
    assert!(prove(&previous, &overflow, n, a, &mut t(), b"order").is_err());
    let mut future = previous.clone();
    future[1] = (1 << (w - 1)) + 1;
    assert!(prove(&future, &carries, n, a, &mut t(), b"order").is_err());
    assert!(prove(&previous, &carries, 32, a, &mut t(), b"order").is_err());
    let verify = |p: &OrderProof| super::verify(n, a, p, &mut t(), b"order");
    for kind in 0..7 {
        let mut bad = proof.clone();
        match kind {
            0 => {
                bad.previous_ab.pop();
            }
            1 => {
                bad.carries_ab.pop();
            }
            2 => {
                bad.carries_c.pop();
            }
            3 => {
                bad.zerocheck.round1_ab.pop();
            }
            4 => {
                bad.zerocheck.round1_c.push(F128::ZERO);
            }
            5 => {
                bad.zerocheck.multilinear_rounds.pop();
            }
            _ => bad.zerocheck.grinding_nonces.push(0),
        }
        assert!(verify(&bad).is_err());
    }
    // Every donor message position and every surfaced source evaluation is
    // challenged. A primitive accepting changed source values still exposes
    // incorrect claims for the caller's mandatory PCS check.
    let count = 128 + 2 * n + 3 + 3 * w - 2;
    for slot in 0..count {
        let mut bad = proof.clone();
        if slot < 64 {
            bad.zerocheck.round1_ab[slot] += F128::ONE;
        } else if slot < 128 {
            bad.zerocheck.round1_c[slot - 64] += F128::ONE;
        } else if slot < 128 + 2 * n {
            let k = slot - 128;
            if k % 2 == 0 {
                bad.zerocheck.multilinear_rounds[k / 2].0 += F128::ONE;
            } else {
                bad.zerocheck.multilinear_rounds[k / 2].1 += F128::ONE;
            }
        } else {
            let k = slot - 128 - 2 * n;
            match k {
                0 => bad.zerocheck.final_a_eval += F128::ONE,
                1 => bad.zerocheck.final_b_eval += F128::ONE,
                2 => bad.zerocheck.final_c_eval += F128::ONE,
                _ => {
                    let b = k - 3;
                    if b < w {
                        bad.previous_ab[b] += K::ONE;
                    } else if b < 2 * w - 1 {
                        bad.carries_ab[b - w] += K::ONE;
                    } else {
                        bad.carries_c[b - (2 * w - 1)] += K::ONE;
                    }
                }
            }
        }
        if let Ok(claims) = verify(&bad) {
            assert!(
                claims.previous_ab != eval_bits(&previous, &claims.point_ab, w)
                    || claims.carries_ab != eval_bits(&carries, &claims.point_ab, w - 1)
                    || claims.carries_c != eval_bits(&carries, &claims.point_c, w - 1)
            );
        }
    }
    assert!(super::verify(n, a, &proof, &mut t(), b"different-label").is_err());
}
