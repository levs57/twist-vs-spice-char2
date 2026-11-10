use super::*;
use om_sumcheck::{combine_sum, eq_table, mle_eval, sumcheck_prove};

fn field(i: usize) -> K {
    K::from_u128((i as u128 + 3).wrapping_mul(0x9e3779b97f4a7c15_6a09e667f3bcc909))
}

fn inputs(n: usize, a: usize, pattern: usize) -> (Vec<u32>, Vec<u32>) {
    let w = width(n, a).unwrap();
    let base = 1u64 << (w - 1);
    let previous = (0..1usize << n)
        .map(|j| {
            let time = base + j as u64;
            match pattern {
                0 => 0,
                1 => (time - 1) as u32,
                _ => ((j as u64 * 13 + 7) % time) as u32,
            }
        })
        .collect::<Vec<_>>();
    let carries = build_carries(&previous, n, a).unwrap();
    (previous, carries)
}

fn eval_bits(words: &[u32], point: &[K], bits: usize) -> Vec<K> {
    (0..bits)
        .map(|b| {
            mle_eval(
                &words
                    .iter()
                    .map(|&x| K::from_bit((x >> b) & 1 == 1))
                    .collect::<Vec<_>>(),
                point,
            )
        })
        .collect()
}

fn dense(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> (OrderProof, OrderClaims) {
    let w = width(n, a).unwrap();
    let (rho, weights) = setup(n, a, tr, base).unwrap();
    let gates = Gates {
        weights: &weights,
        validated_zero: false,
    };
    let mut tables = vec![eq_table(&rho)];
    tables.extend((0..2 * w + 1).map(|_| vec![K::ZERO; 1 << n]));
    for j in 0..1usize << n {
        let mut row = vec![K::ZERO; 2 * w + 1];
        source_row(
            previous[j],
            carries[j],
            (1u64 << (w - 1)) + j as u64,
            &weights,
            &mut row,
        );
        for (column, value) in tables[1..].iter_mut().zip(row) {
            column[j] = value;
        }
    }
    let combine = |v: &[K]| v[0] * gates.value(&v[1..]);
    let initial = combine_sum(&tables, &combine);
    assert_eq!(initial, K::ZERO);
    let (sc, point, values) =
        sumcheck_prove(initial, tables, &combine, 3, tr, &label(base, b"/sc"));
    let p = eval_bits(previous, &point, w);
    let c = eval_bits(carries, &point, w - 1);
    for b in 0..w {
        assert_eq!(values[1 + 2 * b], p[b] + carry_at(&c, w, b));
        assert_eq!(
            values[2 + 2 * b],
            K::ONE + time_bit(&point, w, b) + carry_at(&c, w, b)
        );
    }
    absorb_claims(tr, base, &p, &c);
    (
        OrderProof {
            sc,
            previous: p.clone(),
            carries: c.clone(),
        },
        OrderClaims {
            point,
            previous: p,
            carries: c,
        },
    )
}

#[test]
fn native_carry_words_match_ripple_majority_and_strict_unsigned_order() {
    for w in 1..=7 {
        for time in 1u64 << (w - 1)..1u64 << w {
            for previous in 0u64..1u64 << w {
                let mask = (1u64 << w) - 1;
                let bp = time ^ mask;
                let sum = previous + bp + 1;
                let cin = (sum & mask) ^ previous ^ bp;
                let mut carry = 1u64;
                for b in 0..w {
                    assert_eq!((cin >> b) & 1, carry);
                    let p = (previous >> b) & 1;
                    let i = (time >> b) & 1;
                    let next = (p & (1 ^ i)) | (p & carry) | ((1 ^ i) & carry);
                    assert_eq!((p ^ carry) & (1 ^ i ^ carry), carry ^ next);
                    carry = next;
                }
                assert_eq!(carry == 0, previous < time);
            }
        }
    }
    for (n, a) in [(0, 0), (0, 5), (1, 0), (4, 2), (2, 6)] {
        for pattern in 0..3 {
            let (previous, carries) = inputs(n, a, pattern);
            let w = width(n, a).unwrap();
            for (j, (&p, &word)) in previous.iter().zip(&carries).enumerate() {
                let time = (1u64 << (w - 1)) + j as u64;
                let mut c = 1u64;
                for b in 0..w {
                    assert_eq!(
                        c,
                        if b == 0 {
                            1
                        } else {
                            ((word >> (b - 1)) & 1) as u64
                        }
                    );
                    let x = (p as u64 >> b) & 1;
                    let y = 1 ^ ((time >> b) & 1);
                    c = (x & y) | (x & c) | (y & c);
                }
                assert_eq!(c, 0);
            }
        }
    }
}

#[test]
fn timestamp_bits_are_analytic_low_first_mles() {
    for (n, a) in [(0, 0), (0, 4), (1, 0), (3, 5), (5, 2)] {
        let w = width(n, a).unwrap();
        for kind in 0..3 {
            let point: Vec<K> = (0..n)
                .map(|b| match kind {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => field(b),
                })
                .collect();
            for b in 0..w {
                let truth: Vec<K> = (0..1usize << n)
                    .map(|j| K::from_bit((((1u64 << (w - 1)) + j as u64) >> b) & 1 == 1))
                    .collect();
                assert_eq!(time_bit(&point, w, b), mle_eval(&truth, &point));
            }
        }
    }
}

#[test]
fn ordering_proof_matches_dense_rounds_claims_and_transcript() {
    for (n, a) in [(0, 0), (0, 4), (1, 0), (1, 4), (4, 2), (4, 6)] {
        for pattern in 0..3 {
            let (previous, carries) = inputs(n, a, pattern);
            let mut tr = Transcript::new_default(b"order-oracle");
            let mut reference = tr.clone();
            let actual = prove(&previous, &carries, n, a, &mut tr, b"order").unwrap();
            let expected = dense(&previous, &carries, n, a, &mut reference, b"order");
            assert_eq!(actual, expected);
            assert_eq!(tr, reference);
            let mut verifier = Transcript::new_default(b"order-oracle");
            assert_eq!(
                verify(n, a, &actual.0, &mut verifier, b"order").unwrap(),
                actual.1
            );
            assert_eq!(verifier, tr);
        }
    }
}

#[test]
fn quadratic_cofactor_handles_zero_weights_and_boolean_equality_points() {
    let (n, a) = (4, 3);
    let w = width(n, a).unwrap();
    let (previous, carries) = inputs(n, a, 2);
    for kind in 0..3 {
        for zero in [false, true] {
            let rho: Vec<K> = (0..n)
                .map(|b| match kind {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => field(b + 13),
                })
                .collect();
            let weights: Vec<K> = (0..w)
                .map(|b| {
                    if zero || b == 1 {
                        K::ZERO
                    } else {
                        field(b + 31)
                    }
                })
                .collect();
            let gates = Gates {
                weights: &weights,
                validated_zero: false,
            };
            let row = |j: usize, out: &mut [K]| {
                source_row(
                    previous[j],
                    carries[j],
                    (1u64 << (w - 1)) + j as u64,
                    &weights,
                    out,
                )
            };
            let mut columns = vec![eq_table(&rho)];
            columns.extend((0..2 * w + 1).map(|_| vec![K::ZERO; 1 << n]));
            for j in 0..1usize << n {
                let mut v = vec![K::ZERO; 2 * w + 1];
                row(j, &mut v);
                for (column, value) in columns[1..].iter_mut().zip(v) {
                    column[j] = value;
                }
            }
            let combine = |v: &[K]| v[0] * gates.value(&v[1..]);
            let mut reference = Transcript::new_default(b"order-cofactor");
            let expected = sumcheck_prove(K::ZERO, columns, &combine, 3, &mut reference, b"sc");
            for validated_zero in [false, true] {
                let gates = Gates {
                    weights: &weights,
                    validated_zero,
                };
                let mut tr = Transcript::new_default(b"order-cofactor");
                let actual = eq_factor::prove(&rho, &gates, row, &mut tr, b"sc");
                assert_eq!(actual.0, K::ZERO);
                assert_eq!(&actual.1, &expected.0);
                assert_eq!(&actual.2, &expected.1);
                assert_eq!(actual.3, expected.2[1..]);
                assert_eq!(tr, reference);
            }
        }
    }
}

#[test]
fn invalid_order_carries_shapes_and_tampering_reject_or_expose_false_leaf() {
    assert!(width(32, 0).is_err());
    assert!(width(0, 32).is_err());
    assert!(build_carries(&[], 0, 0).is_err());
    assert!(build_carries(&[1], 0, 0).is_err());
    assert!(build_carries(&[u32::MAX], 0, 4).is_err());
    let (n, a) = (3, 4);
    let w = width(n, a).unwrap();
    let (previous, carries) = inputs(n, a, 2);
    let t = || Transcript::new_default(b"order-tamper");
    let (proof, claims) = prove(&previous, &carries, n, a, &mut t(), b"order").unwrap();
    let check = |p: &OrderProof| verify(n, a, p, &mut t(), b"order");
    let mut bad = proof.clone();
    bad.previous.pop();
    assert!(check(&bad).is_err());
    let mut bad = proof.clone();
    bad.carries.push(K::ZERO);
    assert!(check(&bad).is_err());
    let mut bad = proof.clone();
    bad.sc.round_msgs.pop();
    assert!(check(&bad).is_err());
    let mut bad = proof.clone();
    bad.sc.round_msgs[0].push(K::ZERO);
    assert!(check(&bad).is_err());
    for round in 0..n {
        for slot in 0..3 {
            let mut bad = proof.clone();
            bad.sc.round_msgs[round][slot] += K::ONE;
            if let Ok(out) = check(&bad) {
                assert!(
                    out.previous != eval_bits(&previous, &out.point, w)
                        || out.carries != eval_bits(&carries, &out.point, w - 1)
                );
            }
        }
    }
    for bit in 0..2 * w - 1 {
        let mut bad = proof.clone();
        if bit < w {
            bad.previous[bit] += K::ONE
        } else {
            bad.carries[bit - w] += K::ONE
        };
        if let Ok(out) = check(&bad) {
            assert_eq!(out.point, claims.point);
            assert!(
                out.previous != eval_bits(&previous, &out.point, w)
                    || out.carries != eval_bits(&carries, &out.point, w - 1)
            );
        }
    }
    let mut corrupt = carries.clone();
    corrupt[0] ^= 1;
    assert!(prove(&previous, &corrupt, n, a, &mut t(), b"order").is_err());
    let mut overflow = carries.clone();
    overflow[0] |= 1 << (w - 1);
    assert!(prove(&previous, &overflow, n, a, &mut t(), b"order").is_err());
    let equal: Vec<u32> = (0..1usize << n)
        .map(|j| ((1usize << (w - 1)) + j) as u32)
        .collect();
    assert!(
        prove(
            &equal,
            &vec![(1u32 << (w - 1)) - 1; 1 << n],
            n,
            a,
            &mut t(),
            b"order"
        )
        .is_err()
    );
}
