use super::*;
use om_sumcheck::{bind_low_in_place, combine_sum, mle_eval, sumcheck_prove};

struct Fixture {
    values: Vec<u64>,
    write: Vec<bool>,
    next: Vec<bool>,
    initial: Vec<u64>,
    initial_next: Vec<bool>,
    gp: Vec<K>,
}
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn fixture(n: usize, a: usize, point_kind: usize) -> Fixture {
    let mut state = 0x9a7b_f1c2_d3e4_5678;
    let values = (0..1 << n).map(|_| next(&mut state)).collect();
    let write = (0..1 << n).map(|j| j % 3 == 1).collect();
    let following = (0..1 << n).map(|j| j % 5 <= 1).collect();
    let initial = (0..1 << a).map(|_| next(&mut state)).collect();
    let initial_next = (0..1 << a).map(|j| j % 3 == 2).collect();
    let gp = (0..=n.max(a))
        .map(|j| match point_kind {
            0 => K::ZERO,
            1 => K::ONE,
            2 => K::from_bit(j % 2 == 0),
            _ => K::from_u128(next(&mut state) as u128 | ((next(&mut state) as u128) << 64)),
        })
        .collect();
    Fixture {
        values,
        write,
        next: following,
        initial,
        initial_next,
        gp,
    }
}
fn claims(f: &Fixture) -> MaskClaims {
    let n = f.values.len().trailing_zeros() as usize;
    let a = f.initial.len().trailing_zeros() as usize;
    let cycle = std::array::from_fn(|which| {
        let flags = if which == 0 { &f.write } else { &f.next };
        let masked = f
            .values
            .iter()
            .zip(flags)
            .map(|(&v, &flag)| {
                if flag {
                    K::ZERO
                } else {
                    K::from_u128(v as u128)
                }
            })
            .collect::<Vec<_>>();
        mle_eval(&masked, &f.gp[..n])
    });
    let initial = f
        .initial
        .iter()
        .zip(&f.initial_next)
        .map(|(&v, &flag)| {
            if flag {
                K::ZERO
            } else {
                K::from_u128(v as u128)
            }
        })
        .collect::<Vec<_>>();
    MaskClaims {
        cycle,
        initial: mle_eval(&initial, &f.gp[..a]),
    }
}

/// Explicitly expand independent lower-domain columns into the common cube.
fn dense_columns(f: &Fixture, weights: [K; 3]) -> Vec<Vec<K>> {
    let l = f.gp.len() - 1;
    let mut columns = vec![Vec::new(); 4];
    for j in 0..1usize << l {
        let c = j % f.values.len();
        let m = j % f.initial.len();
        let first = K::from_bit(f.write[c]);
        let next = K::from_bit(f.next[c]);
        let values = [
            K::from_u128(f.values[c] as u128),
            weights[0] * (K::ONE + first) + weights[1] * (K::ONE + next),
            K::from_u128(f.initial[m] as u128),
            weights[2] * (K::ONE + K::from_bit(f.initial_next[m])),
        ];
        for (column, value) in columns.iter_mut().zip(values) {
            column.push(value);
        }
    }
    columns
}
fn reference(f: &Fixture, claim: &MaskClaims) -> (MaskProof, SourceClaims, Transcript) {
    let (a, l) = public_shape(&f.initial, &f.gp).unwrap();
    let n = f.values.len().trailing_zeros() as usize;
    let mut tr = Transcript::new_default(b"masked-oracle");
    let weights = setup(a, l, &f.gp, claim, &mut tr, b"mask");
    let mut tables = vec![eq_table(&f.gp[..l])];
    tables.extend(dense_columns(f, weights));
    let combine = |r: &[K]| r[0] * (r[1] * r[2] + r[3] * r[4]);
    let actual = combine_sum(&tables, &combine);
    assert_eq!(actual, batched_claim(claim, weights));
    let (sc, point, _) = sumcheck_prove(actual, tables, &combine, 3, &mut tr, b"mask/sc");
    let value = mle_eval(
        &f.values
            .iter()
            .map(|&v| K::from_u128(v as u128))
            .collect::<Vec<_>>(),
        &point[..n],
    );
    let proof = MaskProof {
        sc,
        value,
        is_write: mle_eval_bits(&f.write, &point[..n]),
        next_is_write: mle_eval_bits(&f.next, &point[..n]),
        initial_next_is_write: mle_eval_bits(&f.initial_next, &point[..a]),
    };
    let sources = SourceClaims {
        point,
        value,
        is_write: proof.is_write,
        next_is_write: proof.next_is_write,
        initial_next_is_write: proof.initial_next_is_write,
    };
    absorb_sources(&mut tr, b"mask", &proof);
    (proof, sources, tr)
}

#[test]
fn dense_cubic_proofs_and_transcripts_match_for_different_source_domains() {
    for (n, a) in [(0, 0), (0, 3), (3, 0), (1, 3), (3, 1), (3, 3)] {
        for kind in 0..4 {
            let f = fixture(n, a, kind);
            let c = claims(&f);
            let mut tr = Transcript::new_default(b"masked-oracle");
            let (proof, sources) = prove(
                &f.values,
                &f.write,
                &f.next,
                &f.initial,
                &f.initial_next,
                &f.gp,
                &c,
                &mut tr,
                b"mask",
            )
            .unwrap();
            let expected = reference(&f, &c);
            assert_eq!(proof, expected.0, "n={n}, a={a}, kind={kind}");
            assert_eq!(sources, expected.1);
            assert_eq!(tr, expected.2);
            let mut verifier = Transcript::new_default(b"masked-oracle");
            assert_eq!(
                verify(&f.initial, &f.gp, &c, &proof, &mut verifier, b"mask").unwrap(),
                sources
            );
            assert_eq!(verifier, tr);
        }
    }
}

#[test]
fn parallel_replicated_source_matches_dense_oracle() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let f = fixture(14, 3, 3);
        let c = claims(&f);
        let mut tr = Transcript::new_default(b"masked-oracle");
        let actual = prove(
            &f.values,
            &f.write,
            &f.next,
            &f.initial,
            &f.initial_next,
            &f.gp,
            &c,
            &mut tr,
            b"mask",
        )
        .unwrap();
        let expected = reference(&f, &c);
        assert_eq!(actual, (expected.0, expected.1));
        assert_eq!(tr, expected.2);
    });
}

#[test]
fn masked_claims_are_not_products_of_off_cube_mles() {
    let point = vec![K::X, K::ONE];
    let f = Fixture {
        values: vec![0, 1],
        write: vec![false, true],
        next: vec![false, true],
        initial: vec![0, 1],
        initial_next: vec![false, true],
        gp: point,
    };
    let claim = claims(&f);
    assert_eq!(
        claim,
        MaskClaims {
            cycle: [K::ZERO; 2],
            initial: K::ZERO
        }
    );
    let naive = K::X * (K::ONE + K::X);
    assert_ne!(naive, claim.cycle[0]);
    assert_ne!(naive, claim.initial);
    let mut tr = Transcript::new_default(b"masked-oracle");
    let (proof, _) = prove(
        &f.values,
        &f.write,
        &f.next,
        &f.initial,
        &f.initial_next,
        &f.gp,
        &claim,
        &mut tr,
        b"mask",
    )
    .unwrap();
    verify(
        &f.initial,
        &f.gp,
        &claim,
        &proof,
        &mut Transcript::new_default(b"masked-oracle"),
        b"mask",
    )
    .unwrap();
}

#[test]
fn zero_batch_weights_and_boolean_equality_points_match_dense_driver() {
    let f = fixture(3, 2, 2);
    for weights in [
        [K::ZERO; 3],
        [K::ONE, K::ZERO, K::ZERO],
        [K::ZERO, K::ONE, K::ZERO],
        [K::ZERO, K::ZERO, K::ONE],
    ] {
        let columns = dense_columns(&f, weights);
        let l = f.gp.len() - 1;
        let mut tables = vec![eq_table(&f.gp[..l])];
        tables.extend(columns.clone());
        let combine = |r: &[K]| r[0] * (r[1] * r[2] + r[3] * r[4]);
        let initial = combine_sum(&tables, &combine);
        let mut reference = Transcript::new_default(b"masked-zero-batch");
        let expected = sumcheck_prove(initial, tables, &combine, 3, &mut reference, b"sc");
        let mut actual = Transcript::new_default(b"masked-zero-batch");
        let row = |j: usize, out: &mut [K]| {
            for (i, column) in columns.iter().enumerate() {
                out[i] = column[j];
            }
        };
        let result = eq_factor::prove_with_initial_claim(
            &f.gp[..l],
            &Products,
            row,
            Some(initial),
            &mut actual,
            b"sc",
        );
        assert_eq!(result.0, initial);
        assert_eq!(result.1, expected.0);
        assert_eq!(result.2, expected.1);
        assert_eq!(result.3, expected.2[1..]);
        assert_eq!(actual, reference);
    }
}

#[test]
fn cofactor_handles_both_boolean_folds_and_all_endpoint_modes() {
    let mut columns = dense_columns(&fixture(3, 2, 3), [K::X, K::t(7), K::t(13)]);
    for challenge in [K::ZERO, K::ONE, K::ZERO] {
        for pair in 0..columns[0].len() / 2 {
            let left: Vec<_> = columns.iter().map(|c| c[2 * pair]).collect();
            let right: Vec<_> = columns.iter().map(|c| c[2 * pair + 1]).collect();
            let mut both = vec![U256::ZERO; 3];
            Products.accumulate(&vec![K::ONE], 0, &left, &right, Endpoint::Both, &mut both);
            let both = both.into_iter().map(U256::reduce).collect::<Vec<_>>();
            let folded: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| a + challenge * (a + b))
                .collect();
            let coefficients = [both[0], both[0] + both[1] + both[2], both[2]];
            let expected = coefficients
                .iter()
                .rev()
                .fold(K::ZERO, |v, &c| v * challenge + c);
            assert_eq!(Products.value(&folded), expected);
            for endpoint in [Endpoint::Zero, Endpoint::One, Endpoint::Neither] {
                let mut partial = vec![U256::ZERO; 3];
                Products.accumulate(&vec![K::ONE], 0, &left, &right, endpoint, &mut partial);
                assert_eq!(partial[2].reduce(), both[2]);
                if matches!(endpoint, Endpoint::Zero) {
                    assert_eq!(partial[0].reduce(), both[0]);
                }
                if matches!(endpoint, Endpoint::One) {
                    assert_eq!(partial[1].reduce(), both[1]);
                }
            }
        }
        for column in &mut columns {
            bind_low_in_place(column, challenge);
        }
    }
}

#[test]
fn malformed_mutated_and_false_claims_are_rejected() {
    let f = fixture(3, 2, 3);
    let c = claims(&f);
    let mut tr = Transcript::new_default(b"masked-oracle");
    let (proof, _) = prove(
        &f.values,
        &f.write,
        &f.next,
        &f.initial,
        &f.initial_next,
        &f.gp,
        &c,
        &mut tr,
        b"mask",
    )
    .unwrap();
    let rejects = |proof: &MaskProof, c: &MaskClaims| {
        verify(
            &f.initial,
            &f.gp,
            c,
            proof,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask",
        )
        .is_err()
    };
    for round in 0..proof.sc.round_msgs.len() {
        for column in 0..3 {
            let mut bad = proof.clone();
            bad.sc.round_msgs[round][column] += K::ONE;
            assert!(rejects(&bad, &c));
        }
    }
    for field in 0..4 {
        let mut bad = proof.clone();
        match field {
            0 => bad.value += K::ONE,
            1 => bad.is_write += K::ONE,
            2 => bad.next_is_write += K::ONE,
            _ => bad.initial_next_is_write += K::ONE,
        };
        assert!(rejects(&bad, &c));
    }
    let mut bad = proof.clone();
    bad.sc.round_msgs.pop();
    assert!(rejects(&bad, &c));
    let mut bad = proof.clone();
    bad.sc.round_msgs[0].push(K::ZERO);
    assert!(rejects(&bad, &c));
    for field in 0..3 {
        let mut bad = c.clone();
        if field < 2 {
            bad.cycle[field] += K::ONE;
        } else {
            bad.initial += K::ONE;
        }
        assert!(rejects(&proof, &bad));
        if let Ok((bad_proof, _)) = prove(
            &f.values,
            &f.write,
            &f.next,
            &f.initial,
            &f.initial_next,
            &f.gp,
            &bad,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask",
        ) {
            assert!(rejects(&bad_proof, &bad));
        }
    }
    let mut wrong_memory = f.initial.clone();
    wrong_memory[0] ^= 1;
    assert!(
        verify(
            &wrong_memory,
            &f.gp,
            &c,
            &proof,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask"
        )
        .is_err()
    );
    assert!(
        verify(
            &[],
            &f.gp,
            &c,
            &proof,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask"
        )
        .is_err()
    );
    assert!(
        verify(
            &f.initial,
            &[],
            &c,
            &proof,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask"
        )
        .is_err()
    );
    assert!(
        prove(
            &f.values,
            &f.write[..1],
            &f.next,
            &f.initial,
            &f.initial_next,
            &f.gp,
            &c,
            &mut Transcript::new_default(b"masked-oracle"),
            b"mask"
        )
        .is_err()
    );
}
