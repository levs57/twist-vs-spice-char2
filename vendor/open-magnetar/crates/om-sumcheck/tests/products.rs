use om_field::K;
use om_sumcheck::{
    combine_sum, eq_table, product_sum, sumcheck_prove, sumcheck_prove_rounds_eq_mul2,
    sumcheck_prove_rounds_mul2, sumcheck_prove_weighted_mul2,
};
use om_transcript::{Hasher, Transcript};

struct Rng(u64);
impl Rng {
    fn field(&mut self) -> K {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        K::from_u128(u128::from(z) | (u128::from(self.0) << 64))
    }
}

#[test]
fn equality_product_matches_dense_partial_rounds_and_transcript() {
    let mut rng = Rng(0x51a7);
    // n=15 crosses the Rayon path and split-equality block boundaries.
    for n in [0usize, 1, 2, 5, 10, 15] {
        for kind in 0..3 {
            let point: Vec<K> = (0..n)
                .map(|i| match kind {
                    0 => K::from_bit(i % 2 == 0),
                    1 => K::ZERO,
                    _ => rng.field(),
                })
                .collect();
            let f: Vec<K> = (0..1usize << n).map(|_| rng.field()).collect();
            let b = eq_table(&point);
            let claim = product_sum(&f, &b);
            let mut counts = vec![0, n / 2, n];
            counts.sort_unstable();
            counts.dedup();
            for rounds in counts {
                let mut actual_tr = Transcript::new(b"eq-product-oracle", Hasher::Sha256);
                let mut dense_tr = Transcript::new(b"eq-product-oracle", Hasher::Sha256);
                let actual = sumcheck_prove_rounds_eq_mul2(
                    claim,
                    f.clone(),
                    &point,
                    rounds,
                    &mut actual_tr,
                    b"sc",
                );
                let dense = sumcheck_prove_rounds_mul2(
                    claim,
                    vec![f.clone(), b.clone()],
                    rounds,
                    &mut dense_tr,
                    b"sc",
                );
                assert_eq!(actual, dense, "n={n} kind={kind} rounds={rounds}");
                assert_eq!(
                    actual_tr.challenge_k(b"after"),
                    dense_tr.challenge_k(b"after")
                );
            }
        }
    }
}

#[test]
fn weighted_product_matches_generic_with_zero_weights_and_unscaled_terminals() {
    let mut rng = Rng(0x7755);
    for n in [0usize, 1, 3, 11, 14] {
        for groups in [1usize, 2, 3] {
            for all_zero in [false, true] {
                let weights: Vec<K> = (0..groups)
                    .map(|i| {
                        if all_zero || i == 1 {
                            K::ZERO
                        } else {
                            rng.field()
                        }
                    })
                    .collect();
                let tables: Vec<Vec<K>> = (0..2 * groups)
                    .map(|_| (0..1usize << n).map(|_| rng.field()).collect())
                    .collect();
                let combine = |v: &[K]| {
                    (0..groups).fold(K::ZERO, |s, g| s + weights[g] * v[g] * v[groups + g])
                };
                let claim = combine_sum(&tables, &combine);
                let mut actual_tr = Transcript::new(b"weighted-product-oracle", Hasher::Sha256);
                let mut generic_tr = Transcript::new(b"weighted-product-oracle", Hasher::Sha256);
                let actual =
                    sumcheck_prove_weighted_mul2(tables.clone(), &weights, &mut actual_tr, b"sc");
                let generic = sumcheck_prove(claim, tables, &combine, 2, &mut generic_tr, b"sc");
                assert_eq!(actual, generic, "n={n} groups={groups} zero={all_zero}");
                assert_eq!(
                    actual_tr.challenge_k(b"after"),
                    generic_tr.challenge_k(b"after")
                );
            }
        }
    }
}

#[test]
fn delayed_dot_matches_reduced_arithmetic_on_empty_and_parallel_domains() {
    let mut rng = Rng(0xffaa);
    for n in [0usize, 1, 7, 128, 16385] {
        let a: Vec<K> = (0..n).map(|_| rng.field()).collect();
        let b: Vec<K> = (0..n).map(|_| rng.field()).collect();
        assert_eq!(
            product_sum(&a, &b),
            a.iter().zip(&b).fold(K::ZERO, |s, (&a, &b)| s + a * b)
        );
    }
}

#[test]
#[should_panic(expected = "claim inconsistent with equality product")]
fn equality_product_checks_the_original_initial_claim() {
    let point = vec![K::from_u128(0xf1), K::ONE];
    let f = vec![
        K::from_u128(3),
        K::from_u128(5),
        K::from_u128(7),
        K::from_u128(9),
    ];
    let claim = product_sum(&f, &eq_table(&point)) + K::ONE;
    let mut transcript = Transcript::new(b"inconsistent-equality", Hasher::Sha256);
    sumcheck_prove_rounds_eq_mul2(claim, f, &point, 2, &mut transcript, b"sc");
}
