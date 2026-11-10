//! Deterministic interoperability checks using the unmodified upstream APIs.
//! This checks arithmetic and MLE conventions, not protocol soundness or speed.

use flock_field::{F128, f128_slice};
use flock_multilinear::{IndexOrder, eq_table, evaluate, fold_low};
use om_field::K;

fn flock(value: K) -> F128 {
    F128::new(value.value() as u64, (value.value() >> 64) as u64)
}

fn magnetar(value: F128) -> K {
    K::from_u128(u128::from(value.lo) | (u128::from(value.hi) << 64))
}

fn random(state: &mut u128) -> K {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    K::from_u128(*state)
}

fn check_ring_switch(state: &mut u128) {
    use flock_core::pcs::ring_switch as frs;
    use flock_core::zerocheck::PaddingSpec;
    use om_pcs::ring_switch as ors;

    for log_bits in [8usize, 12, 16] {
        let bits: Vec<bool> = (0..1usize << log_bits)
            .map(|_| random(state).value() & 1 != 0)
            .collect();
        let point: Vec<K> = (0..log_bits).map(|_| random(state)).collect();
        let fpoint: Vec<F128> = point.iter().copied().map(flock).collect();
        let packed = ors::pack(&bits);
        let fpacked = flock_core::pcs::pack_witness(&bits, log_bits);
        assert_eq!(
            packed,
            fpacked.iter().copied().map(magnetar).collect::<Vec<_>>()
        );
        let expected = ors::partial_evals(&bits, &point[7..]);
        let ftensor = eq_table(&fpoint[7..], F128::ONE, IndexOrder::LowToHigh);
        let actual = frs::fold_1b_rows_naive(&fpacked, &ftensor);
        assert_eq!(
            expected,
            actual.iter().copied().map(magnetar).collect::<Vec<_>>()
        );
        if log_bits >= 12 {
            let lo_vars = 4;
            let (lo, hi) = frs::build_eq_split(&fpoint[7..], lo_vars);
            let split = frs::fold_1b_rows_split(&fpacked, &lo, &hi, &PaddingSpec::dense(log_bits));
            assert_eq!(actual, split);
        }
        let skip_weights = eq_table(&fpoint[..6], F128::ONE, IndexOrder::LowToHigh);
        let prefix = frs::build_claim_weights_from_skip(&skip_weights, fpoint[6]);
        assert_eq!(
            om_sumcheck::eq_table(&point[..7]),
            prefix.iter().copied().map(magnetar).collect::<Vec<_>>()
        );
        assert_eq!(
            om_sumcheck::mle_eval_bits(&bits, &point),
            magnetar(frs::claim_check(&prefix, &actual))
        );
        let rho: Vec<K> = (0..7).map(|_| random(state)).collect();
        let fweights = eq_table(
            &rho.iter().copied().map(flock).collect::<Vec<_>>(),
            F128::ONE,
            IndexOrder::LowToHigh,
        );
        assert_eq!(
            ors::transpose_bits(&expected),
            frs::tensor_algebra_transpose(&actual)
                .into_iter()
                .map(magnetar)
                .collect::<Vec<_>>()
        );
        let a = ors::a_table(&point[7..], &rho);
        let fa = frs::fold_b128_elems(&ftensor, &fweights);
        assert_eq!(a, fa.iter().copied().map(magnetar).collect::<Vec<_>>());
        let target = frs::inner_product(&frs::tensor_algebra_transpose(&actual), &fweights);
        assert_eq!(target, frs::inner_product(&fpacked, &fa));
        let z: Vec<K> = (7..log_bits).map(|_| random(state)).collect();
        let fz: Vec<F128> = z.iter().copied().map(flock).collect();
        let expected_a = ors::a_eval(&point[7..], &rho, &z);
        assert_eq!(expected_a, om_sumcheck::mle_eval(&a, &z));
        assert_eq!(
            expected_a,
            magnetar(frs::eval_rs_eq(&fpoint[7..], &fz, &fweights))
        );
    }
    println!(
        "PASS: real RingSwitch packing, slice-fold kernels (including split MFR), ordinary-prefix adapter and claim check, tensor transpose, transformed equality table, inner-product target, and succinct weight evaluation agree at 8/12/16 bit-witness variables."
    );
}

fn check_ntt(state: &mut u128) {
    use flock_core::ntt::AdditiveNttF128;

    let mut shapes = 0;
    for log_cols in [1usize, 4, 8, 10] {
        for log_lanes in [0usize, 2, 4] {
            for log_inv_rate in [1usize, 2] {
                let lanes = 1usize << log_lanes;
                let cols = 1usize << log_cols;
                let words = cols << log_inv_rate;
                let message: Vec<K> = (0..cols * lanes).map(|_| random(state)).collect();
                let mut actual = vec![F128::ZERO; words * lanes];
                for (out, input) in actual.iter_mut().zip(&message) {
                    *out = flock(*input);
                }
                let ntt = AdditiveNttF128::standard(log_cols + log_inv_rate);
                ntt.forward_transform_interleaved(&mut actual, lanes);
                let mut expected = vec![K::ZERO; words * lanes];
                for lane in 0..lanes {
                    let lane_message: Vec<K> =
                        (0..cols).map(|col| message[col * lanes + lane]).collect();
                    let encoded = om_pcs::ntt::encode(&lane_message, log_inv_rate);
                    for (q, value) in encoded.into_iter().enumerate() {
                        expected[q * lanes + lane] = value;
                    }
                }
                assert_eq!(
                    expected,
                    actual.into_iter().map(magnetar).collect::<Vec<_>>()
                );
                shapes += 1;
            }
        }
    }
    println!(
        "PASS: {shapes} cross-repository NTT codeword/layout shapes (log2 columns 1/4/8/10, lanes 1/4/16, inverse rates 2/4)."
    );
}

fn main() {
    let mut state = 0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344u128;
    let mut products = 0;
    for i in 0..128 {
        for j in 0..128 {
            let a = K::from_u128(1u128 << i);
            let b = K::from_u128(1u128 << j);
            assert_eq!(a * b, magnetar(flock(a) * flock(b)));
            products += 1;
        }
    }
    for _ in 0..4096 {
        let a = random(&mut state);
        let b = random(&mut state);
        assert_eq!(a * b, magnetar(flock(a) * flock(b)));
        assert_eq!(a + b, magnetar(flock(a) + flock(b)));
        assert_eq!(a.square(), magnetar(flock(a).square()));
        assert_eq!(K::from_bytes(&a.to_bytes()), magnetar(flock(a)));
        products += 1;
    }
    for _ in 0..16 {
        let a = random(&mut state);
        assert!(!a.is_zero());
        assert_eq!(a.inverse(), magnetar(flock(a).inv()));
    }
    let mut mle_shapes = 0;
    for log_n in 0..=14 {
        let point: Vec<K> = (0..log_n).map(|_| random(&mut state)).collect();
        let fpoint: Vec<F128> = point.iter().copied().map(flock).collect();
        let expected = om_sumcheck::eq_table(&point);
        let actual: Vec<K> = eq_table(&fpoint, F128::ONE, IndexOrder::LowToHigh)
            .into_iter()
            .map(magnetar)
            .collect();
        assert_eq!(expected, actual, "equality order at {log_n} variables");
        let table: Vec<K> = (0..1usize << log_n).map(|_| random(&mut state)).collect();
        let ftable: Vec<F128> = table.iter().copied().map(flock).collect();
        assert_eq!(
            om_sumcheck::mle_eval(&table, &point),
            magnetar(evaluate(&ftable, &fpoint, IndexOrder::LowToHigh)),
            "MLE at {log_n} variables"
        );
        if log_n > 0 {
            let r = point[0];
            let mut om_bound = table.clone();
            om_sumcheck::bind_low_in_place(&mut om_bound, r);
            let mut flock_bound = ftable.clone();
            fold_low(&mut flock_bound, flock(r));
            assert_eq!(
                om_bound,
                flock_bound.into_iter().map(magnetar).collect::<Vec<_>>()
            );
            let mut slice_bound = vec![F128::ZERO; table.len() / 2];
            f128_slice::fold_pairs(&ftable, 0, &mut slice_bound, flock(r));
            assert_eq!(
                om_bound,
                slice_bound.into_iter().map(magnetar).collect::<Vec<_>>()
            );
        }
        mle_shapes += 1;
    }
    // Every polynomial basis bit has the same 128-bit packing coordinate.
    for bit in 0..128 {
        let mut bits = [false; 128];
        bits[bit] = true;
        assert_eq!(
            flock(K::pack(&bits)),
            F128::new((1u128 << bit) as u64, ((1u128 << bit) >> 64) as u64)
        );
    }
    println!(
        "PASS: {products} products (all 16,384 basis pairs + 4,096 deterministic random pairs), 4,096 sums/squares/byte roundtrips, 16 nonzero inverses, {mle_shapes} equality/MLE shapes (0..14 variables), 14 low-fold shapes through both Flock APIs, 128 packing coordinates."
    );
    println!(
        "Compile-time Flock scalar PCLMUL backend: {}",
        cfg!(target_feature = "pclmulqdq")
    );
    println!(
        "Compile-time Flock AVX-512 slice-fold backend: {}",
        cfg!(all(
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))
    );

    check_ring_switch(&mut state);
    check_ntt(&mut state);

    // This is a positive reproducer for an upstream optimization-dispatch bug.
    // The closure agrees with multiplication on {0,1}² but differs off the cube.
    // Its composition with a(t)=b(t)=t has degree two, as the API permits.
    let tables = vec![vec![K::ZERO, K::ONE], vec![K::ZERO, K::ONE]];
    let combine = |v: &[K]| v[0] * v[1] + v[0] * (v[0] + K::ONE);
    let actual = om_sumcheck::sumcheck_round_evals(&tables, &combine, 2);
    let expected: Vec<K> = (0..=2).map(|i| combine(&[K::t(i), K::t(i)])).collect();
    assert_eq!(actual[..2], expected[..2]);
    assert_ne!(
        actual[2], expected[2],
        "upstream bug was fixed; update the audit finding"
    );
    assert_eq!(actual[2], K::from_u128(4));
    assert_eq!(expected[2], K::from_u128(2));
    println!(
        "CONFIRMED UPSTREAM BUG: generic degree-2 combine is misidentified as multiplication from Boolean-corner probes: g(t2) returns integer 4, expected integer 2 (no T&S exploit established)."
    );
}
