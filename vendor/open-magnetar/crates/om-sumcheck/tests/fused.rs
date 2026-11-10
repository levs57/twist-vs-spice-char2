use om_field::K;
use om_sumcheck::{
    SumcheckProof, bind_low, bind_low_and_round_evals_into, bind_low_and_round_evals_mul2_into,
    bind_low_in_place, combine_sum, interpolate_eval, sumcheck_init, sumcheck_prove,
    sumcheck_prove_rounds, sumcheck_prove_rounds_mul2, sumcheck_round_evals,
};
use om_transcript::Transcript;

struct Rng(u64);
impl Rng {
    fn field(&mut self) -> K {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        K::from_u128(u128::from(value) | (u128::from(self.0) << 64))
    }
}

fn old_interpolate(ys: &[K], point: K) -> K {
    let mut result = K::ZERO;
    for (i, &value) in ys.iter().enumerate() {
        let mut numerator = K::ONE;
        let mut denominator = K::ONE;
        for j in 0..ys.len() {
            if j != i {
                numerator *= point + K::t(j as u128);
                denominator *= K::t((i ^ j) as u128);
            }
        }
        result += value * numerator / denominator;
    }
    result
}

fn polynomial(coefficients: &[K], point: K) -> K {
    coefficients
        .iter()
        .rev()
        .fold(K::ZERO, |result, &c| result * point + c)
}

#[test]
fn cached_interpolation_matches_original_for_empty_nodes_and_arbitrary_lengths() {
    let mut rng = Rng(0xabba);
    for count in [0usize, 1, 2, 3, 4, 5, 6, 10, 17, 65, 67] {
        let coefficients: Vec<K> = (0..count).map(|_| rng.field()).collect();
        let values: Vec<K> = (0..count)
            .map(|i| polynomial(&coefficients, K::t(i as u128)))
            .collect();
        for _ in 0..3 {
            let point = rng.field();
            assert_eq!(
                interpolate_eval(&values, point),
                polynomial(&coefficients, point)
            );
            assert_eq!(
                interpolate_eval(&values, point),
                old_interpolate(&values, point)
            );
        }
        for i in 0..count {
            assert_eq!(interpolate_eval(&values, K::t(i as u128)), values[i]);
        }
    }
}

#[test]
fn interpolation_cache_can_initialize_concurrently() {
    std::thread::scope(|scope| {
        for worker in 0..8 {
            scope.spawn(move || {
                let mut rng = Rng(worker + 19);
                for count in [13usize, 19, 71] {
                    let coefficients: Vec<K> = (0..count).map(|_| rng.field()).collect();
                    let values: Vec<K> = (0..count)
                        .map(|i| polynomial(&coefficients, K::t(i as u128)))
                        .collect();
                    let point = rng.field();
                    assert_eq!(
                        interpolate_eval(&values, point),
                        polynomial(&coefficients, point)
                    );
                }
            });
        }
    });
}

fn tables(log: usize, width: usize, rng: &mut Rng) -> Vec<Vec<K>> {
    (0..width)
        .map(|_| (0..1usize << log).map(|_| rng.field()).collect())
        .collect()
}

#[test]
fn fused_round_and_bound_tables_match_separate_reference_across_threads() {
    for threads in [1usize, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut rng = Rng(0xfeed);
                for degree in [2usize, 3, 5] {
                    let combine = |values: &[K]| values.iter().copied().fold(K::ONE, |a, b| a * b);
                    for log in [1usize, 2, 3, 10, 14, 16] {
                        let input = tables(log, degree, &mut rng);
                        let mut bound = vec![vec![K::ONE; input[0].len()]; degree];
                        for point in [K::ZERO, K::ONE, rng.field()] {
                            let reference: Vec<Vec<K>> =
                                input.iter().map(|table| bind_low(table, point)).collect();
                            let expected = if log == 1 {
                                Vec::new()
                            } else {
                                sumcheck_round_evals(&reference, &combine, degree)
                            };
                            let claim = combine_sum(&reference, &combine);
                            for known_claim in [None, Some(claim)] {
                                let actual = bind_low_and_round_evals_into(
                                    &input,
                                    &mut bound,
                                    point,
                                    &combine,
                                    degree,
                                    known_claim,
                                );
                                assert_eq!(
                                    bound, reference,
                                    "threads={threads} degree={degree} log={log}"
                                );
                                assert_eq!(
                                    actual, expected,
                                    "threads={threads} degree={degree} log={log}"
                                );
                                if degree == 2 {
                                    let specialized = bind_low_and_round_evals_mul2_into(
                                        &input,
                                        &mut bound,
                                        point,
                                        known_claim,
                                    );
                                    assert_eq!(bound, reference);
                                    assert_eq!(specialized, expected);
                                }
                            }
                        }
                    }
                }
            });
    }
}

#[test]
fn generic_quadratic_is_never_guessed_to_be_multiplication() {
    let input = vec![vec![K::ZERO, K::ONE], vec![K::ZERO, K::ONE]];
    let combine = |v: &[K]| v[0] * v[1] + v[0] * (v[0] + K::ONE);
    assert_eq!(
        sumcheck_round_evals(&input, &combine, 2),
        vec![K::ZERO, K::ONE, K::X]
    );
    let (proof, point, values) = sumcheck_prove(
        K::ONE,
        input,
        &combine,
        2,
        &mut Transcript::new_default(b"general"),
        b"sc",
    );
    let (_, expected) = om_sumcheck::sumcheck_verify(
        K::ONE,
        1,
        2,
        &proof,
        &mut Transcript::new_default(b"general"),
        b"sc",
    )
    .unwrap();
    assert_eq!(expected, combine(&values));
    assert_eq!(expected, point[0]);
    let mut rng = Rng(9123);
    let input = tables(5, 2, &mut rng);
    let r = rng.field();
    let reference: Vec<Vec<K>> = input.iter().map(|table| bind_low(table, r)).collect();
    let mut bound = Vec::new();
    assert_eq!(
        bind_low_and_round_evals_into(&input, &mut bound, r, &combine, 2, None),
        sumcheck_round_evals(&reference, &combine, 2)
    );
}

fn separate_prove(
    claim: K,
    mut tables: Vec<Vec<K>>,
    combine: &(dyn Fn(&[K]) -> K + Sync),
    degree: usize,
    rounds: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, Vec<K>, Vec<Vec<K>>) {
    sumcheck_init(transcript, b"sc", claim, rounds, degree);
    let mut messages = Vec::new();
    let mut point = Vec::new();
    let mut current = claim;
    for _ in 0..rounds {
        let evaluations = sumcheck_round_evals(&tables, combine, degree);
        assert_eq!(evaluations[0] + evaluations[1], current);
        let mut message = vec![evaluations[0]];
        message.extend_from_slice(&evaluations[2..]);
        transcript.absorb_ks(b"sc/msg", &message);
        let r = transcript.challenge_k(b"sc/chal");
        current = old_interpolate(&evaluations, r);
        messages.push(message);
        point.push(r);
        tables = tables.iter().map(|table| bind_low(table, r)).collect();
    }
    (
        SumcheckProof {
            round_msgs: messages,
        },
        point,
        tables,
    )
}

#[test]
fn pipelined_full_and_partial_proofs_preserve_messages_points_and_final_tables() {
    for threads in [1usize, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut rng = Rng(551);
                for degree in [2usize, 3, 5] {
                    let combine = |v: &[K]| v.iter().copied().fold(K::ONE, |a, b| a * b);
                    for log in [0usize, 1, 3, 10, 14] {
                        let input = tables(log, degree, &mut rng);
                        let claim = combine_sum(&input, &combine);
                        let mut counts = vec![0, 1.min(log), 2.min(log), log / 2, log];
                        counts.sort_unstable();
                        counts.dedup();
                        for rounds in counts {
                            let mut expected_tr = Transcript::new_default(b"pipeline");
                            let expected = separate_prove(
                                claim,
                                input.clone(),
                                &combine,
                                degree,
                                rounds,
                                &mut expected_tr,
                            );
                            let mut actual_tr = Transcript::new_default(b"pipeline");
                            let actual = sumcheck_prove_rounds(
                                claim,
                                input.clone(),
                                &combine,
                                degree,
                                rounds,
                                &mut actual_tr,
                                b"sc",
                            );
                            assert_eq!(
                                actual, expected,
                                "degree={degree} log={log} rounds={rounds}"
                            );
                            assert_eq!(actual_tr.state(), expected_tr.state());
                            if degree == 2 {
                                let mut product_tr = Transcript::new_default(b"pipeline");
                                let product = sumcheck_prove_rounds_mul2(
                                    claim,
                                    input.clone(),
                                    rounds,
                                    &mut product_tr,
                                    b"sc",
                                );
                                assert_eq!(product, expected);
                                assert_eq!(product_tr.state(), expected_tr.state());
                            }
                        }
                    }
                }
            });
    }
}

#[test]
fn large_parallel_in_place_binding_matches_direct_even_length_reference() {
    let mut rng = Rng(887);
    for length in [0usize, 2, 6, 1 << 14, (1 << 14) + 2, (1 << 15) + 2] {
        let input: Vec<K> = (0..length).map(|_| rng.field()).collect();
        let point = rng.field();
        let expected: Vec<K> = input
            .chunks_exact(2)
            .map(|pair| pair[0] + point * (pair[0] + pair[1]))
            .collect();
        let mut actual = input;
        bind_low_in_place(&mut actual, point);
        assert_eq!(actual, expected);
    }
}
