//! Independent oracle for the native-domain replacement of common-cube rows.
//! The oracle deliberately retains the old four replicated columns and old
//! Eq-factor driver; its scalar Cofactor shares no native-domain statistics.

use super::*;
use om_sumcheck::{interpolate_eval, mle_eval, sumcheck_init};

struct NativeFixture {
    values: Vec<u64>,
    writes: Vec<bool>,
    next: Vec<bool>,
    memory: Vec<u64>,
    initial_next: Vec<bool>,
    gp: Vec<K>,
}

fn random_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn random_field(state: &mut u64) -> K {
    K::from_u128(u128::from(random_word(state)) | (u128::from(random_word(state)) << 64))
}

fn native_fixture(n: usize, a: usize, point_kind: usize, seed: u64) -> NativeFixture {
    let mut state = seed ^ ((n as u64) << 32) ^ a as u64;
    let values = (0..1usize << n)
        .map(|j| match j % 7 {
            0 => 0,
            1 => u64::MAX,
            _ => random_word(&mut state),
        })
        .collect();
    let memory = (0..1usize << a)
        .map(|j| match j % 5 {
            0 => 1,
            _ => random_word(&mut state),
        })
        .collect();
    let mut flags = |len| {
        (0..len)
            .map(|_| match point_kind {
                1 => false,
                2 => true,
                _ => random_word(&mut state) & 1 != 0,
            })
            .collect()
    };
    let writes = flags(1usize << n);
    let next = flags(1usize << n);
    let initial_next = flags(1usize << a);
    let gp = (0..=n.max(a))
        .map(|bit| match point_kind {
            0 => K::ZERO,
            1 => K::ONE,
            2 => K::from_bit(bit % 2 == 0),
            _ => random_field(&mut state),
        })
        .collect();
    NativeFixture {
        values,
        writes,
        next,
        memory,
        initial_next,
        gp,
    }
}

fn oracle_claims(f: &NativeFixture) -> MaskClaims {
    let masked = |values: &[u64], flags: &[bool]| {
        let point = &f.gp[..values.len().trailing_zeros() as usize];
        let table: Vec<K> = values
            .iter()
            .zip(flags)
            .map(|(&value, &flag)| {
                if flag {
                    K::ZERO
                } else {
                    K::from_u128(value as u128)
                }
            })
            .collect();
        mle_eval(&table, point)
    };
    MaskClaims {
        cycle: [masked(&f.values, &f.writes), masked(&f.values, &f.next)],
        initial: masked(&f.memory, &f.initial_next),
    }
}

/// Freeze the old context, including the full GP point's boundary coordinate.
fn oracle_setup(f: &NativeFixture, claims: &MaskClaims, tr: &mut Transcript) -> [K; 3] {
    let (a, l) = (f.memory.len().trailing_zeros() as usize, f.gp.len() - 1);
    tr.absorb_bytes(b"mask/protocol", b"spice/masked-reduction/replicated-eq/v1");
    tr.absorb_bytes(
        b"mask/dimensions",
        &[a as u64, l as u64, 64]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    tr.absorb_ks(b"mask/gp-point", &f.gp);
    tr.absorb_ks(
        b"mask/claims",
        &[claims.cycle[0], claims.cycle[1], claims.initial],
    );
    std::array::from_fn(|_| tr.challenge_k(b"mask/batch"))
}

fn replicated_columns(f: &NativeFixture, weights: [K; 3]) -> Vec<Vec<K>> {
    let mut columns = vec![Vec::with_capacity(1usize << (f.gp.len() - 1)); 4];
    for row in 0..1usize << (f.gp.len() - 1) {
        let j = row % f.values.len();
        let k = row % f.memory.len();
        let values = [
            K::from_u128(f.values[j] as u128),
            weights[0] * K::from_bit(!f.writes[j]) + weights[1] * K::from_bit(!f.next[j]),
            K::from_u128(f.memory[k] as u128),
            weights[2] * K::from_bit(!f.initial_next[k]),
        ];
        for (column, value) in columns.iter_mut().zip(values) {
            column.push(value);
        }
    }
    columns
}

struct ScalarDenseProducts;
impl Cofactor for ScalarDenseProducts {
    const DEGREE: usize = 2;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        4
    }
    fn value(&self, row: &[K]) -> K {
        row[0] * row[1] + row[2] * row[3]
    }
    fn prepare(&self, low_eq: &[K]) -> Vec<K> {
        low_eq.to_vec()
    }
    fn accumulate(
        &self,
        prepared: &Vec<K>,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    ) {
        let weight = prepared[low];
        if matches!(endpoint, Endpoint::Both | Endpoint::Zero) {
            out[0] += weight.clmul(self.value(left));
        }
        if matches!(endpoint, Endpoint::Both | Endpoint::One) {
            out[1] += weight.clmul(self.value(right));
        }
        let quadratic = (left[0] + right[0]) * (left[1] + right[1])
            + (left[2] + right[2]) * (left[3] + right[3]);
        out[2] += weight.clmul(quadratic);
    }
}

fn old_dense_oracle(f: &NativeFixture, claims: &MaskClaims, tr: &mut Transcript) -> FusedSumcheck {
    let weights = oracle_setup(f, claims, tr);
    let initial =
        weights[0] * claims.cycle[0] + weights[1] * claims.cycle[1] + weights[2] * claims.initial;
    let columns = replicated_columns(f, weights);
    let actual: K = eq_table(&f.gp[..f.gp.len() - 1]).iter().enumerate().fold(
        K::ZERO,
        |sum, (row, &weight)| {
            sum + weight * (columns[0][row] * columns[1][row] + columns[2][row] * columns[3][row])
        },
    );
    assert_eq!(actual, initial);
    let row = |j: usize, out: &mut [K]| {
        for (dst, column) in out.iter_mut().zip(&columns) {
            *dst = column[j];
        }
    };
    let (incoming, sc, point, folded) = eq_factor::prove_with_initial_claim(
        &f.gp[..f.gp.len() - 1],
        &ScalarDenseProducts,
        row,
        Some(initial),
        tr,
        b"mask/sc",
    );
    assert_eq!(incoming, initial);
    (sc, point, weights, folded[0], [folded[1], folded[3]])
}

/// A separate scalar finite-point driver for prescribed round challenges.
/// Transcript sampling still occurs in its normal place; only the returned
/// challenge is overridden. It never uses Eq-prefix normalization or native
/// source exhaustion, so it is an independent oracle for those edge cases.
fn forced_dense_oracle(
    f: &NativeFixture,
    claims: &MaskClaims,
    forced: &[K],
    tr: &mut Transcript,
) -> FusedSumcheck {
    let l = f.gp.len() - 1;
    assert_eq!(forced.len(), l);
    let weights = oracle_setup(f, claims, tr);
    let initial =
        weights[0] * claims.cycle[0] + weights[1] * claims.cycle[1] + weights[2] * claims.initial;
    let mut columns = vec![eq_table(&f.gp[..l])];
    columns.extend(replicated_columns(f, weights));
    let combine = |row: &[K; 5]| row[0] * (row[1] * row[2] + row[3] * row[4]);
    let mut cur = (0..columns[0].len()).fold(K::ZERO, |sum, row| {
        sum + combine(&std::array::from_fn(|column| columns[column][row]))
    });
    assert_eq!(cur, initial);
    sumcheck_init(tr, b"mask/sc", initial, l, 3);
    let mut messages = Vec::with_capacity(l);
    for &challenge in forced {
        let evaluations: Vec<K> = (0..4)
            .map(|node| {
                let at = K::t(node);
                (0..columns[0].len() / 2).fold(K::ZERO, |sum, pair| {
                    let row = std::array::from_fn(|column| {
                        let (a, b) = (columns[column][2 * pair], columns[column][2 * pair + 1]);
                        a + at * (a + b)
                    });
                    sum + combine(&row)
                })
            })
            .collect();
        assert_eq!(evaluations[0] + evaluations[1], cur);
        let message = vec![evaluations[0], evaluations[2], evaluations[3]];
        tr.absorb_ks(b"mask/sc/msg", &message);
        let _unmodified_transcript_draw = tr.challenge_k(b"mask/sc/chal");
        messages.push(message);
        cur = interpolate_eval(&evaluations, challenge);
        for column in &mut columns {
            *column = column
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[0] + pair[1]))
                .collect();
        }
    }
    let folded: [K; 5] = std::array::from_fn(|column| columns[column][0]);
    assert_eq!(cur, combine(&folded));
    (
        SumcheckProof {
            round_msgs: messages,
        },
        forced.to_vec(),
        weights,
        folded[1],
        [folded[2], folded[4]],
    )
}

fn check_fixture(f: &NativeFixture) {
    let claims = oracle_claims(f);
    let mut old_tr = Transcript::new_default(b"native-mask-oracle");
    let old = old_dense_oracle(f, &claims, &mut old_tr);
    let mut native_tr = Transcript::new_default(b"native-mask-oracle");
    let native = prove_sumcheck(
        &f.values,
        &f.writes,
        &f.next,
        &f.memory,
        &f.initial_next,
        &f.gp,
        &claims,
        &mut native_tr,
        b"mask",
    )
    .unwrap();
    assert_eq!(native, old, "complete sumcheck and terminal masks");
    assert_eq!(
        native_tr, old_tr,
        "exact transcript before deferred sources"
    );
    let (n, a) = (
        f.values.len().trailing_zeros() as usize,
        f.memory.len().trailing_zeros() as usize,
    );
    let values: Vec<K> = f.values.iter().map(|&v| K::from_u128(v as u128)).collect();
    let flags = [
        mle_eval_bits(&f.writes, &native.1[..n]),
        mle_eval_bits(&f.next, &native.1[..n]),
        mle_eval_bits(&f.initial_next, &native.1[..a]),
    ];
    assert_eq!(native.3, mle_eval(&values, &native.1[..n]));
    assert_eq!(
        native.4[0],
        native.2[0] * (K::ONE + flags[0]) + native.2[1] * (K::ONE + flags[1])
    );
    assert_eq!(native.4[1], native.2[2] * (K::ONE + flags[2]));
    let mut verifier = Transcript::new_default(b"native-mask-oracle");
    let deferred =
        verify_sumcheck(&f.memory, &f.gp, &claims, &native.0, &mut verifier, b"mask").unwrap();
    assert_eq!(deferred.point, native.1);
    deferred
        .check(native.3, flags[0], flags[1], flags[2])
        .unwrap();
    assert_eq!(verifier, native_tr);
    let mut wrapper_tr = Transcript::new_default(b"native-mask-oracle");
    let (proof, sources) = prove(
        &f.values,
        &f.writes,
        &f.next,
        &f.memory,
        &f.initial_next,
        &f.gp,
        &claims,
        &mut wrapper_tr,
        b"mask",
    )
    .unwrap();
    let expected = MaskProof {
        sc: old.0,
        value: old.3,
        is_write: flags[0],
        next_is_write: flags[1],
        initial_next_is_write: flags[2],
    };
    assert_eq!(proof, expected);
    assert_eq!(sources.point, old.1);
    old_tr.absorb_ks(
        b"mask/sources",
        &[
            expected.value,
            expected.is_write,
            expected.next_is_write,
            expected.initial_next_is_write,
        ],
    );
    assert_eq!(wrapper_tr, old_tr, "complete prover transcript");
    let mut full_verifier = Transcript::new_default(b"native-mask-oracle");
    assert_eq!(
        verify(
            &f.memory,
            &f.gp,
            &claims,
            &proof,
            &mut full_verifier,
            b"mask"
        )
        .unwrap(),
        sources
    );
    assert_eq!(full_verifier, old_tr, "complete proof transcript");
    assert_eq!(
        full_verifier.challenge_k(b"after"),
        old_tr.challenge_k(b"after")
    );
}

#[test]
fn native_domains_match_old_dense_driver_for_unequal_equal_and_zero_dimensions() {
    for (n, a) in [
        (0, 0),
        (0, 4),
        (4, 0),
        (1, 2),
        (2, 1),
        (4, 4),
        (7, 3),
        (3, 7),
        (9, 1),
        (1, 9),
    ] {
        for point_kind in 0..4 {
            check_fixture(&native_fixture(n, a, point_kind, 0x61f0_d35e_9001));
        }
    }
}

#[test]
fn native_domains_match_old_dense_driver_in_sequential_and_parallel_pools() {
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for (n, a) in [(14, 3), (3, 14), (14, 14)] {
                check_fixture(&native_fixture(n, a, 3, 0x93ee_104c));
            }
        });
    }
}

#[test]
fn all_masked_and_public_zero_sources_still_return_their_correct_terminal_values() {
    for (n, a) in [(1, 7), (7, 1), (5, 5)] {
        let mut f = native_fixture(n, a, 3, 0x9173_b4e2);
        f.writes.fill(true);
        f.next.fill(true);
        f.initial_next.fill(true);
        check_fixture(&f);
        f.memory.fill(0);
        f.initial_next.fill(false);
        f.writes.fill(false);
        f.next.fill(false);
        check_fixture(&f);
    }
}

#[test]
fn forced_boolean_challenges_and_zero_equality_prefixes_match_dense_scalar_messages() {
    for (n, a) in [
        (0, 0),
        (0, 4),
        (4, 0),
        (1, 5),
        (5, 1),
        (2, 5),
        (5, 2),
        (4, 4),
    ] {
        let l = n.max(a);
        for point_kind in 0..4 {
            let f = native_fixture(n, a, point_kind, 0x0419_a111_8477);
            let claims = oracle_claims(&f);
            let mut prescriptions: Vec<(Vec<K>, Option<usize>)> = vec![
                (vec![K::ZERO; l], None),
                (vec![K::ONE; l], None),
                ((0..l).map(|bit| K::from_bit(bit % 2 == 0)).collect(), None),
            ];
            if l != 0 {
                // Kill equality before, at and after the smaller source's
                // exhaustion. Subsequent messages vanish but sources must
                // still fold all the way to their correct terminal values.
                let small = n.min(a);
                let mut kill_rounds = vec![0, small.saturating_sub(1), small.min(l - 1), l - 1];
                kill_rounds.sort_unstable();
                kill_rounds.dedup();
                for kill in kill_rounds {
                    let mut forced: Vec<K> = (0..l).map(|bit| K::from_bit(bit % 2 != 0)).collect();
                    forced[kill] = K::ONE + f.gp[kill];
                    prescriptions.push((forced, Some(kill)));
                }
            }
            for (forced, kill) in prescriptions {
                let mut expected_tr = Transcript::new_default(b"forced-native-mask");
                let expected = forced_dense_oracle(&f, &claims, &forced, &mut expected_tr);
                let mut actual_tr = Transcript::new_default(b"forced-native-mask");
                let actual = prove_sumcheck_with_challenges(
                    &f.values,
                    &f.writes,
                    &f.next,
                    &f.memory,
                    &f.initial_next,
                    &f.gp,
                    &claims,
                    &mut actual_tr,
                    b"mask",
                    &forced,
                )
                .unwrap();
                assert_eq!(
                    actual, expected,
                    "n={n} a={a} point_kind={point_kind} kill={kill:?}"
                );
                assert_eq!(actual_tr, expected_tr);
                let values: Vec<K> = f.values.iter().map(|&v| K::from_u128(v as u128)).collect();
                assert_eq!(actual.3, mle_eval(&values, &forced[..n]));
                if let Some(kill) = kill {
                    for message in &actual.0.round_msgs[kill + 1..] {
                        assert!(message.iter().all(|value| value.is_zero()));
                    }
                }
                assert_eq!(
                    actual_tr.challenge_k(b"after"),
                    expected_tr.challenge_k(b"after")
                );
            }
        }
    }
}

#[test]
fn exhausted_native_sources_stay_scalar_without_allocating_common_cube_tables() {
    let common_dimension = 20;
    for dimension in [0, 1, 3, 10] {
        let mut state = 0xe25e_171d ^ dimension as u64;
        let source: [Vec<K>; 2] = std::array::from_fn(|_| {
            (0..1usize << dimension)
                .map(|_| random_field(&mut state))
                .collect()
        });
        let point: Vec<K> = (0..common_dimension)
            .map(|_| random_field(&mut state))
            .collect();
        for skip_statistics in [false, true] {
            let mut native = NativeProduct::new(dimension, |row| [source[0][row], source[1][row]]);
            let _initial_statistics = native.initial_statistics(&point, true);
            assert!(
                native
                    .columns
                    .iter()
                    .chain(&native.scratch)
                    .all(Vec::is_empty)
            );
            let forced: Vec<K> = (0..common_dimension)
                .map(|round| K::from_bit(round % 2 == 0))
                .collect();
            for (round, &challenge) in forced.iter().enumerate() {
                let at_one = round % 2 == 0;
                let next = if skip_statistics { None } else { Some(at_one) };
                let stats = native.bind_and_statistics(round, challenge, &point, next);
                // Two columns, each with at most a half-domain table and one
                // smaller scratch buffer. The common cube has 2^20 rows.
                let stored: usize = native
                    .columns
                    .iter()
                    .chain(&native.scratch)
                    .map(Vec::len)
                    .sum();
                assert!(stored <= 3 * (1usize << dimension));
                if round + 1 >= dimension {
                    let expected = [
                        mle_eval(&source[0], &forced[..dimension]),
                        mle_eval(&source[1], &forced[..dimension]),
                    ];
                    assert_eq!(native.terminal(), expected);
                    assert!(
                        native
                            .columns
                            .iter()
                            .chain(&native.scratch)
                            .all(|column| column.len() <= 2)
                    );
                    let mut expected_stats = [K::ZERO; 3];
                    if let Some(at_one) = next {
                        expected_stats[usize::from(at_one)] = expected[0] * expected[1];
                    }
                    assert_eq!(stats, expected_stats);
                }
            }
        }
    }
}
