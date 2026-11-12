use super::*;
use crate::twist2::{Config, CpuClaims, ReadOrder};

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Replay independently of either RAM prover. Initial values are nonzero,
/// and rows include changing stores and accesses that leave memory unchanged.
fn fixture(n: usize, a: usize, word_log_bits: usize, seed: u64) -> (AccessTrace, Vec<u64>) {
    let mut random = seed | 1;
    let mask = if word_log_bits == 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << word_log_bits)) - 1
    };
    let initial: Vec<_> = (0..1usize << a)
        .map(|_| (next(&mut random) | 1) & mask)
        .collect();
    let mut memory = initial.clone();
    let mut writes = Vec::with_capacity(1usize << n);
    let mut old_values = Vec::with_capacity(1usize << n);
    let mut new_values = Vec::with_capacity(1usize << n);
    for j in 0..1usize << n {
        let x = next(&mut random);
        let address = if j % 8 < 6 {
            ((j / 8) * 7 + (j % 3)) % memory.len()
        } else {
            x as usize % memory.len()
        };
        let old = memory[address];
        let new = if j % 3 == 0 {
            next(&mut random) & mask
        } else {
            old
        };
        writes.push(address as u32);
        old_values.push(old);
        new_values.push(new);
        memory[address] = new;
    }
    (
        AccessTrace {
            initial,
            reads: vec![writes.clone()],
            writes,
            new_values,
            read_values: Some(vec![old_values]),
        },
        memory,
    )
}

/// Direct scalar Lagrange evaluation: does not use SplitEq, eq_table,
/// word::mle, or either protocol's witness preparation.
fn scalar_basis(index: usize, point: &[K]) -> K {
    point.iter().enumerate().fold(K::ONE, |weight, (bit, &r)| {
        weight
            * if (index >> bit) & 1 == 1 {
                r
            } else {
                K::ONE + r
            }
    })
}

fn scalar_eval(values: impl IntoIterator<Item = u64>, point: &[K]) -> K {
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| scalar_basis(index, point) * K::from_u128(value as u128))
        .fold(K::ZERO, |sum, value| sum + value)
}

fn assert_interface(trace: &AccessTrace, interface: &crate::twist2::Interface) {
    let expected = CpuClaims {
        address: scalar_eval(trace.writes.iter().map(|&a| a as u64), &interface.point),
        read: scalar_eval(
            trace.read_values.as_ref().unwrap()[0].iter().copied(),
            &interface.point,
        ),
        write: scalar_eval(trace.new_values.iter().copied(), &interface.point),
    };
    assert_eq!(interface.claims, expected);
}

fn config(
    a: usize,
    c: usize,
    word_log_bits: usize,
    order: ReadOrder,
    public_final: bool,
) -> Config {
    Config {
        address_bits: a,
        chunk_bits: c,
        word_log_bits,
        order,
        public_final,
        ..Config::default()
    }
}

#[test]
fn direct_pushforward_twist_matches_independent_ram_and_cpu_oracles() {
    for (n, a, c, width) in [
        (1, 2, 2, 6),
        (3, 4, 4, 0),
        (6, 4, 2, 3),
        (5, 7, 7, 6), // memory domain wider than the cycle domain
        (9, 6, 3, 6),
    ] {
        let (trace, final_memory) = fixture(n, a, width, 31 * n as u64 + a as u64);
        for order in [ReadOrder::CycleFirst, ReadOrder::AddressFirst] {
            for public_final in [true, false] {
                let cfg = config(a, c, width, order, public_final);
                let result = prove(&cfg, &trace).unwrap();
                let verified = verify(&result.statement, &result.proof).unwrap_or_else(|e| {
                    panic!("n={n} a={a} width={width} {order:?} public={public_final}: {e}")
                });
                assert_eq!(result.final_memory, final_memory);
                assert_eq!(result.interface, verified);
                assert_interface(&trace, &verified);
                assert_eq!(
                    result.statement.final_memory.as_ref(),
                    public_final.then_some(&final_memory)
                );
                let main_populated: usize = result
                    .summary
                    .commitment_components
                    .iter()
                    .map(|component| 1usize << component.log_size)
                    .sum();
                assert_eq!(
                    main_populated,
                    (a + (1usize << width)) * (1usize << n)
                        + usize::from(!public_final) * (1usize << width) * (1usize << a)
                );
                let auxiliary_populated: usize = result
                    .auxiliary_summary
                    .commitment_components
                    .iter()
                    .map(|component| 1usize << component.log_size)
                    .sum();
                assert_eq!(auxiliary_populated, 128 * (1usize << a));
                assert_eq!(result.auxiliary_summary.mosaic_claims, 2);
                for summary in [&result.summary, &result.auxiliary_summary] {
                    assert!(summary.committed_bits.is_power_of_two());
                    assert!(summary.committed_bits >= 1usize << 14);
                    assert_eq!(summary.packed_field_elements, summary.committed_bits / 128);
                }
            }
        }
    }
}

#[test]
fn supplied_and_replayed_reads_make_the_same_complete_proof() {
    let (trace, _) = fixture(7, 6, 3, 0x7715);
    let mut replayed = trace.clone();
    replayed.read_values = None;
    for order in [ReadOrder::CycleFirst, ReadOrder::AddressFirst] {
        let cfg = config(6, 3, 3, order, false);
        let supplied = prove(&cfg, &trace).unwrap();
        let derived = prove(&cfg, &replayed).unwrap();
        assert_eq!(supplied.proof, derived.proof);
        let verified = verify(&derived.statement, &derived.proof).unwrap();
        assert_interface(&trace, &verified);
    }
}

#[test]
fn load_merging_preserves_the_complete_pushforward_proof() {
    let (trace, _) = fixture(8, 6, 6, 0x51a7);
    let cfg = config(6, 3, 6, ReadOrder::AddressFirst, true);
    let merged = prove(&cfg, &trace).unwrap();
    let plain = prove(
        &Config {
            merge_loads: false,
            ..cfg
        },
        &trace,
    )
    .unwrap();
    assert_eq!(merged.proof, plain.proof);
    verify(&plain.statement, &plain.proof).unwrap();
}

#[test]
fn parallel_cubic_and_fraction_reductions_match_scalar_cpu_oracle() {
    let (trace, final_memory) = fixture(14, 8, 6, 0x6a09_e667);
    let cfg = config(8, 4, 6, ReadOrder::AddressFirst, false);
    let result = prove(&cfg, &trace).unwrap();
    let verified = verify(&result.statement, &result.proof).unwrap();
    assert_eq!(result.final_memory, final_memory);
    assert_interface(&trace, &verified);
}

fn rejects(result: &ProveResult, label: &str, edit: impl FnOnce(&mut Proof)) {
    let mut proof = result.proof.clone();
    edit(&mut proof);
    assert!(
        verify(&result.statement, &proof).is_err(),
        "accepted {label}"
    );
}

#[test]
fn every_proof_boundary_authenticates_its_claims() {
    let (trace, _) = fixture(7, 6, 6, 0x243f_6a88);
    let cfg = config(6, 3, 6, ReadOrder::AddressFirst, false);
    let result = prove(&cfg, &trace).unwrap();
    verify(&result.statement, &result.proof).unwrap();

    rejects(&result, "CPU address", |p| p.claims.address += K::ONE);
    rejects(&result, "CPU read", |p| p.claims.read += K::ONE);
    rejects(&result, "CPU write", |p| p.claims.write += K::ONE);
    rejects(&result, "read cycle round", |p| {
        p.read.cycle[1][0] += K::ONE
    });
    rejects(&result, "read address round", |p| {
        p.read.address[1][1] += K::ONE
    });
    rejects(&result, "read address MLE", |p| p.read.ra += K::ONE);
    rejects(&result, "read memory MLE", |p| p.read.val += K::ONE);
    rejects(&result, "private final evaluation", |p| {
        *p.final_eval.as_mut().unwrap() += K::ONE
    });
    for round in 0..result.proof.cycle.sumcheck.round_msgs.len() {
        for element in 0..result.proof.cycle.sumcheck.round_msgs[round].len() {
            rejects(&result, "cubic cycle message", |p| {
                p.cycle.sumcheck.round_msgs[round][element] += K::ONE;
            });
        }
    }
    rejects(&result, "cubic virtual address leaf", |p| {
        p.cycle.ra += K::ONE
    });
    rejects(&result, "cubic increment leaf", |p| p.cycle.inc += K::ONE);

    for root in 0..4 {
        rejects(&result, "rational root", |p| {
            p.pushforward.rational.roots[root] += K::ONE;
        });
    }
    rejects(&result, "rational source layer", |p| {
        p.pushforward.rational.left.layers[0].ac_eval += K::ONE;
    });
    rejects(&result, "rational target layer", |p| {
        p.pushforward.rational.right.layers[0].numerators[0] += K::ONE;
    });
    rejects(&result, "rational product round", |p| {
        p.pushforward.rational.left.layers[1].product.rounds[0][0] += K::ONE;
    });
    rejects(&result, "rational product leaf", |p| {
        p.pushforward.rational.right.layers[1].product.evaluations[0] += K::ONE;
    });
    for bit in 0..cfg.address_bits {
        rejects(&result, "dense address bit leaf", |p| {
            p.pushforward.address_evals[bit] += K::ONE;
        });
    }

    rejects(&result, "main commitment", |p| p.commitment.root[0] ^= 1);
    rejects(&result, "main Mosaic value", |p| p.mosaic.v_g += K::ONE);
    rejects(&result, "main Mosaic round", |p| {
        p.mosaic.sumcheck.round_msgs[0][0] += K::ONE;
    });
    rejects(&result, "main RingSwitch", |p| p.opening.0.s[0] += K::ONE);
    rejects(&result, "main Ligerito", |p| {
        p.opening.1.final_poly[0] += K::ONE
    });
    rejects(&result, "pushforward commitment", |p| {
        p.pushforward.auxiliary.commitment.root[0] ^= 1;
    });
    rejects(&result, "pushforward Mosaic value", |p| {
        p.pushforward.auxiliary.mosaic.v_g += K::ONE;
    });
    rejects(&result, "pushforward Mosaic round", |p| {
        p.pushforward.auxiliary.mosaic.sumcheck.round_msgs[0][0] += K::ONE;
    });
    rejects(&result, "pushforward RingSwitch", |p| {
        p.pushforward.auxiliary.opening.0.s[0] += K::ONE;
    });
    rejects(&result, "pushforward Ligerito", |p| {
        p.pushforward.auxiliary.opening.1.final_poly[0] += K::ONE;
    });
}

#[test]
fn malformed_shapes_and_cross_proof_components_reject() {
    let (trace, _) = fixture(6, 4, 6, 0x1319_8a2e);
    let cfg = config(4, 2, 6, ReadOrder::CycleFirst, true);
    let result = prove(&cfg, &trace).unwrap();
    rejects(&result, "missing dense address leaf", |p| {
        p.pushforward.address_evals.pop();
    });
    rejects(&result, "extra dense address leaf", |p| {
        p.pushforward.address_evals.push(K::ZERO);
    });
    rejects(&result, "missing rational source layer", |p| {
        p.pushforward.rational.left.layers.pop();
    });
    rejects(&result, "missing rational target layer", |p| {
        p.pushforward.rational.right.layers.pop();
    });
    rejects(&result, "missing cubic round", |p| {
        p.cycle.sumcheck.round_msgs.pop();
    });
    rejects(&result, "wrong main domain", |p| p.commitment.n += 1);
    rejects(&result, "wrong auxiliary domain", |p| {
        p.pushforward.auxiliary.commitment.n += 1;
    });
    let (other_trace, _) = fixture(6, 4, 6, 0x299f_31d0);
    let other = prove(&cfg, &other_trace).unwrap();
    rejects(&result, "foreign read reduction", |p| {
        p.read = other.proof.read.clone()
    });
    rejects(&result, "foreign cycle reduction", |p| {
        p.cycle = other.proof.cycle.clone()
    });
    rejects(&result, "foreign pushforward proof", |p| {
        p.pushforward = other.proof.pushforward.clone();
    });
    rejects(&result, "foreign auxiliary authentication", |p| {
        p.pushforward.auxiliary = other.proof.pushforward.auxiliary.clone();
    });
    let mut statement = result.statement.clone();
    statement.initial_memory[1] ^= 1;
    assert!(verify(&statement, &result.proof).is_err());
    let mut statement = result.statement.clone();
    statement.final_memory.as_mut().unwrap()[2] ^= 1;
    assert!(verify(&statement, &result.proof).is_err());
    let mut statement = result.statement.clone();
    statement.config.order = ReadOrder::AddressFirst;
    assert!(verify(&statement, &result.proof).is_err());
    let mut statement = result.statement.clone();
    statement.log_cycles += 1;
    assert!(verify(&statement, &result.proof).is_err());
}

#[test]
fn address_bit_width_is_an_enforced_range_check() {
    let (trace, _) = fixture(5, 4, 6, 0x082e_fa98);
    let cfg = config(4, 2, 6, ReadOrder::AddressFirst, true);
    let mut invalid = trace.clone();
    invalid.writes[0] = 16;
    invalid.reads[0][0] = 16;
    assert!(prove(&cfg, &invalid).is_err());
    let mut invalid = trace.clone();
    invalid.writes[0] = u32::MAX;
    invalid.reads[0][0] = u32::MAX;
    assert!(prove(&cfg, &invalid).is_err());
    let mut invalid = trace;
    invalid.reads[0][0] ^= 1;
    assert!(prove(&cfg, &invalid).is_err());
}

#[test]
fn rangechecked_rational_error_can_have_table_size_minus_one_roots() {
    // Four in-range poles. Construct nonzero residues whose rational sum
    // has exactly the three prescribed roots outside that range. This
    // pins the table-size / field-size error, independent of how often
    // those poles occur in the access trace.
    let labels: Vec<K> = (0..4).map(K::from_u128).collect();
    let roots: Vec<K> = (4..7).map(K::from_u128).collect();
    let numerator = |x: K| roots.iter().fold(K::ONE, |acc, &r| acc * (x + r));
    let residues: Vec<K> = labels
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let derivative = labels
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(K::ONE, |acc, (_, &y)| acc * (x + y));
            numerator(x) / derivative
        })
        .collect();
    assert!(residues.iter().any(|&value| value != K::ZERO));
    for challenge in 4..16 {
        let tau = K::from_u128(challenge);
        let rational = labels
            .iter()
            .zip(&residues)
            .fold(K::ZERO, |sum, (&pole, &weight)| sum + weight / (tau + pole));
        let denominator = labels.iter().fold(K::ONE, |acc, &pole| acc * (tau + pole));
        assert_eq!(rational * denominator, numerator(tau));
        assert_eq!(rational == K::ZERO, challenge < 7);
    }
}

#[test]
fn histogram_pushforward_matches_scalar_rows_with_uneven_parallel_chunks() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (n, a) in [(3, 2), (5, 7), (14, 8)] {
        let (trace, _) = fixture(n, a, 6, 0xa409_3822);
        let mut random = 0x082e_fa98_ec4e_6c89;
        let z: Vec<_> = (0..n)
            .map(|_| K::from_u128(next(&mut random) as u128))
            .collect();
        let rk: Vec<_> = (0..a)
            .map(|_| K::from_u128(next(&mut random) as u128))
            .collect();
        let weights: Vec<_> = (0..1usize << n).map(|j| scalar_basis(j, &z)).collect();
        let pushed = pool.install(|| pushforward(&trace.writes, &weights, 1usize << a));
        let mut expected = vec![K::ZERO; 1usize << a];
        let mut sparse_mle = K::ZERO;
        for (&address, &weight) in trace.writes.iter().zip(&weights) {
            expected[address as usize] += weight;
            sparse_mle += weight * scalar_basis(address as usize, &rk);
        }
        assert_eq!(pushed, expected);
        let pushed_mle = pushed
            .iter()
            .enumerate()
            .fold(K::ZERO, |sum, (k, &weight)| {
                sum + weight * scalar_basis(k, &rk)
            });
        assert_eq!(pushed_mle, sparse_mle);
        assert_eq!(
            pushed
                .iter()
                .copied()
                .fold(K::ZERO, |sum, value| sum + value),
            K::ONE
        );
    }
}

#[test]
fn optimized_cubic_messages_match_generic_multilinear_sumcheck() {
    use om_sumcheck::sumcheck_prove;

    for n in [1, 2, 5, 9, 14] {
        let a = 4;
        let (trace, _) = fixture(n, a, 6, 0x4528_21e6);
        let mut random = 0x38d0_1377_be54_66cf;
        let rj: Vec<_> = (0..n)
            .map(|_| K::from_u128(next(&mut random) as u128))
            .collect();
        let rk: Vec<_> = (0..a)
            .map(|_| K::from_u128(next(&mut random) as u128))
            .collect();
        let inc: Vec<_> = (0..1usize << n)
            .map(|j| {
                if j % 3 == 0 {
                    K::ZERO
                } else {
                    K::from_u128(next(&mut random) as u128)
                }
            })
            .collect();
        let f: Vec<_> = trace
            .writes
            .iter()
            .map(|&address| scalar_basis(address as usize, &rk))
            .collect();
        let eq: Vec<_> = (0..1usize << n).map(|j| scalar_basis(j, &rj)).collect();
        // For Boolean j, LT(j,rj)=sum_{y>j} Eq(rj,y). This independent
        // cumulative-sum oracle avoids the optimized split-LT recurrence.
        let mut lt = vec![K::ZERO; 1usize << n];
        let mut suffix = K::ZERO;
        for j in (0..lt.len()).rev() {
            lt[j] = suffix;
            suffix += eq[j];
        }
        let gamma = K::from_u128(next(&mut random) as u128);
        let delta = K::from_u128(next(&mut random) as u128);
        let tl: Vec<_> = lt.iter().map(|&v| v + gamma).collect();
        let ed: Vec<_> = eq.iter().map(|&v| delta * v).collect();
        let claim = (0..f.len()).fold(K::ZERO, |sum, j| sum + (tl[j] * inc[j] + ed[j]) * f[j]);
        let mut optimized_tr = Transcript::new_default(b"test/direct-pushforward/cubic-oracle");
        let (optimized, point) = prove_cycle(
            &trace.writes,
            &inc,
            &rk,
            &rj,
            claim,
            gamma,
            delta,
            &mut optimized_tr,
        );
        let mut reference_tr = Transcript::new_default(b"test/direct-pushforward/cubic-oracle");
        let (reference, reference_point, values) = sumcheck_prove(
            claim,
            vec![f, inc, tl, ed],
            &|v: &[K]| (v[2] * v[1] + v[3]) * v[0],
            3,
            &mut reference_tr,
            CYCLE_LABEL,
        );
        assert_eq!(optimized.sumcheck, reference, "n={n}");
        assert_eq!(point, reference_point, "n={n}");
        assert_eq!(
            [optimized.ra, optimized.inc],
            [values[0], values[1]],
            "n={n}"
        );
        reference_tr.absorb_ks(b"ram/logup/cycle/evals", &[values[0], values[1]]);
        assert_eq!(
            optimized_tr.challenge_k(b"reference/continuation"),
            reference_tr.challenge_k(b"reference/continuation"),
        );
        let mut verifier_tr = Transcript::new_default(b"test/direct-pushforward/cubic-oracle");
        assert_eq!(
            verify_cycle(&rj, claim, gamma, delta, &optimized, &mut verifier_tr).unwrap(),
            point
        );
    }
}
