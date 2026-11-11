use super::*;
use om_sumcheck::eq_table;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

/// A shared-address trace with a mix of loads (no change) and stores, over
/// `2^hot` hot cells out of `2^a`, so that cells repeat and loads merge.
fn trace(n: usize, a: usize, hot: usize, word_log_bits: usize, seed: u64) -> AccessTrace {
    let mut s = seed;
    let mask = if word_log_bits == 6 {
        u64::MAX
    } else {
        (1u64 << (1 << word_log_bits)) - 1
    };
    let initial: Vec<u64> = (0..1usize << a).map(|_| lcg(&mut s) & mask).collect();
    let mut memory = initial.clone();
    let (mut writes, mut new_values, mut read_values) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..1usize << n {
        let addr = (lcg(&mut s) as usize % (1usize << hot.min(a))) as u32;
        let old = memory[addr as usize];
        let new = if lcg(&mut s).is_multiple_of(3) {
            lcg(&mut s) & mask
        } else {
            old
        };
        memory[addr as usize] = new;
        writes.push(addr);
        new_values.push(new);
        read_values.push(old);
    }
    AccessTrace {
        initial,
        reads: vec![writes.clone()],
        writes,
        new_values,
        read_values: Some(vec![read_values]),
    }
}

fn config(a: usize, c: usize, order: ReadOrder, merge: bool, public_final: bool) -> Config {
    Config {
        address_bits: a,
        chunk_bits: c,
        order,
        merge_loads: merge,
        public_final,
        ..Config::default()
    }
}

const ORDERS: [ReadOrder; 2] = [ReadOrder::CycleFirst, ReadOrder::AddressFirst];

#[test]
fn proves_and_verifies_across_shapes() {
    for (n, a, c, hot) in [
        (1, 4, 2, 4),
        (3, 4, 4, 2),
        (6, 8, 4, 3),
        (9, 8, 2, 8),
        (10, 6, 3, 6),
    ] {
        for order in ORDERS {
            for merge in [false, true] {
                for public_final in [true, false] {
                    let tr = trace(n, a, hot, 6, (n * 31 + a) as u64);
                    let cfg = config(a, c, order, merge, public_final);
                    let result = prove(&cfg, &tr).unwrap();
                    verify(&result.statement, &result.proof).unwrap_or_else(|e| {
                        panic!(
                            "n={n} a={a} c={c} {order:?} merge={merge} public={public_final}: {e}"
                        )
                    });
                }
            }
        }
    }
}

#[test]
fn final_memory_matches_upstream_twist() {
    let tr = trace(8, 8, 5, 6, 7);
    let legacy = crate::prove(
        &crate::Config {
            address_bits: 8,
            chunk_bits: 4,
            ..crate::Config::default()
        },
        &tr,
    )
    .unwrap();
    for order in ORDERS {
        let result = prove(&config(8, 4, order, true, true), &tr).unwrap();
        assert_eq!(result.final_memory, legacy.statement.final_memory);
        assert_eq!(
            result.statement.final_memory.as_ref(),
            Some(&legacy.statement.final_memory)
        );
    }
}

#[test]
fn narrow_words_and_replayed_read_values() {
    for order in ORDERS {
        let mut tr = trace(7, 6, 4, 3, 11);
        let cfg = Config {
            word_log_bits: 3,
            ..config(6, 3, order, true, true)
        };
        let with_values = prove(&cfg, &tr).unwrap();
        tr.read_values = None;
        let replayed = prove(&cfg, &tr).unwrap();
        assert_eq!(with_values.proof, replayed.proof);
        verify(&replayed.statement, &replayed.proof).unwrap();
    }
}

#[test]
fn every_narrow_word_width_verifies_and_rejects_a_wrong_final_memory() {
    for word_log_bits in 0..=2 {
        for order in ORDERS {
            for public_final in [true, false] {
                let tr = trace(7, 6, 4, word_log_bits, 13 + word_log_bits as u64);
                let cfg = Config {
                    word_log_bits,
                    ..config(6, 3, order, true, public_final)
                };
                let result = prove(&cfg, &tr).unwrap();
                verify(&result.statement, &result.proof).unwrap_or_else(|e| {
                    panic!("width 2^{word_log_bits} {order:?} public={public_final}: {e}")
                });
                if public_final {
                    let mut s = result.statement.clone();
                    let memory = s.final_memory.as_mut().unwrap();
                    memory[0] ^= 1;
                    assert!(verify(&s, &result.proof).is_err());
                }
            }
        }
    }
}

#[test]
fn guest_traces_verify() {
    use ram_payload::{GuestOptions, GuestProgram, GuestRows};
    for program in [
        GuestProgram::BinarySearch,
        GuestProgram::Erc20MixedBlock,
        GuestProgram::PointerChase,
    ] {
        let options = GuestOptions {
            program,
            rows: GuestRows::All,
            keys: 8,
            queries: 4,
            size: 8,
            operations: 12,
            ..GuestOptions::default()
        };
        let payload = ram_payload::guest(8, 8, &options).unwrap();
        let tr = AccessTrace {
            initial: payload.initial_memory,
            reads: payload.reads,
            writes: payload.writes,
            new_values: payload.new_values,
            read_values: Some(payload.read_values),
        };
        for order in ORDERS {
            let result = prove(&config(8, 4, order, true, true), &tr).unwrap();
            verify(&result.statement, &result.proof).unwrap();
        }
    }
}

#[test]
fn load_merging_does_not_change_the_proof() {
    let tr = trace(9, 8, 3, 6, 5);
    let merged = prove(&config(8, 4, ReadOrder::AddressFirst, true, true), &tr).unwrap();
    let plain = prove(&config(8, 4, ReadOrder::AddressFirst, false, true), &tr).unwrap();
    assert_eq!(merged.proof, plain.proof);
}

#[test]
fn mosaic_claim_count() {
    // RV, Inc and two claims per component; no address-bit claims.
    let tr = trace(6, 8, 8, 6, 1);
    let result = prove(&config(8, 4, ReadOrder::CycleFirst, true, true), &tr).unwrap();
    assert_eq!(result.summary.mosaic_claims, 2 * 2 + 2);
    let result = prove(&config(8, 4, ReadOrder::CycleFirst, true, false), &tr).unwrap();
    assert_eq!(result.summary.mosaic_claims, 2 * 2 + 3);
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------

fn prove_tampered(
    cfg: &Config,
    tr: &AccessTrace,
    tamper: impl FnOnce(&mut Witness),
) -> ProveResult {
    util::set_lenient();
    let n = tr.writes.len().trailing_zeros() as usize;
    let mut w = build_witness(tr);
    tamper(&mut w);
    prove_witness(cfg, n, &tr.initial, w, Schema::new(cfg, n)).unwrap()
}

fn honest_final(tr: &AccessTrace) -> Vec<u64> {
    build_witness(tr).final_memory
}

/// Find a stored cycle followed by a later read of the same cell.
fn store_then_read(tr: &AccessTrace) -> usize {
    let w = build_witness(tr);
    (0..w.addr.len())
        .find(|&j| w.inc[j] != 0 && w.addr[j + 1..].contains(&w.addr[j]))
        .expect("trace has a store that is read later")
}

#[test]
fn rejects_wrong_inc() {
    // Item 3: with no write term, a wrong Inc is caught by later reads of
    // the cell and by the final-memory claim.
    let tr = trace(8, 6, 3, 6, 21);
    let j = store_then_read(&tr);
    for order in ORDERS {
        for public_final in [true, false] {
            let cfg = config(6, 3, order, false, public_final);
            let honest = honest_final(&tr);
            let result = prove_tampered(&cfg, &tr, |w| {
                w.inc[j] ^= 1;
                w.final_memory = honest.clone();
            });
            assert!(verify(&result.statement, &result.proof).is_err());
        }
    }
}

#[test]
fn rejects_wrong_inc_with_matching_final_memory() {
    // A wrong Inc on the last store to a cell that is never read again, with
    // the final memory updated to match: the trace is then consistent, so
    // the honest statement must reject it, which it does because the final
    // memory in the statement differs.
    let tr = trace(8, 6, 3, 6, 23);
    let w = build_witness(&tr);
    let j = (0..w.addr.len())
        .rev()
        .find(|&j| w.inc[j] != 0)
        .expect("a store");
    let cfg = config(6, 3, ReadOrder::CycleFirst, false, true);
    let honest = prove(&cfg, &tr).unwrap();
    let tampered = prove_tampered(&cfg, &tr, |w| {
        w.inc[j] ^= 4;
        w.final_memory[w.addr[j] as usize] ^= 4;
    });
    // The tampered proof is valid for its own (different) final memory.
    verify(&tampered.statement, &tampered.proof).unwrap();
    assert!(verify(&honest.statement, &tampered.proof).is_err());
}

#[test]
fn rejects_wrong_read_value() {
    let tr = trace(8, 6, 3, 6, 25);
    for order in ORDERS {
        let cfg = config(6, 3, order, false, true);
        let result = prove_tampered(&cfg, &tr, |w| {
            // Keep the written value, change what is read.
            w.rv[5] ^= 2;
            w.inc[5] ^= 2;
        });
        assert!(verify(&result.statement, &result.proof).is_err());
    }
}

#[test]
fn rejects_tampered_public_final_memory() {
    // Item 4: final memory is checked at the read check's address point.
    let tr = trace(7, 6, 6, 6, 27);
    for order in ORDERS {
        let cfg = config(6, 3, order, true, true);
        let result = prove(&cfg, &tr).unwrap();
        for cell in [0usize, 17, 63] {
            let mut statement = result.statement.clone();
            statement.final_memory.as_mut().unwrap()[cell] ^= 1 << 40;
            assert!(verify(&statement, &result.proof).is_err());
        }
    }
}

#[test]
fn rejects_wrong_committed_final_memory() {
    let tr = trace(7, 6, 6, 6, 29);
    let cfg = config(6, 3, ReadOrder::CycleFirst, false, false);
    let result = prove_tampered(&cfg, &tr, |w| w.final_memory[9] ^= 8);
    assert!(verify(&result.statement, &result.proof).is_err());
}

#[test]
fn rejects_modified_proof_and_statement() {
    let tr = trace(7, 6, 4, 6, 31);
    let cfg = config(6, 3, ReadOrder::CycleFirst, true, true);
    let result = prove(&cfg, &tr).unwrap();
    let mut p = result.proof.clone();
    p.rv += K::ONE;
    assert!(verify(&result.statement, &p).is_err());
    let mut p = result.proof.clone();
    p.read.cycle[2][1] += K::ONE;
    assert!(verify(&result.statement, &p).is_err());
    let mut p = result.proof.clone();
    p.read.address[0][0] += K::ONE;
    assert!(verify(&result.statement, &p).is_err());
    let mut p = result.proof.clone();
    p.cycle.a_mirror[0] += K::ONE;
    assert!(verify(&result.statement, &p).is_err());
    let mut p = result.proof.clone();
    p.cycle.inc += K::ONE;
    assert!(verify(&result.statement, &p).is_err());
    let mut statement = result.statement.clone();
    statement.initial_memory[3] ^= 1;
    assert!(verify(&statement, &result.proof).is_err());
    // The binding order is part of the statement.
    let mut statement = result.statement.clone();
    statement.config.order = ReadOrder::AddressFirst;
    assert!(verify(&statement, &result.proof).is_err());
}

/// Mirror-test tables for one component whose rows are given explicitly
/// (`rows[j][u]`), with the other inputs zero so that only the mirror term
/// and the `ra` term remain.
fn mirror_sum(rows: &[Vec<K>], r: &[K], r_j: &[K]) -> K {
    let e = eq_table(r);
    let em = eq_table(&cycle::mirrored(r));
    let ej = eq_table(r_j);
    rows.iter()
        .zip(&ej)
        .map(|(row, &w)| {
            let x: K = row
                .iter()
                .zip(&e)
                .map(|(&v, &q)| v * q)
                .fold(K::ZERO, |a, b| a + b);
            let y: K = row
                .iter()
                .zip(&em)
                .map(|(&v, &q)| v * q)
                .fold(K::ZERO, |a, b| a + b);
            w * x * y
        })
        .fold(K::ZERO, |a, b| a + b)
}

#[test]
fn mirror_identity_accepts_one_hot_and_rejects_others() {
    // Item 5: Σ_j eq(r_j, j) A(r, j) A(1 + r, j) = π(r) exactly for one-hot rows.
    let mut s = 3u64;
    let mut rand_k = || K::from_u128(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let (c, n) = (3usize, 3usize);
    let r: Vec<K> = (0..c).map(|_| rand_k()).collect();
    let r_j: Vec<K> = (0..n).map(|_| rand_k()).collect();
    let one_hot = |u: usize| -> Vec<K> {
        (0..1 << c)
            .map(|v| if v == u { K::ONE } else { K::ZERO })
            .collect()
    };
    let honest: Vec<Vec<K>> = (0..1 << n).map(|j| one_hot((j * 5) % (1 << c))).collect();
    assert_eq!(mirror_sum(&honest, &r, &r_j), cycle::mirror_rhs(&r));
    // A two-hot row.
    let mut two_hot = honest.clone();
    two_hot[3][(3 * 5 + 1) % 8] = K::ONE;
    assert_ne!(mirror_sum(&two_hot, &r, &r_j), cycle::mirror_rhs(&r));
    // A zero row (weight zero).
    let mut zero = honest.clone();
    zero[6] = vec![K::ZERO; 1 << c];
    assert_ne!(mirror_sum(&zero, &r, &r_j), cycle::mirror_rhs(&r));
    // A three-hot row (odd weight, the case a parity check would miss).
    let mut three = honest.clone();
    three[1] = (0..1 << c)
        .map(|v| if v < 3 { K::ONE } else { K::ZERO })
        .collect();
    assert_ne!(mirror_sum(&three, &r, &r_j), cycle::mirror_rhs(&r));
}

#[test]
fn cycle_check_rejects_two_hot_component() {
    // Run the cycle sum-check on tables built from a component with a
    // two-hot row; the claimed sum uses π, so the verifier must reject.
    util::set_lenient();
    let mut s = 9u64;
    let mut rand_k = || K::from_u128(((lcg(&mut s) as u128) << 64) | lcg(&mut s) as u128);
    let (c, n, d) = (2usize, 4usize, 1usize);
    let r_k: Vec<K> = (0..c).map(|_| rand_k()).collect();
    let r_j: Vec<K> = (0..n).map(|_| rand_k()).collect();
    let addr: Vec<u32> = (0..1u32 << n).map(|j| j % 4).collect();
    let inc = vec![K::ZERO; 1 << n];
    let mut tr = Transcript::new_default(b"test");
    let ch = cycle::challenges(&mut tr, d);
    let e = eq_table(&r_k);
    let em = eq_table(&cycle::mirrored(&r_k));
    let eps = ch.eps[0] * ch.delta.inverse();
    let mut a: Vec<K> = addr.iter().map(|&u| e[u as usize]).collect();
    let mut am: Vec<K> = addr.iter().map(|&u| eps * em[u as usize]).collect();
    // Row 5 is two-hot: addresses 1 and 2.
    a[5] = e[1] + e[2];
    am[5] = eps * (em[1] + em[2]);
    let ra = (0..1 << n)
        .map(|j| eq_table(&r_j)[j] * a[j])
        .fold(K::ZERO, |x, y| x + y);
    let claim = cycle::claimed_sum(&ch, K::ZERO, K::ZERO, ra, &r_k, c);
    let tables = cycle::Tables {
        tl: vec![ch.gamma; 1 << n],
        ed: eq_table(&r_j).into_iter().map(|x| ch.delta * x).collect(),
        inc,
        a: vec![a],
        am: vec![am],
    };
    let mut ptr = tr.clone();
    let result = cycle::prove_rows(tables.into_rows(), &[eps], claim, &mut ptr);
    let mut vtr = tr.clone();
    assert!(cycle::verify(&result.proof, &r_j, claim, &ch, d, &mut vtr).is_err());
}

#[test]
fn soundness_terms_at_n20_a16() {
    let config = Config {
        address_bits: 16,
        chunk_bits: 4,
        ..Config::default()
    };
    let statement = Statement {
        config,
        log_cycles: 20,
        initial_memory: vec![0; 1 << 16],
        final_memory: Some(vec![0; 1 << 16]),
    };
    let terms = soundness_terms(&statement);
    let numerator: f64 = terms.iter().map(|t| t.1).sum::<f64>() * 2f64.powi(128);
    let ell = Schema::new(&statement.config, 20).layout.ell;
    // 26 + 92 + 22 + 1 + 120 + 28 + 9, plus the Mosaic rounds.
    assert_eq!(numerator.round() as usize, 298 + 2 * ell);
    assert_eq!(statement.config.components(), 4);
}

#[test]
fn cycle_prover_matches_the_reference_prover() {
    // The split cycle prover must send exactly the reference prover's
    // messages and evaluations, across shapes that exercise the
    // sequential and parallel paths and every block split.
    let mut s = 21u64;
    for &(n, a, c) in &[
        (4usize, 4usize, 2usize),
        (5, 6, 3),
        (7, 8, 4),
        (9, 8, 2),
        (14, 8, 4),
        (15, 12, 4),
        // Chunks wider than LOOKUP_MAX_C build round-0 rows instead.
        (9, 12, 6),
        (13, 16, 8),
    ] {
        fn rand_k(s: &mut u64) -> K {
            K::from_u128(((lcg(s) as u128) << 64) | lcg(s) as u128)
        }
        let d = a / c;
        let r_k: Vec<K> = (0..a).map(|_| rand_k(&mut s)).collect();
        let r_j: Vec<K> = (0..n).map(|_| rand_k(&mut s)).collect();
        let addr: Vec<u32> = (0..1u32 << n)
            .map(|_| (lcg(&mut s) % (1 << a)) as u32)
            .collect();
        // About half the cycles are loads (zero increment).
        let inc: Vec<K> = (0..1usize << n)
            .map(|_| {
                if lcg(&mut s).is_multiple_of(2) {
                    K::ZERO
                } else {
                    rand_k(&mut s)
                }
            })
            .collect();
        let mut tr = Transcript::new_default(b"test");
        let ch = cycle::challenges(&mut tr, d);
        let w = cycle::CycleWitness {
            addr: &addr,
            inc_e: &inc,
            c,
            d,
        };
        let (rows, eps) = cycle::reference_rows(&w, &r_k, &r_j, &ch);
        let claim = cycle::row_sum(&rows);
        let mut t1 = tr.clone();
        let fast = cycle_prover::prove(&w, &r_k, &r_j, claim, &ch, &mut t1);
        let mut t2 = tr.clone();
        let reference = cycle::prove_rows(rows, &eps, claim, &mut t2);
        assert_eq!(fast.proof, reference.proof, "n={n} a={a} c={c}");
        assert_eq!(fast.z, reference.z);
        assert_eq!(
            t1.challenge_k(b"after"),
            t2.challenge_k(b"after"),
            "transcripts diverge at n={n}"
        );
    }
}

#[test]
fn rejects_tampered_cycle_messages_from_the_split_prover() {
    // At n = 8 the cycle check runs the split prover (round-0 tables, Inc
    // skipping on load pairs). Changing any value of any round message, or
    // any final evaluation, must make the verifier reject.
    for (a, c) in [(8usize, 4usize), (12, 6)] {
        let tr = trace(8, a, 8, 6, 77);
        let cfg = config(a, c, ReadOrder::AddressFirst, true, true);
        let result = prove(&cfg, &tr).unwrap();
        verify(&result.statement, &result.proof).unwrap();
        let rounds = result.proof.cycle.sumcheck.round_msgs.len();
        assert_eq!(rounds, 8);
        for i in 0..rounds {
            for k in 0..result.proof.cycle.sumcheck.round_msgs[i].len() {
                let mut p = result.proof.clone();
                p.cycle.sumcheck.round_msgs[i][k] += K::ONE;
                assert!(verify(&result.statement, &p).is_err(), "round {i} value {k}");
            }
        }
        for i in 0..cfg.components() {
            let mut p = result.proof.clone();
            p.cycle.a[i] += K::ONE;
            assert!(verify(&result.statement, &p).is_err());
        }
    }
}
