use om_field::K;
use spice_argument::{
    Config,
    operations::{self, AccessTrace, ProveResult},
};
use std::sync::OnceLock;

fn config(a: usize) -> Config {
    Config {
        address_bits: a,
        chunk_bits: 1,
        ..Config::default()
    }
}
fn trace() -> AccessTrace {
    AccessTrace {
        initial: vec![7, 11, 19, 23],
        addresses: vec![0, 0, 0, 1, 1, 0, 0, 1],
        values: vec![9, 9, 9, 11, 13, 9, 0, 13],
        is_write: vec![true, false, true, false, true, false, true, false],
    }
}
fn fixture() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| operations::prove(&config(2), &trace()).unwrap())
}

#[test]
fn typed_read_write_semantics_write_first_and_precise_pricing() {
    let r = fixture();
    operations::verify(&r.statement, &r.proof).unwrap();
    assert_eq!(r.statement.final_memory, vec![0, 13, 19, 23]);
    let populated: usize = r
        .summary
        .commitment_components
        .iter()
        .map(|c| 1usize << c.log_size)
        .sum();
    assert_eq!(populated, (64 + 2 + 2 * 4 + 1) * 8 + (4 + 1) * 4);
    // Address (2 planes), predecessor (4) and boundary (4) are one chunk each;
    // the flag pair is one chunk; carries (3) split into 2 + 1.
    // Leaf point: address, previous, boundary, flags, initial flag = 5.
    // Mask point: value, flags, initial flag = 3. Order: 1 + 2 + 2 = 5.
    assert_eq!(r.summary.mosaic_claims, 5 + 3 + 5);
    assert!(
        r.summary
            .commitment_components
            .iter()
            .any(|c| c.name == "initial/next_is_write")
    );
    assert_eq!(
        r.summary
            .commitment_components
            .iter()
            .filter(|c| c.name == "input/value")
            .count(),
        1
    );
    assert!(
        !r.summary
            .commitment_components
            .iter()
            .any(|c| c.name == "input/read" || c.name == "input/write")
    );
}

#[test]
fn both_ram_contracts_agree_on_public_state_for_same_typed_events() {
    let t = trace();
    let mut memory = t.initial.clone();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for j in 0..t.addresses.len() {
        let k = t.addresses[j] as usize;
        reads.push(memory[k]);
        if t.is_write[j] {
            memory[k] = t.values[j];
        }
        writes.push(memory[k]);
    }
    let old = spice_argument::AccessTrace {
        initial: t.initial,
        reads: vec![t.addresses.clone()],
        writes: t.addresses,
        new_values: writes,
        read_values: Some(vec![reads]),
    };
    let old = spice_argument::prove(&config(2), &old).unwrap();
    spice_argument::verify(&old.statement, &old.proof).unwrap();
    assert_eq!(old.statement, fixture().statement);
}

#[test]
fn read_only_write_only_mixed_full_width_and_rectangular_domains() {
    for (a, n) in [(1, 0), (2, 2), (3, 1), (1, 4)] {
        for mode in 0..3 {
            let mut memory: Vec<u64> = (0..1usize << a).map(|i| u64::MAX ^ (i as u64)).collect();
            let mut t = AccessTrace {
                initial: memory.clone(),
                addresses: Vec::new(),
                values: Vec::new(),
                is_write: Vec::new(),
            };
            for j in 0..1usize << n {
                let k = (j * 3) % (1usize << a);
                let w = mode == 1 || (mode == 2 && j % 2 == 0);
                let v = if w {
                    (1u64 << 63) ^ j as u64
                } else {
                    memory[k]
                };
                if w {
                    memory[k] = v;
                }
                t.addresses.push(k as u32);
                t.values.push(v);
                t.is_write.push(w);
            }
            let r = operations::prove(&config(a), &t).unwrap();
            operations::verify(&r.statement, &r.proof).unwrap();
            assert_eq!(r.statement.final_memory, memory);
        }
    }
}

#[test]
fn every_mask_and_flag_claim_is_authenticated() {
    let r = fixture();
    for field in 0..6 {
        let mut p = r.proof.clone();
        match field {
            0 => p.evaluations.flags[0] += K::ONE,
            1 => p.evaluations.flags[1] += K::ONE,
            2 => p.evaluations.initial_next_is_write += K::ONE,
            3 => p.masked.cycle[0] += K::ONE,
            4 => p.masked.cycle[1] += K::ONE,
            _ => p.masked.initial += K::ONE,
        };
        assert!(operations::verify(&r.statement, &p).is_err());
    }
    for field in 0..4 {
        let mut p = r.proof.clone();
        match field {
            0 => p.evaluations.mask_value += K::ONE,
            1 => p.evaluations.mask_flags[0] += K::ONE,
            2 => p.evaluations.mask_flags[1] += K::ONE,
            _ => p.evaluations.mask_initial_next_is_write += K::ONE,
        };
        assert!(operations::verify(&r.statement, &p).is_err());
    }
}

#[test]
fn ordering_boundaries_pointers_and_pcs_are_required() {
    let r = fixture();
    let mut s = r.statement.clone();
    s.initial_memory[1] ^= 1;
    assert!(operations::verify(&s, &r.proof).is_err());
    let mut s = r.statement.clone();
    s.final_memory[0] ^= 1;
    assert!(operations::verify(&s, &r.proof).is_err());
    let mut p = r.proof.clone();
    p.evaluations.previous[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.evaluations.final_previous[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.evaluations.carries_c[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.mosaic.v_g += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.opening.0.s[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.opening.1.openings[0].columns[0][0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
}

#[test]
fn inconsistent_typed_reads_and_malformed_proofs_reject() {
    let r = fixture();
    let mut t = trace();
    t.values[1] ^= 1;
    assert!(operations::prove(&config(2), &t).is_err());
    let mut t = trace();
    t.is_write.pop();
    assert!(operations::prove(&config(2), &t).is_err());
    let mut p = r.proof.clone();
    p.evaluations.address.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.evaluations.previous_ab.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.masks.round_msgs.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.evaluations.carries_c.pop();
    assert!(operations::verify(&r.statement, &p).is_err());
}

#[test]
fn deferred_terminal_checks_bind_the_fused_evaluation_message() {
    let r = fixture();
    // Flock's terminal values are checked against the late source values.
    let mut p = r.proof.clone();
    p.order.final_a_eval.lo ^= 1;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.order.final_c_eval.lo ^= 1;
    assert!(operations::verify(&r.statement, &p).is_err());
    // The masked sum-check's last message and the masked claims.
    let mut p = r.proof.clone();
    p.masks.round_msgs.last_mut().unwrap()[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.masked.initial += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    // Every field of the evaluation message.
    let fields = 11;
    for field in 0..fields {
        let mut p = r.proof.clone();
        let e = &mut p.evaluations;
        match field {
            0 => e.address[1] += K::ONE,
            1 => e.previous[3] += K::ONE,
            2 => e.flags[1] += K::ONE,
            3 => e.final_previous[2] += K::ONE,
            4 => e.initial_next_is_write += K::ONE,
            5 => e.mask_value += K::ONE,
            6 => e.mask_flags[0] += K::ONE,
            7 => e.mask_initial_next_is_write += K::ONE,
            8 => e.previous_ab[0] += K::ONE,
            9 => e.carries_ab[2] += K::ONE,
            _ => e.carries_c[1] += K::ONE,
        }
        assert!(
            operations::verify(&r.statement, &p).is_err(),
            "field {field}"
        );
    }
}

fn fixture_2fp() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| {
        operations::prove_with(&config(2), &trace(), &operations::Options::TWO_FINGERPRINTS)
            .unwrap()
    })
}

#[test]
fn two_fingerprints_verify_with_matching_options_only() {
    let two = operations::Options::TWO_FINGERPRINTS;
    let r = fixture_2fp();
    operations::verify_with(&r.statement, &r.proof, &two).unwrap();
    assert_eq!(r.statement, fixture().statement);
    // The verifier chooses the fingerprint count: neither proof passes as the other.
    assert!(operations::verify(&r.statement, &r.proof).is_err());
    assert!(operations::verify_with(&fixture().statement, &fixture().proof, &two).is_err());
    // Same commitment and claims; only the permutation tree doubles.
    assert_eq!(r.summary.committed_bits, fixture().summary.committed_bits);
    assert_eq!(r.summary.mosaic_claims, fixture().summary.mosaic_claims);
    assert_eq!(
        r.proof.permutation.layers.len(),
        fixture().proof.permutation.layers.len() + 1
    );
}

#[test]
fn two_fingerprints_cover_binary_and_radix_four_first_layers() {
    // (a, n) = (2, 3) gives a 5-level tree (binary first layer); (2, 2) and
    // (3, 1) give 4- and 5-level trees.
    for (a, n) in [(2, 3), (2, 2), (3, 1), (1, 0)] {
        let mut memory: Vec<u64> = (0..1u64 << a).map(|i| 100 + i).collect();
        let mut t = AccessTrace {
            initial: memory.clone(),
            addresses: Vec::new(),
            values: Vec::new(),
            is_write: Vec::new(),
        };
        for j in 0..1usize << n {
            let k = (j * 5 + 1) % (1usize << a);
            let w = j % 3 == 0;
            let v = if w { 7 * j as u64 + 1 } else { memory[k] };
            if w {
                memory[k] = v;
            }
            t.addresses.push(k as u32);
            t.values.push(v);
            t.is_write.push(w);
        }
        let two = operations::Options::TWO_FINGERPRINTS;
        let r = operations::prove_with(&config(a), &t, &two).unwrap();
        operations::verify_with(&r.statement, &r.proof, &two).unwrap();
        assert_eq!(r.statement.final_memory, memory);
    }
}

#[test]
fn two_fingerprint_tampering_rejects() {
    let two = operations::Options::TWO_FINGERPRINTS;
    let r = fixture_2fp();
    // Rescale the per-set products while keeping the root product fixed.
    let mut p = r.proof.clone();
    let c = K::from_u128(3);
    p.permutation.layers[0].children[0][0] *= c;
    p.permutation.layers[0].children[0][1] /= c;
    assert!(operations::verify_with(&r.statement, &p, &two).is_err());
    let mut p = r.proof.clone();
    p.permutation.layers.last_mut().unwrap().children[1][0] += K::ONE;
    assert!(operations::verify_with(&r.statement, &p, &two).is_err());
    let mut p = r.proof.clone();
    p.masked.cycle[0] += K::ONE;
    assert!(operations::verify_with(&r.statement, &p, &two).is_err());
    let mut s = r.statement.clone();
    s.final_memory[2] ^= 4;
    assert!(operations::verify_with(&s, &r.proof, &two).is_err());
    let mut s = r.statement.clone();
    s.initial_memory[3] ^= 4;
    assert!(operations::verify_with(&s, &r.proof, &two).is_err());
    assert!(
        operations::prove_with(
            &config(2),
            &trace(),
            &operations::Options {
                fingerprints: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn every_plane_of_a_chunk_is_bound_by_the_combined_claim() {
    let r = fixture();
    // Swapping two planes' values keeps their sum but not the rho combination.
    let mut p = r.proof.clone();
    assert_ne!(p.evaluations.address[0], p.evaluations.address[1]);
    p.evaluations.address.swap(0, 1);
    assert!(operations::verify(&r.statement, &p).is_err());
    for plane in 0..4 {
        let mut p = r.proof.clone();
        p.evaluations.previous[plane] += K::ONE;
        assert!(operations::verify(&r.statement, &p).is_err());
        let mut p = r.proof.clone();
        p.evaluations.previous_ab[plane] += K::ONE;
        assert!(operations::verify(&r.statement, &p).is_err());
    }
    for plane in 0..3 {
        let mut p = r.proof.clone();
        p.evaluations.carries_ab[plane] += K::ONE;
        assert!(operations::verify(&r.statement, &p).is_err());
        let mut p = r.proof.clone();
        p.evaluations.carries_c[plane] += K::ONE;
        assert!(operations::verify(&r.statement, &p).is_err());
    }
}

fn private_options(fingerprints: usize) -> operations::Options {
    operations::Options {
        fingerprints,
        private_final_memory: true,
    }
}

#[test]
fn private_final_memory_verifies_without_the_public_final_memory() {
    for fingerprints in [1, 2] {
        let options = private_options(fingerprints);
        for (a, t) in [
            (2, trace()),
            (
                3,
                AccessTrace {
                    initial: (0..8).map(|i| 100 + i).collect(),
                    addresses: vec![7, 0, 7, 3],
                    values: vec![5, 100, 5, 1],
                    is_write: vec![true, false, false, true],
                },
            ),
        ] {
            let r = operations::prove_with(&config(a), &t, &options).unwrap();
            assert!(r.proof.evaluations.final_value.is_some());
            // The harness still sees the prover's final memory.
            assert_eq!(r.statement.final_memory, fixture_final(&t));
            operations::verify_with(&r.statement, &r.proof, &options).unwrap();
            // The verifier never reads the statement's final memory.
            let mut s = r.statement.clone();
            s.final_memory = Vec::new();
            operations::verify_with(&s, &r.proof, &options).unwrap();
            // Public and private proofs are not interchangeable.
            let public = operations::Options {
                private_final_memory: false,
                ..options
            };
            assert!(operations::verify_with(&r.statement, &r.proof, &public).is_err());
        }
    }
    let public = operations::Options::default();
    assert!(
        operations::verify_with(&fixture().statement, &fixture().proof, &private_options(1))
            .is_err()
    );
    operations::verify_with(&fixture().statement, &fixture().proof, &public).unwrap();
}

fn fixture_final(t: &AccessTrace) -> Vec<u64> {
    let mut memory = t.initial.clone();
    for j in 0..t.addresses.len() {
        if t.is_write[j] {
            memory[t.addresses[j] as usize] = t.values[j];
        }
    }
    memory
}

#[test]
fn private_final_memory_tampering_rejects() {
    let options = private_options(1);
    let r = operations::prove_with(&config(2), &trace(), &options).unwrap();
    // A wrong committed-word evaluation fails the leaf check and the Mosaic claim.
    let mut p = r.proof.clone();
    *p.evaluations.final_value.as_mut().unwrap() += K::ONE;
    assert!(operations::verify_with(&r.statement, &p, &options).is_err());
    // Dropping the evaluation is a shape error.
    let mut p = r.proof.clone();
    p.evaluations.final_value = None;
    assert!(operations::verify_with(&r.statement, &p, &options).is_err());
    // The committed words are bound to the PCS opening.
    let mut p = r.proof.clone();
    p.mosaic = fixture().proof.mosaic.clone();
    assert!(operations::verify_with(&r.statement, &p, &options).is_err());
    // Initial memory stays public.
    let mut s = r.statement.clone();
    s.initial_memory[1] ^= 2;
    assert!(operations::verify_with(&s, &r.proof, &options).is_err());
}
