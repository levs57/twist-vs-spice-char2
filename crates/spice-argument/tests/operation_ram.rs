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
    assert_eq!(r.summary.mosaic_claims, 2 + 5 * 4 + 5);
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
            0 => p.inputs.is_write += K::ONE,
            1 => p.inputs.next_is_write += K::ONE,
            2 => p.inputs.initial_next_is_write += K::ONE,
            3 => p.inputs.masked.cycle[0] += K::ONE,
            4 => p.inputs.masked.cycle[1] += K::ONE,
            _ => p.inputs.masked.initial += K::ONE,
        };
        assert!(operations::verify(&r.statement, &p).is_err());
    }
    for field in 0..4 {
        let mut p = r.proof.clone();
        match field {
            0 => p.masks.value += K::ONE,
            1 => p.masks.is_write += K::ONE,
            2 => p.masks.next_is_write += K::ONE,
            _ => p.masks.initial_next_is_write += K::ONE,
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
    p.inputs.previous[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.inputs.final_previous[0] += K::ONE;
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.order.carries_c[0] += K::ONE;
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
    p.inputs.address.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.order.previous_ab.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.masks.sc.round_msgs.clear();
    assert!(operations::verify(&r.statement, &p).is_err());
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
    p.inputs.masked.cycle[0] += K::ONE;
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
            &operations::Options { fingerprints: 3 }
        )
        .is_err()
    );
}
