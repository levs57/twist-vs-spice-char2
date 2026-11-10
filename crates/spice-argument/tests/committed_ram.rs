use std::sync::OnceLock;

use om_field::K;
use spice_argument::{
    AccessTrace, Config, OrderingMode, OrderingProof, ProveResult, prove, prove_with_ordering,
    verify,
};

fn flock_order(proof: &mut spice_argument::Proof) -> &mut spice_argument::order_flock::OrderProof {
    match &mut proof.order {
        OrderingProof::Flock(p) => p,
        _ => panic!("default ordering must use Flock"),
    }
}

fn trace(a: usize, n: usize, word_log_bits: usize) -> AccessTrace {
    let cells = 1usize << a;
    let t = 1usize << n;
    let mask = if word_log_bits == 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << word_log_bits)) - 1
    };
    let addresses: Vec<_> = (0..t).map(|j| ((j * 5 + j / 3) % cells) as u32).collect();
    AccessTrace {
        initial: (0..cells).map(|j| (j as u64 * 17 + 7) & mask).collect(),
        reads: vec![addresses.clone()],
        writes: addresses,
        new_values: (0..t).map(|j| (j as u64 * 3 + 11) & mask).collect(),
        read_values: None,
    }
}
fn config(a: usize, word_log_bits: usize) -> Config {
    Config {
        address_bits: a,
        chunk_bits: 1,
        word_log_bits,
        ..Config::default()
    }
}
fn fixture() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| prove(&config(3, 6), &trace(3, 4, 6)).unwrap())
}

#[test]
fn end_to_end_matches_twist_public_memory_and_prices_every_private_column() {
    let result = fixture();
    verify(&result.statement, &result.proof).unwrap();
    let twist = ram_argument::prove(&config(3, 6), &trace(3, 4, 6)).unwrap();
    assert_eq!(result.statement, twist.statement);
    ram_argument::verify(&twist.statement, &twist.proof).unwrap();
    let w = 5;
    assert_eq!(result.proof.order.mode(), OrderingMode::FlockBoolean);
    assert_eq!(result.summary.mosaic_claims, 3 + 5 * w);
    let populated: usize = result
        .summary
        .commitment_components
        .iter()
        .map(|c| 1usize << c.log_size)
        .sum();
    assert_eq!(populated, (128 + 3 + 2 * w - 1) * 16 + w * 8);
    assert!(result.summary.committed_bits >= populated);
    assert!(
        result
            .summary
            .commitment_components
            .iter()
            .all(|c| !c.name.contains("counter"))
    );
    assert!(!result.proof.opening.0.sumcheck.round_msgs.is_empty());
    assert!(!result.proof.opening.1.final_poly.is_empty());
}

#[test]
fn rectangular_domains_padding_and_all_value_word_widths_verify() {
    for (a, n) in [(1, 0), (2, 2), (3, 2), (2, 4)] {
        for width in 0..=6 {
            let result = prove(&config(a, width), &trace(a, n, width)).unwrap();
            verify(&result.statement, &result.proof).unwrap();
        }
    }
}

#[test]
fn full_width_zero_unchanged_and_untouched_words_match_twist() {
    let addresses = vec![0, 0, 3, 1, 0, 3, 1, 0];
    let trace = AccessTrace {
        initial: vec![u64::MAX, 1u64 << 63, 0, 7, 19, 23, 29, 31],
        reads: vec![addresses.clone()],
        writes: addresses,
        new_values: vec![u64::MAX, 0, 7, 1u64 << 63, u64::MAX, 0, 0, 1u64 << 63],
        read_values: None,
    };
    let spice = prove(&config(3, 6), &trace).unwrap();
    verify(&spice.statement, &spice.proof).unwrap();
    let twist = ram_argument::prove(&config(3, 6), &trace).unwrap();
    ram_argument::verify(&twist.statement, &twist.proof).unwrap();
    assert_eq!(spice.statement, twist.statement);
    assert_eq!(spice.statement.final_memory[4..], trace.initial[4..]);
}

#[test]
fn public_memory_configuration_and_commitment_are_bound() {
    let r = fixture();
    let mut s = r.statement.clone();
    s.initial_memory[0] ^= 1;
    assert!(verify(&s, &r.proof).is_err());
    let mut s = r.statement.clone();
    s.final_memory[0] ^= 1;
    assert!(verify(&s, &r.proof).is_err());
    let mut s = r.statement.clone();
    s.config.chunk_bits += 1;
    assert!(verify(&s, &r.proof).is_err());
    let mut p = r.proof.clone();
    p.commitment.root[0] ^= 1;
    assert!(verify(&r.statement, &p).is_err());
}

#[test]
fn permutation_and_all_terminal_columns_are_authenticated() {
    let r = fixture();
    let mut p = r.proof.clone();
    p.permutation.roots[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    for column in 0..5 {
        let mut p = r.proof.clone();
        match column {
            0 => p.inputs.address[0] += K::ONE,
            1 => p.inputs.previous[0] += K::ONE,
            2 => p.inputs.final_previous[0] += K::ONE,
            3 => p.inputs.read += K::ONE,
            _ => p.inputs.write += K::ONE,
        }
        assert!(verify(&r.statement, &p).is_err());
    }
}

#[test]
fn order_previous_and_carry_claims_are_authenticated() {
    let r = fixture();
    let mut p = r.proof.clone();
    flock_order(&mut p).previous_ab[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    flock_order(&mut p).carries_ab[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    flock_order(&mut p).carries_c[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    flock_order(&mut p).zerocheck.round1_c[0] += flock_core::field::F128::ONE;
    assert!(verify(&r.statement, &p).is_err());
}

#[test]
fn ripple_baseline_and_flock_prove_same_ram_and_bind_protocol_choice() {
    let result = fixture();
    let baseline =
        prove_with_ordering(&config(3, 6), &trace(3, 4, 6), OrderingMode::Ripple).unwrap();
    verify(&baseline.statement, &baseline.proof).unwrap();
    assert_eq!(baseline.statement, result.statement);
    assert_eq!(
        baseline.summary.committed_bits,
        result.summary.committed_bits
    );
    assert_eq!(
        baseline.summary.commitment_components,
        result.summary.commitment_components
    );
    assert_eq!(
        result.summary.mosaic_claims,
        baseline.summary.mosaic_claims + 4
    );
    let mut p = result.proof.clone();
    p.order = baseline.proof.order.clone();
    assert!(verify(&result.statement, &p).is_err());
    let mut p = baseline.proof.clone();
    p.order = result.proof.order.clone();
    assert!(verify(&baseline.statement, &p).is_err());
}

#[test]
fn mosaic_ringswitch_and_ligerito_are_required() {
    let r = fixture();
    let mut p = r.proof.clone();
    p.mosaic.v_g += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.opening.0.s[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.opening.1.final_poly[0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    p.opening.1.openings[0].columns[0][0] += K::ONE;
    assert!(verify(&r.statement, &p).is_err());
}

#[test]
fn malformed_inputs_and_proofs_return_errors() {
    let r = fixture();
    let mut p = r.proof.clone();
    p.inputs.previous.clear();
    assert!(verify(&r.statement, &p).is_err());
    let mut p = r.proof.clone();
    flock_order(&mut p).carries_c.clear();
    assert!(verify(&r.statement, &p).is_err());
    let mut t = trace(3, 4, 6);
    t.writes[0] = 8;
    assert!(prove(&config(3, 6), &t).is_err());
    let mut t = trace(3, 4, 6);
    t.reads[0][0] ^= 1;
    assert!(prove(&config(3, 6), &t).is_err());
    let mut t = trace(3, 4, 6);
    t.read_values = Some(vec![vec![0; 16]]);
    assert!(prove(&config(3, 6), &t).is_err());
}
