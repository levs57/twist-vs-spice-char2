use std::sync::OnceLock;

use om_field::K;
use ram_argument::{AccessTrace, Config, ProveResult, VirtualizationMode, prove, verify};

fn trace() -> AccessTrace {
    AccessTrace {
        initial: (0..16).map(|i| i * 7).collect(),
        reads: vec![
            (0..32).map(|i| ((i * 5) % 16) as u32).collect(),
            (0..32).map(|i| ((i + 3) % 16) as u32).collect(),
        ],
        writes: (0..32).map(|i| ((i * 3 + 1) % 16) as u32).collect(),
        new_values: (0..32).map(|i| i * 3 + 2).collect(),
        read_values: None,
    }
}

fn config() -> Config {
    Config {
        address_bits: 4,
        chunk_bits: 2,
        read_ports: 2,
        shared_address: false,
        word_log_bits: 3,
        ..Config::default()
    }
}

fn fixture() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| prove(&config(), &trace()).unwrap())
}

#[test]
fn complete_proof_verifies_without_witness() {
    let result = fixture();
    verify(&result.statement, &result.proof).unwrap();
    assert_eq!(
        result.statement.config.virtualization,
        VirtualizationMode::Flat
    );
    assert!(result.proof.twist.virt_tree.is_none());
    assert_eq!(result.summary.cycles, 32);
    assert_eq!(result.summary.address_components, 2);
    assert!(result.summary.committed_bits >= 1 << 14);
    assert_eq!(
        result.summary.committed_bits / 128,
        result.summary.packed_field_elements
    );
    assert!(result.proof.commitment.n > 6);
    assert!(result.summary.mosaic_claims >= 15);
    assert!(
        result
            .summary
            .commitment_components
            .iter()
            .any(|c| c.name == "input/rv0")
    );
    assert!(
        result
            .summary
            .commitment_components
            .iter()
            .all(|c| !c.name.starts_with("input/address"))
    );
    assert_eq!(
        result
            .summary
            .commitment_components
            .iter()
            .map(|c| 1usize << c.log_size)
            .sum::<usize>()
            / result.summary.cycles,
        56,
    );
    assert!(!result.proof.opening.0.sumcheck.round_msgs.is_empty());
    assert!(!result.proof.opening.1.final_poly.is_empty());
}

#[test]
fn public_memory_and_config_are_bound() {
    let result = fixture();
    let mut initial = result.statement.clone();
    initial.initial_memory[0] ^= 1;
    assert!(verify(&initial, &result.proof).is_err());
    let mut final_memory = result.statement.clone();
    final_memory.final_memory[0] ^= 1;
    assert!(verify(&final_memory, &result.proof).is_err());
    let mut config = result.statement.clone();
    config.config.chunk_bits = 1;
    assert!(verify(&config, &result.proof).is_err());
    let mut mode = result.statement.clone();
    mode.config.virtualization = VirtualizationMode::GkrTree;
    assert!(verify(&mode, &result.proof).is_err());
}

#[test]
fn commitment_and_challenge_replay_are_rejected() {
    let result = fixture();
    let mut root = result.proof.clone();
    root.commitment.root[0] ^= 1;
    assert!(verify(&result.statement, &root).is_err());
    let mut point = result.proof.clone();
    point.input_claims.r_o[0] += K::ONE;
    assert!(verify(&result.statement, &point).is_err());
    let mut statement = result.statement.clone();
    statement.log_cycles += 1;
    assert!(verify(&statement, &result.proof).is_err());
}

#[test]
fn every_kind_of_twist_input_claim_is_authenticated() {
    let result = fixture();
    let mut read = result.proof.clone();
    read.input_claims.rv[0] += K::ONE;
    assert!(verify(&result.statement, &read).is_err());
    let mut write = result.proof.clone();
    write.input_claims.wv += K::ONE;
    assert!(verify(&result.statement, &write).is_err());
    let mut address = result.proof.clone();
    address.input_claims.bits[0][0] += K::ONE;
    assert!(verify(&result.statement, &address).is_err());
    let mut write_address = result.proof.clone();
    write_address.input_claims.bits[2][3] += K::ONE;
    assert!(verify(&result.statement, &write_address).is_err());
}

#[test]
fn mosaic_ringswitch_and_ligerito_tampering_is_rejected() {
    let result = fixture();
    let mut mosaic = result.proof.clone();
    mosaic.mosaic.v_g += K::ONE;
    assert!(verify(&result.statement, &mosaic).is_err());
    let mut ring = result.proof.clone();
    ring.opening.0.s[0] += K::ONE;
    assert!(verify(&result.statement, &ring).is_err());
    let mut ligerito = result.proof.clone();
    ligerito.opening.1.final_poly[0] += K::ONE;
    assert!(verify(&result.statement, &ligerito).is_err());
}

#[test]
fn malformed_shapes_return_errors() {
    let result = fixture();
    let mut shape = result.proof.clone();
    shape.input_claims.bits[0].clear();
    assert!(verify(&result.statement, &shape).is_err());
    let mut pcs = result.proof.clone();
    pcs.opening.0.s.clear();
    assert!(verify(&result.statement, &pcs).is_err());
}

#[test]
fn inconsistent_supplied_reads_are_rejected_before_proving() {
    let mut trace = trace();
    trace.read_values = Some(vec![vec![0; 32]; 2]);
    assert!(
        prove(&config(), &trace)
            .unwrap_err()
            .0
            .contains("read value")
    );
}

#[test]
fn invalid_configuration_and_trace_shapes_return_errors() {
    let mut config = config();
    config.chunk_bits = 0;
    assert!(prove(&config, &trace()).is_err());
    config.chunk_bits = 3;
    assert!(prove(&config, &trace()).is_err());
    config.chunk_bits = 2;
    config.log_rate = 1;
    assert!(prove(&config, &trace()).is_err());
    config.log_rate = 2;
    let mut trace = trace();
    trace.writes[0] = 16;
    assert!(prove(&config, &trace).is_err());
    trace.writes.pop();
    assert!(prove(&config, &trace).is_err());
}

#[test]
fn single_component_and_full_word_path_work() {
    let config = Config {
        address_bits: 4,
        chunk_bits: 4,
        shared_address: false,
        ..Config::default()
    };
    let trace = AccessTrace {
        initial: vec![u64::MAX; 16],
        reads: vec![vec![0, 0, 1, 1]],
        writes: vec![0, 1, 0, 1],
        new_values: vec![7, 9, 11, 13],
        read_values: Some(vec![vec![u64::MAX, 7, 9, 9]]),
    };
    let result = prove(&config, &trace).unwrap();
    assert!(result.proof.twist.virt.is_none());
    assert!(result.proof.twist.virt_tree.is_none());
    verify(&result.statement, &result.proof).unwrap();
}

fn shared_fixture() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| {
        let config = Config {
            address_bits: 4,
            chunk_bits: 2,
            ..Config::default()
        };
        let trace = AccessTrace {
            initial: vec![u64::MAX; 16],
            reads: vec![vec![0, 1, 0, 1]],
            writes: vec![0, 1, 0, 1],
            new_values: vec![7, 9, 11, 13],
            read_values: Some(vec![vec![u64::MAX, u64::MAX, 7, 9]]),
        };
        prove(&config, &trace).unwrap()
    })
}

#[test]
fn shared_address_commits_rv_inc_and_one_address_encoding() {
    let result = shared_fixture();
    verify(&result.statement, &result.proof).unwrap();
    let components = &result.summary.commitment_components;
    assert_eq!(components.len(), 4); // Two address chunks, RV, Inc.
    assert!(components.iter().all(|c| c.name != "input/wv"
        && !c.name.starts_with("input/address")
        && !c.name.starts_with("ram/wa/")));
    assert_eq!(
        components
            .iter()
            .map(|c| 1usize << c.log_size)
            .sum::<usize>()
            / result.summary.cycles,
        136
    );
    let claims = &result.proof.input_claims;
    let increments = [u64::MAX ^ 7, u64::MAX ^ 9, 7 ^ 11, 9 ^ 13];
    let inc = om_memcheck::twist::word_mle_eval(&increments, &claims.r_o, &claims.r_c);
    assert_eq!(claims.wv, claims.rv[0] + inc);
    assert_eq!(claims.bits[0], claims.bits[1]);
}

#[test]
fn shared_address_mode_binds_aliases_values_and_internal_address_claims() {
    let result = shared_fixture();
    let mut independent = result.statement.clone();
    independent.config.shared_address = false;
    assert!(verify(&independent, &result.proof).is_err());
    let mut write = result.proof.clone();
    write.input_claims.wv += K::ONE;
    assert!(verify(&result.statement, &write).is_err());
    let mut read = result.proof.clone();
    read.input_claims.rv[0] += K::ONE;
    assert!(verify(&result.statement, &read).is_err());
    let mut one_address = result.proof.clone();
    one_address.input_claims.bits[0][0] += K::ONE;
    assert!(verify(&result.statement, &one_address).is_err());
    let mut both_addresses = result.proof.clone();
    both_addresses.input_claims.bits[0][0] += K::ONE;
    both_addresses.input_claims.bits[1][0] += K::ONE;
    assert!(verify(&result.statement, &both_addresses).is_err());
}

#[test]
fn shared_address_mode_rejects_independent_streams_and_multiple_ports() {
    let mut config = config();
    config.shared_address = true;
    assert!(prove(&config, &trace()).is_err());
    config.read_ports = 1;
    let mut trace = trace();
    trace.reads.truncate(1);
    assert!(
        prove(&config, &trace)
            .unwrap_err()
            .0
            .contains("equal read and write")
    );
}
