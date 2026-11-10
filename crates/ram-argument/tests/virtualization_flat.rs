//! Authenticate the aggregated virtual evaluations through the real PCS.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use om_field::K;
use ram_argument::{AccessTrace, Config, ProveResult, RamProof, VirtualizationMode, prove, verify};
use ram_payload::{GuestOptions, GuestProgram, GuestRows};

fn trace(a: usize, n: usize, ports: usize, shared: bool) -> AccessTrace {
    let initial: Vec<u64> = (0..1usize << a)
        .map(|i| (i as u64).wrapping_mul(0xaed4_963e_125b_d879))
        .collect();
    let mut memory = initial.clone();
    let mut reads = vec![Vec::new(); ports];
    let mut writes = Vec::new();
    let mut new_values = Vec::new();
    for j in 0..1usize << n {
        let address = (j * 13 + (j >> 1) * 7 + 1) % memory.len();
        for (port, read) in reads.iter_mut().enumerate() {
            read.push(if shared {
                address
            } else {
                (j * 3 + port * 17) % memory.len()
            } as u32);
        }
        let next = if j % 3 == 0 {
            memory[address]
        } else {
            memory[address].rotate_left(5) ^ j as u64
        };
        writes.push(address as u32);
        new_values.push(next);
        memory[address] = next;
    }
    AccessTrace {
        initial,
        reads,
        writes,
        new_values,
        read_values: None,
    }
}

fn config(a: usize, c: usize, ports: usize, shared: bool) -> Config {
    Config {
        address_bits: a,
        chunk_bits: c,
        read_ports: ports,
        shared_address: shared,
        virtualization: VirtualizationMode::Flat,
        ..Config::default()
    }
}

fn fixture() -> &'static ProveResult {
    static PROOF: OnceLock<ProveResult> = OnceLock::new();
    PROOF.get_or_init(|| prove(&config(8, 2, 1, true), &trace(8, 5, 1, true)).unwrap())
}

fn rejects(proof: &RamProof) {
    let result = catch_unwind(AssertUnwindSafe(|| verify(&fixture().statement, proof)));
    assert!(result.is_ok(), "malformed proof panicked");
    assert!(result.unwrap().is_err(), "malformed proof accepted");
}

#[test]
fn aggregate_flat_proofs_cover_chunk_counts_cycles_and_ports() {
    for (a, c, n, ports, shared) in [
        (4, 4, 0, 1, true),
        (4, 2, 1, 1, true),
        (6, 2, 3, 1, true),
        (8, 2, 5, 1, true),
        (8, 1, 4, 1, true),
        (16, 1, 2, 1, true),
        (8, 2, 4, 2, false),
    ] {
        let input = trace(a, n, ports, shared);
        let result = prove(&config(a, c, ports, shared), &input).unwrap();
        verify(&result.statement, &result.proof).unwrap();
        let mut expected = input.initial.clone();
        for (&address, &value) in input.writes.iter().zip(&input.new_values) {
            expected[address as usize] = value;
        }
        assert_eq!(result.statement.final_memory, expected);
        assert!(result.proof.twist.virt.is_none() && result.proof.twist.virt_tree.is_none());
        if a == c {
            assert!(result.proof.twist.virt_flat.is_none());
        } else {
            let proof = result.proof.twist.virt_flat.as_ref().unwrap();
            assert_eq!(proof.reduction.sc.round_msgs.len(), n);
            assert!(proof.reduction.sc.round_msgs.iter().all(|m| m.len() == 2));
            assert_eq!(proof.sc.round_msgs.len(), n);
            assert!(proof.sc.round_msgs.iter().all(|m| m.len() == a / c + 1));
            assert_eq!(proof.evals.len(), proof.reduction.evals.len() * (a / c));
        }
    }
}

#[test]
fn every_reduction_product_and_leaf_message_is_authenticated() {
    let base = fixture();
    let flat = base.proof.twist.virt_flat.as_ref().unwrap();
    for reduction in [true, false] {
        let messages = if reduction {
            &flat.reduction.sc.round_msgs
        } else {
            &flat.sc.round_msgs
        };
        for (round, msg) in messages.iter().enumerate() {
            for field in 0..msg.len() {
                let mut proof = base.proof.clone();
                let flat = proof.twist.virt_flat.as_mut().unwrap();
                let sc = if reduction {
                    &mut flat.reduction.sc
                } else {
                    &mut flat.sc
                };
                sc.round_msgs[round][field] += K::ONE;
                rejects(&proof);
            }
        }
    }
    for reduction in [true, false] {
        let len = if reduction {
            flat.reduction.evals.len()
        } else {
            flat.evals.len()
        };
        for field in 0..len {
            let mut proof = base.proof.clone();
            let flat = proof.twist.virt_flat.as_mut().unwrap();
            if reduction {
                flat.reduction.evals[field] += K::ONE;
            } else {
                flat.evals[field] += K::ONE;
            }
            rejects(&proof);
        }
    }
}

#[test]
fn malformed_flat_shapes_and_missing_proofs_reject_without_panics() {
    let base = fixture();
    for reduction in [true, false] {
        let mut proof = base.proof.clone();
        let flat = proof.twist.virt_flat.as_mut().unwrap();
        if reduction {
            flat.reduction.evals.pop();
        } else {
            flat.evals.push(K::ZERO);
        }
        rejects(&proof);
        let mut proof = base.proof.clone();
        let flat = proof.twist.virt_flat.as_mut().unwrap();
        let sc = if reduction {
            &mut flat.reduction.sc
        } else {
            &mut flat.sc
        };
        sc.round_msgs.pop();
        rejects(&proof);
        let mut proof = base.proof.clone();
        let flat = proof.twist.virt_flat.as_mut().unwrap();
        let sc = if reduction {
            &mut flat.reduction.sc
        } else {
            &mut flat.sc
        };
        sc.round_msgs[0].push(K::ZERO);
        rejects(&proof);
    }
    let mut proof = base.proof.clone();
    proof.twist.virt_flat = None;
    rejects(&proof);
}

#[test]
fn new_legacy_and_tree_domains_and_branches_are_distinct() {
    for c in [2, 8] {
        let input = trace(8, 4, 1, true);
        let modes = [
            VirtualizationMode::Flat,
            VirtualizationMode::FlatLegacy,
            VirtualizationMode::GkrTree,
        ];
        let results: Vec<_> = modes
            .iter()
            .map(|&mode| {
                let mut cfg = config(8, c, 1, true);
                cfg.virtualization = mode;
                let result = prove(&cfg, &input).unwrap();
                verify(&result.statement, &result.proof).unwrap();
                result
            })
            .collect();
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.proof.commitment, results[0].proof.commitment);
            for (j, other) in results.iter().enumerate() {
                if i == j {
                    continue;
                }
                let checked =
                    catch_unwind(AssertUnwindSafe(|| verify(&result.statement, &other.proof)));
                assert!(checked.is_ok() && checked.unwrap().is_err());
            }
        }
        if c != 8 {
            let mut mixed = results[0].proof.clone();
            mixed.twist.virt = results[1].proof.twist.virt.clone();
            assert!(verify(&results[0].statement, &mixed).is_err());
            let mut mixed = results[0].proof.clone();
            mixed.twist.virt_tree = results[2].proof.twist.virt_tree.clone();
            assert!(verify(&results[0].statement, &mixed).is_err());
        }
    }
}

#[test]
fn aggregate_flat_proves_all_real_guests_and_row_filters() {
    for program in [
        GuestProgram::BinarySearch,
        GuestProgram::Erc20MixedBlock,
        GuestProgram::PointerChase,
    ] {
        for rows in [GuestRows::All, GuestRows::Memory] {
            let options = GuestOptions {
                program,
                rows,
                keys: 8,
                queries: 4,
                size: 8,
                operations: 12,
                ..GuestOptions::default()
            };
            let payload = ram_payload::guest(6, 8, &options).unwrap();
            let input = AccessTrace {
                initial: payload.initial_memory,
                reads: payload.reads,
                writes: payload.writes,
                new_values: payload.new_values,
                read_values: Some(payload.read_values),
            };
            let result = prove(&config(8, 2, 1, true), &input).unwrap();
            verify(&result.statement, &result.proof).unwrap();
            assert!(result.proof.twist.virt_flat.is_some());
        }
    }
}
