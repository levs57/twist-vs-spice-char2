//! Full committed proofs exercise the tree's terminal claims through Mosaic/PCS.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use om_field::K;
use ram_argument::{
    AccessTrace, Config, ProveResult, RamProof, Statement, VirtualizationMode, prove, verify,
};
use ram_payload::{GuestOptions, GuestProgram, GuestRows};

fn trace(address_bits: usize, log_cycles: usize, ports: usize, shared: bool) -> AccessTrace {
    let cells = 1usize << address_bits;
    let cycles = 1usize << log_cycles;
    let initial: Vec<u64> = (0..cells)
        .map(|i| (i as u64).wrapping_mul(0x94d0_49bb_1331_11eb) ^ 0x1234_5678_9abc_def0)
        .collect();
    let mut memory = initial.clone();
    let mut reads = vec![Vec::with_capacity(cycles); ports];
    let mut read_values = vec![Vec::with_capacity(cycles); ports];
    let mut writes = Vec::with_capacity(cycles);
    let mut new_values = Vec::with_capacity(cycles);
    for j in 0..cycles {
        let write = (j * 13 + (j >> 2) * 7 + 3) % cells;
        for port in 0..ports {
            let read = if shared {
                write
            } else {
                (j * (5 + 2 * port) + 17 * port + 1) % cells
            };
            reads[port].push(read as u32);
            read_values[port].push(memory[read]);
        }
        let value = match j % 4 {
            0 => memory[write], // Unchanged writes still prove an access.
            1 => 0,
            _ => memory[write].rotate_left(11) ^ (j as u64 + 1),
        };
        writes.push(write as u32);
        new_values.push(value);
        memory[write] = value;
    }
    AccessTrace {
        initial,
        reads,
        writes,
        new_values,
        read_values: Some(read_values),
    }
}

fn expected_final(trace: &AccessTrace) -> Vec<u64> {
    let mut memory = trace.initial.clone();
    for (&address, &value) in trace.writes.iter().zip(&trace.new_values) {
        memory[address as usize] = value;
    }
    memory
}

fn config(address_bits: usize, chunk_bits: usize, ports: usize, shared: bool) -> Config {
    Config {
        address_bits,
        chunk_bits,
        read_ports: ports,
        shared_address: shared,
        virtualization: VirtualizationMode::GkrTree,
        ..Config::default()
    }
}

fn fixture() -> &'static ProveResult {
    static RESULT: OnceLock<ProveResult> = OnceLock::new();
    RESULT.get_or_init(|| prove(&config(8, 1, 1, true), &trace(8, 5, 1, true)).unwrap())
}

fn rejects_without_panic(statement: &Statement, proof: &RamProof, case: &str) {
    let checked = catch_unwind(AssertUnwindSafe(|| verify(statement, proof)));
    assert!(checked.is_ok(), "verifier panicked: {case}");
    assert!(checked.unwrap().is_err(), "verifier accepted: {case}");
}

#[test]
fn complete_tree_proofs_cover_depths_small_cycle_counts_and_independent_ports() {
    for (a, chunk, n, ports, shared) in [
        (4, 4, 0, 1, true),  // d=1, no virtualization proof, n=0.
        (4, 2, 1, 1, true),  // d=2, n=1.
        (8, 2, 4, 1, true),  // d=4, shared-address aliases.
        (8, 1, 4, 1, true),  // d=8.
        (8, 2, 4, 2, false), // Separate streams, multiple read ports.
        (16, 1, 2, 1, true), // d=16, four cubic layers.
    ] {
        let trace = trace(a, n, ports, shared);
        let result = prove(&config(a, chunk, ports, shared), &trace).unwrap();
        verify(&result.statement, &result.proof).unwrap();
        assert_eq!(result.statement.final_memory, expected_final(&trace));
        assert!(result.proof.twist.virt.is_none());
        if a == chunk {
            assert!(result.proof.twist.virt_tree.is_none());
        } else {
            let tree = result.proof.twist.virt_tree.as_ref().unwrap();
            assert_eq!(tree.layers.len(), (a / chunk).ilog2() as usize);
            for (layer, node) in tree.layers.iter().enumerate() {
                assert_eq!(node.sumcheck.round_msgs.len(), n);
                assert!(
                    node.sumcheck
                        .round_msgs
                        .iter()
                        .all(|message| message.len() == 3)
                );
                assert_eq!(node.child_evals.len() % (1 << (layer + 1)), 0);
            }
        }
        assert!(!result.proof.opening.1.final_poly.is_empty());
    }
}

#[test]
fn every_tree_layer_and_child_value_is_authenticated() {
    let result = fixture();
    let layers = &result.proof.twist.virt_tree.as_ref().unwrap().layers;
    assert_eq!(layers.len(), 3);
    for (layer, data) in layers.iter().enumerate() {
        for child in 0..data.child_evals.len() {
            let mut proof = result.proof.clone();
            proof.twist.virt_tree.as_mut().unwrap().layers[layer].child_evals[child] += K::ONE;
            rejects_without_panic(
                &result.statement,
                &proof,
                &format!("layer {layer} child {child}"),
            );
        }
        for round in 0..data.sumcheck.round_msgs.len() {
            let mut proof = result.proof.clone();
            proof.twist.virt_tree.as_mut().unwrap().layers[layer]
                .sumcheck
                .round_msgs[round][0] += K::ONE;
            rejects_without_panic(
                &result.statement,
                &proof,
                &format!("layer {layer} round {round}"),
            );
        }
    }
    let mut proof = result.proof.clone();
    proof.mosaic.v_g += K::ONE;
    rejects_without_panic(
        &result.statement,
        &proof,
        "terminal committed-leaf aggregation",
    );
}

#[test]
fn malformed_and_reordered_tree_nodes_return_errors() {
    let result = fixture();
    for layer in 0..3 {
        for add in [false, true] {
            let mut proof = result.proof.clone();
            let values = &mut proof.twist.virt_tree.as_mut().unwrap().layers[layer].child_evals;
            if add {
                values.push(K::ZERO);
            } else {
                values.pop();
            }
            rejects_without_panic(&result.statement, &proof, "wrong child count");

            let mut proof = result.proof.clone();
            let messages = &mut proof.twist.virt_tree.as_mut().unwrap().layers[layer]
                .sumcheck
                .round_msgs;
            if add {
                messages.push(vec![K::ZERO; 3]);
            } else {
                messages.pop();
            }
            rejects_without_panic(&result.statement, &proof, "wrong round count");
        }
        let mut proof = result.proof.clone();
        proof.twist.virt_tree.as_mut().unwrap().layers[layer]
            .sumcheck
            .round_msgs[0]
            .pop();
        rejects_without_panic(&result.statement, &proof, "wrong message degree");
    }
    for add in [false, true] {
        let mut proof = result.proof.clone();
        let layers = &mut proof.twist.virt_tree.as_mut().unwrap().layers;
        if add {
            layers.push(layers[0].clone());
        } else {
            layers.pop();
        }
        rejects_without_panic(&result.statement, &proof, "wrong layer count");
    }
    let mut proof = result.proof.clone();
    proof.twist.virt_tree.as_mut().unwrap().layers.swap(0, 1);
    rejects_without_panic(&result.statement, &proof, "reordered layers");
    let mut proof = result.proof.clone();
    let values = &mut proof.twist.virt_tree.as_mut().unwrap().layers[1].child_evals;
    let other = (1..values.len()).find(|&i| values[i] != values[0]).unwrap();
    values.swap(0, other);
    rejects_without_panic(&result.statement, &proof, "reordered distinct children");
    let mut proof = result.proof.clone();
    proof.twist.virt_tree = None;
    rejects_without_panic(&result.statement, &proof, "missing tree");
}

#[test]
fn mode_and_proof_branch_are_bound_including_the_direct_shortcut() {
    for chunk in [1, 8] {
        let trace = trace(8, 4, 1, true);
        let tree_config = config(8, chunk, 1, true);
        let tree = prove(&tree_config, &trace).unwrap();
        let mut flat_config = tree_config.clone();
        flat_config.virtualization = VirtualizationMode::Flat;
        let flat = prove(&flat_config, &trace).unwrap();
        verify(&flat.statement, &flat.proof).unwrap();
        verify(&tree.statement, &tree.proof).unwrap();
        // Commitment contents are shared; only the selected protocol changes.
        assert_eq!(tree.proof.commitment, flat.proof.commitment);
        rejects_without_panic(
            &flat.statement,
            &tree.proof,
            "tree proof under Flat statement",
        );
        rejects_without_panic(
            &tree.statement,
            &flat.proof,
            "Flat proof under tree statement",
        );
        if chunk != 8 {
            let mut proof = tree.proof.clone();
            proof.twist.virt_flat = flat.proof.twist.virt_flat.clone();
            rejects_without_panic(&tree.statement, &proof, "both branches present");
            proof.twist.virt_tree = None;
            rejects_without_panic(&tree.statement, &proof, "Flat branch substituted for tree");
            let mut proof = flat.proof.clone();
            proof.twist.virt_tree = tree.proof.twist.virt_tree.clone();
            rejects_without_panic(&flat.statement, &proof, "tree branch attached to Flat");
        }
    }
}

#[test]
fn tree_requires_power_of_two_component_count_while_flat_retains_other_counts() {
    let trace = trace(6, 3, 1, true);
    let mut config = config(6, 2, 1, true); // d=3.
    assert!(prove(&config, &trace).is_err());
    config.virtualization = VirtualizationMode::Flat;
    let flat = prove(&config, &trace).unwrap();
    verify(&flat.statement, &flat.proof).unwrap();
    let mut statement = flat.statement.clone();
    statement.config.virtualization = VirtualizationMode::GkrTree;
    rejects_without_panic(&statement, &flat.proof, "non-power-of-two tree statement");
}

#[test]
fn tree_proves_traces_from_each_original_guest_and_memory_filter() {
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
            let payload = ram_payload::guest(8, 8, &options).unwrap();
            let guest = payload.metadata.guest.as_ref().unwrap();
            assert_eq!(guest.output_words, guest.expected_output_words);
            assert_eq!(payload.metadata.padding_cycles, 0);
            let trace = AccessTrace {
                initial: payload.initial_memory,
                reads: payload.reads,
                writes: payload.writes,
                new_values: payload.new_values,
                read_values: Some(payload.read_values),
            };
            let result = prove(&config(8, 2, 1, true), &trace).unwrap();
            verify(&result.statement, &result.proof).unwrap();
            assert_eq!(result.statement.final_memory, expected_final(&trace));
            assert_eq!(
                result.proof.twist.virt_tree.as_ref().unwrap().layers.len(),
                2
            );
        }
    }
}

#[test]
fn tree_binds_internal_parent_mles_away_from_the_boolean_cube() {
    use om_memcheck::onehot::{AddrComps, VirtGroup, VirtTerm, comp_tables, full_table};
    use om_memcheck::virt_tree::{virt_tree_prove, virt_tree_verify};
    use om_sumcheck::{mle_eval, mle_eval_bits};
    use om_transcript::Transcript;

    let (a, d, n) = (4, 4, 1);
    let addresses = [0usize, 15];
    let compact = AddrComps::from_usizes(&addresses, a, d);
    let tables = comp_tables(&addresses, a, d);
    let rk = vec![K::t(2), K::t(3), K::t(4), K::t(5)];
    let z = K::t(7);
    let left = [K::ONE + rk[0], rk[0]];
    let right = [K::ONE + rk[1], rk[1]];
    let parent = [left[0] * right[0], left[1] * right[1]];
    // The next layer must prove this parent MLE, not replace it with the
    // product of the two child MLEs at the preceding challenge.
    assert_eq!(
        mle_eval(&parent, &[z]) + mle_eval(&left, &[z]) * mle_eval(&right, &[z]),
        z.square() + z,
    );
    assert_ne!(z.square() + z, K::ZERO);
    let mut point = rk.clone();
    point.push(z);
    let groups = vec![VirtGroup {
        names: (0..d).map(|i| format!("leaf{i}")).collect(),
        r_k: rk,
        terms: vec![VirtTerm {
            cyc: vec![z],
            value: mle_eval_bits(&full_table(&tables, a, d, 2), &point),
        }],
    }];
    let mut prover = Transcript::new_default(b"off-cube-parent");
    let (proof, claims) = virt_tree_prove(&groups, &[&compact], a, d, n, &mut prover, b"tree");
    let mut verifier = Transcript::new_default(b"off-cube-parent");
    let checked =
        virt_tree_verify(&groups, a, d, n, proof.as_ref(), &mut verifier, b"tree").unwrap();
    assert_eq!(claims, checked);
    assert_eq!(prover.challenge_k(b"after"), verifier.challenge_k(b"after"));
    for (i, claim) in checked.iter().enumerate() {
        assert_eq!(claim.value, mle_eval_bits(&tables[i], &claim.point));
    }
    let mut wrong = groups.clone();
    wrong[0].terms[0].value += K::ONE;
    assert!(
        virt_tree_verify(
            &wrong,
            a,
            d,
            n,
            proof.as_ref(),
            &mut Transcript::new_default(b"off-cube-parent"),
            b"tree",
        )
        .is_err()
    );
}
