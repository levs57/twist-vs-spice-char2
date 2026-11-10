//! RAM proof-value equivalence against a separately compiled backend.
//! Run through scripts/check_sumcheck_optimizations.py.
//! Pinned upstream compares FlatLegacy; an extracted reference also compares GKR.
//! The new Flat protocol is excluded because its proof format changed by design.

use ram_argument as optimized;
use ram_argument_reference as reference;

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn check(
    a: usize,
    c: usize,
    n: usize,
    ports: usize,
    shared: bool,
    pattern: usize,
    seed: u64,
    tree: bool,
) {
    let mut rng = seed;
    let words = 1usize << a;
    let cycles = 1usize << n;
    let initial: Vec<u64> = (0..words)
        .map(|_| if pattern == 0 { 0 } else { next(&mut rng) })
        .collect();
    let mut memory = initial.clone();
    let mut reads = vec![Vec::with_capacity(cycles); ports];
    let mut writes = Vec::with_capacity(cycles);
    let mut values = Vec::with_capacity(cycles);
    for j in 0..cycles {
        let write = match pattern {
            0 => 0,
            1 => next(&mut rng) as usize % words.min(5),
            _ => next(&mut rng) as usize % words,
        };
        for read in &mut reads {
            read.push(if shared {
                write as u32
            } else {
                (next(&mut rng) as usize % words) as u32
            });
        }
        let value = if pattern == 0 || (pattern == 1 && j % 7 != 3) {
            memory[write]
        } else if j % 5 == 0 {
            0
        } else {
            next(&mut rng)
        };
        writes.push(write as u32);
        values.push(value);
        memory[write] = value;
    }
    let config = optimized::Config {
        address_bits: a,
        chunk_bits: c,
        read_ports: ports,
        shared_address: shared,
        virtualization: if tree {
            optimized::VirtualizationMode::GkrTree
        } else {
            optimized::VirtualizationMode::FlatLegacy
        },
        ..Default::default()
    };
    let old_config = reference::Config {
        address_bits: a,
        chunk_bits: c,
        read_ports: ports,
        shared_address: shared,
        virtualization: if tree {
            reference::VirtualizationMode::GkrTree
        } else {
            reference::VirtualizationMode::FlatLegacy
        },
        ..Default::default()
    };
    let trace = optimized::AccessTrace {
        initial: initial.clone(),
        reads: reads.clone(),
        writes: writes.clone(),
        new_values: values.clone(),
        read_values: None,
    };
    let old_trace = reference::AccessTrace {
        initial,
        reads,
        writes,
        new_values: values,
        read_values: None,
    };
    let old = reference::prove(&old_config, &old_trace).unwrap();
    reference::verify(&old.statement, &old.proof).unwrap();
    let new = optimized::prove(&config, &trace).unwrap();
    optimized::verify(&new.statement, &new.proof).unwrap();
    assert_eq!(new.statement.final_memory, memory);
    assert_eq!(new.statement.final_memory, old.statement.final_memory);
    if !tree {
        assert!(new.proof.twist.virt_tree.is_none());
    }
    assert!(new.proof.twist.virt_flat.is_none());
    // All proof members recursively derive Debug over complete field elements,
    // byte arrays and vectors. The two builds have distinct Rust types; compare
    // their complete deterministic value representations rather than casting.
    // The new Flat proof field is absent from both reference types. The tree
    // field is also absent from pinned upstream. Remove only checked None fields;
    // every original proof value and every saved GKR field remains compared.
    let new_debug = format!("{:?}", new.proof);
    assert_eq!(new_debug.matches(", virt_flat: None").count(), 1);
    let new_debug = new_debug.replacen(", virt_flat: None", "", 1);
    let comparable = if cfg!(feature = "extracted-reference") {
        new_debug
    } else {
        assert_eq!(new_debug.matches(", virt_tree: None").count(), 1);
        new_debug.replacen(", virt_tree: None", "", 1)
    };
    assert_eq!(
        comparable,
        format!("{:?}", old.proof),
        "proof mismatch a={a} c={c} n={n} p={ports} shared={shared} pattern={pattern} seed={seed}"
    );
    assert_eq!(format!("{:?}", new.summary), format!("{:?}", old.summary));
    println!(
        "matched a={a} c={c} n={n} ports={ports} shared={shared} pattern={pattern} seed={seed} mode={}",
        if tree { "gkr" } else { "flat-legacy" }
    );
}

fn main() {
    let configurations: [(usize, usize, usize, usize, bool); 10] = [
        (4, 4, 2, 1, true), // Direct virtualization and the smallest supported trace.
        (6, 2, 4, 1, true),
        (8, 4, 6, 1, true),
        (8, 2, 5, 1, true),
        (6, 3, 5, 2, false),
        (8, 4, 5, 1, false),
        (8, 1, 4, 1, true),   // Degree-nine product rounds, including points >= 8.
        (12, 4, 16, 1, true), // Parallel row materialization, dense folds, and sparse cycles.
        (16, 1, 4, 1, true),  // Degree17 FlatLegacy and all four GKR layers.
        (16, 4, 14, 1, true), // Parallel power-of-two component tree.
    ];
    let mut total = 0;
    let modes: &[bool] = if cfg!(feature = "extracted-reference") {
        &[false, true]
    } else {
        &[false]
    };
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for &(a, c, n, ports, shared) in &configurations {
                for &tree in modes {
                    if tree && !(a / c).is_power_of_two() {
                        continue;
                    }
                    for pattern in 0..3 {
                        check(a, c, n, ports, shared, pattern, 17 + pattern as u64, tree);
                        total += 1;
                    }
                }
            }
        });
        println!("completed workers={workers}");
    }
    println!(
        "Matched {total} complete proof values; independently verified both versions for every case."
    );
}
