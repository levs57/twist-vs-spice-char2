//! The byte-identical pinned upstream GKR, with its actual field and transcript.

use std::hint::black_box;
use std::time::Instant;

use primitives::field::F192;

const PAR_THRESHOLD: usize = 1 << 11;
pub(crate) use primitives::log2_ceil_usize;

mod transcript {
    pub use fiat_shamir::transcript::ProverState;
}

#[allow(dead_code)]
#[rustfmt::skip]
#[path = "../../reference/crates/leanvm_core/src/gkr.rs"]
mod gkr;

fn random(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |i: usize, default| {
        args.get(i)
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let (log, threads, samples, padding) =
        (number(0, 18), number(1, 1), number(2, 5), number(3, 0) != 0);
    assert!((1..32).contains(&log) && threads > 0 && samples > 0);
    // SAFETY: set before this process starts any worker or pool initialization.
    unsafe {
        std::env::set_var("LEANVM_NUM_THREADS", threads.to_string());
    }
    assert_eq!(parallel::num_threads(), threads);
    let size = 1usize << log;
    let half = size / 2;
    let boundary = (half / 16).max(1);
    let ranges = if padding {
        vec![half + boundary..size]
    } else {
        Vec::new()
    };
    let input: Vec<F192> = (0..size)
        .map(|i| {
            if ranges.iter().any(|r| r.contains(&i)) {
                F192::ONE
            } else {
                F192::new(
                    random(i as u64),
                    random(i as u64 ^ 0x1234),
                    random(i as u64 ^ 0x5678),
                )
            }
        })
        .collect();
    let active = if padding { half + boundary } else { size };
    let pairs = ((size - (1usize << (log % 2))) / 3).saturating_sub(log / 2);
    println!(
        "backend,field_bits,trees,log_size,logical_leaves_per_tree,active_leaves_per_tree,threads,sample,prepare_ms,prove_ms,total_ms,total_ms_per_tree,quartic_pair_upper_bound_per_tree,fold_row_upper_bound_per_tree,proof_fields,root0,leaf0"
    );
    for iteration in 0..samples + 1 {
        let values = [input.clone(), input.clone(), input.clone()];
        let started = Instant::now();
        let leaves = values.map(|mut v| {
            for range in &ranges {
                assert!(v[range.clone()].iter().all(|&value| value == F192::ONE));
            }
            // Both backends receive the same structural implicit suffix.
            if let Some(last) = ranges.last().filter(|r| r.end == size) {
                v.truncate(last.start.max(1));
            }
            let mut leaf = gkr::Leaves::new(v);
            leaf.ones = ranges.clone();
            leaf
        });
        let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut tr = transcript::ProverState::from_label(b"grandproduct-microbenchmark");
        let prove_started = Instant::now();
        let claims = gkr::prove_product_triple(leaves, &mut tr);
        let prove_ms = prove_started.elapsed().as_secs_f64() * 1000.0;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(claims.point.len(), log);
        let proof = tr.into_proof();
        black_box(&claims);
        if iteration > 0 {
            let root = proof.stream[0];
            let leaf = claims.values[0];
            println!(
                "upstream-radix4,192,3,{log},{size},{active},{threads},{},{prepare_ms:.6},{prove_ms:.6},{total_ms:.6},{:.6},{pairs},{pairs},{},{:016x}{:016x}{:016x},{:016x}{:016x}{:016x}",
                iteration - 1,
                total_ms / 3.0,
                proof.stream.len(),
                root.c2,
                root.c1,
                root.c0,
                leaf.c2,
                leaf.c1,
                leaf.c0
            );
        }
    }
}
