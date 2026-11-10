//! Reproducible port-only benchmark; tuple construction and PCS are excluded.

use std::hint::black_box;
use std::time::Instant;

use lean_grandproduct::{Leaves, prove, verify};
use om_field::K;
use om_transcript::Transcript;

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
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let size = 1usize << log;
    let half = size / 2;
    let boundary = (half / 16).max(1);
    let ranges = if padding {
        vec![half + boundary..size]
    } else {
        Vec::new()
    };
    let input: Vec<K> = (0..size)
        .map(|i| {
            if ranges.iter().any(|r| r.contains(&i)) {
                K::ONE
            } else {
                K::from_u128(
                    u128::from(random(i as u64)) | (u128::from(random(i as u64 ^ 0x1234)) << 64),
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
        let values = [input.clone(), input.clone()];
        let started = Instant::now();
        let leaves = values.map(|v| Leaves::with_identity_ranges(v, ranges.clone()).unwrap());
        let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut tr = Transcript::new_default(b"grandproduct-microbenchmark");
        let prove_started = Instant::now();
        let (proof, claims) = prove(log, leaves, &mut tr, b"gp").unwrap();
        let prove_ms = prove_started.elapsed().as_secs_f64() * 1000.0;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(proof.roots[0], proof.roots[1]);
        let mut verifier = Transcript::new_default(b"grandproduct-microbenchmark");
        assert_eq!(verify(log, &proof, &mut verifier, b"gp").unwrap(), claims);
        assert_eq!(tr.challenge_k(b"after"), verifier.challenge_k(b"after"));
        let fields = 2 + proof
            .layers
            .iter()
            .map(|l| l.rounds.len() * 4 + l.children.iter().map(Vec::len).sum::<usize>())
            .sum::<usize>();
        black_box(&claims);
        if iteration > 0 {
            println!(
                "ported-radix4,128,2,{log},{size},{active},{threads},{},{prepare_ms:.6},{prove_ms:.6},{total_ms:.6},{:.6},{pairs},{pairs},{fields},{:032x},{:032x}",
                iteration - 1,
                total_ms / 2.0,
                proof.roots[0].value(),
                claims.values[0].value()
            );
        }
    }
}
