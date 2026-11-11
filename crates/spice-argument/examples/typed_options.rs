//! Interleaved prove and verify times for typed Spice option sets on a
//! uniform random workload: `cargo run --release --example typed_options --
//! [log_cycles] [address_bits] [repeat]`.

use spice_argument::{
    Config,
    operations::{self, AccessTrace, Options},
};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let n = args.first().copied().unwrap_or(20);
    let a = args.get(1).copied().unwrap_or(16);
    let repeat = args.get(2).copied().unwrap_or(5);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let initial: Vec<u64> = (0..1usize << a).map(|_| next()).collect();
    let mut memory = initial.clone();
    let (mut addresses, mut values, mut is_write) = (vec![], vec![], vec![]);
    for _ in 0..1usize << n {
        let k = (next() as usize) & ((1 << a) - 1);
        let write = next() & 1 == 1;
        if write {
            memory[k] = next();
        }
        addresses.push(k as u32);
        values.push(memory[k]);
        is_write.push(write);
    }
    let trace = AccessTrace {
        initial,
        addresses,
        values,
        is_write,
    };
    let config = Config {
        address_bits: a,
        chunk_bits: 1,
        ..Config::default()
    };
    let sets = [
        ("public", Options::default()),
        ("two-fingerprints", Options::TWO_FINGERPRINTS),
        (
            "private-final",
            Options {
                private_final_memory: true,
                ..Options::default()
            },
        ),
    ];
    let mut prove = vec![vec![]; sets.len()];
    let mut verify = vec![vec![]; sets.len()];
    let mut sizes = vec![operations::ProofSize::default(); sets.len()];
    for round in 0..=repeat {
        for (i, (_, options)) in sets.iter().enumerate() {
            let start = Instant::now();
            let r = operations::prove_with(&config, &trace, options).unwrap();
            let p = start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            operations::verify_with(&r.statement, &r.proof, options).unwrap();
            let v = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(r.statement.final_memory, memory);
            sizes[i] = operations::proof_size(&r.proof);
            if round > 0 {
                prove[i].push(p);
                verify[i].push(v);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for (i, (name, _)) in sets.iter().enumerate() {
        let z = sizes[i];
        println!(
            "{name}: prove {:.1} ms, verify {:.2} ms (median of {repeat}, n={n}, a={a}); \
             proof {} B = pcs {} + mosaic {} + iop {}",
            median(&mut prove[i]),
            median(&mut verify[i]),
            z.total(),
            z.pcs_bytes,
            z.mosaic_bytes,
            z.iop_bytes
        );
    }
}
