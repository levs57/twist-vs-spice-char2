//! Development timing for Twist v2 against the upstream Twist on one
//! workload: `twist2_time <log_cycles> <address_bits> <random|erc20|pointer>
//! <repeats> <methods,comma,separated> [chunk_bits]`.
//! Methods: `legacy`, `cycle`, `address`, `address-nomerge`, `cycle-private`.

use std::time::Instant;

use ram_argument::twist2::{self, ReadOrder};
use ram_argument::{AccessTrace, Config};
use ram_payload::operations::{guest_operations, random_operations};
use ram_payload::{GuestFill, GuestOptions, GuestProgram, GuestRows};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args[1].parse().unwrap();
    let a: u32 = args[2].parse().unwrap();
    let workload = args[3].as_str();
    let repeats: usize = args[4].parse().unwrap();
    let methods: Vec<&str> = args[5].split(',').collect();
    let chunk: usize = args.get(6).map_or(4, |s| s.parse().unwrap());
    let ops = match workload {
        "random" => random_operations(n, a, 1).unwrap(),
        g => {
            let program = match g {
                "erc20" => GuestProgram::Erc20MixedBlock,
                "pointer" => GuestProgram::PointerChase,
                _ => GuestProgram::BinarySearch,
            };
            let options = GuestOptions {
                program,
                rows: GuestRows::Memory,
                size: 64,
                operations: 256,
                keys: 64,
                queries: 16,
                seed: 1,
                max_cycles: 1 << 20,
                fill: GuestFill::Repeat,
            };
            guest_operations(n, a, &options).unwrap()
        }
    };
    let rmw = ops.into_rmw().unwrap();
    let trace = AccessTrace {
        initial: rmw.initial_memory,
        reads: rmw.reads,
        writes: rmw.writes,
        new_values: rmw.new_values,
        read_values: Some(rmw.read_values),
    };
    let legacy = Config {
        address_bits: a as usize,
        chunk_bits: chunk,
        ..Config::default()
    };
    for rep in 0..repeats {
        for &m in &methods {
            let t0 = Instant::now();
            let (metrics, verify_ms, claims, bits) = if m == "legacy" {
                let r = ram_argument::prove(&legacy, &trace).unwrap();
                let p = t0.elapsed();
                let t1 = Instant::now();
                ram_argument::verify(&r.statement, &r.proof).unwrap();
                let v = t1.elapsed().as_secs_f64() * 1e3;
                let _ = p;
                (
                    r.metrics,
                    v,
                    r.summary.mosaic_claims,
                    r.summary.committed_bits,
                )
            } else {
                let mut cfg = twist2::Config::from_legacy(&legacy);
                match m {
                    "cycle" => cfg.order = ReadOrder::CycleFirst,
                    "address" => cfg.order = ReadOrder::AddressFirst,
                    "address-nomerge" => {
                        cfg.order = ReadOrder::AddressFirst;
                        cfg.merge_loads = false;
                    }
                    "cycle-private" => cfg.public_final = false,
                    _ => panic!("unknown method {m}"),
                }
                let r = twist2::prove(&cfg, &trace).unwrap();
                let t1 = Instant::now();
                om_transcript::profiling::clear();
                twist2::verify(&r.statement, &r.proof).unwrap();
                let v = t1.elapsed().as_secs_f64() * 1e3;
                let mut metrics = r.metrics;
                metrics.extend(om_transcript::profiling::take());
                (
                    metrics,
                    v,
                    r.summary.mosaic_claims,
                    r.summary.committed_bits,
                )
            };
            let prove_ms = (t0.elapsed().as_secs_f64() * 1e3) - verify_ms;
            let spans: Vec<String> = metrics
                .iter()
                .filter(|(_, d)| d.as_secs_f64() > 5e-4)
                .map(|(s, d)| format!("{s}={:.1}", d.as_secs_f64() * 1e3))
                .collect();
            println!(
                "rep={rep} {workload} {m}: prove={prove_ms:.1}ms verify={verify_ms:.2}ms claims={claims} bits={bits} | {}",
                spans.join(" ")
            );
        }
    }
}
