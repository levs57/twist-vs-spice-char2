//! Research-only production ablation. It never modifies a prover kernel.
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::time::Instant;

use lookup_argument::{ArgumentProof, Config, LookupTrace, Method, VirtualizationMode};
use om_memcheck::shout::ShoutFlatProof;
use om_transcript::Hasher;
use ram_payload::{GuestOptions, GuestProgram, GuestRows};
use serde_json::{Value, json};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn below(state: &mut u64, bound: usize) -> usize {
    let bound = bound as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let x = splitmix(state);
        if x >= threshold {
            return (x % bound) as usize;
        }
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn hash_bytes(bytes: &[u8]) -> String {
    hex(&Hasher::sha256().digest(bytes))
}
fn hash_words(words: &[u64]) -> String {
    hash_bytes(
        &words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn hash_addresses(addresses: &[u32]) -> String {
    hash_bytes(
        &addresses
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn trace_hash(trace: &LookupTrace) -> String {
    let mut bytes = b"lookup-bench/input/v1".to_vec();
    bytes.extend_from_slice(&(trace.table.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(trace.addresses.len() as u64).to_le_bytes());
    bytes.extend(trace.table.iter().flat_map(|v| v.to_le_bytes()));
    bytes.extend(trace.addresses.iter().flat_map(|v| v.to_le_bytes()));
    hash_bytes(&bytes)
}
fn chunk(address: u32, index: usize) -> u32 {
    (address >> (4 * index)) & 15
}
fn structure(addresses: &[u32]) -> Value {
    let cells: BTreeSet<u32> = addresses.iter().copied().collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut chunks = Vec::new();
    for i in 0..4 {
        let digits: BTreeSet<u32> = addresses.iter().map(|&a| chunk(a, i)).collect();
        let pair_equal = addresses
            .chunks_exact(2)
            .filter(|p| chunk(p[0], i) == chunk(p[1], i))
            .count();
        let adjacent_equal = addresses
            .windows(2)
            .filter(|p| chunk(p[0], i) == chunk(p[1], i))
            .count();
        chunks.push(
            json!({"chunk":i,"unique_digits":digits.len(),"digits":digits,
            "constant_digit":if digits.len()==1{digits.first().copied()}else{None},
            "dyadic_pair_equal_count":pair_equal,"dyadic_pair_count":addresses.len()/2,
            "dyadic_pair_equal_fraction":pair_equal as f64/(addresses.len()/2) as f64,
            "adjacent_equal_fraction":adjacent_equal as f64/(addresses.len()-1) as f64}),
        );
        let existing = groups
            .iter()
            .position(|group| addresses.iter().all(|&a| chunk(a, i) == chunk(a, group[0])));
        if let Some(group) = existing {
            groups[group].push(i)
        } else {
            groups.push(vec![i]);
        }
    }
    // This probe supplies one independent source with four declared chunks.
    // Identical contents remain a useful input statistic, but no longer decide
    // the production OneHot width: only explicit source/slot aliases may share.
    json!({"lookups":addresses.len(),"distinct_addresses":cells.len(),
        "min_address":cells.first(),"max_address":cells.last(),"chunks":chunks,
        "exact_equal_chunk_stream_groups":groups,"distinct_chunk_stream_contents":groups.len(),
        "onehot_sharing_policy":"structural-source-slot-only","unique_onehot_components":4,
        "onehot_field_columns":8,"address_stream_sha256":hash_addresses(addresses)})
}
fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
fn sample(config: &Config, trace: &LookupTrace, iteration: usize, position: usize) -> Value {
    let start = Instant::now();
    let output =
        lookup_argument::prove(Method::Shout, black_box(config), black_box(trace)).unwrap();
    let prove_ms = ms(start.elapsed());
    let start = Instant::now();
    lookup_argument::verify(black_box(&output.statement), black_box(&output.proof)).unwrap();
    let verify_ms = ms(start.elapsed());
    let mut stages = BTreeMap::<String, f64>::new();
    for &(name, duration) in &output.metrics {
        *stages.entry(name.into()).or_default() += ms(duration);
    }
    let virt_ms = stages
        .iter()
        .filter(|(k, _)| k.as_str() == "shout.virtualization" || k.starts_with("virt."))
        .map(|(_, v)| v)
        .sum::<f64>();
    let ArgumentProof::Shout(proof) = &output.proof.argument else {
        panic!("wrong argument")
    };
    let Some(ShoutFlatProof::SinglePoint(virtualization)) = &proof.virt_flat else {
        panic!("expected optimized single-point Flat")
    };
    let fields =
        |proof: &om_sumcheck::SumcheckProof| proof.round_msgs.iter().map(Vec::len).sum::<usize>();
    let summary = &output.summary;
    json!({"iteration":iteration,"case_position":position,"verified":true,"prove_ms":prove_ms,"verify_ms":verify_ms,
        "read_ms":stages.get("shout.read").copied().unwrap_or(0.0),
        "onehot_ms":stages.get("shout.onehot").copied().unwrap_or(0.0),"virtualization_ms":virt_ms,
        "main_commit_ms":ms(output.timings.main_commit),"main_open_ms":ms(output.timings.main_open),
        "exclusive_stage_ms":stages,"trace_sha256":trace_hash(trace),
        "input_proof_stats":{"core_bits":summary.core_bits,"argument_bits":summary.argument_bits,
            "main_committed_bits":summary.main_committed_bits,"main_claims":summary.main_claims,
            "input_address_claims":output.proof.input.address_bits.len(),"ra_evals":proof.ra_evals.len(),
            "read_rounds":proof.read.round_msgs.len(),"read_message_fields":fields(&proof.read),
            "onehot_rounds":proof.onehot.sc.round_msgs.len(),"onehot_message_fields":fields(&proof.onehot.sc),
            "onehot_original_output_components":proof.onehot.evals.len(),
            "virtualization_branch":"flat-single-point","virtualization_rounds":virtualization.sc.round_msgs.len(),
            "virtualization_message_fields":fields(&virtualization.sc),"virtualization_leaf_evals":virtualization.evals.len()}})
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let n: usize = arguments[1].parse().unwrap();
    let threads: usize = arguments[2].parse().unwrap();
    let warmup: usize = arguments[3].parse().unwrap();
    let repeat: usize = arguments[4].parse().unwrap();
    assert!(n >= 1 && n <= 24 && threads > 0 && repeat > 0 && repeat % 2 == 1);
    let (a, chunk_bits, seed) = (16usize, 4usize, 1u64);
    let options = GuestOptions {
        program: GuestProgram::PointerChase,
        rows: GuestRows::Memory,
        ..Default::default()
    };
    let payload = ram_payload::guest(n as u32, a as u32, &options).unwrap();
    let metadata = payload.metadata;
    let original = payload.reads.into_iter().next().unwrap();
    assert_eq!(original.len(), 1 << n);
    let working_set: Vec<u32> = original
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut rom_rng = seed ^ 0x70d158a06a42938e;
    let table: Vec<u64> = (0..1 << a).map(|_| splitmix(&mut rom_rng)).collect();
    let query_values: Vec<u64> = original
        .iter()
        .map(|&address| table[address as usize])
        .collect();
    let permutation_seed = seed ^ 0x72656c6162656c31;
    let mut permutation_rng = permutation_seed;
    let mut permutation: Vec<u32> = (0..1u32 << a).collect();
    for high in (1..permutation.len()).rev() {
        let other = below(&mut permutation_rng, high + 1);
        permutation.swap(high, other);
    }
    assert_eq!(
        permutation.iter().copied().collect::<BTreeSet<_>>().len(),
        1 << a
    );
    let mut relabeled_table = vec![0u64; table.len()];
    for (address, &permuted) in permutation.iter().enumerate() {
        relabeled_table[permuted as usize] = table[address];
    }
    let relabeled: Vec<u32> = original
        .iter()
        .map(|&address| permutation[address as usize])
        .collect();
    for (&address, &expected) in relabeled.iter().zip(&query_values) {
        assert_eq!(relabeled_table[address as usize], expected);
    }
    assert_eq!(
        relabeled.iter().copied().collect::<BTreeSet<_>>().len(),
        working_set.len()
    );
    let mut full_rng = seed ^ 0x66756c6c72616e64;
    let full_random: Vec<u32> = (0..1 << n)
        .map(|_| splitmix(&mut full_rng) as u32 & ((1 << a) - 1))
        .collect();
    let mut same_rng = seed ^ 0x73616d6573657431;
    let same_set_random: Vec<u32> = (0..1 << n)
        .map(|_| working_set[below(&mut same_rng, working_set.len())])
        .collect();
    assert_eq!(
        same_set_random
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>(),
        working_set
    );
    let traces = vec![
        LookupTrace {
            table: table.clone(),
            addresses: original,
            read_values: None,
        },
        LookupTrace {
            table: relabeled_table,
            addresses: relabeled,
            read_values: None,
        },
        LookupTrace {
            table: table.clone(),
            addresses: full_random,
            read_values: None,
        },
        LookupTrace {
            table,
            addresses: same_set_random,
            read_values: None,
        },
    ];
    let names = [
        "pointer-compact",
        "pointer-relabeled",
        "random-full-domain",
        "random-same-working-set",
    ];
    let stats: Vec<Value> = traces
        .iter()
        .map(|trace| structure(&trace.addresses))
        .collect();
    let digests: Vec<String> = traces.iter().map(trace_hash).collect();
    let config = Config {
        address_bits: a,
        chunk_bits,
        virtualization: VirtualizationMode::Flat,
        ..Default::default()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let mut samples: Vec<Vec<Value>> = (0..traces.len()).map(|_| Vec::new()).collect();
    let mut order = Vec::new();
    pool.install(|| {
        for iteration in 0..warmup {
            for (i, trace) in traces.iter().enumerate() {
                sample(&config, trace, iteration, i);
            }
        }
        for iteration in 0..repeat {
            for position in 0..traces.len() {
                let i = (position + iteration) % traces.len();
                let measured = sample(&config, &traces[i], iteration, position);
                assert_eq!(measured["trace_sha256"].as_str(), Some(digests[i].as_str()));
                println!(
                    "{} iteration={} prove={:.3}ms read={:.3}ms onehot={:.3}ms virt={:.3}ms",
                    names[i],
                    iteration,
                    measured["prove_ms"].as_f64().unwrap(),
                    measured["read_ms"].as_f64().unwrap(),
                    measured["onehot_ms"].as_f64().unwrap(),
                    measured["virtualization_ms"].as_f64().unwrap()
                );
                samples[i].push(measured);
                order.push(json!({"iteration":iteration,"position":position,"case":names[i]}));
            }
        }
    });
    let keys = [
        "prove_ms",
        "verify_ms",
        "read_ms",
        "onehot_ms",
        "virtualization_ms",
        "main_commit_ms",
        "main_open_ms",
    ];
    let cases:Vec<Value>=traces.iter().enumerate().map(|(i,trace)|{
        let medians:BTreeMap<&str,f64>=keys.iter().map(|&key|(key,median(samples[i].iter().map(|s|s[key].as_f64().unwrap()).collect()))).collect();
        json!({"name":names[i],"trace_sha256":trace_hash(trace),"query_values_sha256":hash_words(&trace.addresses.iter().map(|&a|trace.table[a as usize]).collect::<Vec<_>>()),
            "structure":stats[i],"samples":samples[i],"median":medians})
    }).collect();
    assert_eq!(
        cases[0]["query_values_sha256"],
        cases[1]["query_values_sha256"]
    );
    let report = json!({"schema":"shout-structure-ablation/v1","research_only":true,
        "config":{"method":"shout","virtualization":"flat-single-point","address_bits":a,"chunk_bits":chunk_bits,"d":a/chunk_bits,
            "log_lookups":n,"lookups":1usize<<n,"threads":threads,"warmup":warmup,"repeat":repeat,"profile":config.profile,"log_rate":config.log_rate,"seed":seed},
        "environment":{"compiled_arch":std::env::consts::ARCH,"compiled_pclmulqdq":cfg!(target_feature="pclmulqdq"),
            "compiled_avx2":cfg!(target_feature="avx2"),"compiled_vpclmulqdq":cfg!(target_feature="vpclmulqdq"),
            "compiled_sha":cfg!(target_feature="sha"),"executable_sha256":hash_bytes(&std::fs::read(std::env::current_exe().unwrap()).unwrap())},
        "source_metadata":metadata,"provenance":{"production_source_sha256":std::env::var("SHOUT_PROBE_SOURCE_SHA256").unwrap()},
        "relabel":{"algorithm":"full-domain Fisher-Yates with rejection-sampled SplitMix64 indices","seed":permutation_seed,
            "domain_size":1usize<<a,"permutation_sha256":hash_addresses(&permutation),"query_value_assertions":1usize<<n,
            "original_and_relabeled_distinct_addresses":working_set.len(),
            "working_set_mapping":working_set.iter().map(|&a|json!([a,permutation[a as usize]])).collect::<Vec<_>>(),
            "preserves":"Every query value and the entire address equality/reuse-distance pattern; numeric geometry changes"},
        "verified_proofs":traces.len()*(warmup+repeat),"measured_case_order":order,"cases":cases,
        "notes":["OneHot retains all four declared chunks; exact equal-content groups are input statistics only. Local Flat constant-factor pruning remains enabled.",
            "Payload generation, permutation, structural statistics and JSON work are outside prover timing.",
            "Every warmup/measured proof is independently verified. Prove duration excludes verification and returned-proof destruction.",
            "Rotating variant order limits fixed-order bias; five samples are descriptive, not confidence intervals.",
            "The relabel changes bit/chunk structure and numeric geometry simultaneously; it does not isolate pure cache effects.",
            "Input proof statistics count IOP fields only where named, not an encoded whole-proof byte length."]});
    std::fs::write(
        &arguments[5],
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}
