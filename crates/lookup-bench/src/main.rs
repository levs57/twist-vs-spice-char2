use lookup_argument::{Config, LookupTrace, Method, Summary, VirtualizationMode};
use om_transcript::Hasher;
use ram_payload::{GuestFill, GuestOptions, GuestProgram, GuestRows};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

const HELP: &str = "Indexed read-only GF128 lookup benchmark: Shout versus weighted Logup*

  --methods shout,logup|both       Ordered methods (default both)
  --patterns random,local,guest|both
                                  Default both means random and local
  --table-bits A                   Public ROM has 2^A u64 words (default 12)
  --address-bits A                 Alias for --table-bits
  --log-lookups N                  Exactly 2^N reads (default 12)
  --log-cycles N                   Alias for --log-lookups
  --chunk-bits C[,C...]            Shout address chunk widths (default 4)
  --virtualization flat|legacy-flat|gkr|both
                                  Shout only; both means flat,gkr (default flat)
  --threads N                     Rayon workers (default 4)
  --warmup N                      Verified warmup proofs/case (default 1)
  --repeat N                      Verified measured proofs/case (default 3)
  --seed N                        Deterministic ROM/pattern seed (default 1)
  --profile provable-120|conjectured-100
                                  PCS policy name, not composed security (default provable-120)
  --log-rate N                    Log2 inverse code rate (default 2)
  --guest-program NAME[,NAME...]   binary-search, erc20-mixed-block, pointer-chase,
                                  or all (default binary-search; guest pattern only)
  --guest-rows memory|all|both      Guest address rows (default memory)
  --guest-size N                   ERC20 accounts/pointer nodes (default 64)
  --guest-operations N             Transfers/pointer steps (default 256)
  --guest-keys N                   Binary-search keys (default 64)
  --guest-queries N                Binary-search queries (default 16)
  --output PATH                   Complete JSON samples and comparisons
  --csv PATH                      One CSV row/proof, including exclusive stages
  --help                          Show this help

All warmup and measured proofs are independently verified. Prover time includes
preparation, statement binding, commitments, reductions, Mosaic, RingSwitch and
Ligerito; payload generation and result destruction are outside it. Commit/open
durations are inclusive; exclusive stages are separate diagnostics. Logup* runs
once per pattern regardless of Shout sweeps. Guest patterns reuse only read-port0
addresses with fresh immutable ROM values, discarding guest RAM values and stores.
Reports include the conservative extra term (T+N)/2^128, excluding other errors;
the inherited provable-120 profile is not a composed 120-bit security claim.
";

#[derive(Clone, Debug, Serialize)]
struct Options {
    methods: Vec<String>,
    patterns: Vec<String>,
    table_bits: usize,
    log_lookups: usize,
    chunk_bits: Vec<usize>,
    virtualization: Vec<String>,
    threads: usize,
    warmup: usize,
    repeat: usize,
    seed: u64,
    profile: String,
    log_rate: usize,
    guest_program: Vec<String>,
    guest_rows: Vec<String>,
    guest_size: usize,
    guest_operations: usize,
    guest_keys: usize,
    guest_queries: usize,
    output: Option<PathBuf>,
    csv: Option<PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            methods: vec!["shout".into(), "logup".into()],
            patterns: vec!["random".into(), "local".into()],
            table_bits: 12,
            log_lookups: 12,
            chunk_bits: vec![4],
            virtualization: vec!["flat".into()],
            threads: 4,
            warmup: 1,
            repeat: 3,
            seed: 1,
            profile: "provable-120".into(),
            log_rate: 2,
            guest_program: vec!["binary-search".into()],
            guest_rows: vec!["memory".into()],
            guest_size: 64,
            guest_operations: 256,
            guest_keys: 64,
            guest_queries: 16,
            output: None,
            csv: None,
        }
    }
}
fn selection(
    value: &str,
    allowed: &[&str],
    expansion: Option<(&str, &[&str])>,
) -> Result<Vec<String>, String> {
    let input = match expansion {
        Some((word, values)) if value == word => values.to_vec(),
        _ => value.split(',').collect(),
    };
    let mut result = Vec::new();
    for value in input {
        if !allowed.contains(&value) {
            return Err(format!("unsupported selection: {value}"));
        }
        if result.iter().any(|old| old == value) {
            return Err(format!("duplicate selection: {value}"));
        }
        result.push(value.to_owned());
    }
    Ok(result)
}
fn options() -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            print!("{HELP}");
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{arg} requires a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid number for {arg}: {value}"))
        };
        match arg.as_str() {
            "--methods" => {
                opts.methods = selection(
                    &value,
                    &["shout", "logup"],
                    Some(("both", &["shout", "logup"])),
                )?
            }
            "--patterns" | "--pattern" => {
                opts.patterns = selection(
                    &value,
                    &["random", "local", "guest"],
                    Some(("both", &["random", "local"])),
                )?
            }
            "--table-bits" | "--address-bits" => opts.table_bits = number()?,
            "--log-lookups" | "--log-cycles" => opts.log_lookups = number()?,
            "--chunk-bits" => {
                opts.chunk_bits = value
                    .split(',')
                    .map(|n| n.parse().map_err(|_| format!("invalid chunk size: {n}")))
                    .collect::<Result<_, _>>()?
            }
            "--virtualization" => {
                opts.virtualization = selection(
                    &value,
                    &["flat", "legacy-flat", "gkr"],
                    Some(("both", &["flat", "gkr"])),
                )?
            }
            "--threads" => opts.threads = number()?,
            "--warmup" => opts.warmup = number()?,
            "--repeat" => opts.repeat = number()?,
            "--seed" => {
                opts.seed = value
                    .parse()
                    .map_err(|_| format!("invalid seed: {value}"))?
            }
            "--profile" => opts.profile = value,
            "--log-rate" => opts.log_rate = number()?,
            "--guest-program" => {
                opts.guest_program = selection(
                    &value,
                    &["binary-search", "erc20-mixed-block", "pointer-chase"],
                    Some((
                        "all",
                        &["binary-search", "erc20-mixed-block", "pointer-chase"],
                    )),
                )?
            }
            "--guest-rows" => {
                opts.guest_rows = selection(
                    &value,
                    &["memory", "all"],
                    Some(("both", &["all", "memory"])),
                )?
            }
            "--guest-size" => opts.guest_size = number()?,
            "--guest-operations" => opts.guest_operations = number()?,
            "--guest-keys" => opts.guest_keys = number()?,
            "--guest-queries" => opts.guest_queries = number()?,
            "--output" => opts.output = Some(value.into()),
            "--csv" => opts.csv = Some(value.into()),
            _ => return Err(format!("unknown option {arg}; use --help")),
        }
    }
    if !(1..=31).contains(&opts.table_bits) || opts.log_lookups > 31 {
        return Err("table bits must be 1..31; log lookups must be 0..31".into());
    }
    if opts.threads == 0 || opts.repeat == 0 {
        return Err("threads and repeat must be positive".into());
    }
    if !["provable-120", "conjectured-100"].contains(&opts.profile.as_str()) {
        return Err("unsupported PCS profile".into());
    }
    let minimum = if opts.profile == "provable-120" { 2 } else { 1 };
    if !(minimum..=8).contains(&opts.log_rate) {
        return Err("invalid log rate for PCS profile".into());
    }
    if opts.methods.iter().any(|m| m == "shout") {
        for (i, &c) in opts.chunk_bits.iter().enumerate() {
            if c == 0 || c > opts.table_bits || !opts.table_bits.is_multiple_of(c) {
                return Err("Shout chunk widths must divide table bits".into());
            }
            if opts.chunk_bits[..i].contains(&c) {
                return Err("duplicate Shout chunk width".into());
            }
            if opts.virtualization.iter().any(|v| v == "gkr")
                && !(opts.table_bits / c).is_power_of_two()
            {
                return Err("GKR requires a power-of-two Shout chunk count".into());
            }
        }
    }
    if opts.patterns.iter().any(|p| p == "guest") && opts.log_lookups < 2 {
        return Err("guest patterns require at least 4 lookups".into());
    }
    Ok(Some(opts))
}
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn fingerprint(table: &[u64], addresses: &[u32]) -> String {
    let lengths = [table.len() as u64, addresses.len() as u64]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let values = table
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let indices = addresses
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    hex(&Hasher::sha256().digest_slices(&[
        b"lookup-bench/input/v1".as_slice(),
        &lengths,
        &values,
        &indices,
    ]))
}
struct Payload {
    name: String,
    trace: LookupTrace,
    digest: String,
    metadata: serde_json::Value,
}
fn payloads(opts: &Options) -> Result<Vec<Payload>, Box<dyn std::error::Error>> {
    let n = 1usize << opts.table_bits;
    let t = 1usize << opts.log_lookups;
    let mut seed = opts.seed ^ 0x70d1_58a0_6a42_938e;
    let table: Vec<u64> = (0..n).map(|_| splitmix(&mut seed)).collect();
    let mut result = Vec::new();
    let mut add = |name: String, addresses: Vec<u32>, metadata: serde_json::Value| {
        let digest = fingerprint(&table, &addresses);
        result.push(Payload {
            name,
            trace: LookupTrace {
                table: table.clone(),
                addresses,
                read_values: None,
            },
            digest,
            metadata,
        });
    };
    for pattern in &opts.patterns {
        if pattern == "guest" {
            for name in &opts.guest_program {
                let program = match name.as_str() {
                    "binary-search" => GuestProgram::BinarySearch,
                    "erc20-mixed-block" => GuestProgram::Erc20MixedBlock,
                    "pointer-chase" => GuestProgram::PointerChase,
                    _ => unreachable!(),
                };
                for rows in &opts.guest_rows {
                    let options = GuestOptions {
                        program,
                        rows: if rows == "memory" {
                            GuestRows::Memory
                        } else {
                            GuestRows::All
                        },
                        size: opts.guest_size,
                        operations: opts.guest_operations,
                        keys: opts.guest_keys,
                        queries: opts.guest_queries,
                        seed: opts.seed,
                        fill: GuestFill::Repeat,
                        ..Default::default()
                    };
                    let payload = ram_payload::guest(
                        opts.log_lookups as u32,
                        opts.table_bits as u32,
                        &options,
                    )?;
                    let metadata = serde_json::json!({"source_ram_pattern":payload.metadata,"use":"Only read-port0 addresses; fresh immutable ROM; RAM values and writes discarded"});
                    let addresses = payload
                        .reads
                        .into_iter()
                        .next()
                        .ok_or("guest trace has no read address stream")?;
                    add(format!("guest/{name}/{rows}"), addresses, metadata);
                }
            }
        } else {
            let mut seed = opts.seed
                ^ if pattern == "random" {
                    0xa6b5_01c3_a312_047f
                } else {
                    0x6d9c_77e5_f2b3_1809
                };
            let addresses = (0..t)
                .map(|j| {
                    let domain = if pattern == "local" && j % 16 != 15 {
                        n.min(64)
                    } else {
                        n
                    };
                    (splitmix(&mut seed) as usize & (domain - 1)) as u32
                })
                .collect();
            add(
                pattern.clone(),
                addresses,
                serde_json::json!({"distribution":if pattern=="random"{"Uniform full ROM"}else{"15/16 first min(N,64) rows; 1/16 uniform full ROM"},"seed":opts.seed}),
            );
        }
    }
    Ok(result)
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct Areas {
    lookups: usize,
    table_rows: usize,
    core_bits: usize,
    argument_bits: usize,
    main_committed_bits: usize,
    main_padding_bits: usize,
    auxiliary_bits: usize,
    auxiliary_committed_bits: usize,
    auxiliary_padding_bits: usize,
    total_committed_bits: usize,
    main_claims: usize,
    auxiliary_claims: usize,
}
impl Areas {
    fn new(s: &Summary) -> Result<Self, String> {
        Ok(Self {
            lookups: s.lookups,
            table_rows: s.table_rows,
            core_bits: s.core_bits,
            argument_bits: s.argument_bits,
            main_committed_bits: s.main_committed_bits,
            main_padding_bits: s
                .main_committed_bits
                .checked_sub(s.core_bits + s.argument_bits)
                .ok_or("main commitment smaller than used area")?,
            auxiliary_bits: s.auxiliary_bits,
            auxiliary_committed_bits: s.auxiliary_committed_bits,
            auxiliary_padding_bits: s
                .auxiliary_committed_bits
                .checked_sub(s.auxiliary_bits)
                .ok_or("auxiliary commitment smaller than used area")?,
            total_committed_bits: s.main_committed_bits + s.auxiliary_committed_bits,
            main_claims: s.main_claims,
            auxiliary_claims: s.auxiliary_claims,
        })
    }
}
#[derive(Debug, Serialize)]
struct Sample {
    iteration: usize,
    prove_ms: f64,
    verify_ms: f64,
    main_commit_ms: f64,
    auxiliary_commit_ms: f64,
    main_open_ms: f64,
    auxiliary_open_ms: f64,
    commit_ms: f64,
    opening_ms: f64,
    remaining_ms: f64,
    prep_ms: f64,
    argument_ms: f64,
    virtualization_ms: f64,
    exclusive_stage_ms: BTreeMap<String, f64>,
    exclusive_stages_total_ms: f64,
    unattributed_prove_ms: f64,
}
#[derive(Debug, Default, PartialEq)]
struct PhaseTotals {
    prep_ms: f64,
    argument_ms: f64,
    virtualization_ms: f64,
}
impl PhaseTotals {
    fn new(stages: &BTreeMap<String, f64>) -> Self {
        let mut totals = Self::default();
        for (name, &duration) in stages {
            if matches!(
                name.as_str(),
                "lookup.witness" | "lookup.pack" | "lookup.statement" | "lookup.claims"
            ) {
                totals.prep_ms += duration;
            }
            if name.starts_with("shout.")
                || name.starts_with("virt.")
                || name.starts_with("lookup.logup")
            {
                totals.argument_ms += duration;
            }
            if name == "shout.virtualization" || name.starts_with("virt.") {
                totals.virtualization_ms += duration;
            }
        }
        totals
    }
}
fn run(
    method: Method,
    config: &Config,
    trace: &LookupTrace,
    iteration: usize,
) -> Result<(Sample, Areas), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let result = lookup_argument::prove(method, black_box(config), black_box(trace))?;
    let prove_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    lookup_argument::verify(black_box(&result.statement), black_box(&result.proof))?;
    let verify_ms = started.elapsed().as_secs_f64() * 1000.0;
    let areas = Areas::new(&result.summary)?;
    let ms = |v: std::time::Duration| v.as_secs_f64() * 1000.0;
    let main_commit_ms = ms(result.timings.main_commit);
    let auxiliary_commit_ms = ms(result.timings.auxiliary_commit);
    let main_open_ms = ms(result.timings.main_open);
    let auxiliary_open_ms = ms(result.timings.auxiliary_open);
    let commit_ms = main_commit_ms + auxiliary_commit_ms;
    let opening_ms = main_open_ms + auxiliary_open_ms;
    let remaining_ms = prove_ms - commit_ms - opening_ms;
    if remaining_ms < -0.000_001 {
        return Err("inclusive commit/open intervals exceed prover time".into());
    }
    let mut exclusive_stage_ms = BTreeMap::new();
    for (name, duration) in result.metrics {
        *exclusive_stage_ms.entry(name.to_owned()).or_insert(0.0) += ms(duration);
    }
    let exclusive_stages_total_ms = exclusive_stage_ms.values().sum();
    let phases = PhaseTotals::new(&exclusive_stage_ms);
    Ok((
        Sample {
            iteration,
            prove_ms,
            verify_ms,
            main_commit_ms,
            auxiliary_commit_ms,
            main_open_ms,
            auxiliary_open_ms,
            commit_ms,
            opening_ms,
            remaining_ms,
            prep_ms: phases.prep_ms,
            argument_ms: phases.argument_ms,
            virtualization_ms: phases.virtualization_ms,
            exclusive_stage_ms,
            exclusive_stages_total_ms,
            unattributed_prove_ms: prove_ms - exclusive_stages_total_ms,
        },
        areas,
    ))
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}
#[derive(Debug, Serialize)]
struct Medians {
    prove_ms: f64,
    verify_ms: f64,
    main_commit_ms: f64,
    auxiliary_commit_ms: f64,
    main_open_ms: f64,
    auxiliary_open_ms: f64,
    commit_ms: f64,
    opening_ms: f64,
    remaining_ms: f64,
    prep_ms: f64,
    argument_ms: f64,
    virtualization_ms: f64,
    unattributed_prove_ms: f64,
    exclusive_stage_ms: BTreeMap<String, f64>,
}
impl Medians {
    fn new(samples: &[Sample]) -> Self {
        let field = |get: fn(&Sample) -> f64| median(samples.iter().map(get).collect());
        let mut stages = BTreeMap::new();
        for s in samples {
            for key in s.exclusive_stage_ms.keys() {
                stages.insert(key.clone(), 0.0);
            }
        }
        for (key, value) in &mut stages {
            *value = median(
                samples
                    .iter()
                    .map(|s| s.exclusive_stage_ms.get(key).copied().unwrap_or(0.0))
                    .collect(),
            );
        }
        Self {
            prove_ms: field(|s| s.prove_ms),
            verify_ms: field(|s| s.verify_ms),
            main_commit_ms: field(|s| s.main_commit_ms),
            auxiliary_commit_ms: field(|s| s.auxiliary_commit_ms),
            main_open_ms: field(|s| s.main_open_ms),
            auxiliary_open_ms: field(|s| s.auxiliary_open_ms),
            commit_ms: field(|s| s.commit_ms),
            opening_ms: field(|s| s.opening_ms),
            remaining_ms: field(|s| s.remaining_ms),
            prep_ms: field(|s| s.prep_ms),
            argument_ms: field(|s| s.argument_ms),
            virtualization_ms: field(|s| s.virtualization_ms),
            unattributed_prove_ms: field(|s| s.unattributed_prove_ms),
            exclusive_stage_ms: stages,
        }
    }
}
#[derive(Debug, Serialize)]
struct LookupBound {
    conservative_degree_numerator: usize,
    field_bits: usize,
    probability_bound: f64,
    negative_log2_bound: f64,
    scope: String,
}
fn logup_bound(t: usize, n: usize) -> LookupBound {
    let degree = t + n;
    LookupBound{conservative_degree_numerator:degree,field_bits:128,probability_bound:(degree as f64)*2f64.powi(-128),negative_log2_bound:128.0-(degree as f64).log2(),scope:"Conservative extra rational-identity term only, conditional on a nonzero defect; excludes poles, batching, other sumchecks and PCS. Valid indexed addresses can give a tighter N-1 rational degree.".into()}
}
#[derive(Debug, Serialize)]
struct Case {
    method: String,
    pattern: String,
    chunk_bits: Option<usize>,
    virtualization: Option<String>,
    trace_sha256: String,
    source_metadata: serde_json::Value,
    areas: Areas,
    logup_extra_bound: Option<LookupBound>,
    samples: Vec<Sample>,
    median: Medians,
}
fn mode(name: &str) -> VirtualizationMode {
    match name {
        "flat" => VirtualizationMode::Flat,
        "legacy-flat" => VirtualizationMode::FlatLegacy,
        "gkr" => VirtualizationMode::GkrTree,
        _ => unreachable!(),
    }
}
fn configurations(opts: &Options) -> Vec<(Method, Option<usize>, Option<String>)> {
    let mut result = Vec::new();
    for method in &opts.methods {
        if method == "logup" {
            result.push((Method::LogupStar, None, None));
        } else {
            for &c in &opts.chunk_bits {
                for v in &opts.virtualization {
                    result.push((Method::Shout, Some(c), Some(v.clone())));
                }
            }
        }
    }
    result
}
fn benchmark(
    opts: &Options,
    payloads: &[Payload],
) -> Result<Vec<Case>, Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for payload in payloads {
        for (method, chunk, virtualization) in configurations(opts) {
            let config = Config {
                address_bits: opts.table_bits,
                chunk_bits: chunk.unwrap_or(0),
                virtualization: virtualization
                    .as_deref()
                    .map(mode)
                    .unwrap_or(VirtualizationMode::Flat),
                profile: opts.profile.clone(),
                log_rate: opts.log_rate,
            };
            let mut expected = None;
            for i in 0..opts.warmup {
                let (_, area) = run(method, &config, &payload.trace, i)?;
                if let Some(old) = &expected {
                    if old != &area {
                        return Err("warmup area changed".into());
                    }
                }
                expected = Some(area);
            }
            let mut samples = Vec::with_capacity(opts.repeat);
            for i in 0..opts.repeat {
                let (sample, area) = run(method, &config, &payload.trace, i)?;
                if let Some(old) = &expected {
                    if old != &area {
                        return Err("measured area changed".into());
                    }
                }
                expected = Some(area);
                samples.push(sample);
            }
            let median = Medians::new(&samples);
            let areas = expected.ok_or("no measured proofs")?;
            println!(
                "pattern={} method={} chunk={} virtualization={} prove={:.3}ms verify={:.3}ms commit={:.3}ms open={:.3}ms iop+prep={:.3}ms main_bits={} aux_bits={}",
                payload.name,
                method.name(),
                chunk.map_or_else(|| "-".into(), |c| c.to_string()),
                virtualization.as_deref().unwrap_or("-"),
                median.prove_ms,
                median.verify_ms,
                median.commit_ms,
                median.opening_ms,
                median.remaining_ms,
                areas.main_committed_bits,
                areas.auxiliary_committed_bits
            );
            println!(
                "  phase medians: preparation={:.3}ms argument={:.3}ms virtualization(total)={:.3}ms read={:.3}ms onehot={:.3}ms unattributed={:.3}ms",
                median.prep_ms,
                median.argument_ms,
                median.virtualization_ms,
                median
                    .exclusive_stage_ms
                    .get("shout.read")
                    .copied()
                    .unwrap_or(0.0),
                median
                    .exclusive_stage_ms
                    .get("shout.onehot")
                    .copied()
                    .unwrap_or(0.0),
                median.unattributed_prove_ms,
            );
            println!("  exclusive stage medians:");
            for (name, duration) in &median.exclusive_stage_ms {
                println!("    {name:<36} {duration:>10.3}ms");
            }
            println!(
                "  Subtotals are computed per proof, then medianed; these overlapping diagnostics and independently medianed rows are not additive. iop+prep includes unattributed preparation/core overhead."
            );
            cases.push(Case {
                method: method.name().into(),
                pattern: payload.name.clone(),
                chunk_bits: chunk,
                virtualization,
                trace_sha256: payload.digest.clone(),
                source_metadata: payload.metadata.clone(),
                logup_extra_bound: if method == Method::LogupStar {
                    Some(logup_bound(areas.lookups, areas.table_rows))
                } else {
                    None
                },
                areas,
                samples,
                median,
            });
        }
    }
    Ok(cases)
}
#[derive(Debug, Serialize)]
struct Comparison {
    shout_case: usize,
    logup_case: usize,
    trace_sha256: String,
    logup_prover_speedup: f64,
    logup_verifier_speedup: f64,
    shout_to_logup_commit_ratio: f64,
    shout_to_logup_opening_ratio: f64,
    shout_to_logup_committed_area_ratio: f64,
}
fn comparisons(cases: &[Case]) -> Vec<Comparison> {
    let mut result = Vec::new();
    for (s, shout) in cases
        .iter()
        .enumerate()
        .filter(|(_, c)| c.method == "shout")
    {
        if let Some((l, logup)) = cases.iter().enumerate().find(|(_, c)| {
            c.method == "logup-star"
                && c.trace_sha256 == shout.trace_sha256
                && c.pattern == shout.pattern
        }) {
            assert_eq!(shout.areas.core_bits, logup.areas.core_bits);
            result.push(Comparison {
                shout_case: s,
                logup_case: l,
                trace_sha256: shout.trace_sha256.clone(),
                logup_prover_speedup: shout.median.prove_ms / logup.median.prove_ms,
                logup_verifier_speedup: shout.median.verify_ms / logup.median.verify_ms,
                shout_to_logup_commit_ratio: shout.median.commit_ms / logup.median.commit_ms,
                shout_to_logup_opening_ratio: shout.median.opening_ms / logup.median.opening_ms,
                shout_to_logup_committed_area_ratio: shout.areas.total_committed_bits as f64
                    / logup.areas.total_committed_bits as f64,
            });
        }
    }
    result
}
fn environment() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("compiled_arch".into(), env::consts::ARCH.into());
    for (name, on) in [
        ("pclmulqdq", cfg!(target_feature = "pclmulqdq")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("vpclmulqdq", cfg!(target_feature = "vpclmulqdq")),
        ("sha", cfg!(target_feature = "sha")),
        ("avx512f", cfg!(target_feature = "avx512f")),
    ] {
        out.insert(format!("compiled_{name}"), on.to_string());
    }
    if let Ok(v) = Command::new("rustc").arg("--version").output() {
        out.insert(
            "rustc_on_path".into(),
            String::from_utf8_lossy(&v.stdout).trim().into(),
        );
    }
    if let Ok(info) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = info.lines().find(|l| l.starts_with("model name")) {
            out.insert(
                "cpu_model".into(),
                line.split_once(':').map_or(line, |(_, v)| v).trim().into(),
            );
        }
    }
    if let Ok(path) = env::current_exe() {
        if let Ok(bytes) = fs::read(path) {
            out.insert(
                "executable_sha256".into(),
                hex(&Hasher::sha256().digest(&bytes)),
            );
        }
    }
    out
}
#[derive(Serialize)]
struct Report {
    schema_version: usize,
    implementation: String,
    options: Options,
    environment: BTreeMap<String, String>,
    provenance: Provenance,
    notes: Vec<String>,
    cases: Vec<Case>,
    comparisons: Vec<Comparison>,
}
#[derive(Serialize)]
struct Provenance {
    open_magnetar: serde_json::Value,
    bitz_logup: serde_json::Value,
    adapter: serde_json::Value,
}
fn write(path: &Path, data: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, data)
}
fn csv_escape(v: &str) -> String {
    if v.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_owned()
    }
}
fn csv(report: &Report) -> String {
    let mut stages = BTreeMap::new();
    for c in &report.cases {
        for s in &c.samples {
            for key in s.exclusive_stage_ms.keys() {
                stages.insert(key.clone(), ());
            }
        }
    }
    let mut header = vec![
        "case",
        "method",
        "pattern",
        "table_bits",
        "log_lookups",
        "threads",
        "seed",
        "profile",
        "log_rate",
        "chunk_bits",
        "virtualization",
        "trace_sha256",
        "iteration",
        "table_rows",
        "lookups",
        "core_bits",
        "argument_bits",
        "main_committed_bits",
        "main_padding_bits",
        "auxiliary_bits",
        "auxiliary_committed_bits",
        "auxiliary_padding_bits",
        "total_committed_bits",
        "main_claims",
        "auxiliary_claims",
        "prove_ms",
        "verify_ms",
        "main_commit_ms",
        "auxiliary_commit_ms",
        "main_open_ms",
        "auxiliary_open_ms",
        "commit_ms",
        "opening_ms",
        "remaining_ms",
        "prep_ms",
        "argument_ms",
        "virtualization_ms",
        "exclusive_stages_total_ms",
        "unattributed_prove_ms",
        "logup_extra_bound_bits",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    header.extend(stages.keys().map(|key| format!("exclusive:{key}")));
    let mut out = header
        .into_iter()
        .map(|v| csv_escape(&v))
        .collect::<Vec<_>>()
        .join(",");
    out.push('\n');
    for (i, c) in report.cases.iter().enumerate() {
        for s in &c.samples {
            let a = &c.areas;
            let mut row = vec![
                i.to_string(),
                c.method.clone(),
                c.pattern.clone(),
                report.options.table_bits.to_string(),
                report.options.log_lookups.to_string(),
                report.options.threads.to_string(),
                report.options.seed.to_string(),
                report.options.profile.clone(),
                report.options.log_rate.to_string(),
                c.chunk_bits.map(|n| n.to_string()).unwrap_or_default(),
                c.virtualization.clone().unwrap_or_default(),
                c.trace_sha256.clone(),
                s.iteration.to_string(),
                a.table_rows.to_string(),
                a.lookups.to_string(),
                a.core_bits.to_string(),
                a.argument_bits.to_string(),
                a.main_committed_bits.to_string(),
                a.main_padding_bits.to_string(),
                a.auxiliary_bits.to_string(),
                a.auxiliary_committed_bits.to_string(),
                a.auxiliary_padding_bits.to_string(),
                a.total_committed_bits.to_string(),
                a.main_claims.to_string(),
                a.auxiliary_claims.to_string(),
            ];
            row.extend(
                [
                    s.prove_ms,
                    s.verify_ms,
                    s.main_commit_ms,
                    s.auxiliary_commit_ms,
                    s.main_open_ms,
                    s.auxiliary_open_ms,
                    s.commit_ms,
                    s.opening_ms,
                    s.remaining_ms,
                    s.prep_ms,
                    s.argument_ms,
                    s.virtualization_ms,
                    s.exclusive_stages_total_ms,
                    s.unattributed_prove_ms,
                ]
                .into_iter()
                .map(|v| format!("{v:.9}")),
            );
            row.push(
                c.logup_extra_bound
                    .as_ref()
                    .map(|b| format!("{:.9}", b.negative_log2_bound))
                    .unwrap_or_default(),
            );
            row.extend(stages.keys().map(|key| {
                format!(
                    "{:.9}",
                    s.exclusive_stage_ms.get(key).copied().unwrap_or(0.0)
                )
            }));
            out.push_str(
                &row.into_iter()
                    .map(|v| csv_escape(&v))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push('\n');
        }
    }
    out
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(opts) = options()? else {
        return Ok(());
    };
    let payloads = payloads(&opts)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build()?;
    let cases = pool.install(|| benchmark(&opts, &payloads).map_err(|e| e.to_string()))?;
    let pairs = comparisons(&cases);
    let provenance = Provenance {
        open_magnetar: serde_json::from_str(include_str!(
            "../../../vendor/open-magnetar/UPSTREAM.json"
        ))?,
        bitz_logup: serde_json::from_str(include_str!("../../../vendor/bitz-logup/UPSTREAM.json"))?,
        adapter: serde_json::from_str(include_str!("../../../audit/stage13-implementation.json"))?,
    };
    let notes = vec![
        "All warmup and measured proofs independently verified. Payload generation and ProveResult destruction excluded from prover time.".into(),
        "Inclusive main/aux commit/open durations are disjoint; remaining_ms includes IOP, witness/setup and statement preparation. Exclusive stage spans are separate diagnostics and must not be added to inclusive timings.".into(),
        "prep_ms sums lookup.witness/pack/statement/claims exclusive spans; argument_ms sums shout.*, virt.* and lookup.logup* exclusive spans. virtualization_ms sums shout.virtualization and all virt.* children, including GKR build/layers. These overlapping phase diagnostics are computed per sample before taking medians; named stage medians need not sum to a phase median.".into(),
        "Both methods share identical public u64 ROM values and address traces with recorded SHA256 fingerprints, read_values=None. Guest-derived cases retain only addresses and discard RAM values/stores.".into(),
        "Shout chunk/virtualization choices are swept; LogupStar has one canonical case per pattern, unused chunk_bits=0 and Flat config. Ratios above1 indicate smaller/faster LogupStar values.".into(),
        "PCS profile names are not composed security guarantees. The conservative extra LogupStar term (T+N)/2^128 excludes other errors; equal overall security is not claimed.".into(),
    ];
    let report = Report {
        schema_version: 1,
        implementation: "gf128-indexed-rom-lookup-v1".into(),
        environment: environment(),
        provenance,
        notes,
        options: opts,
        cases,
        comparisons: pairs,
    };
    if let Some(path) = &report.options.output {
        write(path, &serde_json::to_vec_pretty(&report)?)?;
        println!("JSON: {}", path.display());
    }
    if let Some(path) = &report.options.csv {
        write(path, csv(&report).as_bytes())?;
        println!("CSV: {}", path.display());
    }
    println!(
        "All {} cases verified ({} measured proofs/case plus {} warmups).",
        report.cases.len(),
        report.options.repeat,
        report.options.warmup
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logup_is_not_duplicated_by_shout_sweeps() {
        let opts = Options {
            chunk_bits: vec![1, 2, 4],
            virtualization: vec!["flat".into(), "gkr".into()],
            ..Default::default()
        };
        let configs = configurations(&opts);
        assert_eq!(configs.len(), 7);
        assert_eq!(
            configs
                .iter()
                .filter(|(m, _, _)| *m == Method::LogupStar)
                .count(),
            1
        );
        assert_eq!(configs.last().unwrap().1, None);
    }
    #[test]
    fn security_bound_and_csv_escape() {
        let b = logup_bound(1 << 18, 1 << 16);
        assert!((b.negative_log2_bound - 109.67807190511263).abs() < 1e-10);
        assert_eq!(csv_escape("a,\"b\""), "\"a,\"\"b\"\"\"");
        assert!(selection("shout,shout", &["shout", "logup"], None).is_err());
    }
    #[test]
    fn requested_method_and_chunk_order_are_preserved() {
        let opts = Options {
            methods: vec!["logup".into(), "shout".into()],
            chunk_bits: vec![4, 2, 1],
            ..Default::default()
        };
        let configs = configurations(&opts);
        assert_eq!(configs[0].0, Method::LogupStar);
        assert_eq!(
            configs[1..]
                .iter()
                .map(|c| c.1.unwrap())
                .collect::<Vec<_>>(),
            vec![4, 2, 1]
        );
    }
    fn stage_map(stages: &[(&str, f64)]) -> BTreeMap<String, f64> {
        stages.iter().map(|&(name, ms)| (name.into(), ms)).collect()
    }
    #[test]
    fn phase_totals_include_flat_and_tree_children_once() {
        let stages = stage_map(&[
            ("lookup.witness", 1.0),
            ("lookup.pack", 2.0),
            ("lookup.statement", 3.0),
            ("lookup.claims", 4.0),
            ("shout.read", 5.0),
            ("shout.onehot", 6.0),
            ("shout.virtualization", 7.0),
            ("virt.flat.product", 8.0),
            ("virt.tree.build", 9.0),
            ("virt.tree.layers", 10.0),
            ("lookup.main.commit", 100.0),
            ("pcs.ringswitch", 200.0),
        ]);
        assert_eq!(
            PhaseTotals::new(&stages),
            PhaseTotals {
                prep_ms: 10.0,
                argument_ms: 45.0,
                virtualization_ms: 34.0
            }
        );
    }
    #[test]
    fn logup_argument_excludes_nested_commit_and_open_spans() {
        // Spans report exclusive durations: the parent's 2ms already excludes
        // its auxiliary commitment, opening, and named IOP children.
        let stages = stage_map(&[
            ("lookup.logup", 2.0),
            ("lookup.logup.pushforward", 3.0),
            ("lookup.logup.pack_aux", 5.0),
            ("lookup.logup.rational", 7.0),
            ("lookup.logup.table_product", 11.0),
            ("lookup.aux.commit", 100.0),
            ("lookup.aux.open", 200.0),
            ("lookup.mosaic", 300.0),
            ("pcs.ligerito_open", 400.0),
        ]);
        assert_eq!(
            PhaseTotals::new(&stages),
            PhaseTotals {
                prep_ms: 0.0,
                argument_ms: 28.0,
                virtualization_ms: 0.0
            }
        );
    }
    fn sample_with_stages(stages: &[(&str, f64)]) -> Sample {
        let exclusive_stage_ms = stage_map(stages);
        let phases = PhaseTotals::new(&exclusive_stage_ms);
        let exclusive_stages_total_ms = exclusive_stage_ms.values().sum::<f64>();
        Sample {
            iteration: 0,
            prove_ms: 100.0,
            verify_ms: 1.0,
            main_commit_ms: 0.0,
            auxiliary_commit_ms: 0.0,
            main_open_ms: 0.0,
            auxiliary_open_ms: 0.0,
            commit_ms: 0.0,
            opening_ms: 0.0,
            remaining_ms: 100.0,
            prep_ms: phases.prep_ms,
            argument_ms: phases.argument_ms,
            virtualization_ms: phases.virtualization_ms,
            exclusive_stage_ms,
            exclusive_stages_total_ms,
            unattributed_prove_ms: 100.0 - exclusive_stages_total_ms,
        }
    }
    #[test]
    fn phase_medians_are_derived_from_each_complete_sample() {
        let samples = [
            sample_with_stages(&[("shout.virtualization", 1.0), ("virt.tree.layers", 9.0)]),
            sample_with_stages(&[("shout.virtualization", 9.0), ("virt.tree.layers", 1.0)]),
            sample_with_stages(&[("shout.virtualization", 9.0), ("virt.tree.layers", 9.0)]),
        ];
        let medians = Medians::new(&samples);
        assert_eq!(medians.virtualization_ms, 10.0);
        assert_eq!(medians.argument_ms, 10.0);
        assert_eq!(medians.exclusive_stage_ms.values().sum::<f64>(), 18.0);
        assert_eq!(medians.unattributed_prove_ms, 90.0);
    }
}
