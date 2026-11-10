use ram_argument::{AccessTrace, Config, VirtualizationMode};
use ram_payload::{GuestFill, GuestOptions, GuestProgram, GuestRows, RamPayload};
use serde::Serialize;
use std::{
    collections::BTreeMap, env, fs, hint::black_box, path::PathBuf, process::Command, time::Instant,
};

const HELP: &str = "Standalone GF(2^128) BinaryTwist RAM benchmark

Usage: cargo run --release -p ram-bench -- [options]

  --payload random|guest|both       Payload selection (default both)
  --log-cycles N                    Prove exactly 2^N cycles (default 12)
  --address-bits A                  Memory has 2^A 64-bit words (default 12)
  --chunk-bits C[,C...]             Address bits/component; must divide A (default 4)
  --virtualization flat|gkr|both    Aggregated flat or GKR reduction (default flat);
  --virtualization legacy-flat     Original multipoint flat reference;
                                   GKR requires a power-of-two chunk count
  --memory-model shared|independent
                                   Shared-address RAM by default; independent ports are optional
  --reads P                        Read ports in independent mode (default 1)
  --layout natural|permuted|both    Address relabeling for locality comparison (default natural)
  --threads N                      Rayon worker count (default 4)
  --repeat N                       Measured proofs per case (default 3)
  --warmup N                       Unreported warmup proofs per case (default 1)
  --seed N                         Deterministic payload and relabeling seed (default 1)
  --profile provable-120|conjectured-100
                                   Upstream PCS profile (default provable-120)
  --log-rate N                     Log2 inverse code rate (default 2)
  --guest-keys N                    Binary-search guest sorted-key count (default 64)
  --guest-queries N                 Binary-search guest query count (default 16)
  --guest-program NAME[,NAME...]    binary-search (default), erc20-mixed-block,
                                   pointer-chase, or all
  --guest-size N                    Account/node count for ERC20/pointer chase (default 64)
  --guest-operations N              Transfer/chase step count (default 256)
  --guest-rows all|memory|both       Include dummy accesses (default all), remove them,
                                   or compare both; ordered lists also accepted
  --guest-fill repeat|halt          Repeat the RAM pattern (default) or append idle rows
  --output PATH                    Save complete JSON samples and metadata
  --csv PATH                       Save one CSV row per measured proof
  --help                           Print this help

Every proof is verified. Payload generation and address relabeling are outside
the timed prover. Witness preparation, packing, commitments, all RAM reductions,
Mosaic, RingSwitch and Ligerito opening are inside it. All arithmetic remains
GF128. The inherited provable-120 profile is not a claim of full 128-bit security.
The guest is a real upstream RV64I program; this benchmark proves RAM consistency,
not CPU execution. Shared RAM reads then writes the same address; guest payloads
require that mode. Reports list unique committed columns and padding.
";

#[derive(Clone, Debug, Serialize)]
struct Options {
    payload: String,
    log_cycles: usize,
    address_bits: usize,
    chunk_bits: Vec<usize>,
    virtualization: String,
    memory_model: String,
    reads: usize,
    layout: String,
    threads: usize,
    repeat: usize,
    warmup: usize,
    seed: u64,
    profile: String,
    log_rate: usize,
    guest_keys: usize,
    guest_queries: usize,
    guest_program: String,
    guest_size: usize,
    guest_operations: usize,
    guest_rows: String,
    guest_fill: String,
    output: Option<PathBuf>,
    csv: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            payload: "both".into(),
            log_cycles: 12,
            address_bits: 12,
            chunk_bits: vec![4],
            virtualization: "flat".into(),
            memory_model: "shared".into(),
            reads: 1,
            layout: "natural".into(),
            threads: 4,
            repeat: 3,
            warmup: 1,
            seed: 1,
            profile: "provable-120".into(),
            log_rate: 2,
            guest_keys: 64,
            guest_queries: 16,
            guest_program: "binary-search".into(),
            guest_size: 64,
            guest_operations: 256,
            guest_rows: "all".into(),
            guest_fill: "repeat".into(),
            output: None,
            csv: None,
        }
    }
}

fn options() -> Result<Option<Options>, String> {
    let mut options = Options::default();
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
            "--payload" => options.payload = value,
            "--log-cycles" => options.log_cycles = number()?,
            "--address-bits" => options.address_bits = number()?,
            "--chunk-bits" => {
                options.chunk_bits = value
                    .split(',')
                    .map(|v| {
                        v.parse::<usize>()
                            .map_err(|_| format!("invalid chunk size: {v}"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--reads" => options.reads = number()?,
            "--virtualization" => options.virtualization = value,
            "--memory-model" => options.memory_model = value,
            "--layout" => options.layout = value,
            "--threads" => options.threads = number()?,
            "--repeat" => options.repeat = number()?,
            "--warmup" => options.warmup = number()?,
            "--seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| format!("invalid seed: {value}"))?
            }
            "--profile" => options.profile = value,
            "--log-rate" => options.log_rate = number()?,
            "--guest-keys" => options.guest_keys = number()?,
            "--guest-queries" => options.guest_queries = number()?,
            "--guest-program" => options.guest_program = value,
            "--guest-size" => options.guest_size = number()?,
            "--guest-operations" => options.guest_operations = number()?,
            "--guest-rows" => options.guest_rows = value,
            "--guest-fill" => options.guest_fill = value,
            "--output" => options.output = Some(value.into()),
            "--csv" => options.csv = Some(value.into()),
            _ => return Err(format!("unknown option {arg}; use --help")),
        }
    }
    if !["random", "guest", "both"].contains(&options.payload.as_str()) {
        return Err("--payload must be random, guest, or both".into());
    }
    if !["natural", "permuted", "both"].contains(&options.layout.as_str()) {
        return Err("--layout must be natural, permuted, or both".into());
    }
    if options.repeat == 0 || options.threads == 0 || options.reads == 0 {
        return Err("repeat, threads, and reads must be positive".into());
    }
    if options.address_bits == 0
        || options.address_bits > 31
        || !(2..=30).contains(&options.log_cycles)
    {
        return Err("address-bits must be 1..31 and log-cycles 2..30".into());
    }
    if options.chunk_bits.is_empty()
        || options
            .chunk_bits
            .iter()
            .any(|&c| c == 0 || !options.address_bits.is_multiple_of(c))
    {
        return Err("each positive chunk-bits value must divide address-bits".into());
    }
    if options.payload != "random" && options.reads != 1 {
        return Err(
            "guest payload requires --reads 1; select --payload random for other port counts"
                .into(),
        );
    }
    if virtualizations(&options.virtualization)?.contains(&VirtualizationMode::GkrTree)
        && options
            .chunk_bits
            .iter()
            .any(|&c| !(options.address_bits / c).is_power_of_two())
    {
        return Err(
            "GKR virtualization requires address-bits / chunk-bits to be a power of two".into(),
        );
    }
    if !["shared", "independent"].contains(&options.memory_model.as_str()) {
        return Err("--memory-model must be shared or independent".into());
    }
    if !["repeat", "halt"].contains(&options.guest_fill.as_str()) {
        return Err("--guest-fill must be repeat or halt".into());
    }
    guest_programs(&options.guest_program)?;
    let rows = guest_rows(&options.guest_rows)?;
    if options.payload != "random"
        && options.guest_fill == "halt"
        && rows.contains(&GuestRows::Memory)
    {
        return Err(
            "--guest-rows memory requires --guest-fill repeat; HALTED rows are dummy accesses"
                .into(),
        );
    }
    if options.memory_model == "shared" && options.reads != 1 {
        return Err("shared-address RAM requires --reads 1; use --memory-model independent for separate read ports".into());
    }
    if options.memory_model == "independent" && options.payload != "random" {
        return Err(
            "independent-port mode requires --payload random; guest traces use shared-address RAM"
                .into(),
        );
    }
    if !["provable-120", "conjectured-100"].contains(&options.profile.as_str()) {
        return Err("--profile must be provable-120 or conjectured-100".into());
    }
    let minimum_rate = if options.profile == "provable-120" {
        2
    } else {
        1
    };
    if !(minimum_rate..=8).contains(&options.log_rate) {
        return Err(format!(
            "--log-rate must be {minimum_rate}..8 for this profile"
        ));
    }
    Ok(Some(options))
}

fn guest_programs(value: &str) -> Result<Vec<GuestProgram>, String> {
    if value == "all" {
        return Ok(vec![
            GuestProgram::BinarySearch,
            GuestProgram::Erc20MixedBlock,
            GuestProgram::PointerChase,
        ]);
    }
    let mut programs = Vec::new();
    for name in value.split(',') {
        let program = match name {
            "binary-search" => GuestProgram::BinarySearch,
            "erc20-mixed-block" => GuestProgram::Erc20MixedBlock,
            "pointer-chase" => GuestProgram::PointerChase,
            _ => {
                return Err(format!(
                    "unknown guest program {name}; use binary-search, erc20-mixed-block, pointer-chase, or all"
                ));
            }
        };
        if programs.contains(&program) {
            return Err(format!("duplicate guest program {name}"));
        }
        programs.push(program);
    }
    Ok(programs)
}

fn guest_rows(value: &str) -> Result<Vec<GuestRows>, String> {
    if value == "both" {
        return Ok(vec![GuestRows::All, GuestRows::Memory]);
    }
    let mut selections = Vec::new();
    for name in value.split(',') {
        let rows = match name {
            "all" => GuestRows::All,
            "memory" => GuestRows::Memory,
            _ => {
                return Err(format!(
                    "unknown guest row selection {name}; use all, memory, or both"
                ));
            }
        };
        if selections.contains(&rows) {
            return Err(format!("duplicate guest row selection {name}"));
        }
        selections.push(rows);
    }
    Ok(selections)
}

fn row_name(rows: GuestRows) -> &'static str {
    match rows {
        GuestRows::All => "all",
        GuestRows::Memory => "memory",
    }
}

fn virtualizations(value: &str) -> Result<Vec<VirtualizationMode>, String> {
    if value == "both" {
        return Ok(vec![VirtualizationMode::Flat, VirtualizationMode::GkrTree]);
    }
    let mut modes = Vec::new();
    for name in value.split(',') {
        let mode = match name {
            "flat" => VirtualizationMode::Flat,
            "legacy-flat" => VirtualizationMode::FlatLegacy,
            "gkr" | "gkr-tree" => VirtualizationMode::GkrTree,
            _ => {
                return Err(format!(
                    "unknown virtualization {name}; use flat, gkr, legacy-flat, or both"
                ));
            }
        };
        if modes.contains(&mode) {
            return Err(format!("duplicate virtualization {name}"));
        }
        modes.push(mode);
    }
    Ok(modes)
}

fn virtualization_name(mode: VirtualizationMode) -> &'static str {
    match mode {
        VirtualizationMode::Flat => "flat",
        VirtualizationMode::FlatLegacy => "legacy-flat",
        VirtualizationMode::GkrTree => "gkr-tree",
    }
}

fn virtualization_time(stages: &BTreeMap<String, f64>) -> f64 {
    stages
        .iter()
        .filter(|(name, _)| name.as_str() == "twist.virtualization" || name.starts_with("virt."))
        .map(|(_, value)| value)
        .sum()
}

#[derive(Serialize)]
struct Sample {
    prover_ms: f64,
    virtualization_ms: f64,
    commitment_ms: f64,
    everything_else_ms: f64,
    commitment_fraction: f64,
    verifier_ms: f64,
    unattributed_ms: f64,
    stages_ms: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct Case {
    payload: String,
    guest_program: Option<GuestProgram>,
    guest_rows: Option<GuestRows>,
    layout: String,
    chunk_bits: usize,
    address_components: usize,
    virtualization: String,
    virtualization_stats: VirtualizationStats,
    payload_generation_ms: f64,
    source_payload_metadata: serde_json::Value,
    address_relabeling: String,
    work_normalization: WorkNormalization,
    proof_summary: ProofSummary,
    samples: Vec<Sample>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
struct VirtualizationStats {
    sumchecks: usize,
    sumcheck_rounds: usize,
    round_message_field_elements: usize,
    node_evaluation_field_elements: usize,
    /// GF128 elements only; excludes framing and in-memory Vec overhead.
    field_payload_bytes: usize,
}

impl VirtualizationStats {
    fn from_proof(proof: &ram_argument::RamProof) -> Self {
        let mut result = Self::default();
        if let Some(flat) = &proof.twist.virt {
            result.sumchecks = 1;
            result.sumcheck_rounds = flat.sc.round_msgs.len();
            result.round_message_field_elements = flat.sc.round_msgs.iter().map(Vec::len).sum();
            result.node_evaluation_field_elements = flat.evals.len();
        }
        if let Some(flat) = &proof.twist.virt_flat {
            result.sumchecks = 2;
            result.sumcheck_rounds = flat.reduction.sc.round_msgs.len() + flat.sc.round_msgs.len();
            result.round_message_field_elements = flat
                .reduction
                .sc
                .round_msgs
                .iter()
                .chain(&flat.sc.round_msgs)
                .map(Vec::len)
                .sum();
            result.node_evaluation_field_elements = flat.reduction.evals.len() + flat.evals.len();
        }
        if let Some(tree) = &proof.twist.virt_tree {
            result.sumchecks = tree.layers.len();
            for layer in &tree.layers {
                result.sumcheck_rounds += layer.sumcheck.round_msgs.len();
                result.round_message_field_elements += layer
                    .sumcheck
                    .round_msgs
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                result.node_evaluation_field_elements += layer.child_evals.len();
            }
        }
        result.field_payload_bytes =
            16 * (result.round_message_field_elements + result.node_evaluation_field_elements);
        result
    }
}

#[derive(Serialize)]
struct WorkNormalization {
    memory_access_cycles: usize,
    dummy_access_cycles: usize,
    memory_access_fraction: f64,
    /// Multiply any measured milliseconds by this to obtain ms / million RAM-access cycles.
    ms_to_ms_per_million_memory_accesses: Option<f64>,
}

impl WorkNormalization {
    fn from_payload(payload: &RamPayload) -> Self {
        let memory_access_cycles = payload
            .metadata
            .guest
            .as_ref()
            .map_or(payload.metadata.cycles, |guest| guest.memory_access_cycles);
        Self::new(payload.metadata.cycles, memory_access_cycles)
    }

    fn new(cycles: usize, memory_access_cycles: usize) -> Self {
        Self {
            memory_access_cycles,
            dummy_access_cycles: cycles - memory_access_cycles,
            memory_access_fraction: memory_access_cycles as f64 / cycles as f64,
            ms_to_ms_per_million_memory_accesses: (memory_access_cycles > 0)
                .then(|| 1_000_000.0 / memory_access_cycles as f64),
        }
    }
}

#[derive(Serialize)]
struct CostComparison {
    all_ms: f64,
    memory_only_ms: f64,
    memory_only_equal_work_ms: f64,
    estimated_saving_fraction: f64,
}

impl CostComparison {
    fn new(all_ms: f64, memory_only_ms: f64, scale: f64) -> Self {
        let memory_only_equal_work_ms = memory_only_ms * scale;
        Self {
            all_ms,
            memory_only_ms,
            memory_only_equal_work_ms,
            estimated_saving_fraction: if all_ms > 0.0 {
                1.0 - memory_only_equal_work_ms / all_ms
            } else {
                0.0
            },
        }
    }
}

#[derive(Serialize)]
struct DummyComparison {
    guest_program: GuestProgram,
    layout: String,
    chunk_bits: usize,
    virtualization: String,
    all_memory_access_cycles: usize,
    memory_only_memory_access_cycles: usize,
    /// Exact ratio of useful RAM-access counts in the two proved traces, including tails.
    memory_only_equal_work_scale: f64,
    prover: CostComparison,
    commitment: CostComparison,
    everything_else: CostComparison,
    stages: BTreeMap<String, CostComparison>,
}

#[derive(Serialize)]
struct ModeCostComparison {
    flat_ms: f64,
    gkr_tree_ms: f64,
    speedup: Option<f64>,
    saving_fraction: Option<f64>,
}

impl ModeCostComparison {
    fn new(flat_ms: f64, gkr_tree_ms: f64) -> Self {
        Self {
            flat_ms,
            gkr_tree_ms,
            speedup: (gkr_tree_ms > 0.0).then(|| flat_ms / gkr_tree_ms),
            saving_fraction: (flat_ms > 0.0).then(|| 1.0 - gkr_tree_ms / flat_ms),
        }
    }
}

#[derive(Serialize)]
struct VirtualizationComparison {
    payload: String,
    guest_program: Option<GuestProgram>,
    guest_rows: Option<GuestRows>,
    layout: String,
    chunk_bits: usize,
    address_components: usize,
    prover: ModeCostComparison,
    virtualization: ModeCostComparison,
    verifier: ModeCostComparison,
    flat_stats: VirtualizationStats,
    gkr_tree_stats: VirtualizationStats,
}

#[derive(Default, Serialize)]
struct ProofSummary {
    cycles: usize,
    memory_cells: usize,
    address_components: usize,
    chunk_bits: usize,
    committed_bits: usize,
    packed_field_elements: usize,
    mosaic_claims: usize,
    commitment_components: Vec<CommitmentComponent>,
    commitment_footprint: CommitmentFootprint,
}

#[derive(Default, Serialize)]
struct CommitmentFootprint {
    address_encoding_bits: usize,
    read_write_value_bits: usize,
    increment_bits: usize,
    raw_binary_address_bits: usize,
    other_bits: usize,
    populated_bits: usize,
    padding_bits: usize,
    populated_bits_per_cycle: usize,
    padded_bits_per_cycle: usize,
}

#[derive(Serialize)]
struct CommitmentComponent {
    name: String,
    log_size: usize,
}

impl From<&ram_argument::Summary> for ProofSummary {
    fn from(value: &ram_argument::Summary) -> Self {
        let mut footprint = CommitmentFootprint::default();
        for component in &value.commitment_components {
            let bits = 1usize << component.log_size;
            footprint.populated_bits += bits;
            if component.name == "ram/inc" {
                footprint.increment_bits += bits;
            } else if component.name.starts_with("ram/ra") || component.name.starts_with("ram/wa") {
                footprint.address_encoding_bits += bits;
            } else if component.name.starts_with("input/rv") || component.name == "input/wv" {
                footprint.read_write_value_bits += bits;
            } else if component.name.starts_with("input/address") {
                footprint.raw_binary_address_bits += bits;
            } else {
                footprint.other_bits += bits;
            }
        }
        footprint.padding_bits = value.committed_bits - footprint.populated_bits;
        footprint.populated_bits_per_cycle = footprint.populated_bits / value.cycles;
        footprint.padded_bits_per_cycle = value.committed_bits / value.cycles;
        Self {
            cycles: value.cycles,
            memory_cells: value.memory_cells,
            address_components: value.address_components,
            chunk_bits: value.chunk_bits,
            committed_bits: value.committed_bits,
            packed_field_elements: value.packed_field_elements,
            mosaic_claims: value.mosaic_claims,
            commitment_components: value
                .commitment_components
                .iter()
                .map(|c| CommitmentComponent {
                    name: c.name.clone(),
                    log_size: c.log_size,
                })
                .collect(),
            commitment_footprint: footprint,
        }
    }
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    implementation: String,
    vendor_patch_sha256: String,
    upstream_revision: &'static str,
    field: &'static str,
    options: Options,
    environment: BTreeMap<String, String>,
    cases: Vec<Case>,
    dummy_comparisons: Vec<DummyComparison>,
    virtualization_comparisons: Vec<VirtualizationComparison>,
}

fn environment() -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    values.insert("architecture".into(), env::consts::ARCH.into());
    values.insert("os".into(), env::consts::OS.into());
    values.insert(
        "compiled_pclmulqdq".into(),
        cfg!(target_feature = "pclmulqdq").to_string(),
    );
    values.insert(
        "compiled_avx2".into(),
        cfg!(target_feature = "avx2").to_string(),
    );
    values.insert(
        "compiled_vpclmulqdq".into(),
        cfg!(target_feature = "vpclmulqdq").to_string(),
    );
    values.insert(
        "compiled_sha".into(),
        cfg!(target_feature = "sha").to_string(),
    );
    values.insert(
        "compiled_avx512f".into(),
        cfg!(target_feature = "avx512f").to_string(),
    );
    if let Ok(output) = Command::new("rustc").arg("--version").output() {
        values.insert(
            "rustc_on_path".into(),
            String::from_utf8_lossy(&output.stdout).trim().into(),
        );
    }
    if let Ok(info) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = info.lines().find(|l| l.starts_with("model name")) {
            values.insert(
                "cpu_model".into(),
                line.split_once(':').map_or(line, |(_, v)| v).trim().into(),
            );
        }
    }
    values
}

fn random_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// A permutation preserves the complete memory relation while changing locality.
fn relabel(payload: &RamPayload, seed: u64) -> RamPayload {
    let mut copy = payload.clone();
    let mut map: Vec<usize> = (0..copy.initial_memory.len()).collect();
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    if state == 0 {
        state = 1;
    }
    for i in (1..map.len()).rev() {
        let j = (random_u64(&mut state) % (i as u64 + 1)) as usize;
        map.swap(i, j);
    }
    for (i, &value) in payload.initial_memory.iter().enumerate() {
        copy.initial_memory[map[i]] = value;
    }
    for port in &mut copy.reads {
        for address in port {
            *address = map[*address as usize] as u32;
        }
    }
    for address in &mut copy.writes {
        *address = map[*address as usize] as u32;
    }
    copy
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn run(options: &Options) -> Result<Report, Box<dyn std::error::Error>> {
    let payloads: &[&str] = match options.payload.as_str() {
        "both" => &["random", "guest"],
        "random" => &["random"],
        _ => &["guest"],
    };
    let layouts: &[&str] = match options.layout.as_str() {
        "both" => &["natural", "permuted"],
        "natural" => &["natural"],
        _ => &["permuted"],
    };
    let mut cases = Vec::new();
    let mut workloads = Vec::new();
    for &name in payloads {
        if name == "random" {
            workloads.push((name, None, None));
        } else {
            for program in guest_programs(&options.guest_program)? {
                for rows in guest_rows(&options.guest_rows)? {
                    workloads.push((name, Some(program), Some(rows)));
                }
            }
        }
    }
    for (name, program, rows) in workloads {
        let start = Instant::now();
        let payload = if name == "random" {
            if options.memory_model == "shared" {
                ram_payload::random_ram(
                    options.log_cycles as u32,
                    options.address_bits as u32,
                    options.seed,
                )?
            } else {
                ram_payload::random(
                    options.log_cycles as u32,
                    options.address_bits as u32,
                    options.reads,
                    options.seed,
                )?
            }
        } else {
            ram_payload::guest(
                options.log_cycles as u32,
                options.address_bits as u32,
                &GuestOptions {
                    program: program.unwrap(),
                    rows: rows.unwrap(),
                    size: options.guest_size,
                    operations: options.guest_operations,
                    keys: options.guest_keys,
                    queries: options.guest_queries,
                    seed: options.seed,
                    // Build a checked source pattern independently of the requested
                    // RAM-trace length, which can be shorter or much longer.
                    max_cycles: 1usize << 20,
                    fill: if options.guest_fill == "repeat" {
                        GuestFill::Repeat
                    } else {
                        GuestFill::Halt
                    },
                },
            )?
        };
        let generation_ms = start.elapsed().as_secs_f64() * 1000.0;
        for &layout in layouts {
            let data = if layout == "permuted" {
                relabel(&payload, options.seed)
            } else {
                payload.clone()
            };
            let trace = AccessTrace {
                initial: data.initial_memory.clone(),
                reads: data.reads.clone(),
                writes: data.writes.clone(),
                new_values: data.new_values.clone(),
                read_values: Some(data.read_values.clone()),
            };
            for &chunk_bits in &options.chunk_bits {
                for virtualization in virtualizations(&options.virtualization)? {
                    eprintln!(
                        "{name}{}/{layout}/{}/{}: 2^{} cycles, 2^{} words, {chunk_bits}-bit components, {} workers",
                        program.map_or(String::new(), |p| format!(
                            "/{}/{}",
                            p.name(),
                            row_name(rows.unwrap())
                        )),
                        options.memory_model,
                        virtualization_name(virtualization),
                        options.log_cycles,
                        options.address_bits,
                        options.threads
                    );
                    let config = Config {
                        virtualization,
                        shared_address: options.memory_model == "shared",
                        address_bits: options.address_bits,
                        chunk_bits,
                        read_ports: options.reads,
                        word_log_bits: 6,
                        profile: options.profile.clone(),
                        log_rate: options.log_rate,
                    };
                    let mut samples = Vec::new();
                    let mut summary = ProofSummary::default();
                    let mut virtualization_stats = VirtualizationStats::default();
                    for iteration in 0..options.warmup + options.repeat {
                        let start = Instant::now();
                        let result = ram_argument::prove(&config, black_box(&trace))?;
                        let prover_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let start = Instant::now();
                        ram_argument::verify(
                            black_box(&result.statement),
                            black_box(&result.proof),
                        )?;
                        let verifier_ms = start.elapsed().as_secs_f64() * 1000.0;
                        if iteration >= options.warmup {
                            let mut stages_ms = BTreeMap::new();
                            for (stage, duration) in result.metrics {
                                *stages_ms.entry(stage.to_string()).or_insert(0.0) +=
                                    duration.as_secs_f64() * 1000.0;
                            }
                            let commitment_ms = stages_ms.get("pcs.commit").copied().unwrap_or(0.0);
                            if commitment_ms == 0.0 {
                                return Err(
                                    "commitment timing is missing; enable om-transcript profiling"
                                        .into(),
                                );
                            }
                            let attributed: f64 = stages_ms.values().sum();
                            if attributed > prover_ms * 1.01 {
                                return Err(
                                    "stage timings overlap or exceed measured prover time".into()
                                );
                            }
                            summary = ProofSummary::from(&result.summary);
                            let stats = VirtualizationStats::from_proof(&result.proof);
                            if !samples.is_empty() && stats != virtualization_stats {
                                return Err(
                                    "virtualization proof shape changed across repetitions".into(),
                                );
                            }
                            virtualization_stats = stats;
                            samples.push(Sample {
                                prover_ms,
                                virtualization_ms: virtualization_time(&stages_ms),
                                commitment_ms,
                                everything_else_ms: (prover_ms - commitment_ms).max(0.0),
                                commitment_fraction: commitment_ms / prover_ms,
                                verifier_ms,
                                unattributed_ms: (prover_ms - attributed).max(0.0),
                                stages_ms,
                            });
                        }
                    }
                    cases.push(Case {
                    payload: name.into(),
                    guest_program: program,
                    guest_rows: rows,
                    layout: layout.into(),
                    chunk_bits,
                    address_components: options.address_bits / chunk_bits,
                    virtualization: virtualization_name(virtualization).into(),
                    virtualization_stats,
                    payload_generation_ms: generation_ms,
                    source_payload_metadata: serde_json::to_value(&data.metadata)?,
                    work_normalization: WorkNormalization::from_payload(&data),
                    address_relabeling: if layout == "permuted" {
                        "Full-domain Fisher-Yates bijection; xorshift64(13,7,17), initial state seed XOR 0x9e3779b97f4a7c15 (zero replaced by 1). Source compact_to_original mapping precedes this bijection.".into()
                    } else { "Identity; source compact_to_original mapping applies directly.".into() },
                    proof_summary: summary,
                    samples,
                });
                }
            }
        }
    }
    let dummy_comparisons = compare_dummy_cases(&cases);
    let virtualization_comparisons = compare_virtualization_cases(&cases);
    let provenance: serde_json::Value =
        serde_json::from_str(include_str!("../../../vendor/open-magnetar/UPSTREAM.json"))?;
    Ok(Report {
        schema: "memarg-stage-1.2/v6",
        implementation: provenance["implementation"]
            .as_str()
            .ok_or("vendor provenance is missing the implementation identifier")?
            .into(),
        vendor_patch_sha256: provenance["local_patch_sha256"]
            .as_str()
            .ok_or("vendor provenance is missing the local patch hash")?
            .into(),
        upstream_revision: "febfc0e66f5809e034dd1afac78202bd4882de27",
        field: "GF(2^128), X^128+X^7+X^2+X+1",
        options: options.clone(),
        environment: environment(),
        cases,
        dummy_comparisons,
        virtualization_comparisons,
    })
}

fn compare_dummy_cases(cases: &[Case]) -> Vec<DummyComparison> {
    cases
        .iter()
        .filter(|case| case.guest_rows == Some(GuestRows::All))
        .filter_map(|all| {
            let memory = cases.iter().find(|case| {
                case.guest_rows == Some(GuestRows::Memory)
                    && case.guest_program == all.guest_program
                    && case.layout == all.layout
                    && case.chunk_bits == all.chunk_bits
                    && case.virtualization == all.virtualization
            })?;
            let all_count = all.work_normalization.memory_access_cycles;
            let memory_count = memory.work_normalization.memory_access_cycles;
            if all_count == 0 || memory_count == 0 {
                return None;
            }
            let scale = all_count as f64 / memory_count as f64;
            let compare = |metric: fn(&Sample) -> f64| {
                CostComparison::new(
                    median(all.samples.iter().map(metric).collect()),
                    median(memory.samples.iter().map(metric).collect()),
                    scale,
                )
            };
            let stage_names: std::collections::BTreeSet<_> = all
                .samples
                .iter()
                .chain(&memory.samples)
                .flat_map(|sample| sample.stages_ms.keys())
                .collect();
            let stages = stage_names
                .into_iter()
                .map(|stage| {
                    let stage_median = |case: &Case| {
                        median(
                            case.samples
                                .iter()
                                .map(|sample| sample.stages_ms.get(stage).copied().unwrap_or(0.0))
                                .collect(),
                        )
                    };
                    (
                        stage.clone(),
                        CostComparison::new(stage_median(all), stage_median(memory), scale),
                    )
                })
                .collect();
            Some(DummyComparison {
                guest_program: all.guest_program.unwrap(),
                layout: all.layout.clone(),
                chunk_bits: all.chunk_bits,
                virtualization: all.virtualization.clone(),
                all_memory_access_cycles: all_count,
                memory_only_memory_access_cycles: memory_count,
                memory_only_equal_work_scale: scale,
                prover: compare(|sample| sample.prover_ms),
                commitment: compare(|sample| sample.commitment_ms),
                everything_else: compare(|sample| sample.everything_else_ms),
                stages,
            })
        })
        .collect()
}

fn compare_virtualization_cases(cases: &[Case]) -> Vec<VirtualizationComparison> {
    cases
        .iter()
        .filter(|case| case.virtualization == "flat")
        .filter_map(|flat| {
            let tree = cases.iter().find(|case| {
                case.virtualization == "gkr-tree"
                    && case.payload == flat.payload
                    && case.guest_program == flat.guest_program
                    && case.guest_rows == flat.guest_rows
                    && case.layout == flat.layout
                    && case.chunk_bits == flat.chunk_bits
                    && case.source_payload_metadata == flat.source_payload_metadata
            })?;
            let compare = |metric: fn(&Sample) -> f64| {
                ModeCostComparison::new(
                    median(flat.samples.iter().map(metric).collect()),
                    median(tree.samples.iter().map(metric).collect()),
                )
            };
            Some(VirtualizationComparison {
                payload: flat.payload.clone(),
                guest_program: flat.guest_program,
                guest_rows: flat.guest_rows,
                layout: flat.layout.clone(),
                chunk_bits: flat.chunk_bits,
                address_components: flat.address_components,
                prover: compare(|s| s.prover_ms),
                virtualization: compare(|s| s.virtualization_ms),
                verifier: compare(|s| s.verifier_ms),
                flat_stats: flat.virtualization_stats.clone(),
                gkr_tree_stats: tree.virtualization_stats.clone(),
            })
        })
        .collect()
}

fn print_report(report: &Report) {
    println!("Verified GF128 RAM proofs; times are medians in milliseconds.");
    for case in &report.cases {
        let metric = |f: fn(&Sample) -> f64| median(case.samples.iter().map(f).collect());
        println!(
            "\n{}{}/{}/{}: chunk={} bits, d={}, samples={}",
            case.payload,
            case.guest_program.map_or(String::new(), |p| format!(
                "/{}/{}",
                p.name(),
                row_name(case.guest_rows.unwrap())
            )),
            case.layout,
            case.virtualization,
            case.chunk_bits,
            case.address_components,
            case.samples.len()
        );
        if let (Some(real), Some(padding), Some(total)) = (
            case.source_payload_metadata
                .get("real_cycles")
                .and_then(|v| v.as_u64()),
            case.source_payload_metadata
                .get("padding_cycles")
                .and_then(|v| v.as_u64()),
            case.source_payload_metadata
                .get("cycles")
                .and_then(|v| v.as_u64()),
        ) {
            println!(
                "  payload cycles      {real} active + {padding} idle ({:.1}% padding)",
                100.0 * padding as f64 / total as f64
            );
        }
        if case.payload == "guest" {
            let data = &case.source_payload_metadata;
            if let (Some(mode), Some(source), Some(pattern), Some(full), Some(tail)) = (
                data.get("trace_mode").and_then(|v| v.as_str()),
                data.get("source_cycles").and_then(|v| v.as_u64()),
                data.get("pattern_cycles").and_then(|v| v.as_u64()),
                data.get("full_pattern_repetitions")
                    .and_then(|v| v.as_u64()),
                data.get("tail_pattern_cycles").and_then(|v| v.as_u64()),
            ) {
                println!(
                    "  guest pattern       {mode}: {source} VM cycles, {pattern} selected rows; {full} full copies + {tail} tail rows"
                );
            }
            let work = &case.work_normalization;
            println!(
                "  actual RAM work     {} memory + {} dummy rows ({:.2}% memory)",
                work.memory_access_cycles,
                work.dummy_access_cycles,
                100.0 * work.memory_access_fraction
            );
            if let Some(guest) = data.get("guest") {
                println!(
                    "  trace values        {} zero reads ({} dummy), {} unchanged writes, {} distinct words",
                    guest["zero_read_cycles"],
                    guest["dummy_zero_read_cycles"],
                    guest["unchanged_write_cycles"],
                    guest["distinct_accessed_words"]
                );
            }
        }
        println!("  prover total       {:10.3}", metric(|s| s.prover_ms));
        println!(
            "  virtualization     {:10.3}",
            metric(|s| s.virtualization_ms)
        );
        println!(
            "  virtual proof       {} sumchecks, {} rounds, {} field bytes ({} round elements + {} node evaluations)",
            case.virtualization_stats.sumchecks,
            case.virtualization_stats.sumcheck_rounds,
            case.virtualization_stats.field_payload_bytes,
            case.virtualization_stats.round_message_field_elements,
            case.virtualization_stats.node_evaluation_field_elements
        );
        if let Some(scale) = case.work_normalization.ms_to_ms_per_million_memory_accesses {
            println!(
                "  ms / million RAM   {:10.3}",
                metric(|s| s.prover_ms) * scale
            );
        }
        println!(
            "  commitment         {:10.3}  ({:.1}% median share)",
            metric(|s| s.commitment_ms),
            100.0 * metric(|s| s.commitment_fraction)
        );
        println!(
            "  everything else    {:10.3}",
            metric(|s| s.everything_else_ms)
        );
        println!("  verifier           {:10.3}", metric(|s| s.verifier_ms));
        let footprint = &case.proof_summary.commitment_footprint;
        let total_bits = case.proof_summary.committed_bits as f64;
        println!(
            "  committed witness  {:.1} KiB, {} populated / {} padded bits per cycle",
            total_bits / 8192.0,
            footprint.populated_bits_per_cycle,
            footprint.padded_bits_per_cycle
        );
        println!(
            "    column area: address encoding {:.2}%, values {:.2}%, increments {:.2}%, padding {:.2}%",
            100.0 * footprint.address_encoding_bits as f64 / total_bits,
            100.0 * footprint.read_write_value_bits as f64 / total_bits,
            100.0 * footprint.increment_bits as f64 / total_bits,
            100.0 * footprint.padding_bits as f64 / total_bits
        );
        if let Some(first) = case.samples.first() {
            for stage in first.stages_ms.keys() {
                let value = median(
                    case.samples
                        .iter()
                        .map(|s| s.stages_ms.get(stage).copied().unwrap_or(0.0))
                        .collect(),
                );
                println!("    {stage:26} {value:10.3}");
            }
        }
        println!(
            "    unattributed               {:10.3}",
            metric(|s| s.unattributed_ms)
        );
    }
    if !report.dummy_comparisons.is_empty() {
        println!(
            "\nEqual-RAM-work estimates: memory-only timings scaled by the exact useful-row ratio."
        );
        println!("These are proportional estimates; fixed per-proof costs do not scale linearly.");
    }
    for pair in &report.dummy_comparisons {
        println!(
            "\n{}/{}/{}: chunk={}, {} / {} useful rows, scale={:.6}",
            pair.guest_program.name(),
            pair.layout,
            pair.virtualization,
            pair.chunk_bits,
            pair.all_memory_access_cycles,
            pair.memory_only_memory_access_cycles,
            pair.memory_only_equal_work_scale
        );
        for (name, cost) in [
            ("prover", &pair.prover),
            ("commitment", &pair.commitment),
            ("everything else", &pair.everything_else),
        ]
        .into_iter()
        .chain(
            pair.stages
                .iter()
                .filter(|(name, _)| name.as_str() == "twist.rw.cycles")
                .map(|(name, cost)| (name.as_str(), cost)),
        ) {
            println!(
                "  {name:20} all {:9.3} ms; memory {:9.3} ms; scaled {:9.3} ms; estimated saving {:6.2}%",
                cost.all_ms,
                cost.memory_only_ms,
                cost.memory_only_equal_work_ms,
                100.0 * cost.estimated_saving_fraction
            );
        }
    }
    if !report.virtualization_comparisons.is_empty() {
        println!("\nMatched flat / GKR-tree comparison at the same RAM workload:");
    }
    for pair in &report.virtualization_comparisons {
        println!(
            "\n{}{}/{}/{}: chunk={}, d={}",
            pair.payload,
            pair.guest_program
                .map_or(String::new(), |p| format!("/{}", p.name())),
            pair.guest_rows.map_or("", row_name),
            pair.layout,
            pair.chunk_bits,
            pair.address_components
        );
        for (name, cost) in [
            ("prover", &pair.prover),
            ("virtualization", &pair.virtualization),
            ("verifier", &pair.verifier),
        ] {
            println!(
                "  {name:16} flat {:9.3} ms; tree {:9.3} ms; speedup {}; saving {}",
                cost.flat_ms,
                cost.gkr_tree_ms,
                cost.speedup.map_or("n/a".into(), |x| format!("{x:.3}x")),
                cost.saving_fraction
                    .map_or("n/a".into(), |x| format!("{:.2}%", 100.0 * x))
            );
        }
    }
}

fn save_csv(path: &PathBuf, report: &Report) -> std::io::Result<()> {
    let mut stages = std::collections::BTreeSet::new();
    for case in &report.cases {
        for sample in &case.samples {
            stages.extend(sample.stages_ms.keys().cloned());
        }
    }
    let mut text = "payload,guest_program,guest_rows,layout,virtualization,chunk_bits,address_components,iteration,memory_access_cycles,dummy_access_cycles,memory_access_fraction,ms_to_ms_per_million_memory_accesses,prover_ms,virtualization_ms,commitment_ms,everything_else_ms,verifier_ms,unattributed_ms".to_string();
    for stage in &stages {
        text.push(',');
        text.push_str(stage);
    }
    text.push('\n');
    for case in &report.cases {
        for (i, sample) in case.samples.iter().enumerate() {
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{:.9},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                case.payload,
                case.guest_program.map_or("", GuestProgram::name),
                case.guest_rows.map_or("", row_name),
                case.layout,
                case.virtualization,
                case.chunk_bits,
                case.address_components,
                i,
                case.work_normalization.memory_access_cycles,
                case.work_normalization.dummy_access_cycles,
                case.work_normalization.memory_access_fraction,
                case.work_normalization
                    .ms_to_ms_per_million_memory_accesses
                    .map(|scale| format!("{scale:.9}"))
                    .unwrap_or_default(),
                sample.prover_ms,
                sample.virtualization_ms,
                sample.commitment_ms,
                sample.everything_else_ms,
                sample.verifier_ms,
                sample.unattributed_ms
            ));
            for stage in &stages {
                text.push_str(&format!(
                    ",{:.6}",
                    sample.stages_ms.get(stage).copied().unwrap_or(0.0)
                ));
            }
            text.push('\n');
        }
    }
    fs::write(path, text)
}

fn main() {
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let Some(options) =
            options().map_err(|s| std::io::Error::new(std::io::ErrorKind::InvalidInput, s))?
        else {
            return Ok(());
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()?;
        let report = pool
            .install(|| run(&options).map_err(|e| e.to_string()))
            .map_err(std::io::Error::other)?;
        print_report(&report);
        if let Some(path) = &options.output {
            fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
        }
        if let Some(path) = &options.csv {
            save_csv(path, &report)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ram-bench: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(program: GuestProgram, rows: GuestRows, useful: usize, total_ms: f64) -> Case {
        Case {
            payload: "guest".into(),
            guest_program: Some(program),
            guest_rows: Some(rows),
            layout: "natural".into(),
            chunk_bits: 4,
            address_components: 4,
            virtualization: "flat".into(),
            virtualization_stats: VirtualizationStats::default(),
            payload_generation_ms: 0.0,
            source_payload_metadata: serde_json::Value::Null,
            address_relabeling: "identity".into(),
            work_normalization: WorkNormalization::new(1024, useful),
            proof_summary: ProofSummary::default(),
            samples: vec![Sample {
                prover_ms: total_ms,
                virtualization_ms: total_ms * 0.4,
                commitment_ms: total_ms * 0.2,
                everything_else_ms: total_ms * 0.8,
                commitment_fraction: 0.2,
                verifier_ms: 1.0,
                unattributed_ms: 0.0,
                stages_ms: BTreeMap::from([("twist.rw.cycles".into(), total_ms * 0.1)]),
            }],
        }
    }

    #[test]
    fn dummy_comparisons_use_exact_work_counts_and_match_programs() {
        let cases = vec![
            case(GuestProgram::PointerChase, GuestRows::All, 301, 40.0),
            case(GuestProgram::Erc20MixedBlock, GuestRows::Memory, 1024, 50.0),
            case(GuestProgram::PointerChase, GuestRows::Memory, 1024, 60.0),
            case(GuestProgram::Erc20MixedBlock, GuestRows::All, 512, 10.0),
        ];
        let comparisons = compare_dummy_cases(&cases);
        assert_eq!(comparisons.len(), 2);
        let pointer = &comparisons[0];
        assert_eq!(pointer.memory_only_equal_work_scale, 301.0 / 1024.0);
        assert_eq!(
            pointer.prover.memory_only_equal_work_ms,
            60.0 * 301.0 / 1024.0
        );
        assert_eq!(
            pointer.stages["twist.rw.cycles"].memory_only_equal_work_ms,
            6.0 * 301.0 / 1024.0
        );
        // Retain a negative estimate when the memory-only variant costs more per useful row.
        assert_eq!(comparisons[1].prover.estimated_saving_fraction, -1.5);
    }

    #[test]
    fn no_useful_rows_have_no_normalized_cost_or_comparison() {
        let normalization = WorkNormalization::new(1024, 0);
        assert_eq!(normalization.ms_to_ms_per_million_memory_accesses, None);
        assert_eq!(normalization.dummy_access_cycles, 1024);
        assert_eq!(normalization.memory_access_fraction, 0.0);
        let cases = vec![
            case(GuestProgram::BinarySearch, GuestRows::All, 0, 40.0),
            case(GuestProgram::BinarySearch, GuestRows::Memory, 1024, 60.0),
        ];
        assert!(compare_dummy_cases(&cases).is_empty());
    }

    #[test]
    fn mode_and_dummy_comparisons_keep_the_other_dimension_fixed() {
        let flat_all = case(GuestProgram::PointerChase, GuestRows::All, 301, 40.0);
        let flat_memory = case(GuestProgram::PointerChase, GuestRows::Memory, 1024, 60.0);
        let mut tree_all = case(GuestProgram::PointerChase, GuestRows::All, 301, 20.0);
        tree_all.virtualization = "gkr-tree".into();
        let mut tree_memory = case(GuestProgram::PointerChase, GuestRows::Memory, 1024, 180.0);
        tree_memory.virtualization = "gkr-tree".into();
        let cases = vec![flat_all, tree_memory, flat_memory, tree_all];
        let modes = compare_virtualization_cases(&cases);
        assert_eq!(modes.len(), 2);
        assert_eq!(modes[0].prover.speedup, Some(2.0));
        assert_eq!(modes[0].virtualization.speedup, Some(2.0));
        assert_eq!(modes[1].prover.saving_fraction, Some(-2.0));
        let dummy = compare_dummy_cases(&cases);
        assert_eq!(dummy.len(), 2);
        assert_eq!(dummy[0].virtualization, "flat");
        assert_eq!(dummy[0].prover.memory_only_ms, 60.0);
        assert_eq!(dummy[1].virtualization, "gkr-tree");
        assert_eq!(dummy[1].prover.memory_only_ms, 180.0);
    }

    #[test]
    fn virtualization_total_includes_exclusive_claim_and_product_children() {
        let stages = BTreeMap::from([
            ("twist.virtualization".into(), 2.0),
            ("virt.tree.build".into(), 3.0),
            ("virt.tree.layers".into(), 5.0),
            ("virt.flat.claims".into(), 11.0),
            ("virt.claim_reduction".into(), 13.0),
            ("virt.flat.product".into(), 17.0),
            ("twist.onehot".into(), 7.0),
        ]);
        assert_eq!(virtualization_time(&stages), 51.0);
    }
}
