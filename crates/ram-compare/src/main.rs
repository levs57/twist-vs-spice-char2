use om_transcript::Hasher;
use om_transcript::profiling;
use ram_argument::{AccessTrace, Config, Statement, Summary, VirtualizationMode};
use ram_payload::operations::{OperationPayload, guest_operations, random_operations};
use ram_payload::{GuestFill, GuestOptions, GuestProgram, GuestRows, RamPayload};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

mod platform;
use platform::{LoadWindow, load_average};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// The harness's own sources, hashed into every report so a result names the
/// exact comparison code that produced it.
const SOURCES: &[(&str, &[u8])] = &[
    ("crates/ram-compare/Cargo.toml", include_bytes!("../Cargo.toml")),
    ("crates/ram-compare/build.rs", include_bytes!("../build.rs")),
    ("crates/ram-compare/src/main.rs", include_bytes!("main.rs")),
    ("crates/ram-compare/src/platform.rs", include_bytes!("platform.rs")),
];
fn sources_sha256() -> String {
    let mut bytes = Vec::new();
    for (name, contents) in SOURCES {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&Hasher::sha256().digest(contents));
    }
    hex(&Hasher::sha256().digest(&bytes))
}
const HELP: &str = "Chronological shared-address RAM: BinaryTwist Flat versus ordered Spice
  --methods twist,spice,spice-ripple,spice-typed|both
                                  Ordered methods; both=twist,spice (default)
  --trace-format rmw|operations    Default rmw; spice-typed requires operations
  --patterns random,guest|both     Alias --payload; default both
  --address-bits A                 2^A public 64-bit words (default 12)
  --log-cycles N                   2^N rows in the chosen trace format (default 12)
  --chunk-bits C[,C...]             Twist Flat only; Spice runs once/pattern (default 4)
  --threads N --warmup N --repeat N (defaults 8,1,3)
  --seed N --profile NAME --log-rate N (defaults 1,provable-120,2)
  --guest-program NAME[,NAME...]   binary-search,erc20-mixed-block,pointer-chase,all
  --guest-rows all|memory|both      Ordered lists accepted (default all)
  --guest-fill repeat|halt          Default repeat; memory+halt is invalid
  --guest-size N --guest-operations N (defaults 64,256)
  --guest-keys N --guest-queries N  (defaults 64,16)
  --output PATH --csv PATH
Spice uses the Flock Boolean-circuit argument; spice-ripple selects its prior
Eq-factor ordering proof. Both have identical committed columns.
spice-typed uses one value plus explicit read/write flags. In operations mode,
all methods use the same T load/store rows; baseline reads write back their
old value and baseline writes acquire an internal old-value read.
Every warmup/measured proof is independently verified. Complete prover time
includes witness/link construction, public boundaries, packing, IOP and PCS.
Reports retain measured padded costs and separately show commitment-padding
estimates. Only pcs.commit is normalized; all other work stays fully charged.
Payload generation/result destruction are excluded. GF128 GP soundness depends
on T+N; the PCS profile is not a composed security or equal-security claim.
";

#[derive(Clone, Debug, Serialize)]
struct Options {
    methods: Vec<String>,
    trace_format: String,
    patterns: Vec<String>,
    address_bits: usize,
    log_cycles: usize,
    chunk_bits: Vec<usize>,
    threads: usize,
    warmup: usize,
    repeat: usize,
    seed: u64,
    profile: String,
    log_rate: usize,
    guest_program: Vec<String>,
    guest_rows: Vec<String>,
    guest_fill: String,
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
            methods: vec!["twist".into(), "spice".into()],
            trace_format: "rmw".into(),
            patterns: vec!["random".into(), "guest".into()],
            address_bits: 12,
            log_cycles: 12,
            chunk_bits: vec![4],
            threads: 8,
            warmup: 1,
            repeat: 3,
            seed: 1,
            profile: "provable-120".into(),
            log_rate: 2,
            guest_program: vec!["binary-search".into()],
            guest_rows: vec!["all".into()],
            guest_fill: "repeat".into(),
            guest_size: 64,
            guest_operations: 256,
            guest_keys: 64,
            guest_queries: 16,
            output: None,
            csv: None,
        }
    }
}
fn select(value: &str, allowed: &[&str], expansion: &[&str]) -> Result<Vec<String>> {
    let input = if value == "both" || value == "all" {
        expansion.to_vec()
    } else {
        value.split(',').collect()
    };
    let mut selected = Vec::new();
    for name in input {
        if !allowed.contains(&name) || selected.iter().any(|old| old == name) {
            return Err(format!("unsupported/duplicate selection: {name}").into());
        }
        selected.push(name.to_owned());
    }
    Ok(selected)
}
fn options() -> Result<Option<Options>> {
    let mut o = Options::default();
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
                .map_err(|_| format!("invalid {arg}: {value}"))
        };
        match arg.as_str() {
            "--methods" => {
                o.methods = select(
                    &value,
                    &["twist", "spice", "spice-ripple", "spice-typed"],
                    &["twist", "spice"],
                )?
            }
            "--trace-format" => o.trace_format = value,
            "--patterns" | "--payload" => {
                o.patterns = select(&value, &["random", "guest"], &["random", "guest"])?
            }
            "--address-bits" => o.address_bits = number()?,
            "--log-cycles" => o.log_cycles = number()?,
            "--chunk-bits" => {
                o.chunk_bits = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--threads" => o.threads = number()?,
            "--warmup" => o.warmup = number()?,
            "--repeat" => o.repeat = number()?,
            "--seed" => o.seed = value.parse()?,
            "--profile" => o.profile = value,
            "--log-rate" => o.log_rate = number()?,
            "--guest-program" => {
                o.guest_program = select(
                    &value,
                    &["binary-search", "erc20-mixed-block", "pointer-chase"],
                    &["binary-search", "erc20-mixed-block", "pointer-chase"],
                )?
            }
            "--guest-rows" => {
                o.guest_rows = select(&value, &["all", "memory"], &["all", "memory"])?
            }
            "--guest-fill" => o.guest_fill = value,
            "--guest-size" => o.guest_size = number()?,
            "--guest-operations" => o.guest_operations = number()?,
            "--guest-keys" => o.guest_keys = number()?,
            "--guest-queries" => o.guest_queries = number()?,
            "--output" => o.output = Some(value.into()),
            "--csv" => o.csv = Some(value.into()),
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    validate_options(&o)?;
    Ok(Some(o))
}

fn validate_options(o: &Options) -> Result<()> {
    if !["rmw", "operations"].contains(&o.trace_format.as_str()) {
        return Err("trace format must be rmw or operations".into());
    }
    if o.methods.iter().any(|m| m == "spice-typed") && o.trace_format != "operations" {
        return Err("spice-typed requires --trace-format operations".into());
    }
    if !(1..=31).contains(&o.address_bits) || o.log_cycles > 31 || o.threads == 0 || o.repeat == 0 {
        return Err("address bits must be 1..31; log cycles 0..31; threads/repeat positive".into());
    }
    let minimum = match o.profile.as_str() {
        "provable-120" => 2,
        "conjectured-100" => 1,
        _ => return Err("unsupported PCS profile".into()),
    };
    if !(minimum..=8).contains(&o.log_rate) {
        return Err("invalid log rate".into());
    }
    if o.chunk_bits.is_empty()
        || o.methods.iter().any(|m| m == "twist")
            && o.chunk_bits.iter().enumerate().any(|(i, &c)| {
                c == 0
                    || c > o.address_bits
                    || !o.address_bits.is_multiple_of(c)
                    || o.chunk_bits[..i].contains(&c)
            })
    {
        return Err("distinct positive chunk sizes must divide address bits".into());
    }
    if !["repeat", "halt"].contains(&o.guest_fill.as_str()) {
        return Err("guest fill must be repeat or halt".into());
    }
    if o.patterns.iter().any(|p| p == "guest")
        && (o.log_cycles < 2
            || o.guest_fill == "halt" && o.guest_rows.iter().any(|r| r == "memory"))
    {
        return Err(
            "guest requires at least four rows; memory filtering requires repeat fill".into(),
        );
    }
    Ok(())
}

fn program(name: &str) -> GuestProgram {
    match name {
        "binary-search" => GuestProgram::BinarySearch,
        "erc20-mixed-block" => GuestProgram::Erc20MixedBlock,
        "pointer-chase" => GuestProgram::PointerChase,
        _ => unreachable!(),
    }
}
fn workloads(o: &Options) -> Vec<(String, Option<(GuestProgram, GuestRows)>)> {
    let mut work = Vec::new();
    for pattern in &o.patterns {
        if pattern == "random" {
            work.push((pattern.clone(), None));
        } else {
            for p in &o.guest_program {
                for r in &o.guest_rows {
                    work.push((
                        format!("guest/{p}/{r}"),
                        Some((
                            program(p),
                            if r == "all" {
                                GuestRows::All
                            } else {
                                GuestRows::Memory
                            },
                        )),
                    ));
                }
            }
        }
    }
    work
}
fn guest_options(o: &Options, program: GuestProgram, rows: GuestRows) -> GuestOptions {
    GuestOptions {
        program,
        rows,
        size: o.guest_size,
        operations: o.guest_operations,
        keys: o.guest_keys,
        queries: o.guest_queries,
        seed: o.seed,
        max_cycles: 1 << 20,
        fill: if o.guest_fill == "repeat" {
            GuestFill::Repeat
        } else {
            GuestFill::Halt
        },
    }
}

fn payload(o: &Options, guest: Option<(GuestProgram, GuestRows)>) -> Result<RamPayload> {
    Ok(if let Some((program, rows)) = guest {
        ram_payload::guest(
            o.log_cycles as u32,
            o.address_bits as u32,
            &guest_options(o, program, rows),
        )?
    } else {
        ram_payload::random_ram(o.log_cycles as u32, o.address_bits as u32, o.seed)?
    })
}
fn operation_payload(
    o: &Options,
    guest: Option<(GuestProgram, GuestRows)>,
) -> Result<OperationPayload> {
    Ok(if let Some((program, rows)) = guest {
        guest_operations(
            o.log_cycles as u32,
            o.address_bits as u32,
            &guest_options(o, program, rows),
        )?
    } else {
        random_operations(o.log_cycles as u32, o.address_bits as u32, o.seed)?
    })
}
fn configurations(o: &Options) -> Vec<(String, Option<usize>)> {
    o.methods
        .iter()
        .flat_map(|method| {
            if method.starts_with("spice") {
                vec![(method.clone(), None)]
            } else {
                o.chunk_bits
                    .iter()
                    .map(|&c| (method.clone(), Some(c)))
                    .collect()
            }
        })
        .collect()
}
fn ordering_name(method: &str) -> Option<&'static str> {
    match method {
        "twist" => None,
        "spice" => Some("flock-boolean"),
        "spice-ripple" => Some("ripple-eq-factor"),
        "spice-typed" => Some("single-value-flock"),
        _ => unreachable!("validated method"),
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn trace_hash(trace: &AccessTrace) -> String {
    let mut bytes = Vec::new();
    for size in [trace.initial.len(), trace.writes.len(), trace.reads.len()] {
        bytes.extend_from_slice(&(size as u64).to_le_bytes());
    }
    for &v in &trace.initial {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for &v in trace.reads.iter().flatten().chain(&trace.writes) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for &v in trace
        .new_values
        .iter()
        .chain(trace.read_values.iter().flatten().flatten())
    {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    hex(&Hasher::sha256().digest(&bytes))
}

fn operation_trace_hash(trace: &OperationPayload) -> String {
    let mut bytes = b"ram-compare/operations/v1".to_vec();
    for size in [trace.initial_memory.len(), trace.addresses.len()] {
        bytes.extend_from_slice(&(size as u64).to_le_bytes());
    }
    for &value in &trace.initial_memory {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for ((&address, &value), &write) in trace
        .addresses
        .iter()
        .zip(&trace.values)
        .zip(&trace.is_write)
    {
        bytes.extend_from_slice(&address.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes.push(u8::from(write));
    }
    hex(&Hasher::sha256().digest(&bytes))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Component {
    name: String,
    log_size: usize,
    bits: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Area {
    cycles: usize,
    memory_cells: usize,
    populated_bits: usize,
    committed_bits: usize,
    padding_bits: usize,
    populated_fraction: f64,
    packed_field_elements: usize,
    mosaic_claims: usize,
    components: Vec<Component>,
}
impl Area {
    fn new(summary: &Summary, method: &str, o: &Options) -> Result<Self> {
        let components: Vec<_> = summary
            .commitment_components
            .iter()
            .map(|c| Component {
                name: c.name.clone(),
                log_size: c.log_size,
                bits: 1usize << c.log_size,
            })
            .collect();
        let populated_bits = components.iter().map(|c| c.bits).sum::<usize>();
        if method.starts_with("spice") {
            let b = o.address_bits.max(o.log_cycles) + 1;
            let (cycle_bits, boundary_bits) = if method == "spice-typed" {
                (64 + o.address_bits + 2 * b + 1, b + 1)
            } else {
                (128 + o.address_bits + 2 * b - 1, b)
            };
            let expected =
                cycle_bits * (1usize << o.log_cycles) + boundary_bits * (1usize << o.address_bits);
            if populated_bits != expected {
                return Err("Spice component area differs from predecessor/carry formula".into());
            }
        }
        if populated_bits > summary.committed_bits {
            return Err("components exceed commitment area".into());
        }
        Ok(Self {
            cycles: summary.cycles,
            memory_cells: summary.memory_cells,
            populated_bits,
            committed_bits: summary.committed_bits,
            padding_bits: summary.committed_bits - populated_bits,
            populated_fraction: populated_bits as f64 / summary.committed_bits as f64,
            packed_field_elements: summary.packed_field_elements,
            mosaic_claims: summary.mosaic_claims,
            components,
        })
    }
}
/// Commitment-only linear normalization, not a measured proof over an
/// unpadded domain. All non-commitment work remains fully charged.
#[derive(Debug, Serialize)]
struct PaddingAdjustedEstimate {
    prover_ms: f64,
    commit_ms: f64,
    removed_padding_ms: f64,
}
const ADJUSTED_TIMES: [&str; 3] = ["prover_ms", "commit_ms", "removed_padding_ms"];
impl PaddingAdjustedEstimate {
    fn new(prover_ms: f64, commit_ms: f64, fraction: f64) -> Self {
        let adjusted_commit_ms = commit_ms * fraction;
        let removed_padding_ms = commit_ms - adjusted_commit_ms;
        Self {
            prover_ms: prover_ms - removed_padding_ms,
            commit_ms: adjusted_commit_ms,
            removed_padding_ms,
        }
    }
    fn times(&self) -> [f64; 3] {
        [self.prover_ms, self.commit_ms, self.removed_padding_ms]
    }
    fn medians(samples: &[Sample]) -> Self {
        let values: Vec<_> = (0..ADJUSTED_TIMES.len())
            .map(|i| {
                median(
                    samples
                        .iter()
                        .map(|s| s.padding_adjusted_estimate.times()[i])
                        .collect(),
                )
            })
            .collect();
        Self {
            prover_ms: values[0],
            commit_ms: values[1],
            removed_padding_ms: values[2],
        }
    }
}
#[derive(Debug, Serialize)]
struct Sample {
    iteration: usize,
    prover_ms: f64,
    verifier_ms: f64,
    commit_ms: f64,
    opening_ms: f64,
    iop_and_prep_ms: f64,
    preparation_ms: f64,
    argument_ms: f64,
    ordering_ms: f64,
    mask_ms: f64,
    mosaic_ms: f64,
    unattributed_ms: f64,
    stages_ms: BTreeMap<String, f64>,
    padding_adjusted_estimate: PaddingAdjustedEstimate,
    /// Exclusive verifier spans: the profiler is cleared before `verify`
    /// and read right after it. Empty when the verifier has no spans.
    verifier_stages_ms: BTreeMap<String, f64>,
    /// Load averages immediately before proving and after verifying.
    load: LoadWindow,
}
const TIMES: [&str; 12] = [
    "prover_ms",
    "verifier_ms",
    "commit_ms",
    "opening_ms",
    "iop_and_prep_ms",
    "preparation_ms",
    "argument_ms",
    "ordering_ms",
    "mask_ms",
    "mosaic_ms",
    "unattributed_ms",
    "exclusive_total_ms",
];
impl Sample {
    fn new(
        iteration: usize,
        prover_ms: f64,
        verifier_ms: f64,
        metrics: Vec<(&'static str, Duration)>,
        populated_fraction: f64,
    ) -> Result<Self> {
        if !populated_fraction.is_finite() || populated_fraction <= 0.0 || populated_fraction > 1.0
        {
            return Err("invalid populated/committed fraction".into());
        }
        let stages_ms = stage_map(metrics);
        let get = |name| stages_ms.get(name).copied().unwrap_or(0.0);
        let commit_ms = get("pcs.commit");
        let opening_ms = get("pcs.open") + get("pcs.ringswitch") + get("pcs.ligerito_open");
        if !stages_ms.contains_key("pcs.commit") || !stages_ms.contains_key("pcs.open") {
            return Err("missing commit/open profiling spans".into());
        }
        let preparation_ms = stages_ms
            .iter()
            .filter(|(name, _)| {
                [
                    "ram.witness",
                    "ram.pack",
                    "ram.statement",
                    "ram.claims",
                    "spice.witness",
                    "spice.pack",
                    "spice.statement",
                    "spice.claims",
                ]
                .contains(&name.as_str())
            })
            .map(|(_, v)| v)
            .sum::<f64>();
        let total = stages_ms.values().sum::<f64>();
        let unattributed_ms = prover_ms - total;
        let argument_ms = total - commit_ms - opening_ms - preparation_ms;
        let ordering_ms = stages_ms
            .iter()
            .filter(|(name, _)| name.starts_with("spice.order"))
            .map(|(_, &value)| value)
            .sum();
        let mask_ms = stages_ms
            .iter()
            .filter(|(name, _)| name.starts_with("spice.mask"))
            .map(|(_, &value)| value)
            .sum();
        let mosaic_ms = stages_ms
            .iter()
            .filter(|(name, _)| name.starts_with("mosaic."))
            .map(|(_, &value)| value)
            .sum();
        if unattributed_ms < -1e-6
            || argument_ms < -1e-6
            || prover_ms - commit_ms - opening_ms < -1e-6
        {
            return Err("inconsistent exclusive timing partition".into());
        }
        let padding_adjusted_estimate =
            PaddingAdjustedEstimate::new(prover_ms, commit_ms, populated_fraction);
        Ok(Self {
            iteration,
            prover_ms,
            verifier_ms,
            commit_ms,
            opening_ms,
            iop_and_prep_ms: prover_ms - commit_ms - opening_ms,
            preparation_ms,
            argument_ms,
            ordering_ms,
            mask_ms,
            mosaic_ms,
            unattributed_ms,
            stages_ms,
            padding_adjusted_estimate,
            verifier_stages_ms: BTreeMap::new(),
            load: LoadWindow {
                start: None,
                end: None,
            },
        })
    }
    fn times(&self) -> [f64; 12] {
        [
            self.prover_ms,
            self.verifier_ms,
            self.commit_ms,
            self.opening_ms,
            self.iop_and_prep_ms,
            self.preparation_ms,
            self.argument_ms,
            self.ordering_ms,
            self.mask_ms,
            self.mosaic_ms,
            self.unattributed_ms,
            self.stages_ms.values().sum(),
        ]
    }
}
/// Sums exclusive span durations by name, in milliseconds.
fn stage_map(metrics: Vec<(&'static str, Duration)>) -> BTreeMap<String, f64> {
    let mut stages = BTreeMap::new();
    for (name, duration) in metrics {
        *stages.entry(name.to_owned()).or_insert(0.0) += duration.as_secs_f64() * 1000.0;
    }
    stages
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[m - 1] + values[m]) / 2.0
    } else {
        values[m]
    }
}
#[derive(Debug, Serialize)]
struct Medians {
    #[serde(flatten)]
    times: BTreeMap<String, f64>,
    stages_ms: BTreeMap<String, f64>,
    verifier_stages_ms: BTreeMap<String, f64>,
    padding_adjusted_estimate: PaddingAdjustedEstimate,
}
fn stage_medians(
    samples: &[Sample],
    get: impl Fn(&Sample) -> &BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let mut stages = BTreeMap::new();
    for sample in samples {
        for name in get(sample).keys() {
            stages.insert(name.clone(), 0.0);
        }
    }
    for (name, value) in &mut stages {
        *value = median(
            samples
                .iter()
                .map(|s| get(s).get(name).copied().unwrap_or(0.0))
                .collect(),
        );
    }
    stages
}
impl Medians {
    fn new(samples: &[Sample]) -> Self {
        let times = TIMES
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                (
                    name.into(),
                    median(samples.iter().map(|s| s.times()[i]).collect()),
                )
            })
            .collect();
        Self {
            times,
            stages_ms: stage_medians(samples, |s| &s.stages_ms),
            verifier_stages_ms: stage_medians(samples, |s| &s.verifier_stages_ms),
            padding_adjusted_estimate: PaddingAdjustedEstimate::medians(samples),
        }
    }
}
/// Times one verification and collects the verifier's own profiling spans.
/// The profiler is cleared first so no prover span leaks into the result.
fn timed_verify(
    verify: impl FnOnce() -> Result<()>,
) -> Result<(f64, Vec<(&'static str, Duration)>)> {
    profiling::clear();
    let start = Instant::now();
    verify()?;
    let verifier_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((verifier_ms, profiling::take()))
}
fn observed(
    iteration: usize,
    prover_ms: f64,
    verifier_ms: f64,
    verifier_metrics: Vec<(&'static str, Duration)>,
    statement: &Statement,
    metrics: Vec<(&'static str, Duration)>,
    summary: &Summary,
    trace: &AccessTrace,
    method: &str,
    o: &Options,
) -> Result<(Sample, Area, Vec<u64>)> {
    if statement.initial_memory != trace.initial || statement.log_cycles != o.log_cycles {
        return Err("prover returned a different public input".into());
    }
    let area = Area::new(summary, method, o)?;
    let mut sample = Sample::new(
        iteration,
        prover_ms,
        verifier_ms,
        metrics,
        area.populated_fraction,
    )?;
    sample.verifier_stages_ms = stage_map(verifier_metrics);
    Ok((
        sample,
        area,
        statement.final_memory.clone(),
    ))
}
fn sample(
    method: &str,
    config: &Config,
    trace: &AccessTrace,
    operation_trace: Option<&spice_argument::operations::AccessTrace>,
    iteration: usize,
    o: &Options,
) -> Result<(Sample, Area, Vec<u64>)> {
    let start = load_average();
    let (mut sample, area, final_memory) =
        measure(method, config, trace, operation_trace, iteration, o)?;
    sample.load = LoadWindow {
        start,
        end: load_average(),
    };
    Ok((sample, area, final_memory))
}
fn measure(
    method: &str,
    config: &Config,
    trace: &AccessTrace,
    operation_trace: Option<&spice_argument::operations::AccessTrace>,
    iteration: usize,
    o: &Options,
) -> Result<(Sample, Area, Vec<u64>)> {
    let start = Instant::now();
    if method == "twist" {
        let result = ram_argument::prove(black_box(config), black_box(trace))?;
        let prover_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (verifier_ms, verifier_metrics) = timed_verify(|| {
            ram_argument::verify(black_box(&result.statement), black_box(&result.proof))?;
            Ok(())
        })?;
        observed(
            iteration,
            prover_ms,
            verifier_ms,
            verifier_metrics,
            &result.statement,
            result.metrics,
            &result.summary,
            trace,
            method,
            o,
        )
    } else if method == "spice-typed" {
        let operation_trace = operation_trace.ok_or("spice-typed requires an operation trace")?;
        let result =
            spice_argument::operations::prove(black_box(config), black_box(operation_trace))?;
        let prover_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (verifier_ms, verifier_metrics) = timed_verify(|| {
            spice_argument::operations::verify(black_box(&result.statement), black_box(&result.proof))?;
            Ok(())
        })?;
        observed(
            iteration,
            prover_ms,
            verifier_ms,
            verifier_metrics,
            &result.statement,
            result.metrics,
            &result.summary,
            trace,
            method,
            o,
        )
    } else {
        let result = if method == "spice-ripple" {
            spice_argument::prove_with_ordering(
                black_box(config),
                black_box(trace),
                spice_argument::OrderingMode::Ripple,
            )?
        } else {
            spice_argument::prove(black_box(config), black_box(trace))?
        };
        let prover_ms = start.elapsed().as_secs_f64() * 1000.0;
        let (verifier_ms, verifier_metrics) = timed_verify(|| {
            spice_argument::verify(black_box(&result.statement), black_box(&result.proof))?;
            Ok(())
        })?;
        observed(
            iteration,
            prover_ms,
            verifier_ms,
            verifier_metrics,
            &result.statement,
            result.metrics,
            &result.summary,
            trace,
            method,
            o,
        )
    }
}
#[derive(Debug, Serialize)]
struct Case {
    method: String,
    ordering: Option<String>,
    pattern: String,
    trace_format: String,
    chunk_bits: Option<usize>,
    trace_sha256: String,
    operation_trace_sha256: Option<String>,
    metadata: serde_json::Value,
    /// Load averages at the start and end of this case's measurements.
    load: LoadWindow,
    area: Area,
    grand_product_extra_bound: Option<serde_json::Value>,
    samples: Vec<Sample>,
    median: Medians,
}
fn fmt_load(load: Option<platform::LoadAverage>) -> String {
    load.map_or("n/a".into(), |l| format!("{:.2}/{:.2}/{:.2}", l.one, l.five, l.fifteen))
}
fn benchmark(o: &Options) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    for (pattern, guest) in workloads(o) {
        let (pattern, data, operation_trace, operation_digest, metadata, expected_final) =
            if o.trace_format == "operations" {
                let data = operation_payload(o, guest)?;
                let digest = operation_trace_hash(&data);
                let final_memory = data.replay()?;
                let metadata = serde_json::to_value(&data.metadata)?;
                let rmw = data.clone().into_rmw()?;
                let typed = spice_argument::operations::AccessTrace {
                    initial: data.initial_memory,
                    addresses: data.addresses,
                    values: data.values,
                    is_write: data.is_write,
                };
                (
                    format!("{pattern}/operations"),
                    rmw,
                    Some(typed),
                    Some(digest),
                    metadata,
                    Some(final_memory),
                )
            } else {
                let data = payload(o, guest)?;
                let metadata = serde_json::to_value(&data.metadata)?;
                (pattern, data, None, None, metadata, None)
            };
        let trace = AccessTrace {
            initial: data.initial_memory,
            reads: data.reads,
            writes: data.writes,
            new_values: data.new_values,
            read_values: Some(data.read_values),
        };
        let digest = trace_hash(&trace);
        let mut expected_final = expected_final;
        for (method, chunk_bits) in configurations(o) {
            let config = Config {
                address_bits: o.address_bits,
                chunk_bits: chunk_bits.unwrap_or(o.chunk_bits[0]),
                virtualization: VirtualizationMode::Flat,
                read_ports: 1,
                shared_address: true,
                word_log_bits: 6,
                profile: o.profile.clone(),
                log_rate: o.log_rate,
            };
            eprintln!(
                "{pattern}/{method}/chunk={chunk_bits:?}: T=2^{} N=2^{} threads={}",
                o.log_cycles, o.address_bits, o.threads
            );
            let case_start = load_average();
            let mut area = None;
            let mut samples = Vec::new();
            for i in 0..o.warmup + o.repeat {
                let (sample, current, final_memory) = sample(
                    &method,
                    &config,
                    &trace,
                    operation_trace.as_ref(),
                    i.saturating_sub(o.warmup),
                    o,
                )?;
                if expected_final
                    .as_ref()
                    .is_some_and(|old| old != &final_memory)
                {
                    return Err("methods/repetitions disagree on public final memory".into());
                }
                expected_final = Some(final_memory);
                if area.as_ref().is_some_and(|old| old != &current) {
                    return Err("commitment summary changed across proofs".into());
                }
                area = Some(current);
                if i >= o.warmup {
                    samples.push(sample);
                }
            }
            let load = LoadWindow {
                start: case_start,
                end: load_average(),
            };
            let median = Medians::new(&samples);
            let area = area.ok_or("missing proof")?;
            println!(
                "{pattern} {method} chunk={chunk_bits:?}: prove={:.3}ms verify={:.3}ms commit={:.3}ms open={:.3}ms IOP+prep={:.3}ms bits={}/{} populated/padded",
                median.times["prover_ms"],
                median.times["verifier_ms"],
                median.times["commit_ms"],
                median.times["opening_ms"],
                median.times["iop_and_prep_ms"],
                area.populated_bits,
                area.committed_bits
            );
            println!(
                "  preparation={:.3}ms argument={:.3}ms ordering(total)={:.3}ms mask(total)={:.3}ms mosaic(total)={:.3}ms unattributed={:.3}ms (independent medians)",
                median.times["preparation_ms"],
                median.times["argument_ms"],
                median.times["ordering_ms"],
                median.times["mask_ms"],
                median.times["mosaic_ms"],
                median.times["unattributed_ms"]
            );
            println!("  load(1/5/15m) start={} end={}", fmt_load(load.start), fmt_load(load.end));
            let adjusted = &median.padding_adjusted_estimate;
            println!(
                "  commitment-padding-adjusted ESTIMATE (populated/padded={:.4}): prove={:.3}ms commit={:.3}ms; modeled commitment padding={:.3}ms; all other costs unchanged",
                area.populated_fraction,
                adjusted.prover_ms,
                adjusted.commit_ms,
                adjusted.removed_padding_ms,
            );
            for (name, value) in &median.stages_ms {
                println!("    {name:<38} {value:>10.3}ms");
            }
            if !median.verifier_stages_ms.is_empty() {
                println!("  verifier stages (exclusive medians):");
                for (name, value) in &median.verifier_stages_ms {
                    println!("    {name:<38} {value:>10.3}ms");
                }
            }
            let l = area.cycles + area.memory_cells;
            if method.starts_with("spice") {
                println!(
                    "  fingerprint_bound_bits≈{:.3} (GP term only; excludes other checks)",
                    128.0 - (l as f64).log2()
                );
            }
            let bound = method.starts_with("spice").then(|| serde_json::json!({"field_bits":128,"degree_numerator":l,"probability_bound":(l as f64)*2f64.powi(-128),"fingerprint_bound_bits":128.0-(l as f64).log2(),"scope":"GP fingerprint term only; excludes order/mask/GKR/batching/PCS/hash errors; not an equal-security claim"}));
            let ordering = ordering_name(&method).map(str::to_owned);
            cases.push(Case {
                method,
                ordering,
                pattern: pattern.clone(),
                trace_format: o.trace_format.clone(),
                chunk_bits,
                trace_sha256: digest.clone(),
                operation_trace_sha256: operation_digest.clone(),
                metadata: metadata.clone(),
                load,
                area,
                grand_product_extra_bound: bound,
                samples,
                median,
            });
        }
    }
    Ok(cases)
}
#[derive(Debug, Serialize)]
struct Pair {
    twist_case: usize,
    spice_case: usize,
    spice_method: String,
    trace_sha256: String,
    operation_trace_sha256: Option<String>,
    spice_prover_speedup: f64,
    spice_prover_speedup_padding_adjusted_estimate: f64,
    spice_verifier_speedup: f64,
    twist_to_spice_populated_ratio: f64,
    twist_to_spice_committed_ratio: f64,
}
fn pairs(cases: &[Case]) -> Vec<Pair> {
    cases
        .iter()
        .enumerate()
        .filter(|(_, c)| c.method == "twist")
        .flat_map(|(t, twist)| {
            cases
                .iter()
                .enumerate()
                .filter(move |(_, c)| {
                    c.method.starts_with("spice")
                        && c.pattern == twist.pattern
                        && c.trace_format == twist.trace_format
                        && c.trace_sha256 == twist.trace_sha256
                        && c.operation_trace_sha256 == twist.operation_trace_sha256
                })
                .map(move |(s, spice)| Pair {
                    twist_case: t,
                    spice_case: s,
                    spice_method: spice.method.clone(),
                    trace_sha256: twist.trace_sha256.clone(),
                    operation_trace_sha256: twist.operation_trace_sha256.clone(),
                    spice_prover_speedup: twist.median.times["prover_ms"]
                        / spice.median.times["prover_ms"],
                    spice_prover_speedup_padding_adjusted_estimate: twist
                        .median
                        .padding_adjusted_estimate
                        .prover_ms
                        / spice.median.padding_adjusted_estimate.prover_ms,
                    spice_verifier_speedup: twist.median.times["verifier_ms"]
                        / spice.median.times["verifier_ms"],
                    twist_to_spice_populated_ratio: twist.area.populated_bits as f64
                        / spice.area.populated_bits as f64,
                    twist_to_spice_committed_ratio: twist.area.committed_bits as f64
                        / spice.area.committed_bits as f64,
                })
        })
        .collect()
}
#[derive(Serialize)]
struct Report {
    schema_version: usize,
    implementation: String,
    options: Options,
    environment: serde_json::Value,
    provenance: serde_json::Value,
    notes: Vec<String>,
    cases: Vec<Case>,
    comparisons: Vec<Pair>,
    verified_proofs: usize,
}
fn escaped(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}
fn csv(report: &Report) -> String {
    let mut stages = BTreeMap::new();
    let mut verifier_stages = BTreeMap::new();
    for c in &report.cases {
        for s in &c.samples {
            for name in s.stages_ms.keys() {
                stages.insert(name.clone(), ());
            }
            for name in s.verifier_stages_ms.keys() {
                verifier_stages.insert(name.clone(), ());
            }
        }
    }
    let mut header: Vec<_> = [
        "case",
        "method",
        "ordering",
        "pattern",
        "trace_format",
        "chunk_bits",
        "trace_sha256",
        "operation_trace_sha256",
        "iteration",
        "address_bits",
        "log_cycles",
        "threads",
        "populated_bits",
        "committed_bits",
        "padding_bits",
        "populated_fraction",
        "mosaic_claims",
        "load1_start",
        "load1_end",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    header.extend(TIMES.into_iter().map(str::to_owned));
    header.extend(
        ADJUSTED_TIMES
            .into_iter()
            .map(|name| format!("padding_adjusted_estimate:{name}")),
    );
    header.extend(stages.keys().map(|s| format!("exclusive:{s}")));
    header.extend(verifier_stages.keys().map(|s| format!("verifier:{s}")));
    let mut output = header
        .iter()
        .map(|s| escaped(s))
        .collect::<Vec<_>>()
        .join(",")
        + "\n";
    for (i, c) in report.cases.iter().enumerate() {
        for s in &c.samples {
            let mut row = vec![
                i.to_string(),
                c.method.clone(),
                c.ordering.clone().unwrap_or_default(),
                c.pattern.clone(),
                c.trace_format.clone(),
                c.chunk_bits.map(|v| v.to_string()).unwrap_or_default(),
                c.trace_sha256.clone(),
                c.operation_trace_sha256.clone().unwrap_or_default(),
                s.iteration.to_string(),
                report.options.address_bits.to_string(),
                report.options.log_cycles.to_string(),
                report.options.threads.to_string(),
                c.area.populated_bits.to_string(),
                c.area.committed_bits.to_string(),
                c.area.padding_bits.to_string(),
                format!("{:.12}", c.area.populated_fraction),
            ];
            row.push(c.area.mosaic_claims.to_string());
            for load in [s.load.start, s.load.end] {
                row.push(load.map(|l| format!("{:.2}", l.one)).unwrap_or_default());
            }
            row.extend(s.times().into_iter().map(|v| format!("{v:.9}")));
            row.extend(
                s.padding_adjusted_estimate
                    .times()
                    .into_iter()
                    .map(|v| format!("{v:.9}")),
            );
            row.extend(
                stages
                    .keys()
                    .map(|name| format!("{:.9}", s.stages_ms.get(name).copied().unwrap_or(0.0))),
            );
            row.extend(verifier_stages.keys().map(|name| {
                format!(
                    "{:.9}",
                    s.verifier_stages_ms.get(name).copied().unwrap_or(0.0)
                )
            }));
            output.push_str(&row.iter().map(|s| escaped(s)).collect::<Vec<_>>().join(","));
            output.push('\n');
        }
    }
    output
}
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}
fn main() -> Result<()> {
    let Some(o) = options()? else {
        return Ok(());
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(o.threads)
        .build()?;
    let cases = pool.install(|| benchmark(&o).map_err(|e| e.to_string()))?;
    let comparisons = pairs(&cases);
    let verified_proofs = cases.len() * (o.warmup + o.repeat);
    let environment = platform::environment(
        hex(&Hasher::sha256().digest(&fs::read(env::current_exe()?)?)),
        sources_sha256(),
    );
    let provenance = serde_json::json!({
        "open_magnetar":serde_json::from_str::<serde_json::Value>(include_str!("../../../vendor/open-magnetar/UPSTREAM.json"))?,
        "lean_grandproduct":serde_json::from_str::<serde_json::Value>(include_str!("../../../vendor/lean-grandproduct/UPSTREAM.json"))?,
        "flock_boolean":serde_json::from_str::<serde_json::Value>(include_str!("../../../vendor/flock-boolean/UPSTREAM.json"))?,
        "adapter":serde_json::from_str::<serde_json::Value>(include_str!("../../../audit/stage14-implementation.json"))?,
        "typed_adapter":serde_json::from_str::<serde_json::Value>(include_str!("../../../audit/stage15-implementation.json"))?,
    });
    let report = Report { schema_version:4,implementation:"typed-spice-vs-twist-gf128-v4".into(),options:o,environment,provenance,notes:vec!["Every warmup/measured proof independently verified; identical initial/access trace and exact final memory checked across methods.".into(),"Operations mode preserves explicit load/store kinds and exactly T operation rows. All baselines use the same rows after replay conversion; reads write back their old word and writes acquire an internal old-value read. It does not infer kinds from equal values. Operation flags and values are included in operation_trace_sha256.".into(),"Inclusive commit/open and Mosaic totals are reconstructed from the exclusive span partition; IOP+prep includes preparation, argument and unattributed time. Derived actual/estimated sample totals are medianed independently.".into(),"Padding-adjusted values model commitment padding only: r=populated_bits/committed_bits multiplies measured pcs.commit; the estimated prover total subtracts only that modeled commitment saving. These are linear cost estimates, not measured unpadded proofs or achievable speedups. Fixed overhead and domain-size-dependent logarithmic commitment work are approximated. Opening, Mosaic, ordering, other RAM IOP, packing, public boundaries, verification and unattributed time remain fully charged. Actual padded timings and all exclusive stages are preserved.".into(),"Spice commits access/final predecessors and ripple carries. Typed Spice commits one value, is-write and next-is-write flags, with an additional initial next-is-write bitplane; mask reduction and actual component/Mosaic padding costs are included. Initial/final memory are public and transcript-bound.".into(),"Current GP field is GF128. Its (T+N)/2^128 term and other errors prevent inferring equal/composed 120-bit security from the inherited PCS profile. Larger-field GP requires a separately authenticated extension opening adapter.".into()],cases,comparisons,verified_proofs };
    if let Some(path) = &report.options.output {
        write(path, &serde_json::to_vec_pretty(&report)?)?;
        println!("JSON: {}", path.display());
    }
    if let Some(path) = &report.options.csv {
        write(path, csv(&report).as_bytes())?;
        println!("CSV: {}", path.display());
    }
    for pair in &report.comparisons {
        let twist = &report.cases[pair.twist_case];
        println!(
            "{} Twist({:?})/{} ratios: prover={:.3}× commitment-padding-adjusted ESTIMATE={:.3}× verifier={:.3}× populated={:.3}× committed={:.3}×",
            twist.pattern,
            twist.chunk_bits,
            pair.spice_method,
            pair.spice_prover_speedup,
            pair.spice_prover_speedup_padding_adjusted_estimate,
            pair.spice_verifier_speedup,
            pair.twist_to_spice_populated_ratio,
            pair.twist_to_spice_committed_ratio
        );
    }
    println!(
        "Verified {verified_proofs} proofs across {} cases, including warmups.",
        report.cases.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn method_order_preserved_spice_not_repeated_by_chunks() {
        let o = Options {
            methods: vec!["spice".into(), "twist".into()],
            chunk_bits: vec![4, 2, 1],
            ..Default::default()
        };
        assert_eq!(
            configurations(&o),
            vec![
                ("spice".into(), None),
                ("twist".into(), Some(4)),
                ("twist".into(), Some(2)),
                ("twist".into(), Some(1))
            ]
        );
    }
    #[test]
    fn ripple_baseline_has_one_canonical_case_and_an_explicit_ordering_id() {
        let o = Options {
            methods: vec!["spice-ripple".into(), "twist".into(), "spice".into()],
            chunk_bits: vec![2, 1],
            ..Default::default()
        };
        assert_eq!(
            configurations(&o),
            vec![
                ("spice-ripple".into(), None),
                ("twist".into(), Some(2)),
                ("twist".into(), Some(1)),
                ("spice".into(), None)
            ]
        );
        assert_eq!(ordering_name("twist"), None);
        assert_eq!(ordering_name("spice"), Some("flock-boolean"));
        assert_eq!(ordering_name("spice-ripple"), Some("ripple-eq-factor"));
        assert!(
            select(
                "spice,spice-ripple",
                &["twist", "spice", "spice-ripple"],
                &["twist", "spice"]
            )
            .is_ok()
        );
    }
    #[test]
    fn nested_pcs_spans_form_disjoint_inclusive_accounting() {
        let metrics = vec![
            ("pcs.commit", 10),
            ("pcs.open", 1),
            ("pcs.ringswitch", 2),
            ("pcs.ligerito_open", 3),
            ("spice.pack", 4),
            ("spice.order", 5),
            ("spice.permutation", 6),
        ];
        let sample = Sample::new(
            0,
            40.0,
            1.0,
            metrics
                .into_iter()
                .map(|(n, v)| (n, Duration::from_millis(v)))
                .collect(),
            1.0,
        )
        .unwrap();
        assert_eq!(sample.commit_ms, 10.0);
        assert_eq!(sample.opening_ms, 6.0);
        assert_eq!(sample.preparation_ms, 4.0);
        assert_eq!(sample.argument_ms, 11.0);
        assert_eq!(sample.ordering_ms, 5.0);
        assert_eq!(sample.unattributed_ms, 9.0);
        assert_eq!(sample.iop_and_prep_ms, 24.0);
    }
    #[test]
    fn total_ordering_is_medianed_per_sample_including_children() {
        let sample = |parent, pack| {
            Sample::new(
                0,
                100.0,
                1.0,
                vec![
                    ("pcs.commit", Duration::from_millis(1)),
                    ("pcs.open", Duration::from_millis(1)),
                    ("spice.order", Duration::from_millis(parent)),
                    ("spice.order.pack", Duration::from_millis(pack)),
                ],
                1.0,
            )
            .unwrap()
        };
        let medians = Medians::new(&[sample(1, 9), sample(9, 1), sample(9, 9)]);
        assert_eq!(medians.times["ordering_ms"], 10.0);
        assert_eq!(
            medians.stages_ms["spice.order"] + medians.stages_ms["spice.order.pack"],
            18.0
        );
    }
    fn padding_sample(fraction: f64) -> Sample {
        Sample::new(
            0,
            100.0,
            2.0,
            vec![
                ("pcs.commit", 16),
                ("pcs.open", 1),
                ("pcs.ringswitch", 8),
                ("pcs.ligerito_open", 12),
                ("mosaic.batch", 1),
                ("mosaic.region_sums", 9),
                ("mosaic.initial", 2),
                ("mosaic.bind_g", 4),
                ("mosaic.build_e", 3),
                ("mosaic.dense", 8),
                ("spice.pack", 2),
                ("spice.order", 7),
            ]
            .into_iter()
            .map(|(name, ms)| (name, Duration::from_millis(ms)))
            .collect(),
            fraction,
        )
        .unwrap()
    }
    #[test]
    fn no_padding_preserves_actual_commitment_and_total() {
        let sample = padding_sample(1.0);
        let adjusted = &sample.padding_adjusted_estimate;
        assert_eq!(adjusted.prover_ms, sample.prover_ms);
        assert_eq!(adjusted.commit_ms, sample.commit_ms);
        assert_eq!(adjusted.removed_padding_ms, 0.0);
    }
    #[test]
    fn padding_estimate_only_discounts_commitment_and_preserves_all_other_work() {
        let sample = padding_sample(0.0625);
        let adjusted = &sample.padding_adjusted_estimate;
        assert_eq!(adjusted.removed_padding_ms, 15.0);
        assert_eq!(adjusted.prover_ms, 85.0);
        assert_eq!(adjusted.commit_ms, 1.0);
        assert_eq!(
            adjusted.prover_ms - adjusted.commit_ms,
            sample.prover_ms - sample.commit_ms,
        );
        assert_eq!(sample.opening_ms, 21.0);
        assert_eq!(sample.mosaic_ms, 27.0);
        assert_eq!(sample.ordering_ms, 7.0);
        assert_eq!(sample.preparation_ms, 2.0);
        assert_eq!(sample.verifier_ms, 2.0);
        assert_eq!(sample.stages_ms["mosaic.bind_g"], 4.0);
        assert_eq!(sample.stages_ms["mosaic.dense"], 8.0);
        assert_eq!(sample.prover_ms, 100.0);
        assert_eq!(sample.commit_ms, 16.0);
        let json = serde_json::to_value(adjusted).unwrap();
        assert_eq!(json.as_object().unwrap().len(), 3);
        for name in ["opening_ms", "mosaic_ms", "ordering_ms", "stages_ms"] {
            assert!(json.get(name).is_none());
        }
    }
    #[test]
    fn adjusted_totals_are_medianed_per_sample() {
        let sample = |prover, commit| {
            Sample::new(
                0,
                prover,
                1.0,
                vec![
                    ("pcs.commit", Duration::from_millis(commit)),
                    ("pcs.open", Duration::ZERO),
                ],
                0.5,
            )
            .unwrap()
        };
        let medians = Medians::new(&[sample(100.0, 0), sample(100.0, 80), sample(200.0, 80)]);
        let adjusted = &medians.padding_adjusted_estimate;
        assert_eq!(adjusted.prover_ms, 100.0);
        assert_eq!(adjusted.removed_padding_ms, 40.0);
        assert_eq!(adjusted.commit_ms, 40.0);
        assert_ne!(
            adjusted.prover_ms,
            medians.times["prover_ms"] - 0.5 * medians.times["commit_ms"],
        );
    }
    #[test]
    fn full_one_row_flock_and_ripple_cases_share_trace_and_commitment_area() {
        let options = Options {
            methods: vec!["spice-ripple".into(), "spice".into()],
            patterns: vec!["random".into()],
            address_bits: 2,
            log_cycles: 0,
            chunk_bits: vec![1],
            threads: 2,
            warmup: 0,
            repeat: 1,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .unwrap();
        // The workload generator requires four rows, while both proof APIs
        // support one. Exercise the benchmark's actual measurement/verification
        // path directly so this also covers virtual zero-cycle extension.
        let trace = AccessTrace {
            initial: vec![1, 2, 3, 4],
            reads: vec![vec![2]],
            writes: vec![2],
            new_values: vec![99],
            read_values: Some(vec![vec![3]]),
        };
        let config = Config {
            address_bits: 2,
            chunk_bits: 1,
            ..Default::default()
        };
        let runs = pool
            .install(|| {
                options
                    .methods
                    .iter()
                    .map(|method| {
                        sample(method, &config, &trace, None, 0, &options)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .unwrap();
        let ((_, ripple, old_final), (_, flock, new_final)) = (&runs[0], &runs[1]);
        assert_eq!(old_final, new_final);
        assert_eq!(new_final, &vec![1, 2, 99, 4]);
        assert_eq!(ripple.components, flock.components);
        assert_eq!(ripple.populated_bits, flock.populated_bits);
        assert_eq!(ripple.committed_bits, flock.committed_bits);
        assert_eq!(flock.mosaic_claims - ripple.mosaic_claims, 2);
    }
    #[test]
    fn verifier_spans_are_isolated_from_prover_spans() {
        drop(profiling::Span::new("prove.leftover"));
        let (ms, metrics) = timed_verify(|| {
            let _outer = profiling::Span::new("verify.outer");
            drop(profiling::Span::new("verify.inner"));
            Ok(())
        })
        .unwrap();
        assert!(ms >= 0.0);
        let stages = stage_map(metrics);
        assert_eq!(
            stages.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["verify.inner", "verify.outer"]
        );
        let mut a = padding_sample(1.0);
        let mut b = padding_sample(1.0);
        a.verifier_stages_ms = BTreeMap::from([("verify.x".into(), 1.0)]);
        b.verifier_stages_ms = BTreeMap::from([("verify.x".into(), 3.0)]);
        let medians = Medians::new(&[a, b]);
        assert_eq!(medians.verifier_stages_ms["verify.x"], 2.0);
    }
    #[test]
    fn guest_order_and_csv_quoting() {
        let o = Options {
            patterns: vec!["guest".into()],
            guest_program: vec!["pointer-chase".into()],
            guest_rows: vec!["memory".into(), "all".into()],
            ..Default::default()
        };
        assert_eq!(
            workloads(&o)
                .iter()
                .map(|w| w.0.as_str())
                .collect::<Vec<_>>(),
            vec!["guest/pointer-chase/memory", "guest/pointer-chase/all"]
        );
        assert_eq!(escaped("a,\"b"), "\"a,\"\"b\"");
    }

    #[test]
    fn typed_method_requires_explicit_operation_format() {
        assert_eq!(Options::default().trace_format, "rmw");
        let mut o = Options {
            methods: vec!["spice-typed".into()],
            ..Default::default()
        };
        assert!(
            validate_options(&o)
                .unwrap_err()
                .to_string()
                .contains("requires")
        );
        o.trace_format = "operations".into();
        assert!(validate_options(&o).is_ok());
        assert_eq!(configurations(&o), vec![("spice-typed".into(), None)]);
        assert_eq!(ordering_name("spice-typed"), Some("single-value-flock"));
        o.trace_format = "automatic".into();
        assert!(validate_options(&o).is_err());
        assert!(
            select(
                "spice-typed,spice",
                &["twist", "spice", "spice-ripple", "spice-typed"],
                &["twist", "spice"]
            )
            .is_ok()
        );
    }

    #[test]
    fn operation_hash_keeps_unchanged_write_distinct_from_read() {
        let mut read = random_operations(2, 2, 19).unwrap();
        read.addresses.fill(0);
        read.values.fill(read.initial_memory[0]);
        read.is_write.fill(false);
        read.metadata.read_operations = 4;
        read.metadata.write_operations = 0;
        let mut write = read.clone();
        write.is_write[0] = true;
        write.metadata.read_operations = 3;
        write.metadata.write_operations = 1;
        assert_ne!(operation_trace_hash(&read), operation_trace_hash(&write));
        let into_trace = |data: OperationPayload| {
            let data = data.into_rmw().unwrap();
            AccessTrace {
                initial: data.initial_memory,
                reads: data.reads,
                writes: data.writes,
                new_values: data.new_values,
                read_values: Some(data.read_values),
            }
        };
        assert_eq!(
            trace_hash(&into_trace(read)),
            trace_hash(&into_trace(write))
        );
    }

    #[test]
    fn typed_and_baseline_cases_use_the_same_operation_rows_and_final_memory() {
        let options = Options {
            methods: vec![
                "spice-typed".into(),
                "twist".into(),
                "spice".into(),
                "spice-ripple".into(),
            ],
            trace_format: "operations".into(),
            patterns: vec!["random".into()],
            address_bits: 2,
            log_cycles: 2,
            chunk_bits: vec![1],
            threads: 2,
            warmup: 0,
            repeat: 1,
            ..Default::default()
        };
        let cases = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| benchmark(&options).map_err(|e| e.to_string()))
            .unwrap();
        assert_eq!(cases.len(), 4);
        for case in &cases {
            assert_eq!(case.pattern, "random/operations");
            assert_eq!(case.trace_format, "operations");
            assert_eq!(case.area.cycles, 4);
            assert_eq!(case.metadata["operations"], 4);
            assert_eq!(case.trace_sha256, cases[0].trace_sha256);
            assert_eq!(case.operation_trace_sha256, cases[0].operation_trace_sha256);
            assert!(case.operation_trace_sha256.is_some());
            if cfg!(unix) {
                assert!(case.load.start.is_some() && case.load.end.is_some());
                assert!(case.samples.iter().all(|s| s.load.start.is_some()));
            }
        }
        assert_eq!(
            cases[0].area.populated_bits,
            (64 + 2 + 2 * 3 + 1) * 4 + (3 + 1) * 4
        );
        assert_eq!(
            cases[2].area.populated_bits,
            (128 + 2 + 2 * 3 - 1) * 4 + 3 * 4
        );
        assert!(cases[0].median.times["mask_ms"] > 0.0);
        let comparisons = pairs(&cases);
        assert_eq!(comparisons.len(), 3);
        assert_eq!(
            comparisons
                .iter()
                .map(|p| p.spice_method.as_str())
                .collect::<Vec<_>>(),
            vec!["spice-typed", "spice", "spice-ripple"]
        );
        let report = Report {
            schema_version: 4,
            implementation: "test".into(),
            options,
            environment: serde_json::Value::Null,
            provenance: serde_json::Value::Null,
            notes: vec![],
            cases,
            comparisons,
            verified_proofs: 4,
        };
        let table = csv(&report);
        let rows: Vec<_> = table.lines().collect();
        assert_eq!(rows.len(), 5);
        assert!(rows[0].contains("operation_trace_sha256"));
        assert!(rows[0].contains("mask_ms"));
        assert!(rows[0].contains("populated_fraction"));
        assert!(!rows[0].contains("ordering_gate_fraction"));
        assert!(!rows[0].contains("padding_adjusted_estimate:ordering_ms"));
        assert!(rows[0].contains("padding_adjusted_estimate:prover_ms"));
        assert!(rows[0].contains("padding_adjusted_estimate:commit_ms"));
        assert!(!rows[0].contains("padding_adjusted_estimate:opening_ms"));
        assert!(!rows[0].contains("padding_adjusted_estimate:exclusive:"));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["schema_version"], 4);
        assert!(
            json["cases"][0]["samples"][0]["padding_adjusted_estimate"]["prover_ms"].is_number()
        );
        assert!(json["cases"][0]["median"]["padding_adjusted_estimate"]["prover_ms"].is_number());
        assert!(
            json["comparisons"][0]["spice_prover_speedup_padding_adjusted_estimate"].is_number()
        );
        assert!(
            rows.iter()
                .all(|line| line.split(',').count() == rows[0].split(',').count())
        );
    }
}
