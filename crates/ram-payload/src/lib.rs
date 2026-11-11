//! Reproducible RAM workloads. Guest execution uses the original open-magnetar VM.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use om_elf::{Params, build_m0, extract_output, load};
use om_isa::bytecode::{HALTED_UPC, VirtualMachine, expand, store_apply};
use serde::{Deserialize, Serialize};

pub mod locality;
pub mod operations;

pub const UPSTREAM_COMMIT: &str = "febfc0e66f5809e034dd1afac78202bd4882de27";
pub const GUEST_ELF: &[u8] = include_bytes!("../../../payloads/open-magnetar/binary-search.elf");
const GUEST_ADDRESS_BITS: u32 = 16;
const GUEST_INPUT_BITS: u32 = 12;
const GUEST_OUTPUT_BITS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum GuestProgram {
    #[default]
    BinarySearch,
    Erc20MixedBlock,
    PointerChase,
}

impl GuestProgram {
    pub fn name(self) -> &'static str {
        match self {
            Self::BinarySearch => "binary-search",
            Self::Erc20MixedBlock => "erc20-mixed-block",
            Self::PointerChase => "pointer-chase",
        }
    }

    pub fn elf(self) -> &'static [u8] {
        match self {
            Self::BinarySearch => GUEST_ELF,
            Self::Erc20MixedBlock => {
                include_bytes!("../../../payloads/open-magnetar/erc20-mixed-block.elf")
            }
            Self::PointerChase => {
                include_bytes!("../../../payloads/open-magnetar/pointer-chase.elf")
            }
        }
    }

    fn params(self) -> Params {
        Params::new(
            GUEST_ADDRESS_BITS,
            GUEST_INPUT_BITS,
            if self == Self::Erc20MixedBlock {
                11
            } else {
                GUEST_OUTPUT_BITS
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GuestRows {
    #[default]
    All,
    Memory,
}

/// Word addresses and 64-bit values; all reads occur before the write in a cycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamPayload {
    pub initial_memory: Vec<u64>,
    pub reads: Vec<Vec<u32>>,
    pub writes: Vec<u32>,
    pub new_values: Vec<u64>,
    pub read_values: Vec<Vec<u64>>,
    pub metadata: PayloadMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadMetadata {
    pub kind: String,
    /// `shared-address` matches the VM RAM port; `independent-ports` is generic Twist.
    pub memory_model: String,
    pub seed: u64,
    pub address_bits: u32,
    pub cycles: usize,
    pub read_ports: usize,
    pub real_cycles: usize,
    pub padding_cycles: usize,
    pub trace_mode: String,
    pub source_cycles: usize,
    pub pattern_cycles: usize,
    pub full_pattern_repetitions: usize,
    pub tail_pattern_cycles: usize,
    pub replayed_cycles: usize,
    pub guest: Option<GuestMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMetadata {
    pub program: String,
    pub upstream_commit: String,
    pub elf_sha256: String,
    pub keys: usize,
    pub queries: usize,
    pub rows: GuestRows,
    pub size: usize,
    pub operations: usize,
    pub max_cycles: usize,
    pub original_address_bits: u32,
    /// Compact word index -> original VM word index. Original word zero stays zero.
    pub compact_to_original: Vec<u32>,
    pub input_words: Vec<u64>,
    pub output_words: Vec<u64>,
    pub expected_output_words: Vec<u64>,
    /// Instructions retired by the single source VM execution, never replayed instructions.
    pub retired_instructions: u64,
    pub source_memory_access_cycles: usize,
    pub memory_access_cycles: usize,
    pub dummy_access_cycles: usize,
    pub zero_read_cycles: usize,
    pub dummy_zero_read_cycles: usize,
    pub unchanged_write_cycles: usize,
    pub distinct_accessed_words: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GuestFill {
    /// Repeat recorded RAM addresses/stores against continuously updated memory.
    #[default]
    Repeat,
    /// Preserve the exact source execution and append VM HALTED rows.
    Halt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestOptions {
    pub program: GuestProgram,
    pub rows: GuestRows,
    pub size: usize,
    pub operations: usize,
    pub keys: usize,
    pub queries: usize,
    pub seed: u64,
    pub max_cycles: usize,
    pub fill: GuestFill,
}

impl Default for GuestOptions {
    fn default() -> Self {
        Self {
            program: GuestProgram::BinarySearch,
            rows: GuestRows::All,
            size: 64,
            operations: 256,
            keys: 64,
            queries: 16,
            seed: 1,
            max_cycles: 1 << 20,
            fill: GuestFill::Repeat,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadError(pub String);

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PayloadError {}

fn dimensions(log_cycles: u32, address_bits: u32) -> Result<(usize, usize), PayloadError> {
    if !(2..usize::BITS).contains(&log_cycles) {
        return Err(PayloadError(
            "log_cycles must be at least 2 and fit usize".into(),
        ));
    }
    if address_bits == 0 || address_bits > 32 || address_bits >= usize::BITS {
        return Err(PayloadError(
            "address_bits must be in 1..=32 and fit usize".into(),
        ));
    }
    Ok((1usize << log_cycles, 1usize << address_bits))
}

/// SplitMix64 fixes the payload stream independently of a rand crate/version.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// One uniformly sampled RAM address per cycle: read its old word, then replace it.
pub fn random_ram(
    log_cycles: u32,
    address_bits: u32,
    seed: u64,
) -> Result<RamPayload, PayloadError> {
    random_with_addressing(log_cycles, address_bits, RandomAddressing::Shared, seed)
}

/// Uniform independent read/write addresses for the explicit generic-port mode.
pub fn random(
    log_cycles: u32,
    address_bits: u32,
    read_ports: usize,
    seed: u64,
) -> Result<RamPayload, PayloadError> {
    random_with_addressing(
        log_cycles,
        address_bits,
        RandomAddressing::Independent(read_ports),
        seed,
    )
}

#[derive(Clone, Copy)]
enum RandomAddressing {
    Shared,
    Independent(usize),
}

fn random_with_addressing(
    log_cycles: u32,
    address_bits: u32,
    addressing: RandomAddressing,
    seed: u64,
) -> Result<RamPayload, PayloadError> {
    let (cycles, words) = dimensions(log_cycles, address_bits)?;
    let (read_ports, memory_model) = match addressing {
        RandomAddressing::Shared => (1, "shared-address"),
        RandomAddressing::Independent(ports) => (ports, "independent-ports"),
    };
    if read_ports == 0 {
        return Err(PayloadError("read_ports must be positive".into()));
    }
    let mut rng = SplitMix64(seed);
    let initial_memory: Vec<u64> = (0..words).map(|_| rng.next()).collect();
    let mut memory = initial_memory.clone();
    let mut reads = vec![Vec::with_capacity(cycles); read_ports];
    let mut read_values = vec![Vec::with_capacity(cycles); read_ports];
    let mut writes = Vec::with_capacity(cycles);
    let mut new_values = Vec::with_capacity(cycles);
    for cycle in 0..cycles {
        for (port, values) in reads.iter_mut().zip(&mut read_values) {
            let addr = (rng.next() as usize) & (words - 1);
            port.push(addr as u32);
            values.push(memory[addr]);
        }
        let addr = match addressing {
            RandomAddressing::Shared => reads[0][cycle] as usize,
            RandomAddressing::Independent(_) => (rng.next() as usize) & (words - 1),
        };
        let value = rng.next();
        writes.push(addr as u32);
        new_values.push(value);
        memory[addr] = value;
    }
    Ok(RamPayload {
        initial_memory,
        reads,
        writes,
        new_values,
        read_values,
        metadata: PayloadMetadata {
            kind: "uniform-random".into(),
            memory_model: memory_model.into(),
            seed,
            address_bits,
            cycles,
            read_ports,
            real_cycles: cycles,
            padding_cycles: 0,
            trace_mode: "random".into(),
            source_cycles: cycles,
            pattern_cycles: cycles,
            full_pattern_repetitions: 1,
            tail_pattern_cycles: 0,
            replayed_cycles: 0,
            guest: None,
        },
    })
}

fn binary_search_input(options: &GuestOptions) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
    if options.keys == 0 || options.queries == 0 || options.max_cycles == 0 {
        return Err(PayloadError(
            "guest keys, queries and max_cycles must be positive".into(),
        ));
    }
    let input_words = options
        .keys
        .checked_add(options.queries)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| PayloadError("guest input size overflow".into()))?;
    if input_words >= 1 << GUEST_INPUT_BITS || options.queries >= 1 << GUEST_OUTPUT_BITS {
        return Err(PayloadError(
            "guest input/output does not fit the embedded ELF ABI (I=12,O=8)".into(),
        ));
    }
    let mut words = Vec::with_capacity(input_words);
    words.push(options.keys as u64);
    words.extend((0..options.keys).map(|i| 1000 + 2 * i as u64));
    words.push(options.queries as u64);
    let mut expected = Vec::with_capacity(options.queries);
    let mut rng = SplitMix64(options.seed);
    for query in 0..options.queries {
        let index = rng.next() % options.keys as u64;
        if query % 2 == 0 {
            words.push(1000 + 2 * index);
            expected.push(index);
        } else {
            words.push(1001 + 2 * index);
            expected.push(u64::MAX);
        }
    }
    Ok((words, expected))
}

fn check_guest_capacity(
    input_words: usize,
    output_words: usize,
    program: GuestProgram,
) -> Result<(), PayloadError> {
    let params = program.params();
    if input_words >= 1usize << params.input_log_words
        || output_words >= 1usize << params.output_log_words
    {
        return Err(PayloadError(format!(
            "{} input/output exceeds embedded ELF ABI capacity",
            program.name()
        )));
    }
    Ok(())
}

fn pointer_chase_input(options: &GuestOptions) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
    let n = options.size;
    let input_len = n
        .checked_add(2)
        .ok_or_else(|| PayloadError("pointer input size overflow".into()))?;
    check_guest_capacity(input_len, 2, GuestProgram::PointerChase)?;
    if options.operations > options.max_cycles {
        return Err(PayloadError(
            "pointer steps exceed the source max_cycles budget".into(),
        ));
    }
    let mut rng = SplitMix64(options.seed);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next() as usize % (i + 1);
        order.swap(i, j);
    }
    let mut next = vec![0u64; n];
    for i in 0..n {
        next[order[i]] = order[(i + 1) % n] as u64;
    }
    let mut current = 0usize;
    let mut xor = 0u64;
    for _ in 0..options.operations {
        current = next[current] as usize;
        xor ^= current as u64;
    }
    let mut words = Vec::with_capacity(input_len);
    words.push(n as u64);
    words.extend(next);
    words.push(options.operations as u64);
    Ok((words, vec![current as u64, xor]))
}

fn erc20_input(options: &GuestOptions) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
    let n = options.size;
    if n > 1024 {
        return Err(PayloadError(
            "erc20 account count exceeds upstream cap 1024".into(),
        ));
    }
    let input_len = options
        .operations
        .checked_mul(3)
        .and_then(|m| m.checked_add(n))
        .and_then(|m| m.checked_add(2))
        .ok_or_else(|| PayloadError("erc20 input size overflow".into()))?;
    check_guest_capacity(input_len, n + 1, GuestProgram::Erc20MixedBlock)?;
    let mut rng = SplitMix64(options.seed);
    let balances: Vec<u64> = (0..n).map(|_| 1_000_000 + rng.next() % 1_000_000).collect();
    let mut words = Vec::with_capacity(input_len);
    words.push(n as u64);
    words.extend(&balances);
    words.push(options.operations as u64);
    for operation in 0..options.operations {
        let pool = if operation % 4 == 3 { n } else { n.min(8) };
        let mut from = rng.next() as usize % pool;
        let mut to = if pool > 1 {
            (from + 1 + rng.next() as usize % (pool - 1)) % pool
        } else {
            from
        };
        let mut amount = 1 + rng.next() % 1000;
        match operation % 10 {
            7 => to = from,
            8 => from = n,
            9 => amount = u64::MAX,
            _ => {}
        }
        words.extend([from as u64, to as u64, amount]);
    }
    // Independent host execution of the input triples, including failed and self transfers.
    let mut expected = balances;
    let mut successes = 0;
    for transfer in words[n + 2..].chunks_exact(3) {
        let (from, to, amount) = (transfer[0] as usize, transfer[1] as usize, transfer[2]);
        if from < n && to < n && expected[from] >= amount {
            if from != to {
                expected[from] -= amount;
                expected[to] = expected[to].wrapping_add(amount);
            }
            successes += 1;
        }
    }
    expected.insert(0, successes);
    Ok((words, expected))
}

fn guest_input(options: &GuestOptions) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
    if options.max_cycles == 0 {
        return Err(PayloadError("guest max_cycles must be positive".into()));
    }
    if options.program != GuestProgram::BinarySearch
        && (options.size == 0 || options.operations == 0)
    {
        return Err(PayloadError(
            "guest size and operations must be positive".into(),
        ));
    }
    match options.program {
        GuestProgram::BinarySearch => binary_search_input(options),
        GuestProgram::Erc20MixedBlock => erc20_input(options),
        GuestProgram::PointerChase => pointer_chase_input(options),
    }
}

/// Execute the selected unchanged upstream ELF to HALT on fresh input.
///
/// Repeat fills the requested length by replaying source RAM addresses and masked
/// stores against current memory. Halt appends actual VM HALTED rows instead.
pub fn guest(
    log_cycles: u32,
    address_bits: u32,
    options: &GuestOptions,
) -> Result<RamPayload, PayloadError> {
    let (cycles, words) = dimensions(log_cycles, address_bits)?;
    if options.rows == GuestRows::Memory && options.fill == GuestFill::Halt {
        return Err(PayloadError(
            "memory-only guest rows cannot use HALTED padding; select repeat fill".into(),
        ));
    }
    let (input_words, expected_output_words) = guest_input(options)?;
    let input: Vec<u8> = input_words.iter().flat_map(|x| x.to_le_bytes()).collect();
    let params = options.program.params();
    let elf = options.program.elf();
    let image = load(elf, &params).map_err(|e| PayloadError(format!("guest ELF: {e}")))?;
    let m0 = build_m0(&image.ram_words, &input, &params)
        .map_err(|e| PayloadError(format!("guest input: {e}")))?;
    let mut vm = VirtualMachine::new(expand(&image.code, image.entry_pc), &m0, GUEST_ADDRESS_BITS);
    let mut result = vm
        .run(options.max_cycles)
        .map_err(|e| PayloadError(format!("guest execution: {e:?}")))?;
    let source_cycles = result.trace.len();
    if result.exit_code != 0 {
        return Err(PayloadError(format!(
            "guest exited with code {}",
            result.exit_code
        )));
    }
    if options.fill == GuestFill::Halt && source_cycles >= cycles {
        return Err(PayloadError(format!(
            "guest needs {source_cycles} real microcycles plus at least one HALTED row; requested {cycles}; increase log_cycles",
        )));
    }
    let output = extract_output(&result.ram, &params)
        .map_err(|e| PayloadError(format!("guest output: {e}")))?;
    let expected: Vec<u8> = expected_output_words
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    if output != expected {
        return Err(PayloadError(format!(
            "{} guest output differs from expected answers",
            options.program.name()
        )));
    }
    let source_memory_access_cycles = result.trace.iter().filter(|r| r.rom.mem).count();
    if options.fill == GuestFill::Halt {
        let halted = vm.program.rom[HALTED_UPC as usize];
        while result.trace.len() < cycles {
            result.trace.push(
                vm.step(&halted)
                    .map_err(|e| PayloadError(format!("guest padding: {e:?}")))?,
            );
        }
    }

    let mut original_addresses = BTreeSet::from([0usize]);
    original_addresses.extend(
        m0.iter()
            .filter_map(|(&addr, &value)| (value != 0).then_some(addr)),
    );
    original_addresses.extend(result.trace.iter().map(|r| r.ram_idx));
    if original_addresses.len() > words {
        return Err(PayloadError(format!(
            "guest requires {} distinct initialized/accessed words; address_bits={address_bits} allows {words}",
            original_addresses.len()
        )));
    }
    let compact_to_original: Vec<u32> =
        original_addresses.iter().map(|&addr| addr as u32).collect();
    let compact: BTreeMap<usize, u32> = original_addresses
        .into_iter()
        .enumerate()
        .map(|(i, a)| (a, i as u32))
        .collect();
    let mut initial_memory = vec![0; words];
    for (&addr, &value) in &m0 {
        if value != 0 {
            initial_memory[compact[&addr] as usize] = value;
        }
    }
    let pattern: Vec<_> = result.trace[..source_cycles]
        .iter()
        .filter(|record| options.rows == GuestRows::All || record.rom.mem)
        .collect();
    let pattern_cycles = pattern.len();
    if pattern_cycles == 0 {
        return Err(PayloadError("selected guest row pattern is empty".into()));
    }
    let mut memory = initial_memory.clone();
    let mut addresses = Vec::with_capacity(cycles);
    let mut old_values = Vec::with_capacity(cycles);
    let mut new_values = Vec::with_capacity(cycles);
    let mut memory_access_cycles = 0;
    let mut dummy_zero_read_cycles = 0;
    for cycle in 0..cycles {
        let record = if options.fill == GuestFill::Halt {
            &result.trace[cycle]
        } else {
            pattern[cycle % pattern_cycles]
        };
        let address = compact[&record.ram_idx];
        let current = memory[address as usize];
        let new = replayed_value(
            record.store_width,
            record.store_off,
            record.store_src,
            current,
        );
        if (options.fill == GuestFill::Halt || cycle < pattern_cycles)
            && (current != record.ram_old || new != record.ram_new)
        {
            return Err(PayloadError(format!(
                "source RAM replay differs from VM at retained cycle {cycle}"
            )));
        }
        addresses.push(address);
        old_values.push(current);
        new_values.push(new);
        memory[address as usize] = new;
        memory_access_cycles += usize::from(record.rom.mem);
        dummy_zero_read_cycles += usize::from(!record.rom.mem && current == 0);
    }
    let zero_read_cycles = old_values.iter().filter(|&&value| value == 0).count();
    let unchanged_write_cycles = old_values
        .iter()
        .zip(&new_values)
        .filter(|(old, new)| old == new)
        .count();
    let distinct_accessed_words = addresses.iter().copied().collect::<BTreeSet<_>>().len();
    let repeated = options.fill == GuestFill::Repeat;
    let payload = RamPayload {
        initial_memory,
        reads: vec![addresses.clone()],
        writes: addresses,
        new_values,
        read_values: vec![old_values],
        metadata: PayloadMetadata {
            kind: format!("{}-guest", options.program.name()),
            memory_model: "shared-address".into(),
            seed: options.seed,
            address_bits,
            cycles,
            read_ports: 1,
            real_cycles: if repeated { cycles } else { source_cycles },
            padding_cycles: if repeated { 0 } else { cycles - source_cycles },
            trace_mode: if repeated {
                "repeat-ram-pattern"
            } else {
                "vm-halt"
            }
            .into(),
            source_cycles,
            pattern_cycles,
            full_pattern_repetitions: if repeated { cycles / pattern_cycles } else { 1 },
            tail_pattern_cycles: if repeated { cycles % pattern_cycles } else { 0 },
            replayed_cycles: if repeated {
                cycles.saturating_sub(pattern_cycles)
            } else {
                0
            },
            guest: Some(GuestMetadata {
                program: format!("open-magnetar/guest/examples/{}.c", options.program.name()),
                upstream_commit: UPSTREAM_COMMIT.into(),
                elf_sha256: om_elf::Hasher::Sha256
                    .digest(elf)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect(),
                keys: options.keys,
                queries: options.queries,
                rows: options.rows,
                size: options.size,
                operations: options.operations,
                max_cycles: options.max_cycles,
                original_address_bits: GUEST_ADDRESS_BITS,
                compact_to_original,
                input_words,
                output_words: output
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                expected_output_words,
                retired_instructions: result.instret,
                source_memory_access_cycles,
                memory_access_cycles,
                dummy_access_cycles: cycles - memory_access_cycles,
                zero_read_cycles,
                dummy_zero_read_cycles,
                unchanged_write_cycles,
                distinct_accessed_words,
            }),
        },
    };
    payload.replay()?;
    Ok(payload)
}

fn replayed_value(store_width: Option<u8>, store_off: u8, store_src: u64, current: u64) -> u64 {
    match store_width {
        Some(width) => store_apply(width, u32::from(store_off), current, store_src),
        None => current,
    }
}

#[cfg(test)]
mod tests {
    use super::replayed_value;

    #[test]
    fn replay_preserves_reads_and_unwritten_byte_lanes() {
        let current = 0x1122_3344_5566_7788;
        assert_eq!(replayed_value(None, 0, 0, current), current);
        assert_eq!(
            replayed_value(Some(0), 1, 0xaabb, current),
            0x1122_3344_5566_bb88
        );
        assert_eq!(
            replayed_value(Some(1), 2, 0xaabb_ccdd, current),
            0x1122_3344_ccdd_7788
        );
        assert_eq!(
            replayed_value(Some(2), 4, 0xdead_beef_cafe_babe, current),
            0xcafe_babe_5566_7788
        );
        assert_eq!(
            replayed_value(Some(3), 0, 0xfeed_face_cafe_babe, current),
            0xfeed_face_cafe_babe
        );
        // A later store in one period changes a lane the earlier store must retain next period.
        let first_store = replayed_value(Some(0), 1, 0xaa, current);
        let period_end = replayed_value(Some(0), 6, 0xbb, first_store);
        assert_eq!(period_end, 0x11bb_3344_5566_aa88);
        assert_eq!(replayed_value(None, 0, 0, period_end), period_end);
        assert_eq!(replayed_value(Some(0), 1, 0xaa, period_end), period_end);
    }
}

impl RamPayload {
    /// Validate every supplied read against read-before-write replay and return final memory.
    pub fn replay(&self) -> Result<Vec<u64>, PayloadError> {
        let cycles = self.writes.len();
        if self.new_values.len() != cycles
            || self.reads.len() != self.read_values.len()
            || self.reads.is_empty()
            || self.reads.iter().any(|r| r.len() != cycles)
            || self.read_values.iter().any(|r| r.len() != cycles)
        {
            return Err(PayloadError(
                "payload columns have inconsistent lengths".into(),
            ));
        }
        let mut memory = self.initial_memory.clone();
        for cycle in 0..cycles {
            for (port, (addresses, values)) in self.reads.iter().zip(&self.read_values).enumerate()
            {
                let actual = memory.get(addresses[cycle] as usize).ok_or_else(|| {
                    PayloadError(format!(
                        "read address out of range at cycle {cycle}, port {port}"
                    ))
                })?;
                if *actual != values[cycle] {
                    return Err(PayloadError(format!(
                        "read value mismatch at cycle {cycle}, port {port}: replay={actual}, supplied={}",
                        values[cycle]
                    )));
                }
            }
            let slot = memory.get_mut(self.writes[cycle] as usize).ok_or_else(|| {
                PayloadError(format!("write address out of range at cycle {cycle}"))
            })?;
            *slot = self.new_values[cycle];
        }
        Ok(memory)
    }
}
