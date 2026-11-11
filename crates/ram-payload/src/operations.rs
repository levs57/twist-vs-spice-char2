//! Operation-aware 64-bit RAM traces, independent of the legacy RMW schema.
//!
//! Loads observe the containing word. A narrow store becomes a read followed
//! by a full-word write computed by the original VM's masked-store helper.
//! Operation indices are the new timestamps; source VM indices remain in the
//! normalization metadata rather than being presented as CPU instructions.

use std::collections::{BTreeMap, BTreeSet};

use om_elf::{build_m0, extract_output, load};
use om_isa::bytecode::{HALTED_UPC, TraceRecord, VirtualMachine, expand, store_apply};
use serde::{Deserialize, Serialize};

use super::{
    GUEST_ADDRESS_BITS, GuestFill, GuestMetadata, GuestOptions, GuestRows, PayloadError,
    PayloadMetadata, RamPayload, SplitMix64, UPSTREAM_COMMIT, dimensions, guest_input,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationPayload {
    pub initial_memory: Vec<u64>,
    pub addresses: Vec<u32>,
    /// Read result for a read; replacement full word for a write.
    pub values: Vec<u64>,
    pub is_write: Vec<bool>,
    pub metadata: OperationMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationMetadata {
    pub kind: String,
    pub seed: u64,
    pub address_bits: u32,
    pub operations: usize,
    pub read_operations: usize,
    pub write_operations: usize,
    pub memory_operations: usize,
    pub dummy_read_operations: usize,
    pub narrow_store_read_operations: usize,
    pub narrow_store_write_operations: usize,
    pub trace_mode: String,
    /// Full VM execution before filtering, expansion, replay or truncation.
    pub source_vm_cycles: usize,
    /// Original VM rows retained by the selected row policy.
    pub selected_source_rows: usize,
    pub source_load_rows: usize,
    pub source_full_store_rows: usize,
    pub source_narrow_store_rows: usize,
    /// One selected template, after narrow stores have been expanded.
    pub pattern_operations: usize,
    pub full_pattern_repetitions: usize,
    pub tail_pattern_operations: usize,
    pub replayed_operations: usize,
    pub padding_operations: usize,
    /// Completely emitted selected source rows, counting replayed copies.
    pub completed_source_rows: usize,
    pub partial_source_operation: Option<PartialSourceOperation>,
    pub guest: Option<GuestMetadata>,
}

/// The final prefix may contain a narrow store's read but omit its write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialSourceOperation {
    pub source_cycle: usize,
    pub pattern_iteration: usize,
    pub emitted_micro_operations: u8,
    pub total_micro_operations: u8,
}

impl OperationPayload {
    fn replay_inner<const KEEP_OLD: bool>(&self) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
        let count = self.addresses.len();
        if count == 0
            || !count.is_power_of_two()
            || self.values.len() != count
            || self.is_write.len() != count
            || self.metadata.operations != count
            || self.metadata.padding_operations > count
        {
            return Err(PayloadError("operation trace shape mismatch".into()));
        }
        let bits = self.metadata.address_bits;
        if bits == 0
            || bits > 32
            || bits >= usize::BITS
            || self.initial_memory.len() != 1usize << bits
        {
            return Err(PayloadError(
                "operation initial-memory shape mismatch".into(),
            ));
        }
        let mut memory = self.initial_memory.clone();
        let mut old_values = if KEEP_OLD {
            Vec::with_capacity(count)
        } else {
            Vec::new()
        };
        for (i, ((&address, &value), &write)) in self
            .addresses
            .iter()
            .zip(&self.values)
            .zip(&self.is_write)
            .enumerate()
        {
            let word = memory
                .get_mut(address as usize)
                .ok_or_else(|| PayloadError(format!("operation address out of range at {i}")))?;
            if KEEP_OLD {
                old_values.push(*word);
            }
            if write {
                *word = value;
            } else if *word != value {
                return Err(PayloadError(format!("incorrect operation read at {i}")));
            }
        }
        Ok((old_values, memory))
    }

    pub fn replay(&self) -> Result<Vec<u64>, PayloadError> {
        self.replay_inner::<false>().map(|(_, memory)| memory)
    }

    /// Validate/replay once and return both old values and final memory, so a
    /// caller preparing multiple argument representations can reuse the pass.
    pub fn replay_with_old_values(&self) -> Result<(Vec<u64>, Vec<u64>), PayloadError> {
        self.replay_inner::<true>()
    }

    /// Translate the same operation rows for Twist: reads write back their old
    /// word; writes acquire an internal old-value read before replacement.
    pub fn into_rmw(self) -> Result<RamPayload, PayloadError> {
        let (old_values, _) = self.replay_with_old_values()?;
        let mut guest = self.metadata.guest;
        if let Some(guest) = &mut guest {
            guest.zero_read_cycles = old_values.iter().filter(|&&v| v == 0).count();
            guest.unchanged_write_cycles = old_values
                .iter()
                .zip(&self.values)
                .filter(|(a, b)| a == b)
                .count();
        }
        Ok(RamPayload {
            initial_memory: self.initial_memory,
            reads: vec![self.addresses.clone()],
            writes: self.addresses,
            new_values: self.values,
            read_values: vec![old_values],
            metadata: PayloadMetadata {
                kind: self.metadata.kind,
                memory_model: "shared-address".into(),
                seed: self.metadata.seed,
                address_bits: self.metadata.address_bits,
                cycles: self.metadata.operations,
                read_ports: 1,
                real_cycles: self.metadata.operations - self.metadata.padding_operations,
                padding_cycles: self.metadata.padding_operations,
                trace_mode: self.metadata.trace_mode,
                source_cycles: self.metadata.source_vm_cycles,
                pattern_cycles: self.metadata.pattern_operations,
                full_pattern_repetitions: self.metadata.full_pattern_repetitions,
                tail_pattern_cycles: self.metadata.tail_pattern_operations,
                replayed_cycles: self.metadata.replayed_operations,
                guest,
            },
        })
    }
}

#[cfg(test)]
mod tests;

/// Independent uniform addresses and unbiased per-operation read/write coins.
/// The actual read/write counts are reported; no adjacent RMW pair is forced.
pub fn random_operations(
    log_operations: u32,
    address_bits: u32,
    seed: u64,
) -> Result<OperationPayload, PayloadError> {
    let (count, words) = dimensions(log_operations, address_bits)?;
    let mut rng = SplitMix64(seed);
    let initial_memory = (0..words).map(|_| rng.next()).collect::<Vec<_>>();
    let mut memory = initial_memory.clone();
    let mut addresses = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    let mut is_write = Vec::with_capacity(count);
    for _ in 0..count {
        let address = (rng.next() as usize) & (words - 1);
        let write = rng.next() & 1 != 0;
        let candidate = rng.next();
        let value = if write { candidate } else { memory[address] };
        memory[address] = value;
        addresses.push(address as u32);
        values.push(value);
        is_write.push(write);
    }
    let writes = is_write.iter().filter(|&&write| write).count();
    Ok(OperationPayload {
        initial_memory,
        addresses,
        values,
        is_write,
        metadata: OperationMetadata {
            kind: "uniform-random-operations".into(),
            seed,
            address_bits,
            operations: count,
            read_operations: count - writes,
            write_operations: writes,
            memory_operations: count,
            dummy_read_operations: 0,
            narrow_store_read_operations: 0,
            narrow_store_write_operations: 0,
            trace_mode: "random-operations".into(),
            source_vm_cycles: 0,
            selected_source_rows: 0,
            source_load_rows: 0,
            source_full_store_rows: 0,
            source_narrow_store_rows: 0,
            pattern_operations: count,
            full_pattern_repetitions: 1,
            tail_pattern_operations: 0,
            replayed_operations: 0,
            padding_operations: 0,
            completed_source_rows: 0,
            partial_source_operation: None,
            guest: None,
        },
    })
}

#[derive(Clone, Copy)]
struct Store {
    width: u8,
    offset: u8,
    source: u64,
}

#[derive(Clone, Copy)]
struct PatternOperation {
    address: u32,
    store: Option<Store>,
    memory: bool,
    source_cycle: usize,
    part: u8,
    parts: u8,
}

fn append_record(
    pattern: &mut Vec<PatternOperation>,
    source_cycle: usize,
    record: &TraceRecord,
    address: u32,
) {
    let base = PatternOperation {
        address,
        store: None,
        memory: record.rom.mem,
        source_cycle,
        part: 1,
        parts: 1,
    };
    match record.store_width {
        None => pattern.push(base),
        Some(width) => {
            let store = Store {
                width,
                offset: record.store_off,
                source: record.store_src,
            };
            if width == 3 && record.store_off == 0 {
                pattern.push(PatternOperation {
                    store: Some(store),
                    ..base
                });
            } else {
                pattern.push(PatternOperation {
                    part: 1,
                    parts: 2,
                    ..base
                });
                pattern.push(PatternOperation {
                    store: Some(store),
                    part: 2,
                    parts: 2,
                    ..base
                });
            }
        }
    }
}

struct FilledPattern {
    addresses: Vec<u32>,
    values: Vec<u64>,
    is_write: Vec<bool>,
    memory_operations: usize,
    dummy_zero_reads: usize,
    zero_old_values: usize,
    unchanged: usize,
    narrow_reads: usize,
    narrow_writes: usize,
    completed_rows: usize,
    partial: Option<PartialSourceOperation>,
}

fn fill_pattern(
    initial: &[u64],
    pattern: &[PatternOperation],
    count: usize,
    repeat: bool,
) -> FilledPattern {
    let mut memory = initial.to_vec();
    let mut filled = FilledPattern {
        addresses: Vec::with_capacity(count),
        values: Vec::with_capacity(count),
        is_write: Vec::with_capacity(count),
        memory_operations: 0,
        dummy_zero_reads: 0,
        zero_old_values: 0,
        unchanged: 0,
        narrow_reads: 0,
        narrow_writes: 0,
        completed_rows: 0,
        partial: None,
    };
    for i in 0..count {
        let op = if repeat || i < pattern.len() {
            pattern[i % pattern.len()]
        } else {
            // The original VM HALTED row reads word zero and changes no RAM.
            PatternOperation {
                address: 0,
                store: None,
                memory: false,
                source_cycle: 0,
                part: 1,
                parts: 1,
            }
        };
        let current = memory[op.address as usize];
        let value = op.store.map_or(current, |s| {
            store_apply(s.width, u32::from(s.offset), current, s.source)
        });
        if op.store.is_some() {
            memory[op.address as usize] = value;
        }
        filled.addresses.push(op.address);
        filled.values.push(value);
        filled.is_write.push(op.store.is_some());
        filled.memory_operations += usize::from(op.memory);
        filled.dummy_zero_reads += usize::from(!op.memory && current == 0);
        filled.zero_old_values += usize::from(current == 0);
        filled.unchanged += usize::from(current == value);
        filled.narrow_reads += usize::from(op.parts == 2 && op.part == 1);
        filled.narrow_writes += usize::from(op.parts == 2 && op.part == 2);
        filled.completed_rows += usize::from(op.part == op.parts && (repeat || i < pattern.len()));
        if i + 1 == count && op.part < op.parts {
            filled.partial = Some(PartialSourceOperation {
                source_cycle: op.source_cycle,
                pattern_iteration: i / pattern.len(),
                emitted_micro_operations: op.part,
                total_micro_operations: op.parts,
            });
        }
    }
    filled
}

/// Execute and check the original ELF, then replay its normalized word pattern.
/// The complete source execution is checked even when the requested output is
/// only a prefix. Replayed rows describe RAM operations, not new CPU execution.
pub fn guest_operations(
    log_operations: u32,
    address_bits: u32,
    options: &GuestOptions,
) -> Result<OperationPayload, PayloadError> {
    let (count, words) = dimensions(log_operations, address_bits)?;
    if options.rows == GuestRows::Memory && options.fill == GuestFill::Halt {
        return Err(PayloadError(
            "memory-only operations cannot use HALTED padding".into(),
        ));
    }
    let (input_words, expected_output_words) = guest_input(options)?;
    let input = input_words
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let params = options.program.params();
    let elf = options.program.elf();
    let image = load(elf, &params).map_err(|e| PayloadError(format!("guest ELF: {e}")))?;
    let initial = build_m0(&image.ram_words, &input, &params)
        .map_err(|e| PayloadError(format!("guest input: {e}")))?;
    let mut vm = VirtualMachine::new(
        expand(&image.code, image.entry_pc),
        &initial,
        GUEST_ADDRESS_BITS,
    );
    let result = vm
        .run(options.max_cycles)
        .map_err(|e| PayloadError(format!("guest execution: {e:?}")))?;
    if result.exit_code != 0 {
        return Err(PayloadError(format!(
            "guest exited with code {}",
            result.exit_code
        )));
    }
    let output = extract_output(&result.ram, &params)
        .map_err(|e| PayloadError(format!("guest output: {e}")))?;
    let expected = expected_output_words
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    if output != expected {
        return Err(PayloadError(
            "guest output differs from independent expected answers".into(),
        ));
    }
    let halted = vm
        .step(&vm.program.rom[HALTED_UPC as usize].to_owned())
        .map_err(|e| PayloadError(format!("guest HALTED row: {e:?}")))?;
    if halted.rom.mem
        || halted.store_width.is_some()
        || halted.ram_idx != 0
        || halted.ram_old != halted.ram_new
    {
        return Err(PayloadError(
            "unexpected original VM HALTED RAM behavior".into(),
        ));
    }
    let source_cycles = result.trace.len();
    let source_memory_rows = result.trace.iter().filter(|r| r.rom.mem).count();
    let mut original_addresses = BTreeSet::from([0usize]);
    original_addresses.extend(initial.iter().filter_map(|(&a, &v)| (v != 0).then_some(a)));
    original_addresses.extend(result.trace.iter().map(|r| r.ram_idx));
    if original_addresses.len() > words {
        return Err(PayloadError(format!(
            "guest needs {} initialized/accessed words; address_bits={address_bits} allows {words}",
            original_addresses.len()
        )));
    }
    let compact_to_original = original_addresses
        .iter()
        .map(|&a| a as u32)
        .collect::<Vec<_>>();
    let compact: BTreeMap<usize, u32> = original_addresses
        .into_iter()
        .enumerate()
        .map(|(i, a)| (a, i as u32))
        .collect();
    let mut initial_memory = vec![0; words];
    for (&address, &value) in &initial {
        if value != 0 {
            initial_memory[compact[&address] as usize] = value;
        }
    }
    let mut source_memory = initial_memory.clone();
    let mut pattern = Vec::new();
    let (mut selected, mut loads, mut full_stores, mut narrow_stores) = (0, 0, 0, 0);
    for (cycle, record) in result.trace.iter().enumerate() {
        if options.rows == GuestRows::Memory && !record.rom.mem {
            continue;
        }
        let address = compact[&record.ram_idx];
        let current = source_memory[address as usize];
        let new = record.store_width.map_or(current, |width| {
            store_apply(
                width,
                u32::from(record.store_off),
                current,
                record.store_src,
            )
        });
        if current != record.ram_old || new != record.ram_new {
            return Err(PayloadError(format!(
                "normalized source differs from VM at cycle {cycle}"
            )));
        }
        source_memory[address as usize] = new;
        selected += 1;
        match record.store_width {
            None if record.rom.mem => loads += 1,
            Some(3) if record.store_off == 0 => full_stores += 1,
            Some(_) => narrow_stores += 1,
            None => {}
        }
        append_record(&mut pattern, cycle, record, address);
    }
    if pattern.is_empty() {
        return Err(PayloadError("selected operation pattern is empty".into()));
    }
    if options.fill == GuestFill::Halt && pattern.len() > count {
        return Err(PayloadError(format!(
            "guest needs {} normalized operations; requested {count}",
            pattern.len()
        )));
    }
    let repeat = options.fill == GuestFill::Repeat;
    let filled = fill_pattern(&initial_memory, &pattern, count, repeat);
    let writes = filled.is_write.iter().filter(|&&v| v).count();
    let distinct = filled
        .addresses
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    Ok(OperationPayload {
        initial_memory,
        addresses: filled.addresses,
        values: filled.values,
        is_write: filled.is_write,
        metadata: OperationMetadata {
            kind: format!("{}-guest-operations", options.program.name()),
            seed: options.seed,
            address_bits,
            operations: count,
            read_operations: count - writes,
            write_operations: writes,
            memory_operations: filled.memory_operations,
            dummy_read_operations: count - filled.memory_operations,
            narrow_store_read_operations: filled.narrow_reads,
            narrow_store_write_operations: filled.narrow_writes,
            trace_mode: if repeat {
                "repeat-normalized-ram-operations"
            } else {
                "normalized-vm-halt"
            }
            .into(),
            source_vm_cycles: source_cycles,
            selected_source_rows: selected,
            source_load_rows: loads,
            source_full_store_rows: full_stores,
            source_narrow_store_rows: narrow_stores,
            pattern_operations: pattern.len(),
            full_pattern_repetitions: if repeat { count / pattern.len() } else { 1 },
            tail_pattern_operations: if repeat { count % pattern.len() } else { 0 },
            replayed_operations: if repeat {
                count.saturating_sub(pattern.len())
            } else {
                0
            },
            padding_operations: if repeat { 0 } else { count - pattern.len() },
            completed_source_rows: filled.completed_rows,
            partial_source_operation: filled.partial,
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
                source_memory_access_cycles: source_memory_rows,
                memory_access_cycles: filled.memory_operations,
                dummy_access_cycles: count - filled.memory_operations,
                zero_read_cycles: filled.zero_old_values,
                dummy_zero_read_cycles: filled.dummy_zero_reads,
                unchanged_write_cycles: filled.unchanged,
                distinct_accessed_words: distinct,
            }),
        },
    })
}
