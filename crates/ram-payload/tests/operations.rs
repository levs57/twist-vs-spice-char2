use om_elf::{Params, build_m0, load};
use om_isa::bytecode::{VirtualMachine, expand};
use ram_payload::operations::{OperationPayload, guest_operations, random_operations};
use ram_payload::{GuestFill, GuestOptions, GuestProgram, GuestRows};

fn options(program: GuestProgram, rows: GuestRows) -> GuestOptions {
    GuestOptions {
        program,
        rows,
        keys: 8,
        queries: 3,
        size: 8,
        operations: 12,
        seed: 73,
        ..GuestOptions::default()
    }
}

fn check_replay(payload: &OperationPayload) {
    let mut memory = payload.initial_memory.clone();
    let mut reads = 0;
    let mut writes = 0;
    for ((&address, &value), &write) in payload
        .addresses
        .iter()
        .zip(&payload.values)
        .zip(&payload.is_write)
    {
        if write {
            memory[address as usize] = value;
            writes += 1;
        } else {
            assert_eq!(memory[address as usize], value);
            reads += 1;
        }
    }
    assert_eq!(payload.replay().unwrap(), memory);
    assert_eq!(payload.metadata.read_operations, reads);
    assert_eq!(payload.metadata.write_operations, writes);
    assert_eq!(reads + writes, payload.metadata.operations);
    assert_eq!(
        payload.metadata.memory_operations + payload.metadata.dummy_read_operations,
        reads + writes
    );
    let rmw = payload.clone().into_rmw().unwrap();
    assert_eq!(rmw.reads[0], payload.addresses);
    assert_eq!(rmw.writes, payload.addresses);
    assert_eq!(rmw.new_values, payload.values);
    assert_eq!(rmw.replay().unwrap(), memory);
}

#[test]
fn random_operations_are_deterministic_independent_and_translate_without_expansion() {
    let payload = random_operations(12, 5, 73).unwrap();
    assert_eq!(payload, random_operations(12, 5, 73).unwrap());
    assert_eq!(payload.addresses.len(), 1 << 12);
    assert!(payload.metadata.read_operations > 0 && payload.metadata.write_operations > 0);
    assert!(
        payload
            .addresses
            .chunks_exact(2)
            .any(|pair| pair[0] != pair[1])
    );
    check_replay(&payload);
    let json = serde_json::to_string(&payload).unwrap();
    assert_eq!(
        serde_json::from_str::<OperationPayload>(&json).unwrap(),
        payload
    );
    assert_ne!(random_operations(12, 5, 74).unwrap(), payload);
}

/// Independent VM oracle: classification and first-period values come from the
/// original executed records, not the operation generator's private template.
#[test]
fn all_guests_normalize_exact_original_memory_records_and_keep_stable_mapping() {
    for program in [
        GuestProgram::BinarySearch,
        GuestProgram::Erc20MixedBlock,
        GuestProgram::PointerChase,
    ] {
        let mut mapping = None;
        for rows in [GuestRows::All, GuestRows::Memory] {
            let opts = options(program, rows);
            let payload = guest_operations(12, 12, &opts).unwrap();
            let meta = payload.metadata.guest.as_ref().unwrap();
            assert_eq!(meta.output_words, meta.expected_output_words);
            if let Some((initial, original)) = &mapping {
                assert_eq!(&payload.initial_memory, initial);
                assert_eq!(&meta.compact_to_original, original);
            } else {
                mapping = Some((
                    payload.initial_memory.clone(),
                    meta.compact_to_original.clone(),
                ));
            }
            let params = Params::new(
                16,
                12,
                if program == GuestProgram::Erc20MixedBlock {
                    11
                } else {
                    8
                },
            );
            let image = load(program.elf(), &params).unwrap();
            let input = meta
                .input_words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let initial = build_m0(&image.ram_words, &input, &params).unwrap();
            let mut vm = VirtualMachine::new(expand(&image.code, image.entry_pc), &initial, 16);
            let result = vm.run(opts.max_cycles).unwrap();
            assert_eq!(payload.metadata.source_vm_cycles, result.trace.len());
            let mut expected = Vec::new();
            let mut selected = 0;
            for record in &result.trace {
                if rows == GuestRows::Memory && !record.rom.mem {
                    continue;
                }
                selected += 1;
                let address = meta
                    .compact_to_original
                    .binary_search(&(record.ram_idx as u32))
                    .unwrap() as u32;
                match record.store_width {
                    None => expected.push((address, record.ram_old, false)),
                    Some(3) if record.store_off == 0 => {
                        expected.push((address, record.ram_new, true))
                    }
                    Some(_) => {
                        expected.push((address, record.ram_old, false));
                        expected.push((address, record.ram_new, true));
                    }
                }
            }
            assert_eq!(payload.metadata.selected_source_rows, selected);
            assert_eq!(payload.metadata.pattern_operations, expected.len());
            assert_eq!(
                expected.len(),
                selected + payload.metadata.source_narrow_store_rows
            );
            for (i, &(address, value, write)) in
                expected.iter().take(payload.addresses.len()).enumerate()
            {
                assert_eq!(
                    (payload.addresses[i], payload.values[i], payload.is_write[i]),
                    (address, value, write),
                    "program={program:?},rows={rows:?},i={i}"
                );
            }
            assert_eq!(
                payload.metadata.full_pattern_repetitions * expected.len()
                    + payload.metadata.tail_pattern_operations,
                payload.addresses.len()
            );
            assert_eq!(payload.metadata.padding_operations, 0);
            if rows == GuestRows::Memory {
                assert_eq!(payload.metadata.dummy_read_operations, 0);
            }
            check_replay(&payload);
        }
    }
}

#[test]
fn repeat_prefixes_and_halt_are_valid_and_invalid_inputs_reject() {
    let opts = options(GuestProgram::BinarySearch, GuestRows::Memory);
    let small = guest_operations(4, 12, &opts).unwrap();
    let large = guest_operations(10, 12, &opts).unwrap();
    assert_eq!(small.initial_memory, large.initial_memory);
    assert_eq!(small.addresses, large.addresses[..16]);
    assert_eq!(small.values, large.values[..16]);
    assert_eq!(small.is_write, large.is_write[..16]);
    check_replay(&small);
    let halt_opts = GuestOptions {
        rows: GuestRows::All,
        fill: GuestFill::Halt,
        ..opts
    };
    let halted = guest_operations(12, 12, &halt_opts).unwrap();
    let start = halted.metadata.pattern_operations;
    assert!(halted.metadata.padding_operations > 0);
    assert!(halted.addresses[start..].iter().all(|&a| a == 0));
    assert!(halted.is_write[start..].iter().all(|&write| !write));
    check_replay(&halted);
    assert!(guest_operations(4, 12, &halt_opts).is_err());
    assert!(
        guest_operations(
            12,
            12,
            &GuestOptions {
                rows: GuestRows::Memory,
                ..halt_opts
            }
        )
        .is_err()
    );
    assert!(
        guest_operations(
            12,
            12,
            &GuestOptions {
                max_cycles: 1,
                ..opts
            }
        )
        .is_err()
    );
    assert!(guest_operations(12, 1, &opts).is_err());
    assert!(random_operations(1, 4, 0).is_err());
    assert!(random_operations(4, 0, 0).is_err());
    let mut bad = random_operations(4, 3, 0).unwrap();
    bad.addresses[0] = 8;
    assert!(bad.into_rmw().is_err());
    let mut bad = random_operations(4, 3, 0).unwrap();
    bad.is_write[0] = false;
    bad.values[0] = bad.initial_memory[bad.addresses[0] as usize] ^ 1;
    assert!(bad.replay().is_err());
}
