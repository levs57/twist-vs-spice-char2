use ram_payload::{
    GUEST_ELF, GuestFill, GuestOptions, GuestProgram, GuestRows, RamPayload, guest, random,
    random_ram,
};

#[test]
fn random_ram_reads_and_replaces_the_same_word() {
    let payload = random_ram(8, 3, 73).unwrap();
    assert_eq!(payload, random_ram(8, 3, 73).unwrap());
    assert_eq!(payload.metadata.memory_model, "shared-address");
    assert_eq!(payload.metadata.read_ports, 1);
    assert_eq!(payload.reads[0], payload.writes);
    let mut memory = payload.initial_memory.clone();
    for (cycle, &address) in payload.writes.iter().enumerate() {
        let word = &mut memory[address as usize];
        assert_eq!(payload.read_values[0][cycle], *word);
        *word = payload.new_values[cycle];
    }
    assert_eq!(payload.replay().unwrap(), memory);
    let generic = random(8, 3, 1, 73).unwrap();
    assert_eq!(generic.metadata.memory_model, "independent-ports");
    assert_ne!(generic.reads[0], generic.writes);
    assert_eq!(payload.initial_memory, generic.initial_memory);
}

#[test]
fn random_is_reproducible_and_reads_precede_writes() {
    let payload = random(8, 5, 3, 73).unwrap();
    assert_eq!(payload, random(8, 5, 3, 73).unwrap());
    assert_ne!(payload.writes, random(8, 5, 3, 74).unwrap().writes);
    payload.replay().unwrap();
    assert_eq!(payload.reads.len(), 3);
    assert_eq!(payload.metadata.memory_model, "independent-ports");
    let encoded = serde_json::to_vec(&payload).unwrap();
    assert_eq!(
        payload,
        serde_json::from_slice::<RamPayload>(&encoded).unwrap()
    );
    let mut tampered = payload;
    tampered.read_values[2][4] ^= 1;
    assert!(
        tampered
            .replay()
            .unwrap_err()
            .to_string()
            .contains("read value mismatch")
    );
}

#[test]
fn actual_guest_outputs_and_ram_values_are_checked() {
    let options = GuestOptions {
        keys: 64,
        queries: 8,
        seed: 1,
        max_cycles: 1 << 14,
        fill: GuestFill::Halt,
        ..GuestOptions::default()
    };
    let payload = guest(14, 12, &options).unwrap();
    assert_eq!(payload.metadata.memory_model, "shared-address");
    let meta = payload.metadata.guest.as_ref().unwrap();
    assert_eq!(meta.output_words, meta.expected_output_words);
    assert_eq!(meta.output_words[1], u64::MAX);
    assert_eq!(meta.output_words[3], u64::MAX);
    assert_eq!(meta.original_address_bits, 16);
    assert_eq!(meta.compact_to_original[0], 0);
    assert!(meta.compact_to_original.windows(2).all(|p| p[0] < p[1]));
    assert_eq!(payload.reads[0], payload.writes);
    assert!(payload.metadata.real_cycles > 100);
    assert!(meta.memory_access_cycles > 0);
    assert!(meta.memory_access_cycles < payload.metadata.real_cycles);
    payload.replay().unwrap();
    let first_idle = payload.metadata.real_cycles;
    assert!(payload.writes[first_idle..].iter().all(|&addr| addr == 0));
    assert_eq!(
        &payload.read_values[0][first_idle..],
        &payload.new_values[first_idle..]
    );
    assert_eq!(payload.metadata.padding_cycles, (1 << 14) - first_idle);
    println!(
        "guest real_cycles={}, compact_words={}, memory_access_cycles={}, minimum_log_cycles={}",
        first_idle,
        meta.compact_to_original.len(),
        meta.memory_access_cycles,
        (first_idle + 1).next_power_of_two().ilog2()
    );
}

#[test]
fn guest_limits_do_not_silently_truncate_or_repeat_execution() {
    let options = GuestOptions {
        keys: 8,
        queries: 2,
        seed: 2,
        max_cycles: 1 << 20,
        fill: GuestFill::Halt,
        ..GuestOptions::default()
    };
    assert!(
        guest(2, 8, &options)
            .unwrap_err()
            .to_string()
            .contains("real microcycles")
    );
    assert!(
        guest(14, 2, &options)
            .unwrap_err()
            .to_string()
            .contains("distinct initialized/accessed words")
    );
    assert!(
        guest(
            14,
            8,
            &GuestOptions {
                max_cycles: 1,
                ..options
            }
        )
        .unwrap_err()
        .to_string()
        .contains("MaxCyclesExceeded")
    );
    let one = guest(14, 8, &options).unwrap();
    let two = guest(15, 8, &options).unwrap();
    assert_eq!(one.metadata.real_cycles, two.metadata.real_cycles);
    assert_eq!(one.writes, two.writes[..1 << 14]);
    assert_eq!(one.new_values, two.new_values[..1 << 14]);
}

#[test]
fn repeated_guest_pattern_keeps_current_memory_across_periods_and_tail() {
    let options = GuestOptions {
        keys: 64,
        queries: 16,
        seed: 1,
        max_cycles: 1 << 14,
        fill: GuestFill::Repeat,
        ..GuestOptions::default()
    };
    let short = guest(12, 12, &options).unwrap();
    let long = guest(14, 12, &options).unwrap();
    let prefix = guest(4, 12, &options).unwrap();
    let halted = guest(
        12,
        12,
        &GuestOptions {
            fill: GuestFill::Halt,
            ..options
        },
    )
    .unwrap();
    let meta = &short.metadata;
    let source = meta.source_cycles;
    assert_eq!(GuestOptions::default().fill, GuestFill::Repeat);
    assert_eq!(meta.trace_mode, "repeat-ram-pattern");
    assert_eq!(meta.real_cycles, 1 << 12);
    assert_eq!(meta.padding_cycles, 0);
    assert_eq!(meta.source_cycles, halted.metadata.real_cycles);
    assert_eq!(meta.full_pattern_repetitions, (1 << 12) / source);
    assert_eq!(meta.tail_pattern_cycles, (1 << 12) % source);
    assert_eq!(meta.replayed_cycles, (1 << 12) - source);
    assert!(long.metadata.full_pattern_repetitions > 1);
    assert!(long.metadata.tail_pattern_cycles > 0);
    assert_eq!(prefix.metadata.full_pattern_repetitions, 0);
    assert_eq!(prefix.metadata.tail_pattern_cycles, 1 << 4);
    assert_eq!(prefix.metadata.replayed_cycles, 0);
    assert_eq!(prefix.metadata.source_cycles, source);
    assert_eq!(prefix.metadata.padding_cycles, 0);
    assert_eq!(prefix.writes, long.writes[..1 << 4]);
    assert_eq!(prefix.read_values[0], long.read_values[0][..1 << 4]);
    assert_eq!(prefix.new_values, long.new_values[..1 << 4]);
    assert_eq!(short.writes[..source], halted.writes[..source]);
    assert_eq!(
        short.read_values[0][..source],
        halted.read_values[0][..source]
    );
    assert_eq!(short.new_values[..source], halted.new_values[..source]);
    assert_eq!(short.writes, long.writes[..1 << 12]);
    assert_eq!(short.read_values[0], long.read_values[0][..1 << 12]);
    assert_eq!(short.new_values, long.new_values[..1 << 12]);
    for cycle in source..long.writes.len() {
        assert_eq!(long.writes[cycle], long.writes[cycle % source]);
    }
    // Later source-pattern reads see writes made during the previous period.
    assert!(
        (source..long.writes.len())
            .any(|i| long.read_values[0][i] != long.read_values[0][i % source])
    );
    // A recorded read-only row preserves the new current value, not its cached source value.
    assert!((source..long.writes.len()).any(|i| {
        let source_i = i % source;
        long.read_values[0][source_i] == long.new_values[source_i]
            && long.read_values[0][i] != long.read_values[0][source_i]
            && long.new_values[i] == long.read_values[0][i]
    }));
    short.replay().unwrap();
    long.replay().unwrap();
    let guest_meta = long.metadata.guest.as_ref().unwrap();
    assert_eq!(guest_meta.output_words, guest_meta.expected_output_words);
    assert_eq!(
        guest_meta.retired_instructions,
        halted.metadata.guest.as_ref().unwrap().retired_instructions
    );
    assert!(
        guest_meta.memory_access_cycles
            >= guest_meta.source_memory_access_cycles * long.metadata.full_pattern_repetitions
    );
    assert!(guest_meta.memory_access_cycles <= long.metadata.real_cycles);
    println!(
        "source={source}, repeat_cycles={}, full_patterns={}, tail={}, replayed={}",
        long.metadata.real_cycles,
        long.metadata.full_pattern_repetitions,
        long.metadata.tail_pattern_cycles,
        long.metadata.replayed_cycles
    );
}

fn assert_guest_stats(payload: &RamPayload) {
    let meta = payload.metadata.guest.as_ref().unwrap();
    assert_eq!(
        meta.memory_access_cycles + meta.dummy_access_cycles,
        payload.writes.len()
    );
    assert_eq!(
        meta.zero_read_cycles,
        payload.read_values[0]
            .iter()
            .filter(|&&value| value == 0)
            .count()
    );
    assert_eq!(
        meta.unchanged_write_cycles,
        payload.read_values[0]
            .iter()
            .zip(&payload.new_values)
            .filter(|(old, new)| old == new)
            .count()
    );
    assert_eq!(
        meta.distinct_accessed_words,
        payload
            .writes
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
    assert!(meta.dummy_zero_read_cycles <= meta.zero_read_cycles);
    assert!(meta.dummy_zero_read_cycles <= meta.dummy_access_cycles);
}

#[test]
fn pointer_chase_is_one_permutation_cycle_with_checked_outputs() {
    let options = GuestOptions {
        program: GuestProgram::PointerChase,
        rows: GuestRows::Memory,
        size: 17,
        operations: 39,
        ..GuestOptions::default()
    };
    let payload = guest(12, 10, &options).unwrap();
    let meta = payload.metadata.guest.as_ref().unwrap();
    let next = &meta.input_words[1..1 + options.size];
    let mut visited = std::collections::BTreeSet::new();
    let mut current = 0;
    for _ in 0..options.size {
        assert!(visited.insert(current));
        current = next[current] as usize;
    }
    assert_eq!(current, 0);
    assert_eq!(visited.len(), options.size);
    let mut expected_xor = 0;
    for _ in 0..options.operations {
        current = next[current] as usize;
        expected_xor ^= current as u64;
    }
    assert_eq!(meta.output_words, [current as u64, expected_xor]);
    assert_eq!(meta.output_words, meta.expected_output_words);
    assert_eq!(
        payload.metadata.pattern_cycles,
        meta.source_memory_access_cycles
    );
    assert!(payload.metadata.source_cycles > payload.metadata.pattern_cycles);
    assert_eq!(meta.memory_access_cycles, 1 << 12);
    assert_eq!(meta.dummy_access_cycles, 0);
    assert_eq!(meta.dummy_zero_read_cycles, 0);
    assert_guest_stats(&payload);
    payload.replay().unwrap();
}

#[test]
fn erc20_balances_change_and_memory_filter_retains_real_word_zero() {
    let options = GuestOptions {
        program: GuestProgram::Erc20MixedBlock,
        size: 16,
        operations: 30,
        ..GuestOptions::default()
    };
    let all = guest(15, 12, &options).unwrap();
    let memory = guest(
        15,
        12,
        &GuestOptions {
            rows: GuestRows::Memory,
            ..options
        },
    )
    .unwrap();
    let prefix = guest(
        8,
        12,
        &GuestOptions {
            rows: GuestRows::Memory,
            ..options
        },
    )
    .unwrap();
    let all_meta = all.metadata.guest.as_ref().unwrap();
    let meta = memory.metadata.guest.as_ref().unwrap();
    assert_eq!(all.initial_memory, memory.initial_memory);
    assert_eq!(all_meta.compact_to_original, meta.compact_to_original);
    assert_eq!(all.metadata.source_cycles, memory.metadata.source_cycles);
    assert_eq!(meta.output_words[0], 24); // 8 successes per ten: distinct and self transfers.
    let initial = &meta.input_words[1..1 + options.size];
    assert!(initial.iter().all(|&balance| balance >= 1_000_000));
    let mut balances: std::collections::BTreeMap<usize, u64> =
        initial.iter().copied().enumerate().collect();
    let mut successes = 0;
    for transfer in meta.input_words[options.size + 2..].chunks_exact(3) {
        let (from, to, amount) = (transfer[0] as usize, transfer[1] as usize, transfer[2]);
        if let (Some(&balance), true) = (balances.get(&from), to < options.size) {
            if balance >= amount {
                if from != to {
                    *balances.get_mut(&from).unwrap() -= amount;
                    *balances.get_mut(&to).unwrap() += amount;
                }
                successes += 1;
            }
        }
    }
    let expected: Vec<_> = std::iter::once(successes)
        .chain(balances.values().copied())
        .collect();
    assert_eq!(meta.output_words, expected);
    assert_eq!(meta.output_words, meta.expected_output_words);
    assert_ne!(&meta.output_words[1..], initial);
    assert_eq!(
        meta.output_words[1..].iter().sum::<u64>(),
        initial.iter().sum::<u64>()
    );

    // Use the original VM's row flag as the oracle; address zero can be genuine memory.
    let params = om_elf::Params::new(16, 12, 11);
    let image = om_elf::load(options.program.elf(), &params).unwrap();
    let input: Vec<u8> = meta
        .input_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let m0 = om_elf::build_m0(&image.ram_words, &input, &params).unwrap();
    let mut vm = om_isa::bytecode::VirtualMachine::new(
        om_isa::bytecode::expand(&image.code, image.entry_pc),
        &m0,
        16,
    );
    let result = vm.run(options.max_cycles).unwrap();
    let retained: Vec<_> = result.trace.iter().filter(|row| row.rom.mem).collect();
    assert_eq!(retained.len(), memory.metadata.pattern_cycles);
    assert_eq!(result.trace.len(), memory.metadata.source_cycles);
    assert!(retained.iter().any(|row| row.ram_idx == 0));
    for (cycle, record) in retained.iter().enumerate() {
        assert_eq!(
            meta.compact_to_original[memory.writes[cycle] as usize] as usize,
            record.ram_idx
        );
        assert_eq!(memory.read_values[0][cycle], record.ram_old);
        assert_eq!(memory.new_values[cycle], record.ram_new);
    }
    // Across periods and the final prefix, removing dummy transitions must yield
    // the same evolving RAM sequence as the memory-only payload. This also checks
    // the exact useful-access denominator used by the proportional benchmark.
    let all_memory_positions: Vec<_> = (0..all.writes.len())
        .filter(|&cycle| result.trace[cycle % result.trace.len()].rom.mem)
        .collect();
    assert_eq!(all_meta.memory_access_cycles, all_memory_positions.len());
    for (memory_cycle, &all_cycle) in all_memory_positions.iter().enumerate() {
        assert_eq!(memory.writes[memory_cycle], all.writes[all_cycle]);
        assert_eq!(
            memory.read_values[0][memory_cycle],
            all.read_values[0][all_cycle]
        );
        assert_eq!(memory.new_values[memory_cycle], all.new_values[all_cycle]);
    }
    let expected_dummy_zeros = (0..all.writes.len())
        .filter(|&cycle| {
            !result.trace[cycle % result.trace.len()].rom.mem && all.read_values[0][cycle] == 0
        })
        .count();
    assert_eq!(all_meta.dummy_zero_read_cycles, expected_dummy_zeros);
    assert!(memory.writes.iter().any(|&address| address == 0));
    assert_eq!(meta.memory_access_cycles, memory.writes.len());
    assert_eq!(meta.dummy_access_cycles, 0);
    assert_eq!(meta.dummy_zero_read_cycles, 0);
    assert!(meta.zero_read_cycles > 0); // Legitimate zero-valued loads/stores remain.
    assert!(all_meta.dummy_access_cycles > 0);
    assert_eq!(all.metadata.pattern_cycles, all.metadata.source_cycles);
    assert_eq!(prefix.writes, memory.writes[..1 << 8]);
    assert_eq!(prefix.read_values[0], memory.read_values[0][..1 << 8]);
    assert_eq!(prefix.new_values, memory.new_values[..1 << 8]);
    assert_eq!(
        memory.metadata.full_pattern_repetitions,
        (1 << 15) / retained.len()
    );
    assert_eq!(
        memory.metadata.tail_pattern_cycles,
        (1 << 15) % retained.len()
    );
    assert_guest_stats(&all);
    assert_guest_stats(&memory);
    memory.replay().unwrap();
    assert_eq!(
        memory,
        guest(
            15,
            12,
            &GuestOptions {
                rows: GuestRows::Memory,
                ..options
            }
        )
        .unwrap()
    );
    println!(
        "erc20 source={}, pattern={}, distinct={}, zero_reads={}, dummy={}",
        memory.metadata.source_cycles,
        memory.metadata.pattern_cycles,
        meta.distinct_accessed_words,
        meta.zero_read_cycles,
        meta.dummy_access_cycles
    );
}

#[test]
fn guest_program_options_reject_invalid_limits_and_padding_modes() {
    assert_eq!(
        serde_json::to_string(&GuestProgram::Erc20MixedBlock).unwrap(),
        "\"erc20-mixed-block\""
    );
    assert_eq!(
        serde_json::to_string(&GuestRows::Memory).unwrap(),
        "\"memory\""
    );
    let options = GuestOptions {
        program: GuestProgram::PointerChase,
        rows: GuestRows::Memory,
        ..GuestOptions::default()
    };
    assert!(
        guest(
            12,
            12,
            &GuestOptions {
                fill: GuestFill::Halt,
                ..options
            }
        )
        .unwrap_err()
        .to_string()
        .contains("memory-only")
    );
    assert!(guest(12, 12, &GuestOptions { size: 0, ..options }).is_err());
    assert!(
        guest(
            12,
            12,
            &GuestOptions {
                operations: 0,
                ..options
            }
        )
        .is_err()
    );
    assert!(
        guest(
            12,
            12,
            &GuestOptions {
                size: 4094,
                ..options
            }
        )
        .unwrap_err()
        .to_string()
        .contains("capacity")
    );
    assert!(
        guest(
            12,
            12,
            &GuestOptions {
                operations: 100,
                max_cycles: 10,
                ..options
            }
        )
        .unwrap_err()
        .to_string()
        .contains("budget")
    );
    let erc = GuestOptions {
        program: GuestProgram::Erc20MixedBlock,
        ..options
    };
    assert!(
        guest(12, 12, &GuestOptions { size: 1025, ..erc })
            .unwrap_err()
            .to_string()
            .contains("1024")
    );
    assert!(
        guest(
            12,
            12,
            &GuestOptions {
                operations: 2000,
                ..erc
            }
        )
        .unwrap_err()
        .to_string()
        .contains("capacity")
    );
}

#[test]
fn embedded_elf_and_original_source_match_provenance_manifest() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../../../payloads/open-magnetar/fixture.json")).unwrap();
    let digest: String = om_elf::Hasher::Sha256
        .digest(GUEST_ELF)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(manifest["elf_sha256"], digest);
    assert_eq!(manifest["upstream_commit"], ram_payload::UPSTREAM_COMMIT);
    let sources: [(&str, &[u8]); 4] = [
        (
            "examples/binary-search.c",
            include_bytes!("../../../payloads/open-magnetar/examples/binary-search.c"),
        ),
        (
            "rt/start.S",
            include_bytes!("../../../payloads/open-magnetar/rt/start.S"),
        ),
        (
            "rt/om.h",
            include_bytes!("../../../payloads/open-magnetar/rt/om.h"),
        ),
        (
            "link.ld",
            include_bytes!("../../../payloads/open-magnetar/link.ld"),
        ),
    ];
    for (name, bytes) in sources {
        let digest: String = om_elf::Hasher::Sha256
            .digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(manifest["source_sha256"][name], digest, "{name}");
    }
    let new_fixtures: [(GuestProgram, &str, &str, &[u8]); 2] = [
        (
            GuestProgram::Erc20MixedBlock,
            include_str!("../../../payloads/open-magnetar/erc20-mixed-block.fixture.json"),
            "examples/erc20-mixed-block.c",
            include_bytes!("../../../payloads/open-magnetar/examples/erc20-mixed-block.c"),
        ),
        (
            GuestProgram::PointerChase,
            include_str!("../../../payloads/open-magnetar/pointer-chase.fixture.json"),
            "examples/pointer-chase.c",
            include_bytes!("../../../payloads/open-magnetar/examples/pointer-chase.c"),
        ),
    ];
    for (program, json, source_name, source_bytes) in new_fixtures {
        let manifest: serde_json::Value = serde_json::from_str(json).unwrap();
        let digest: String = om_elf::Hasher::Sha256
            .digest(program.elf())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(manifest["elf_sha256"], digest, "{} ELF", program.name());
        assert_eq!(manifest["upstream_commit"], ram_payload::UPSTREAM_COMMIT);
        assert_eq!(
            manifest["elf_bytes"].as_u64(),
            Some(program.elf().len() as u64)
        );
        let sources: [(&str, &[u8]); 4] = [
            (source_name, source_bytes),
            (
                "rt/start.S",
                include_bytes!("../../../payloads/open-magnetar/rt/start.S"),
            ),
            (
                "rt/om.h",
                include_bytes!("../../../payloads/open-magnetar/rt/om.h"),
            ),
            (
                "link.ld",
                include_bytes!("../../../payloads/open-magnetar/link.ld"),
            ),
        ];
        for (name, bytes) in sources {
            let digest: String = om_elf::Hasher::Sha256
                .digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                manifest["source_sha256"][name],
                digest,
                "{} {name}",
                program.name()
            );
        }
    }
}
