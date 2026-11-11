use super::*;

#[test]
fn one_replay_returns_validated_old_values_and_final_memory() {
    let mut payload = random_operations(2, 1, 19).unwrap();
    payload.initial_memory = vec![11, 22];
    payload.addresses = vec![0, 1, 0, 1];
    payload.values = vec![11, 33, 44, 33];
    payload.is_write = vec![false, true, true, false];
    assert_eq!(
        payload.replay_with_old_values().unwrap(),
        (vec![11, 22, 11, 33], vec![44, 33])
    );
    assert_eq!(payload.replay().unwrap(), vec![44, 33]);
    payload.values[3] = 22;
    assert!(payload.replay_with_old_values().is_err());
    assert!(payload.replay().is_err());
}

fn operation(
    address: u32,
    store: Option<Store>,
    cycle: usize,
    part: u8,
    parts: u8,
) -> PatternOperation {
    PatternOperation {
        address,
        store,
        memory: true,
        source_cycle: cycle,
        part,
        parts,
    }
}

#[test]
fn narrow_stores_preserve_other_lanes_and_read_current_memory_on_repetition() {
    let initial = [0x8877_6655_4433_2211];
    for (width, offset, source, expected) in [
        (0, 3, 0xab, 0x8877_6655_ab33_2211),
        (1, 2, 0xcdef, 0x8877_6655_cdef_2211),
        (2, 4, 0x0123_4567, 0x0123_4567_4433_2211),
    ] {
        let pattern = [
            operation(0, None, 7, 1, 2),
            operation(
                0,
                Some(Store {
                    width,
                    offset,
                    source,
                }),
                7,
                2,
                2,
            ),
        ];
        let filled = fill_pattern(&initial, &pattern, 4, true);
        assert_eq!(filled.is_write, [false, true, false, true]);
        assert_eq!(filled.values, [initial[0], expected, expected, expected]);
        assert_eq!(filled.narrow_reads, 2);
        assert_eq!(filled.narrow_writes, 2);
        assert_eq!(filled.completed_rows, 2);
        assert_eq!(filled.partial, None);
    }
}

#[test]
fn partial_source_store_stops_before_its_write_and_halt_reads_preserve_word_zero() {
    let pattern = [
        operation(0, None, 4, 1, 1),
        operation(0, None, 7, 1, 2),
        operation(
            0,
            Some(Store {
                width: 0,
                offset: 0,
                source: 0x34,
            }),
            7,
            2,
            2,
        ),
    ];
    let filled = fill_pattern(&[0x12], &pattern, 8, true);
    assert_eq!(
        filled.values,
        [0x12, 0x12, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34]
    );
    assert_eq!(filled.narrow_reads, 3);
    assert_eq!(filled.narrow_writes, 2);
    assert_eq!(filled.completed_rows, 5);
    assert_eq!(
        filled.partial,
        Some(PartialSourceOperation {
            source_cycle: 7,
            pattern_iteration: 2,
            emitted_micro_operations: 1,
            total_micro_operations: 2,
        })
    );
    let halted = fill_pattern(&[0xdead_beef], &pattern[..1], 4, false);
    assert_eq!(halted.values, [0xdead_beef; 4]);
    assert_eq!(halted.is_write, [false; 4]);
    assert_eq!(halted.memory_operations, 1);
    assert_eq!(halted.completed_rows, 1);
}

#[test]
fn unchanged_full_store_keeps_its_write_kind() {
    let pattern = [operation(
        0,
        Some(Store {
            width: 3,
            offset: 0,
            source: 17,
        }),
        9,
        1,
        1,
    )];
    let filled = fill_pattern(&[17], &pattern, 4, true);
    assert_eq!(filled.values, [17; 4]);
    assert_eq!(filled.is_write, [true; 4]);
    assert_eq!(filled.narrow_reads, 0);
    assert_eq!(filled.completed_rows, 4);
}
