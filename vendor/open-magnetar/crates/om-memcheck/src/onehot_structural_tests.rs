// Regression tests for structural OneHot reuse. Witness contents must not
// decide which independently declared component polynomials receive work.
use super::*;

fn selection(comps: &[&AddrComps]) -> (usize, Vec<usize>) {
    let flat: Vec<_> = comps
        .iter()
        .flat_map(|&component| (0..component.d()).map(move |index| (component, index)))
        .collect();
    let (sources, original_to_source) = onehot_component_sources(&flat);
    (sources.len(), original_to_source)
}

fn fixture_addresses(ac: usize, n: usize, pattern: usize) -> Vec<usize> {
    let mask = (1usize << ac) - 1;
    (0..1usize << n)
        .map(|j| match pattern {
            0 => 0,
            // The compact pointer's narrow numeric range, including its
            // coincident all-zero high chunks at a=16/ac=4.
            1 => 1 + j % 70.min((1usize << (4 * ac)) - 1),
            // Every chunk is the same constant digit.
            2 => (0..4).fold(0, |address, i| address | ((3 & mask) << (i * ac))),
            // All four component streams are equal but change with the cycle.
            3 => {
                let digit = (j * 7 + 3) & mask;
                (0..4).fold(0, |address, i| address | (digit << (i * ac)))
            }
            _ => unreachable!(),
        })
        .collect()
}

fn compare_original_proof(comps: &[&AddrComps], ac: usize, n: usize, honest: bool) {
    let names: Vec<_> = (0..comps.len() * 4)
        .map(|i| format!("original/component/{i}"))
        .collect();
    let mut expected_tr = Transcript::new_default(b"structural-onehot-test");
    let mut actual_tr = expected_tr.clone();
    let expected = dense_onehot(comps, &names, ac, n, &mut expected_tr);
    let actual = onehot_prove(comps, &names, ac, n, &mut actual_tr, b"onehot-oracle");
    assert_eq!(actual, expected, "ac={ac}, n={n}");
    assert_eq!(actual.0.evals.len(), names.len());
    let expected_after = expected_tr.challenge_k(b"after");
    assert_eq!(actual_tr.challenge_k(b"after"), expected_after);

    let mut verifier_tr = Transcript::new_default(b"structural-onehot-test");
    let verified = onehot_verify(&names, ac, n, &actual.0, &mut verifier_tr, b"onehot-oracle");
    if honest {
        assert_eq!(verified.unwrap(), actual.1);
        assert_eq!(verifier_tr.challenge_k(b"after"), expected_after);
    } else {
        assert!(verified.is_err(), "invalid Bits witness accepted");
    }
}

#[test]
fn four_declared_chunks_stay_four_regardless_of_their_contents() {
    for n in [0, 1, 4, 6] {
        for ac in [1, 2, 4] {
            for pattern in 0..4 {
                let addresses = fixture_addresses(ac, n, pattern);
                let compact = AddrComps::from_usizes(&addresses, 4 * ac, 4);
                let bits = AddrComps::Bits {
                    tables: comp_tables(&addresses, 4 * ac, 4),
                    ac,
                };
                assert_eq!(selection(&[&compact]), (4, vec![0, 1, 2, 3]));
                assert_eq!(selection(&[&bits]), (4, vec![0, 1, 2, 3]));
            }
        }
    }
}

#[test]
fn explicit_shared_stream_reuses_same_chunk_but_equal_clones_do_not() {
    for ac in [1, 2, 4] {
        let addresses = fixture_addresses(ac, 4, 3);
        let compact = AddrComps::from_usizes(&addresses, 4 * ac, 4);
        let independent = compact.clone();
        assert_eq!(compact, independent);
        let read = &compact;
        let known_shared_write = read;
        assert_eq!(
            selection(&[read, known_shared_write]),
            (4, vec![0, 1, 2, 3, 0, 1, 2, 3])
        );
        assert_eq!(selection(&[read, &independent]), (8, (0..8).collect()));
        assert_eq!(
            selection(&[read, known_shared_write, &independent]),
            (8, vec![0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7])
        );
        assert!(components_equal(read, known_shared_write));
        assert!(!components_equal(read, &independent));

        let bits = AddrComps::Bits {
            tables: comp_tables(&addresses, 4 * ac, 4),
            ac,
        };
        let bits_clone = bits.clone();
        assert_eq!(
            selection(&[&bits, &bits]),
            (4, vec![0, 1, 2, 3, 0, 1, 2, 3])
        );
        assert_eq!(selection(&[&bits, &bits_clone]), (8, (0..8).collect()));
        assert_eq!(selection(&[read, &bits]), (8, (0..8).collect()));
        assert!(!components_equal(&bits, &bits_clone));
        assert!(!components_equal(read, &bits));
    }
}

#[test]
fn structural_selection_preserves_full_dense_proofs_and_transcripts() {
    for n in [0, 1, 4] {
        for ac in [1, 2, 4] {
            for pattern in 0..4 {
                let addresses = fixture_addresses(ac, n, pattern);
                let compact = AddrComps::from_usizes(&addresses, 4 * ac, 4);
                let independent = compact.clone();
                let bits = AddrComps::Bits {
                    tables: comp_tables(&addresses, 4 * ac, 4),
                    ac,
                };
                compare_original_proof(&[&compact], ac, n, true);
                compare_original_proof(&[&compact, &compact], ac, n, true);
                compare_original_proof(&[&compact, &independent], ac, n, true);
                compare_original_proof(&[&compact, &bits], ac, n, true);
            }
        }
    }
}

#[test]
fn invalid_bits_keep_distinct_components_and_match_unmerged_oracle() {
    for n in [0, 1, 4] {
        for ac in [1, 2, 4] {
            let addresses = fixture_addresses(ac, n, 3);
            let mut tables = comp_tables(&addresses, 4 * ac, 4);
            let actual = addresses[0] & ((1 << ac) - 1);
            // Make the first row contain two hot entries. Equal chunk tables
            // elsewhere and equal clones must not hide this invalid row.
            tables[0][(actual + 1) % (1 << ac)] = true;
            let invalid = AddrComps::Bits { tables, ac };
            let independent = invalid.clone();
            assert_eq!(selection(&[&invalid]), (4, vec![0, 1, 2, 3]));
            assert_eq!(
                selection(&[&invalid, &invalid]),
                (4, vec![0, 1, 2, 3, 0, 1, 2, 3])
            );
            assert_eq!(selection(&[&invalid, &independent]), (8, (0..8).collect()));
            compare_original_proof(&[&invalid], ac, n, false);
            compare_original_proof(&[&invalid, &invalid], ac, n, false);
            compare_original_proof(&[&invalid, &independent], ac, n, false);
        }
    }
}
