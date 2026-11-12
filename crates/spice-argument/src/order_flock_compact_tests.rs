//! Independent bit-major packing and conditional terminal-binding oracles.
//! These fixtures allocate only O(w * 2^n) bits, never a 2^a RAM table.

use super::*;
use om_sumcheck::{eq_table, mle_eval_bits};

fn compact_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn compact_inputs(n: usize, a: usize, pattern: usize) -> (Vec<u32>, Vec<u32>) {
    let w = n.max(a) + 1;
    let base = 1u64 << (w - 1);
    let mut state = 0x71a9_0806_fed1 ^ a as u64;
    let previous: Vec<u32> = (0..1usize << n)
        .map(|j| {
            let time = base + j as u64;
            match pattern {
                0 => 0,
                1 => (time - 1) as u32,
                _ => (compact_random(&mut state) % time) as u32,
            }
        })
        .collect();
    let carries = previous
        .iter()
        .enumerate()
        .map(|(j, &p)| {
            // Independent Boolean full-adder on p + !time + 1.
            let time = base + j as u64;
            let mut cin = true;
            let mut word = 0u32;
            for bit in 0..w {
                if bit != 0 && cin {
                    word |= 1u32 << (bit - 1);
                }
                let x = (p >> bit) & 1 != 0;
                let y = (time >> bit) & 1 == 0;
                cin = (x && y) || (x && cin) || (y && cin);
            }
            assert!(!cin, "unsigned p < time");
            word
        })
        .collect();
    (previous, carries)
}

fn compact_dimensions(n: usize, a: usize) -> (usize, usize, usize) {
    let w = n.max(a) + 1;
    let gate_bits = w.next_power_of_two().trailing_zeros() as usize;
    (w, gate_bits, n + gate_bits)
}

/// Boolean construction q = j + 2^n * b, with all unused gate planes zero.
fn compact_dense_rows(previous: &[u32], carries: &[u32], n: usize, a: usize) -> [Vec<u8>; 3] {
    let (w, _, m) = compact_dimensions(n, a);
    let mut rows = std::array::from_fn(|_| vec![0u8; (1usize << m) / 8]);
    for j in 0..1usize << n {
        let time = (1u64 << (w - 1)) + j as u64;
        for b in 0..w {
            let cin = if b == 0 {
                1
            } else {
                (carries[j] >> (b - 1)) & 1
            };
            let cout = if b + 1 == w { 0 } else { (carries[j] >> b) & 1 };
            let bits = [
                ((previous[j] >> b) & 1) ^ cin,
                1 ^ ((time >> b) as u32 & 1) ^ cin,
                cin ^ cout,
            ];
            assert_eq!(bits[0] & bits[1], bits[2]);
            let q = j + (b << n);
            for (out, bit) in rows.iter_mut().zip(bits) {
                out[q / 8] |= (bit as u8) << (q % 8);
            }
        }
    }
    rows
}

fn compact_functional(n: usize, claim: &ZerocheckClaim, at_c: bool) -> (Vec<K>, Vec<K>) {
    let point = if at_c {
        &claim.r_rest
    } else {
        &claim.mlv_challenges
    };
    let mut cycle = vec![K::ZERO; K_SKIP];
    cycle.extend(point[..n - K_SKIP].iter().copied().map(from_flock));
    let weights = lagrange_weights_naive(K_SKIP, claim.z)
        .into_iter()
        .map(from_flock)
        .collect();
    (cycle, weights)
}

/// The public low-64 functional is applied to real cycle bits, not interpreted
/// as Eq at the placeholder zeros in `point[..6]`.
fn compact_eval_planes(words: &[u32], width: usize, point: &[K], prefix: &[K]) -> Vec<K> {
    assert_eq!(prefix.len(), 64);
    let high = eq_table(&point[K_SKIP..]);
    let mut sums = vec![K::ZERO; width];
    for (j, &word) in words.iter().enumerate() {
        let weight = prefix[j % 64] * high[j / 64];
        for (bit, sum) in sums.iter_mut().enumerate() {
            if (word >> bit) & 1 != 0 {
                *sum += weight;
            }
        }
    }
    sums
}

fn compact_sources(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    claim: &ZerocheckClaim,
) -> (Vec<K>, Vec<K>, Vec<K>) {
    let w = n.max(a) + 1;
    let (ab, prefix) = compact_functional(n, claim, false);
    let (c, prefix_c) = compact_functional(n, claim, true);
    assert_eq!(prefix, prefix_c);
    (
        compact_eval_planes(previous, w, &ab, &prefix),
        compact_eval_planes(carries, w - 1, &ab, &prefix),
        compact_eval_planes(carries, w - 1, &c, &prefix),
    )
}

fn compact_raw_oracle(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    initial: Transcript,
    add_unused_gate: bool,
) -> (ZerocheckProof, ZerocheckClaim, Transcript) {
    let (w, _, m) = compact_dimensions(n, a);
    let mut rows = compact_dense_rows(previous, carries, n, a);
    if add_unused_gate {
        assert!(w < w.next_power_of_two());
        // A=B=C=1 is still a valid Boolean gate. Its extra uncommitted
        // contribution must be rejected by terminal source authentication.
        let at = (w << n) + 17;
        for row in &mut rows {
            row[at / 8] |= 1 << (at % 8);
        }
    }
    let mut tr = initial;
    let _shape = setup(n, a, &mut tr, b"order").unwrap();
    let mut challenger = OmChallenger { tr };
    let (proof, terminal) = prove_packed_padded(
        &rows[0],
        &rows[1],
        &rows[2],
        m,
        &PaddingSpec::dense(m),
        &mut challenger,
    );
    (proof, terminal, challenger.tr)
}

#[test]
fn compact_dynamic_widths_match_independent_boolean_donor_proofs_and_transcripts() {
    let n = 13;
    for a in [13, 20, 21, 23, 31] {
        for pattern in 0..3 {
            let (previous, carries) = compact_inputs(n, a, pattern);
            let initial = Transcript::new_default(b"compact-boolean-oracle");
            let (oracle_proof, terminal, mut oracle_tr) =
                compact_raw_oracle(&previous, &carries, n, a, initial.clone(), false);
            let mut tr = initial;
            let (proof, claims) = prove(&previous, &carries, n, a, &mut tr, b"order").unwrap();
            assert_eq!(proof.zerocheck, oracle_proof, "a={a}, pattern={pattern}");
            let (p, ab, c) = compact_sources(&previous, &carries, n, a, &terminal);
            assert_eq!(proof.previous_ab, p);
            assert_eq!(proof.carries_ab, ab);
            assert_eq!(proof.carries_c, c);
            let (point_ab, weights) = compact_functional(n, &terminal, false);
            let (point_c, _) = compact_functional(n, &terminal, true);
            assert_eq!(claims.point_ab, point_ab);
            assert_eq!(claims.point_c, point_c);
            assert_eq!(claims.prefix_weights.as_ref(), Some(&weights));
            oracle_tr.absorb_ks(b"order/previous-ab", &p);
            oracle_tr.absorb_ks(b"order/carries-ab", &ab);
            oracle_tr.absorb_ks(b"order/carries-c", &c);
            assert_eq!(tr, oracle_tr);
            let mut verifier = Transcript::new_default(b"compact-boolean-oracle");
            assert_eq!(
                verify(n, a, &proof, &mut verifier, b"order").unwrap(),
                claims
            );
            assert_eq!(verifier, tr);
        }
    }
}

#[test]
fn compact_source_functional_is_not_eq_at_the_six_placeholder_coordinates() {
    let (n, a) = (13, 20);
    let (previous, carries) = compact_inputs(n, a, 2);
    let (_, terminal) = prove_zerocheck(
        &previous,
        &carries,
        n,
        a,
        &mut Transcript::new_default(b"compact-non-eq"),
        b"order",
    )
    .unwrap();
    let (point, prefix) = compact_functional(n, &terminal, false);
    let weighted = compact_eval_planes(&previous, 21, &point, &prefix);
    let ordinary: Vec<K> = (0..21)
        .map(|bit| {
            let table: Vec<bool> = previous
                .iter()
                .map(|&word| (word >> bit) & 1 != 0)
                .collect();
            mle_eval_bits(&table, &point)
        })
        .collect();
    assert!(weighted.iter().zip(&ordinary).any(|(a, b)| a != b));
    assert_ne!(prefix, eq_table(&point[..K_SKIP]));
}

#[test]
fn compact_terminal_binding_rejects_each_modified_source_and_unused_gate_contribution() {
    let (n, a) = (13, 20);
    let (previous, carries) = compact_inputs(n, a, 2);
    let (_, terminal) = prove_zerocheck(
        &previous,
        &carries,
        n,
        a,
        &mut Transcript::new_default(b"compact-bindings"),
        b"order",
    )
    .unwrap();
    let (p, ab, c) = compact_sources(&previous, &carries, n, a, &terminal);
    check_sources(n, a, &terminal, &p, &ab, &c).unwrap();
    for (group, len) in [(0, p.len()), (1, ab.len()), (2, c.len())] {
        for bit in 0..len {
            let (mut p, mut ab, mut c) = (p.clone(), ab.clone(), c.clone());
            match group {
                0 => p[bit] += K::ONE,
                1 => ab[bit] += K::ONE,
                _ => c[bit] += K::ONE,
            }
            assert!(
                check_sources(n, a, &terminal, &p, &ab, &c).is_err(),
                "group={group}, bit={bit}"
            );
        }
    }
    assert!(check_sources(n, a, &terminal, &p[..p.len() - 1], &ab, &c).is_err());
    let initial = Transcript::new_default(b"compact-unused-gate");
    let (proof, bad_terminal, _) =
        compact_raw_oracle(&previous, &carries, n, a, initial.clone(), true);
    // The extra A=B=C gate satisfies Flock's Boolean relation by itself.
    let verified = verify_zerocheck_with(n, a, &proof, &mut initial.clone(), b"order").unwrap();
    assert_eq!(verified.z, bad_terminal.z);
    let (p, ab, c) = compact_sources(&previous, &carries, n, a, &verified);
    assert!(check_sources(n, a, &verified, &p, &ab, &c).is_err());
}

fn compact_packed_functional(rows: &[u8], low: &[K], high_point: &[K]) -> K {
    let high = eq_table(high_point);
    rows.chunks_exact(8)
        .zip(high)
        .fold(K::ZERO, |total, (bytes, weight)| {
            if weight.is_zero() {
                return total;
            }
            let mut word = u64::from_le_bytes(bytes.try_into().unwrap());
            let mut value = K::ZERO;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                value += low[bit];
            }
            total + weight * value
        })
}

#[test]
fn compact_public_time_moments_handle_phi_nodes_boolean_points_and_gate_padding() {
    use flock_core::field::PHI_8_TABLE;
    let (n, a) = (13, 20);
    let (_, _, m) = compact_dimensions(n, a);
    let (previous, carries) = compact_inputs(n, a, 2);
    let rows = compact_dense_rows(&previous, &carries, n, a);
    let mut state = 0x8a81_1234_5a55;
    let mut random_field = || {
        to_flock(K::from_u128(
            u128::from(compact_random(&mut state)) | (u128::from(compact_random(&mut state)) << 64),
        ))
    };
    let random_z = random_field();
    for kind in 0..3 {
        let ab: Vec<F128> = (0..m - K_SKIP)
            .map(|bit| match kind {
                0 => F128::ZERO,
                1 => {
                    if bit % 2 == 0 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                }
                _ => random_field(),
            })
            .collect();
        let c: Vec<F128> = match kind {
            0 => vec![F128::ONE; m - K_SKIP],
            1 => ab.iter().map(|&x| F128::ONE + x).collect(),
            _ => (0..m - K_SKIP).map(|_| random_field()).collect(),
        };
        let ab_k: Vec<K> = ab.iter().copied().map(from_flock).collect();
        let c_k: Vec<K> = c.iter().copied().map(from_flock).collect();
        for z in [F128::ZERO, F128::ONE, PHI_8_TABLE[17], random_z] {
            let low: Vec<K> = lagrange_weights_naive(K_SKIP, z)
                .into_iter()
                .map(from_flock)
                .collect();
            let claim = ZerocheckClaim {
                z,
                mlv_challenges: ab.clone(),
                r_rest: c.clone(),
                a_eval: to_flock(compact_packed_functional(&rows[0], &low, &ab_k)),
                b_eval: to_flock(compact_packed_functional(&rows[1], &low, &ab_k)),
                c_eval: to_flock(compact_packed_functional(&rows[2], &low, &c_k)),
            };
            let (p, ab, c) = compact_sources(&previous, &carries, n, a, &claim);
            let sources = check_sources(n, a, &claim, &p, &ab, &c).unwrap();
            assert_eq!(sources.prefix_weights, Some(low));
        }
    }
}

#[test]
fn compact_terminal_bitplanes_bind_through_mixed_functional_and_ordinary_mosaic() {
    use om_mosaic::{
        Claim, PrefixClaim, batch_prove_packed_with_prefixes, batch_verify_with_prefixes, build_g,
        compute_layout,
    };
    use om_sumcheck::PackedBits;
    use std::collections::BTreeMap;

    let (n, a) = (13, 20);
    let w = n.max(a) + 1;
    let (previous, carries) = compact_inputs(n, a, 2);
    let (_, terminal) = prove_zerocheck(
        &previous,
        &carries,
        n,
        a,
        &mut Transcript::new_default(b"compact-mosaic-source"),
        b"order",
    )
    .unwrap();
    let (point_ab, prefix) = compact_functional(n, &terminal, false);
    let (point_c, _) = compact_functional(n, &terminal, true);
    let (p, ab, c) = compact_sources(&previous, &carries, n, a, &terminal);
    check_sources(n, a, &terminal, &p, &ab, &c).unwrap();
    let tables: BTreeMap<i64, Vec<bool>> = (0..w)
        .map(|bit| {
            (
                bit as i64,
                previous.iter().map(|&word| word >> bit & 1 != 0).collect(),
            )
        })
        .chain((0..w - 1).map(|bit| {
            (
                (w + bit) as i64,
                carries.iter().map(|&word| word >> bit & 1 != 0).collect(),
            )
        }))
        .collect();
    let layout = compute_layout(&tables.keys().map(|&id| (id, n)).collect::<Vec<_>>(), 14);
    let g = build_g(&layout, &tables);
    let packed = PackedBits::from_bools(&g);
    let mut prefix_claims = Vec::new();
    for (values, first_id, point) in [(&p, 0, &point_ab), (&ab, w, &point_ab), (&c, w, &point_c)] {
        for (bit, &value) in values.iter().enumerate() {
            prefix_claims.push(PrefixClaim {
                claim: Claim {
                    comp_id: (first_id + bit) as i64,
                    point: point.clone(),
                    value,
                },
                prefix_weights: prefix.clone(),
            });
        }
    }
    let ordinary = vec![Claim {
        comp_id: 0,
        point: point_ab.clone(),
        value: mle_eval_bits(&tables[&0], &point_ab),
    }];
    assert_ne!(ordinary[0].value, prefix_claims[0].claim.value);
    let mut prover = Transcript::new_default(b"compact-mosaic-binding");
    let (proof, point, value) =
        batch_prove_packed_with_prefixes(&layout, &ordinary, &prefix_claims, &packed, &mut prover);
    assert_eq!(value, mle_eval_bits(&g, &point));
    let mut verifier = Transcript::new_default(b"compact-mosaic-binding");
    assert_eq!(
        batch_verify_with_prefixes(&layout, &ordinary, &prefix_claims, &proof, &mut verifier)
            .unwrap(),
        (point, value),
    );
    assert_eq!(prover, verifier);
    for change in 0..5 {
        let mut bad = prefix_claims.clone();
        match change {
            0 => bad[0].prefix_weights = eq_table(&bad[0].claim.point[..K_SKIP]),
            1 => bad[0].prefix_weights[0] += K::ONE,
            2 => bad[0].claim.value += K::ONE,
            3 => bad[0].claim.point[K_SKIP] += K::ONE,
            _ => bad.swap(0, 1),
        }
        assert!(
            batch_verify_with_prefixes(
                &layout,
                &ordinary,
                &bad,
                &proof,
                &mut Transcript::new_default(b"compact-mosaic-binding"),
            )
            .is_err(),
            "mutation={change}"
        );
    }
}
