//! Port of `reference/tests/test_mosaic.py`, plus the 256-claim cap
//! (spec 05 §5.4) which the Python suite enforces in code but does
//! not test.

use std::collections::BTreeMap;

use om_field::K;
use om_mosaic::{
    BatchProof, Claim, MAX_CLAIMS, batch_prove, batch_prove_packed, batch_verify, build_g,
    build_g_packed, compute_layout, lift_point,
};
use om_sumcheck::mle_eval_bits;
use om_transcript::Transcript;

/// Deterministic splitmix64 stream (no `rand` crate), as in the other
/// crates' tests; stands in for Python's `random.Random(77)`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn k(&mut self) -> K {
        K::from_u128(u128::from(self.next_u64()) | (u128::from(self.next_u64()) << 64))
    }
    fn ks(&mut self, n: usize) -> Vec<K> {
        (0..n).map(|_| self.k()).collect()
    }
    fn bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

const COMPS: [(i64, usize); 7] = [(5, 3), (2, 4), (9, 3), (1, 0), (7, 1), (3, 4), (4, 2)];

fn setup(
    rng: &mut Rng,
    ell_min: usize,
) -> (om_mosaic::Layout, BTreeMap<i64, Vec<bool>>, Vec<bool>) {
    let lay = compute_layout(&COMPS, ell_min);
    let tables: BTreeMap<i64, Vec<bool>> = COMPS
        .iter()
        .map(|&(cid, ls)| (cid, (0..1usize << ls).map(|_| rng.bit()).collect()))
        .collect();
    let g = build_g(&lay, &tables);
    (lay, tables, g)
}

fn claims(rng: &mut Rng, tables: &BTreeMap<i64, Vec<bool>>) -> Vec<Claim> {
    // Component 2 gets three claims (COMPS + [(2, 4), (2, 4)]).
    let mut cs = COMPS.to_vec();
    cs.extend_from_slice(&[(2, 4), (2, 4)]);
    cs.iter()
        .map(|&(cid, ls)| {
            let r = rng.ks(ls);
            Claim {
                comp_id: cid,
                value: mle_eval_bits(&tables[&cid], &r),
                point: r,
            }
        })
        .collect()
}

#[test]
fn layout_order_alignment_disjoint() {
    let mut rng = Rng(77);
    let (lay, _, _) = setup(&mut rng, 0);
    let order: Vec<i64> = lay.placements.iter().map(|p| p.comp_id).collect();
    assert_eq!(order, vec![2, 3, 5, 9, 4, 7, 1]);
    let total: usize = COMPS.iter().map(|&(_, ls)| 1usize << ls).sum(); // 55
    assert_eq!(lay.ell, 6);
    assert!((1usize << lay.ell) >= total);
    let mut end = 0;
    for p in &lay.placements {
        assert_eq!(p.offset % (1usize << p.log_size), 0);
        assert_eq!(p.offset, end);
        assert_eq!(p.suffix, p.offset >> p.log_size);
        end += 1usize << p.log_size;
    }
}

#[test]
fn layout_ell_min_and_order_independence() {
    let a = compute_layout(&COMPS, 9);
    let mut rev = COMPS;
    rev.reverse();
    let b = compute_layout(&rev, 9);
    assert_eq!(a, b);
    assert_eq!(a.ell, 9);
}

#[test]
fn layout_single() {
    assert_eq!(compute_layout(&[(1, 0)], 0).ell, 0);
}

#[test]
#[should_panic(expected = "duplicate component id")]
fn layout_duplicates_rejected() {
    compute_layout(&[(1, 2), (1, 3)], 0);
}

#[test]
#[should_panic(expected = "empty component list")]
fn layout_empty_rejected() {
    compute_layout(&[], 0);
}

#[test]
#[should_panic(expected = "bad component")]
fn layout_negative_id_rejected() {
    // In particular the Spartan whole-W sentinel -1 is no component.
    compute_layout(&[(-1, 2)], 0);
}

#[test]
#[should_panic(expected = "bad component")]
fn layout_id_at_2_pow_32_rejected() {
    compute_layout(&[(1i64 << 32, 2)], 0);
}

#[test]
fn lifting_eq4() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    for &(cid, ls) in &COMPS {
        let r = rng.ks(ls);
        let c = Claim {
            comp_id: cid,
            value: mle_eval_bits(&tables[&cid], &r),
            point: r,
        };
        assert_eq!(mle_eval_bits(&g, &lift_point(&lay, &c).unwrap()), c.value);
    }
}

#[test]
fn batching_completeness() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let cl = claims(&mut rng, &tables);
    let (proof, r_g, v_g) = batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
    let (r_g2, v_g2) = batch_verify(&lay, &cl, &proof, &mut Transcript::new_default(b"m")).unwrap();
    assert_eq!((&r_g, v_g), (&r_g2, v_g2));
    assert_eq!(mle_eval_bits(&g, &r_g), v_g);
}

#[test]
fn batching_negative() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let cl = claims(&mut rng, &tables);
    let (proof, _, _) = batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));

    let mut p2 = proof.clone();
    p2.sumcheck.round_msgs[2][0] += K::ONE;
    assert!(batch_verify(&lay, &cl, &p2, &mut Transcript::new_default(b"m")).is_err());

    let p2 = BatchProof {
        v_g: proof.v_g + K::ONE,
        ..proof.clone()
    };
    assert!(batch_verify(&lay, &cl, &p2, &mut Transcript::new_default(b"m")).is_err());

    let mut cl2 = cl.clone();
    cl2[3] = Claim {
        value: cl[3].value + K::ONE,
        ..cl[3].clone()
    };
    assert!(batch_verify(&lay, &cl2, &proof, &mut Transcript::new_default(b"m")).is_err());

    let unknown = vec![Claim {
        comp_id: 1234,
        point: vec![],
        value: K::ZERO,
    }];
    assert!(batch_verify(&lay, &unknown, &proof, &mut Transcript::new_default(b"m")).is_err());

    let wrong_len = vec![Claim {
        comp_id: 2,
        point: vec![K::ONE],
        value: K::ZERO,
    }];
    assert!(batch_verify(&lay, &wrong_len, &proof, &mut Transcript::new_default(b"m")).is_err());
}

#[test]
#[should_panic]
fn false_claim_not_provable() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let mut cl = claims(&mut rng, &tables);
    cl[0] = Claim {
        value: cl[0].value + K::ONE,
        ..cl[0].clone()
    };
    batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
}

#[test]
fn claim_cap_at_256_accepted() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let cl: Vec<Claim> = (0..MAX_CLAIMS)
        .map(|i| {
            let (cid, ls) = COMPS[i % COMPS.len()];
            let r = rng.ks(ls);
            Claim {
                comp_id: cid,
                value: mle_eval_bits(&tables[&cid], &r),
                point: r,
            }
        })
        .collect();
    let (proof, r_g, v_g) = batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
    let out = batch_verify(&lay, &cl, &proof, &mut Transcript::new_default(b"m")).unwrap();
    assert_eq!(out, (r_g, v_g));
}

#[test]
#[should_panic(expected = "too many Mosaic claims")]
fn claim_cap_300_prove_rejected() {
    let mut rng = Rng(77);
    let (lay, _, g) = setup(&mut rng, 0);
    let cl = vec![
        Claim {
            comp_id: 1,
            point: vec![],
            value: K::ZERO,
        };
        300
    ];
    batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
}

#[test]
fn claim_cap_300_verify_rejected() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let cl = claims(&mut rng, &tables);
    let (proof, _, _) = batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
    let many = vec![
        Claim {
            comp_id: 1,
            point: vec![],
            value: K::ZERO,
        };
        300
    ];
    let err = batch_verify(&lay, &many, &proof, &mut Transcript::new_default(b"m")).unwrap_err();
    assert!(err.message().contains("too many Mosaic claims"), "{err}");
}

#[test]
fn batching_packed_matches_bool() {
    let mut rng = Rng(77);
    let (lay, tables, g) = setup(&mut rng, 0);
    let cl = claims(&mut rng, &tables);
    // The packed assembly is the same bits.
    let packed_tables: BTreeMap<i64, om_sumcheck::PackedBits> = tables
        .iter()
        .map(|(&cid, tb)| (cid, om_sumcheck::PackedBits::from_bools(tb)))
        .collect();
    let g_packed = build_g_packed(&lay, &packed_tables);
    assert_eq!(g_packed.to_bools(), g);
    // And the packed batch proof is byte-identical to the bool one.
    let (p1, r1, v1) = batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
    let (p2, r2, v2) = batch_prove_packed(&lay, &cl, &g_packed, &mut Transcript::new_default(b"m"));
    assert_eq!((p1, r1, v1), (p2, r2, v2));
}

/// The batching sum-check straight from its definition (spec 05 §5.4):
/// full `E` and `G` tables over `2^ell`, bound low variable first.
fn dense_batch_prove(
    lay: &om_mosaic::Layout,
    cl: &[Claim],
    g: &[bool],
    tr: &mut Transcript,
) -> (BatchProof, Vec<K>, K) {
    use om_sumcheck::{SumcheckProof, batch_claims, interpolate_eval, sumcheck_init};
    let values: Vec<K> = cl.iter().map(|c| c.value).collect();
    let (alphas, claim) = batch_claims(&values, tr, b"mos/batch");
    let mut e = om_mosaic::eq_sum_table(lay, cl, &alphas);
    let mut gt: Vec<K> = g.iter().map(|&b| K::from_bit(b)).collect();
    sumcheck_init(tr, b"mos/batch", claim, lay.ell, 2);
    let (mut msgs, mut rs, mut cur) = (Vec::new(), Vec::new(), claim);
    for _ in 0..lay.ell {
        let mut ev = [K::ZERO; 3];
        for i in 0..gt.len() / 2 {
            for (k, slot) in ev.iter_mut().enumerate() {
                let x = K::t(k as u128);
                *slot += (e[2 * i] + x * (e[2 * i] + e[2 * i + 1]))
                    * (gt[2 * i] + x * (gt[2 * i] + gt[2 * i + 1]));
            }
        }
        assert_eq!(ev[0] + ev[1], cur);
        let msg = vec![ev[0], ev[2]];
        tr.absorb_ks(b"mos/batch/msg", &msg);
        let c = tr.challenge_k(b"mos/batch/chal");
        cur = interpolate_eval(&ev, c);
        msgs.push(msg);
        rs.push(c);
        om_sumcheck::bind_low_in_place(&mut e, c);
        om_sumcheck::bind_low_in_place(&mut gt, c);
    }
    tr.absorb_k(b"mos/vg", gt[0]);
    (
        BatchProof {
            sumcheck: SumcheckProof { round_msgs: msgs },
            v_g: gt[0],
        },
        rs,
        gt[0],
    )
}

#[test]
fn batching_split_matches_dense_definition() {
    // Mixed sizes, including components smaller than the bit-level
    // phase and claims spread over several components, against the
    // dense definition for every split point of the two phases.
    let comps: [(i64, usize); 9] = [
        (0, 9),
        (1, 8),
        (2, 8),
        (3, 6),
        (4, 3),
        (5, 1),
        (6, 0),
        (7, 4),
        (8, 2),
    ];
    for (seed, ell_min) in [(1u64, 0usize), (2, 12), (3, 0)] {
        let mut rng = Rng(seed);
        let lay = compute_layout(&comps, ell_min);
        let tables: BTreeMap<i64, Vec<bool>> = comps
            .iter()
            .map(|&(cid, ls)| {
                let dense = seed == 3 && cid == 0;
                let bits = (0..1usize << ls)
                    .map(|_| dense || rng.next_u64().is_multiple_of(5))
                    .collect();
                (cid, bits)
            })
            .collect();
        let g = build_g(&lay, &tables);
        let packed = om_sumcheck::PackedBits::from_bools(&g);
        let mut cl = Vec::new();
        for (i, &(cid, ls)) in comps.iter().enumerate() {
            for _ in 0..1 + i % 3 {
                let r = rng.ks(ls);
                cl.push(Claim {
                    comp_id: cid,
                    value: mle_eval_bits(&tables[&cid], &r),
                    point: r,
                });
            }
        }
        let want = dense_batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"m"));
        for m in [0, 1, 2, 3, 4, 7, 8, 10, lay.ell, lay.ell + 3] {
            let got = om_mosaic::batch_prove_packed_with(
                &lay,
                &cl,
                &packed,
                &mut Transcript::new_default(b"m"),
                m,
            );
            assert_eq!(got, want, "seed {seed} ell {} bit rounds {m}", lay.ell);
        }
    }
}

#[test]
fn duplicate_internal_basis_terms_preserve_public_slots_and_transcript() {
    let mut rng = Rng(0x9917);
    let (lay, tables, g) = setup(&mut rng, 12);
    let mut cl = claims(&mut rng, &tables);
    let originals = cl.clone();
    cl.extend(originals.iter().cycle().take(21).cloned());
    // Same point on distinct placements must remain distinct basis functions;
    // Boolean points also exercise zero equality prefixes in packed rounds.
    for cid in [2i64, 3] {
        let point = vec![K::ZERO, K::ONE, K::ZERO, K::ONE];
        let claim = Claim {
            comp_id: cid,
            value: mle_eval_bits(&tables[&cid], &point),
            point,
        };
        cl.extend([claim.clone(), claim.clone(), claim]);
    }
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    let mut dense_tr = Transcript::new_default(b"duplicate-mosaic");
    let expected = dense_batch_prove(&lay, &cl, &g, &mut dense_tr);
    let expected_tail = dense_tr.challenge_k(b"after");
    for m in [0, 1, 3, 8, 10, lay.ell, lay.ell + 1] {
        let mut actual_tr = Transcript::new_default(b"duplicate-mosaic");
        let actual = om_mosaic::batch_prove_packed_with(&lay, &cl, &packed, &mut actual_tr, m);
        assert_eq!(actual, expected, "split={m}");
        assert_eq!(actual_tr.challenge_k(b"after"), expected_tail);
    }
}

#[test]
fn duplicate_point_with_inconsistent_value_keeps_original_claim_check() {
    let mut rng = Rng(0xa31);
    let (lay, tables, g) = setup(&mut rng, 8);
    let mut cl = claims(&mut rng, &tables);
    let mut wrong = cl[0].clone();
    wrong.value += K::ONE;
    cl.push(wrong);
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    for m in [0, 1, 10] {
        assert!(
            std::panic::catch_unwind(|| om_mosaic::batch_prove_packed_with(
                &lay,
                &cl,
                &packed,
                &mut Transcript::new_default(b"wrong-duplicate"),
                m,
            ))
            .is_err()
        );
    }
}

/// Claims that differ in low coordinates share their contracted G table, but
/// retain distinct Eq prefixes and public slots. Two equal-size placements
/// deliberately use the same points; the small placements also share a block
/// when the split is above their own dimensions.
fn shared_suffix_fixture() -> (om_mosaic::Layout, Vec<bool>, Vec<Claim>) {
    let mut rng = Rng(0x5aff_17);
    let comps = [(0, 9), (1, 9), (2, 5), (3, 4), (4, 0)];
    let lay = compute_layout(&comps, 0);
    let tables: BTreeMap<i64, Vec<bool>> = comps
        .iter()
        .map(|&(cid, log)| {
            let table = (0..1usize << log)
                .map(|_| match cid {
                    2 => false,
                    3 | 4 => true,
                    _ => rng.bit(),
                })
                .collect();
            (cid, table)
        })
        .collect();
    let shared = rng.ks(9);
    let mut claims = Vec::new();
    for (cid, log) in comps {
        for variant in 0..3 {
            let mut point = shared[..log].to_vec();
            for (i, coordinate) in point.iter_mut().take(4).enumerate() {
                *coordinate = match variant {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ if i == 0 => rng.k(),
                    _ => K::from_bit(i % 2 == 0),
                };
            }
            claims.push(Claim {
                comp_id: cid,
                value: mle_eval_bits(&tables[&cid], &point),
                point,
            });
        }
    }
    // Preserve an additional exact duplicate as well as the suffix aliases.
    claims.push(claims[0].clone());
    let g = build_g(&lay, &tables);
    (lay, g, claims)
}

#[test]
fn shared_suffix_distinct_prefixes_and_placements_match_dense_transcript() {
    let (lay, g, claims) = shared_suffix_fixture();
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    let mut dense_tr = Transcript::new_default(b"shared-suffix");
    let expected = dense_batch_prove(&lay, &claims, &g, &mut dense_tr);
    let expected_tail = dense_tr.challenge_k(b"after");
    for m in [0, 1, 3, 4, 6, 8, 9, 10, lay.ell, lay.ell + 3] {
        let mut actual_tr = Transcript::new_default(b"shared-suffix");
        let actual = om_mosaic::batch_prove_packed_with(&lay, &claims, &packed, &mut actual_tr, m);
        assert_eq!(actual, expected, "split={m}");
        assert_eq!(actual_tr.challenge_k(b"after"), expected_tail);
        assert_eq!(
            batch_verify(
                &lay,
                &claims,
                &actual.0,
                &mut Transcript::new_default(b"shared-suffix")
            )
            .unwrap(),
            (actual.1, actual.2)
        );
    }
}

#[test]
fn shared_suffix_false_value_keeps_every_public_claim_checked() {
    let (lay, g, claims) = shared_suffix_fixture();
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    let (proof, _, _) = batch_prove_packed(
        &lay,
        &claims,
        &packed,
        &mut Transcript::new_default(b"false-shared-suffix"),
    );
    let mut wrong = claims.clone();
    // The second claim shares a suffix, but has a different prefix and value.
    wrong[1].value += K::ONE;
    assert!(
        batch_verify(
            &lay,
            &wrong,
            &proof,
            &mut Transcript::new_default(b"false-shared-suffix")
        )
        .is_err()
    );
    for m in [0, 4, 6, 10, lay.ell] {
        assert!(
            std::panic::catch_unwind(|| om_mosaic::batch_prove_packed_with(
                &lay,
                &wrong,
                &packed,
                &mut Transcript::new_default(b"false-shared-suffix"),
                m,
            ))
            .is_err(),
            "false claim passed split={m}"
        );
    }
}

#[test]
fn shared_suffix_parallel_regions_match_dense_transcript() {
    let mut rng = Rng(0x9090_77);
    let comps = [(0, 16), (1, 16), (2, 3)];
    let lay = compute_layout(&comps, 0);
    let tables: BTreeMap<i64, Vec<bool>> = comps
        .iter()
        .map(|&(cid, log)| {
            let table = (0..1usize << log)
                .map(|_| match cid {
                    0 => rng.bit(),
                    1 => rng.next_u64().is_multiple_of(17),
                    _ => true,
                })
                .collect();
            (cid, table)
        })
        .collect();
    let shared = rng.ks(16);
    let mut claims = Vec::new();
    for cid in [0, 1] {
        for variant in 0..4 {
            let mut point = shared.clone();
            for (i, coordinate) in point.iter_mut().take(10).enumerate() {
                *coordinate = match variant {
                    0 => K::ZERO,
                    1 => K::ONE,
                    2 => K::from_bit(i % 2 == 0),
                    _ => rng.k(),
                };
            }
            claims.push(Claim {
                comp_id: cid,
                value: mle_eval_bits(&tables[&cid], &point),
                point,
            });
        }
    }
    let small_point = vec![K::ZERO, K::ONE, rng.k()];
    claims.push(Claim {
        comp_id: 2,
        value: mle_eval_bits(&tables[&2], &small_point),
        point: small_point,
    });
    let g = build_g(&lay, &tables);
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    let mut dense_tr = Transcript::new_default(b"parallel-shared-suffix");
    let expected = dense_batch_prove(&lay, &claims, &g, &mut dense_tr);
    let expected_tail = dense_tr.challenge_k(b"after");
    // Both large regions meet PAR_BITS; run several worker counts so the
    // grouped scans and their lazy-product reductions exercise parallel paths.
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for m in [0, 6, 10, 12] {
            pool.install(|| {
                let mut tr = Transcript::new_default(b"parallel-shared-suffix");
                let actual = om_mosaic::batch_prove_packed_with(&lay, &claims, &packed, &mut tr, m);
                assert_eq!(actual, expected, "workers={workers} split={m}");
                assert_eq!(tr.challenge_k(b"after"), expected_tail);
            });
        }
    }
}

/// Components large enough for the eight-row subset-sum scan (rows of at
/// least 64 bits and at least eight rows per block), at densities from
/// empty to full, give the dense definition's proof for every split.
#[test]
fn packed_scan_matches_dense_at_every_density() {
    let comps: Vec<(i64, usize)> = vec![(0, 15), (1, 14), (2, 13), (3, 12), (4, 14)];
    // Bit set with probability keep / 64; component 4 alternates dense and
    // sparse stretches so both paths run within one region.
    let keeps = [32u64, 4, 64, 0, 0];
    let mut rng = Rng(0x5eed);
    let lay = compute_layout(&comps, 0);
    let tables: BTreeMap<i64, Vec<bool>> = comps
        .iter()
        .map(|&(cid, ls)| {
            let bits = (0..1usize << ls)
                .map(|i| {
                    let keep = if cid == 4 {
                        if (i >> 9) % 2 == 0 { 48 } else { 2 }
                    } else {
                        keeps[cid as usize]
                    };
                    rng.next_u64() % 64 < keep
                })
                .collect();
            (cid, bits)
        })
        .collect();
    let g = build_g(&lay, &tables);
    let packed = om_sumcheck::PackedBits::from_bools(&g);
    let mut cl = Vec::new();
    for (i, &(cid, ls)) in comps.iter().enumerate() {
        for _ in 0..1 + i % 3 {
            let r = rng.ks(ls);
            cl.push(Claim {
                comp_id: cid,
                value: mle_eval_bits(&tables[&cid], &r),
                point: r,
            });
        }
    }
    let want = dense_batch_prove(&lay, &cl, &g, &mut Transcript::new_default(b"scan"));
    for m in [5, 6, 7, 8, 10] {
        let got = om_mosaic::batch_prove_packed_with(
            &lay,
            &cl,
            &packed,
            &mut Transcript::new_default(b"scan"),
            m,
        );
        assert_eq!(got, want, "bit rounds {m}");
    }
}
