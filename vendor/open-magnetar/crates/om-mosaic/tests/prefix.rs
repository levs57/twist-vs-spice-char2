use std::collections::BTreeMap;

use om_field::K;
use om_mosaic::{
    BIT_ROUNDS, BatchProof, Claim, Layout, PrefixClaim, batch_prove_packed,
    batch_prove_packed_with_prefixes, batch_verify, batch_verify_with_prefixes, build_g,
    compute_layout,
};
use om_sumcheck::{PackedBits, batch_claims, eq_table, mle_eval_bits, sumcheck_prove};
use om_transcript::Transcript;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn k(&mut self) -> K {
        K::from_u128(u128::from(self.next()) | (u128::from(self.next()) << 64))
    }
    fn point(&mut self, n: usize) -> Vec<K> {
        (0..n).map(|_| self.k()).collect()
    }
}

fn functional(table: &[bool], point: &[K], weights: &[K]) -> K {
    let p = weights.len().trailing_zeros() as usize;
    let high = eq_table(&point[p..]);
    table
        .iter()
        .enumerate()
        .filter(|(_, bit)| **bit)
        .fold(K::ZERO, |sum, (i, _)| {
            sum + weights[i & (weights.len() - 1)] * high[i >> p]
        })
}

/// Independent transcript framing oracle for the explicit-prefix API.
fn bind_context(
    layout: &Layout,
    ordinary: &[Claim],
    prefixes: &[PrefixClaim],
    tr: &mut Transcript,
) {
    if prefixes.is_empty() {
        return;
    }
    tr.absorb_bytes(b"mos/prefix/protocol", b"mos/prefix/low-table-v1");
    let shape: Vec<u8> = [layout.ell, ordinary.len(), prefixes.len()]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(b"mos/prefix/shape", &shape);
    let by_id = layout.by_id();
    for (claim, weights) in ordinary.iter().map(|c| (c, &[][..])).chain(
        prefixes
            .iter()
            .map(|p| (&p.claim, p.prefix_weights.as_slice())),
    ) {
        let pl = by_id[&claim.comp_id];
        let descriptor: Vec<u8> = [claim.comp_id as u64, pl.log_size as u64, pl.offset as u64]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        tr.absorb_bytes(b"mos/prefix/component", &descriptor);
        tr.absorb_ks(b"mos/prefix/point", &claim.point);
        tr.absorb_ks(b"mos/prefix/weights", weights);
    }
}

/// Materialize the complete functional E on the whole global cube and run
/// the generic scalar sumcheck, without the packed phase or basis sharing.
fn dense(
    layout: &Layout,
    ordinary: &[Claim],
    prefixes: &[PrefixClaim],
    g: &[bool],
    tr: &mut Transcript,
) -> (BatchProof, Vec<K>, K) {
    bind_context(layout, ordinary, prefixes, tr);
    let values: Vec<K> = ordinary
        .iter()
        .map(|c| c.value)
        .chain(prefixes.iter().map(|p| p.claim.value))
        .collect();
    let (alphas, claim) = batch_claims(&values, tr, b"mos/batch");
    let by_id = layout.by_id();
    let mut e = vec![K::ZERO; g.len()];
    for (claim, &alpha) in ordinary.iter().zip(&alphas) {
        let pl = by_id[&claim.comp_id];
        for (i, weight) in eq_table(&claim.point).into_iter().enumerate() {
            e[pl.offset + i] += alpha * weight;
        }
    }
    for (prefix, &alpha) in prefixes.iter().zip(&alphas[ordinary.len()..]) {
        let pl = by_id[&prefix.claim.comp_id];
        let p = prefix.prefix_weights.len().trailing_zeros() as usize;
        let high = eq_table(&prefix.claim.point[p..]);
        for i in 0..1usize << pl.log_size {
            e[pl.offset + i] +=
                alpha * prefix.prefix_weights[i & (prefix.prefix_weights.len() - 1)] * high[i >> p];
        }
    }
    let gt: Vec<K> = g.iter().copied().map(K::from_bit).collect();
    let (sc, point, finals) = sumcheck_prove(
        claim,
        vec![e, gt],
        &|row| row[0] * row[1],
        2,
        tr,
        b"mos/batch",
    );
    tr.absorb_k(b"mos/vg", finals[1]);
    (
        BatchProof {
            sumcheck: sc,
            v_g: finals[1],
        },
        point,
        finals[1],
    )
}

fn fixture(large: bool) -> (Layout, Vec<bool>, Vec<Claim>, Vec<PrefixClaim>) {
    let mut rng = Rng(0x8795_1342_1117);
    let comps = if large {
        vec![(0, 16), (1, 16), (2, 6), (3, 2), (4, 0)]
    } else {
        vec![(0, 12), (1, 11), (2, 6), (3, 2), (4, 0)]
    };
    let layout = compute_layout(&comps, 0);
    let tables: BTreeMap<i64, Vec<bool>> = comps
        .iter()
        .map(|&(id, n)| {
            (
                id,
                (0..1usize << n)
                    .map(|_| rng.next() % (id as u64 + 2) == 0)
                    .collect(),
            )
        })
        .collect();
    let mut ordinary = Vec::new();
    for &(id, n) in &comps {
        let point = rng.point(n);
        ordinary.push(Claim {
            comp_id: id,
            value: mle_eval_bits(&tables[&id], &point),
            point,
        });
    }
    let mut prefixes: Vec<PrefixClaim> = Vec::new();
    for (id, p) in [(0, 6usize), (0, 6), (0, 10), (1, 6), (2, 6), (3, 2), (4, 0)] {
        let n = comps.iter().find(|&&(cid, _)| cid == id).unwrap().1;
        // The two six-bit prefixes of component zero share their Eq suffix
        // but have different functionals. Low point placeholders are random.
        let point = if id == 0 && prefixes.len() == 1 {
            prefixes[0].claim.point.clone()
        } else {
            rng.point(n)
        };
        let weights: Vec<K> = (0..1usize << p)
            .map(|i| match id {
                1 => K::ZERO,
                2 => K::from_bit(i == 17),
                _ => rng.k(),
            })
            .collect();
        let value = functional(&tables[&id], &point, &weights);
        prefixes.push(PrefixClaim {
            claim: Claim {
                comp_id: id,
                point,
                value,
            },
            prefix_weights: weights,
        });
    }
    prefixes.push(prefixes[0].clone());
    let g = build_g(&layout, &tables);
    (layout, g, ordinary, prefixes)
}

fn compare(layout: &Layout, g: &[bool], ordinary: &[Claim], prefixes: &[PrefixClaim]) {
    let packed = PackedBits::from_bools(g);
    let mut want_tr = Transcript::new_default(b"prefix-oracle");
    let want = dense(layout, ordinary, prefixes, g, &mut want_tr);
    let mut fast_tr = Transcript::new_default(b"prefix-oracle");
    let fast = batch_prove_packed_with_prefixes(layout, ordinary, prefixes, &packed, &mut fast_tr);
    assert_eq!(fast, want);
    let tail = fast_tr.challenge_k(b"after");
    assert_eq!(tail, want_tr.challenge_k(b"after"));
    assert_eq!(mle_eval_bits(g, &fast.1), fast.2);
    let mut verifier_tr = Transcript::new_default(b"prefix-oracle");
    assert_eq!(
        batch_verify_with_prefixes(layout, ordinary, prefixes, &fast.0, &mut verifier_tr).unwrap(),
        (fast.1, fast.2)
    );
    assert_eq!(verifier_tr.challenge_k(b"after"), tail);
}

#[test]
fn arbitrary_prefixes_match_dense_proof_points_and_transcript() {
    let (layout, g, ordinary, prefixes) = fixture(false);
    compare(&layout, &g, &ordinary, &prefixes);
    compare(&layout, &g, &[], &prefixes);
}

#[test]
fn prefix_parallel_paths_match_dense_with_one_and_four_workers() {
    let (layout, g, ordinary, prefixes) = fixture(true);
    for workers in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| compare(&layout, &g, &ordinary, &prefixes));
    }
}

#[test]
fn zero_dimensional_prefixes_and_boolean_high_points_match_dense() {
    for bit in [false, true] {
        let layout = compute_layout(&[(0, 0)], 0);
        let prefix = PrefixClaim {
            claim: Claim {
                comp_id: 0,
                point: vec![],
                value: if bit { K::t(7) } else { K::ZERO },
            },
            prefix_weights: vec![K::t(7)],
        };
        compare(&layout, &[bit], &[], &[prefix]);
    }
    let (layout, g, ordinary, mut prefixes) = fixture(false);
    // Recompute true values by the independent global functional. Boolean
    // coordinates include zero Eq factors throughout the packed phase.
    let by_id = layout.by_id();
    for prefix in &mut prefixes {
        for (i, x) in prefix.claim.point.iter_mut().enumerate() {
            *x = K::from_bit(i % 2 == 0);
        }
        let pl = by_id[&prefix.claim.comp_id];
        prefix.claim.value = functional(
            &g[pl.offset..pl.offset + (1 << pl.log_size)],
            &prefix.claim.point,
            &prefix.prefix_weights,
        );
    }
    compare(&layout, &g, &ordinary, &prefixes);
}

#[test]
fn empty_prefix_list_keeps_ordinary_proof_and_transcript_byte_identical() {
    let (layout, g, ordinary, _) = fixture(false);
    let packed = PackedBits::from_bools(&g);
    let (mut old_tr, mut new_tr) = (
        Transcript::new_default(b"empty"),
        Transcript::new_default(b"empty"),
    );
    let old = batch_prove_packed(&layout, &ordinary, &packed, &mut old_tr);
    let new = batch_prove_packed_with_prefixes(&layout, &ordinary, &[], &packed, &mut new_tr);
    assert_eq!(old, new);
    assert_eq!(old_tr.challenge_k(b"after"), new_tr.challenge_k(b"after"));
    let (mut old_tr, mut new_tr) = (
        Transcript::new_default(b"empty"),
        Transcript::new_default(b"empty"),
    );
    assert_eq!(
        batch_verify(&layout, &ordinary, &old.0, &mut old_tr),
        batch_verify_with_prefixes(&layout, &ordinary, &[], &old.0, &mut new_tr)
    );
    assert_eq!(old_tr.challenge_k(b"after"), new_tr.challenge_k(b"after"));
}

#[test]
fn changed_weights_values_placements_and_messages_are_rejected() {
    let (layout, g, ordinary, prefixes) = fixture(false);
    let packed = PackedBits::from_bools(&g);
    let (proof, _, _) = batch_prove_packed_with_prefixes(
        &layout,
        &ordinary,
        &prefixes,
        &packed,
        &mut Transcript::new_default(b"tamper"),
    );
    let mut variants = Vec::new();
    let mut changed = prefixes.clone();
    changed[0].prefix_weights[17] += K::ONE;
    variants.push(changed);
    let mut changed = prefixes.clone();
    changed[0].claim.value += K::ONE;
    variants.push(changed);
    let mut changed = prefixes.clone();
    changed.swap(0, 1);
    variants.push(changed);
    let mut changed = prefixes.clone();
    changed[0].claim.comp_id = 1;
    changed[0].claim.point.pop();
    variants.push(changed);
    for changed in variants {
        assert!(
            batch_verify_with_prefixes(
                &layout,
                &ordinary,
                &changed,
                &proof,
                &mut Transcript::new_default(b"tamper")
            )
            .is_err()
        );
    }
    let mut changed = proof.clone();
    changed.sumcheck.round_msgs[1][0] += K::ONE;
    assert!(
        batch_verify_with_prefixes(
            &layout,
            &ordinary,
            &prefixes,
            &changed,
            &mut Transcript::new_default(b"tamper")
        )
        .is_err()
    );
    let mut changed = proof.clone();
    changed.v_g += K::ONE;
    assert!(
        batch_verify_with_prefixes(
            &layout,
            &ordinary,
            &prefixes,
            &changed,
            &mut Transcript::new_default(b"tamper")
        )
        .is_err()
    );
    let mut false_claim = prefixes;
    false_claim[0].claim.value += K::ONE;
    assert!(
        std::panic::catch_unwind(|| batch_prove_packed_with_prefixes(
            &layout,
            &ordinary,
            &false_claim,
            &packed,
            &mut Transcript::new_default(b"false-prefix")
        ))
        .is_err()
    );
}

#[test]
fn malformed_prefix_shapes_rejected_before_transcript_interaction() {
    let (layout, g, ordinary, prefixes) = fixture(false);
    let packed = PackedBits::from_bools(&g);
    let (proof, _, _) = batch_prove_packed_with_prefixes(
        &layout,
        &ordinary,
        &prefixes,
        &packed,
        &mut Transcript::new_default(b"malformed"),
    );
    let mut bad = Vec::new();
    for weights in [vec![], vec![K::ONE; 3], vec![K::ONE; 1 << (BIT_ROUNDS + 1)]] {
        let mut p = prefixes[0].clone();
        p.prefix_weights = weights;
        bad.push(p);
    }
    let mut p = prefixes[5].clone();
    p.prefix_weights = vec![K::ONE; 8];
    bad.push(p);
    let mut p = prefixes[0].clone();
    p.claim.comp_id = -1;
    bad.push(p);
    let mut p = prefixes[0].clone();
    p.claim.point.pop();
    bad.push(p);
    for p in bad {
        let mut tr = Transcript::new_default(b"malformed");
        let mut untouched = tr.clone();
        assert!(
            batch_verify_with_prefixes(&layout, &ordinary, &[p.clone()], &proof, &mut tr).is_err()
        );
        assert_eq!(tr.challenge_k(b"after"), untouched.challenge_k(b"after"));
        assert!(
            std::panic::catch_unwind(|| batch_prove_packed_with_prefixes(
                &layout,
                &ordinary,
                &[p],
                &packed,
                &mut Transcript::new_default(b"malformed")
            ))
            .is_err()
        );
    }
}
