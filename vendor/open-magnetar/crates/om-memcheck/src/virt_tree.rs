//! Experimental balanced-product-tree address virtualization.
//!
//! Every internal table is the *pointwise* product of its two children on the
//! Boolean cycle cube. One cubic sum-check per level reduces batched claims on
//! those tables to claims on the next level. In particular, the verifier never
//! replaces the multilinear extension of a pointwise product by a product of
//! multilinear extensions at an off-cube point. Intermediate tables are not
//! committed; the returned leaf claims must be authenticated by the caller.

use om_field::{K, U256};
use om_sumcheck::{SumcheckProof, VerificationError, eq_eval, eq_table, sumcheck_verify};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::lazy::{SplitEq, sumcheck_prove_rows_products};
use crate::onehot::{
    AddrComps, EvalClaim, PAR_THRESHOLD, VirtGroup, components_equal, split_point, virt_claims,
    virt_direct,
};
use crate::sublabel;

/// One tree level, from root toward leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtTreeLayerProof {
    /// A cubic sum-check in the `n` cycle variables, batched over all nodes.
    pub sumcheck: SumcheckProof,
    /// Child evaluations at this layer's shared cycle point. Group-major, then
    /// natural left-to-right node order; layer `l` has `2^(l+1)` per group.
    pub child_evals: Vec<K>,
}

/// `log2(d)` layers. The final layer's children are the original components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtTreeProof {
    pub layers: Vec<VirtTreeLayerProof>,
}

fn invalid(message: &str) -> VerificationError {
    VerificationError::new(message)
}

fn validate_groups(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
) -> Result<usize, VerificationError> {
    if d == 0
        || !d.is_power_of_two()
        || a < d
        || !a.is_multiple_of(d)
        || n >= usize::BITS as usize
        || a / d >= usize::BITS as usize
    {
        return Err(invalid("virt tree: invalid dimensions"));
    }
    groups
        .len()
        .checked_mul(d)
        .ok_or_else(|| invalid("virt tree: too many groups"))?;
    let mut terms = 0usize;
    for group in groups {
        if group.names.len() != d
            || group.r_k.len() != a
            || group.terms.iter().any(|term| term.cyc.len() != n)
        {
            return Err(invalid("virt tree: malformed group"));
        }
        terms = terms
            .checked_add(group.terms.len())
            .ok_or_else(|| invalid("virt tree: too many terms"))?;
    }
    Ok(terms)
}

/// Bind the tree protocol, dimensions and input claims before any batching
/// challenge. Names matter because they identify the eventual committed leaves.
fn bind_context(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    tr: &mut Transcript,
    label: &[u8],
) {
    tr.absorb_bytes(&sublabel(label, b"/protocol"), b"product-tree/gkr/v1");
    let mut dimensions = Vec::with_capacity(32);
    for value in [a, d, n, groups.len()] {
        dimensions.extend_from_slice(&(value as u64).to_le_bytes());
    }
    tr.absorb_bytes(&sublabel(label, b"/dimensions"), &dimensions);
    for group in groups {
        for name in &group.names {
            tr.absorb_bytes(&sublabel(label, b"/name"), name.as_bytes());
        }
        tr.absorb_ks(&sublabel(label, b"/address"), &group.r_k);
        tr.absorb_bytes(
            &sublabel(label, b"/term-count"),
            &(group.terms.len() as u64).to_le_bytes(),
        );
        for term in &group.terms {
            tr.absorb_ks(&sublabel(label, b"/cycle"), &term.cyc);
            tr.absorb_k(&sublabel(label, b"/value"), term.value);
        }
    }
}

fn layer_label(label: &[u8], layer: usize) -> Vec<u8> {
    sublabel(label, format!("/layer/{layer}").as_bytes())
}

pub(crate) fn component_shape(comps: &AddrComps, a: usize, d: usize, n: usize) -> bool {
    let cycles = 1usize << n;
    match comps {
        AddrComps::Addrs {
            addrs,
            a: ca,
            d: cd,
        } => {
            *ca == a
                && *cd == d
                && a <= 32
                && addrs.len() == cycles
                && addrs.iter().all(|&addr| (addr as u64) < (1u64 << a))
        }
        AddrComps::Bits { tables, ac } => {
            *ac == a / d
                && tables.len() == d
                && cycles
                    .checked_mul(1usize << ac)
                    .is_some_and(|len| tables.iter().all(|table| table.len() == len))
        }
    }
}

/// Build the non-root internal levels bottom-up. Leaves stay in their compact
/// address/bit representation. Level `s-1` has subtrees of `2^s` components.
fn build_levels(
    comps: &[&AddrComps],
    representatives: &[usize],
    chunk_eqs: &[Vec<Vec<K>>],
    d: usize,
    n: usize,
) -> Vec<Vec<Vec<K>>> {
    let _span = Span::new("virt.tree.build");
    let cycles = 1usize << n;
    let mut levels: Vec<Vec<Vec<K>>> = Vec::new();
    let mut children_per_group = d;
    while children_per_group > 2 {
        let nodes_per_group = children_per_group / 2;
        let previous = levels.last();
        let make_table = |node: usize| {
            let group = node / nodes_per_group;
            let within = node % nodes_per_group;
            let child = |side: usize, j: usize| {
                if let Some(previous) = previous {
                    previous[group * children_per_group + 2 * within + side][j]
                } else {
                    comps[representatives[group]].row_eval(
                        2 * within + side,
                        j,
                        &chunk_eqs[group][2 * within + side],
                    )
                }
            };
            if cycles == 1 {
                return vec![child(0, 0) * child(1, 0)];
            }
            let pair = |index: usize| {
                let j = 2 * index;
                om_field::mul_pairs(
                    [child(0, j), child(0, j + 1)],
                    [child(1, j), child(1, j + 1)],
                )
            };
            // Indexed collection initializes each page on its producing worker;
            // flattening pairs retains that allocation without a serial copy.
            let pairs: Vec<[K; 2]> = if cycles >= PAR_THRESHOLD {
                (0..cycles / 2)
                    .into_par_iter()
                    .with_min_len(512)
                    .map(pair)
                    .collect()
            } else {
                (0..cycles / 2).map(pair).collect()
            };
            pairs.into_flattened()
        };
        let node_count = representatives.len() * nodes_per_group;
        // Parallelise across tables too, since small cycle domains can still
        // have many nodes. Rayon nested iterators share the same worker pool.
        let next = if node_count.saturating_mul(cycles) >= PAR_THRESHOLD {
            (0..node_count).into_par_iter().map(make_table).collect()
        } else {
            (0..node_count).map(make_table).collect()
        };
        levels.push(next);
        children_per_group = nodes_per_group;
    }
    levels
}

/// Reduce full-address claims through a balanced product tree. `d` must be a
/// power of two. The component/claim shapes are prover preconditions.
#[allow(clippy::too_many_arguments)]
pub fn virt_tree_prove(
    groups: &[VirtGroup],
    comps_per_group: &[&AddrComps],
    a: usize,
    d: usize,
    n: usize,
    transcript: &mut Transcript,
    label: &[u8],
) -> (Option<VirtTreeProof>, Vec<EvalClaim>) {
    let nterms = validate_groups(groups, a, d, n).expect("virt_tree_prove: dimensions/groups");
    assert_eq!(
        groups.len(),
        comps_per_group.len(),
        "virt_tree_prove: groups"
    );
    assert!(
        comps_per_group
            .iter()
            .all(|comps| component_shape(comps, a, d, n)),
        "virt_tree_prove: component shape"
    );
    bind_context(groups, a, d, n, transcript, label);
    if d == 1 {
        return (None, virt_direct(groups));
    }
    let nu = transcript.challenge_ks(&sublabel(label, b"/nu"), nterms);
    let mut representatives: Vec<usize> = Vec::new();
    let mut original_to_unique = Vec::with_capacity(groups.len());
    let mut weighted_points: Vec<Vec<(K, &[K])>> = Vec::new();
    let mut term_index = 0;
    for (gi, group) in groups.iter().enumerate() {
        let existing = representatives.iter().position(|&other| {
            group.r_k == groups[other].r_k
                && components_equal(comps_per_group[gi], comps_per_group[other])
        });
        let unique = existing.unwrap_or_else(|| {
            representatives.push(gi);
            weighted_points.push(Vec::new());
            representatives.len() - 1
        });
        original_to_unique.push(unique);
        for term in &group.terms {
            if let Some((weight, _)) = weighted_points[unique]
                .iter_mut()
                .find(|(_, point)| *point == term.cyc)
            {
                *weight += nu[term_index];
            } else {
                weighted_points[unique].push((nu[term_index], &term.cyc));
            }
            term_index += 1;
        }
    }
    let coefficients: Vec<Vec<SplitEq>> = weighted_points
        .iter()
        .map(|points| {
            points
                .iter()
                .map(|(weight, point)| {
                    let mut eq = SplitEq::new(point);
                    for value in &mut eq.lo {
                        *value *= *weight;
                    }
                    eq
                })
                .collect()
        })
        .collect();
    let chunk_eqs: Vec<Vec<Vec<K>>> = representatives
        .iter()
        .map(|&gi| {
            split_point(&groups[gi].r_k, a, d)
                .iter()
                .map(|point| eq_table(point))
                .collect()
        })
        .collect();
    let unique_groups = representatives.len();
    let mut levels = build_levels(comps_per_group, &representatives, &chunk_eqs, d, n);
    let height = d.trailing_zeros() as usize;
    let mut proof = VirtTreeProof {
        layers: Vec::with_capacity(height),
    };
    let mut point = Vec::new();
    let mut weights = Vec::new();
    let mut known_initial_claim = None;
    for layer in 0..height {
        let _span = Span::new("virt.tree.layers");
        let children_per_group = 1usize << (layer + 1);
        let parents_per_group = children_per_group / 2;
        let internal = levels.last();
        let child = |unique: usize, child: usize, j: usize| {
            if let Some(internal) = internal {
                internal[unique * children_per_group + child][j]
            } else {
                comps_per_group[representatives[unique]].row_eval(
                    child,
                    j,
                    &chunk_eqs[unique][child],
                )
            }
        };
        let ll = layer_label(label, layer);
        let (_, sumcheck, next_point, mles) = if layer == 0 {
            let row = |j: usize, out: &mut [K]| {
                for unique in 0..unique_groups {
                    out[unique] = coefficients[unique]
                        .iter()
                        .fold(K::ZERO, |acc, eq| acc + eq.at(j));
                    out[unique_groups + 2 * unique] = child(unique, 0, j);
                    out[unique_groups + 2 * unique + 1] = child(unique, 1, j);
                }
            };
            // With no groups there is still the zero polynomial. A dummy zero
            // coefficient/node keeps the lazy row interface nonempty.
            if unique_groups == 0 {
                let zero = |_: usize, out: &mut [K]| out.fill(K::ZERO);
                sumcheck_prove_rows_products(n, &zero, 1, 2, transcript, &sublabel(&ll, b"/sc"))
            } else {
                sumcheck_prove_rows_products(
                    n,
                    &row,
                    unique_groups,
                    2,
                    transcript,
                    &sublabel(&ll, b"/sc"),
                )
            }
        } else {
            let row = |j: usize, out: &mut [K]| {
                for unique in 0..unique_groups {
                    for node in 0..children_per_group {
                        out[unique * children_per_group + node] = child(unique, node, j);
                    }
                }
            };
            crate::gkr_kernels::prove(
                &point,
                row,
                &weights,
                known_initial_claim,
                transcript,
                &sublabel(&ll, b"/sc"),
            )
        };
        let offset = if layer == 0 { unique_groups } else { 0 };
        let child_evals: Vec<K> = original_to_unique
            .iter()
            .flat_map(|&unique| {
                mles[offset + unique * children_per_group
                    ..offset + (unique + 1) * children_per_group]
                    .iter()
                    .copied()
            })
            .collect();
        transcript.absorb_ks(&sublabel(&ll, b"/children"), &child_evals);
        point = next_point;
        if layer + 1 < height {
            // Sample only after all original child slots have been absorbed.
            // Keep original draw order even when equal tables are merged.
            let lambda =
                transcript.challenge_ks(&sublabel(&ll, b"/lambda-next"), child_evals.len());
            // These are this prover's own MLEs of the next layer's parent
            // tables, not external claims. Derive the next initial claim in
            // original-slot order before merging alias weights. On large
            // domains its saved endpoint scan outweighs inverse setup; keep
            // the old actual-first-sum/no-extra-inverse path for small cases.
            known_initial_claim = if 1usize << n >= PAR_THRESHOLD {
                Some(
                    lambda
                        .iter()
                        .zip(&child_evals)
                        .fold(U256::ZERO, |sum, (&weight, &value)| {
                            sum + weight.clmul(value)
                        })
                        .reduce(),
                )
            } else {
                None
            };
            weights = vec![K::ZERO; unique_groups * children_per_group];
            for (original, &unique) in original_to_unique.iter().enumerate() {
                for node in 0..children_per_group {
                    weights[unique * children_per_group + node] +=
                        lambda[original * children_per_group + node];
                }
            }
        }
        proof.layers.push(VirtTreeLayerProof {
            sumcheck,
            child_evals,
        });
        if parents_per_group < d / 2 {
            levels.pop();
        }
    }
    let leaves = &proof.layers.last().expect("nonempty tree").child_evals;
    let claims = virt_claims(groups, d, &point, leaves, a);
    (Some(proof), claims)
}

/// Verify the tree and return its terminal committed-component claims.
/// All proof/group shapes are checked before indexing or transcript mutation.
#[allow(clippy::too_many_arguments)]
pub fn virt_tree_verify(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    proof: Option<&VirtTreeProof>,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<Vec<EvalClaim>, VerificationError> {
    let nterms = validate_groups(groups, a, d, n)?;
    let height = d.trailing_zeros() as usize;
    if d == 1 {
        if proof.is_some() {
            return Err(invalid("virt tree: unexpected proof for d=1"));
        }
        bind_context(groups, a, d, n, transcript, label);
        return Ok(virt_direct(groups));
    }
    let proof = proof.ok_or_else(|| invalid("virt tree: missing proof"))?;
    if proof.layers.len() != height {
        return Err(invalid("virt tree: wrong layer count"));
    }
    for (layer, current) in proof.layers.iter().enumerate() {
        let count = groups
            .len()
            .checked_mul(1usize << (layer + 1))
            .ok_or_else(|| invalid("virt tree: too many nodes"))?;
        if current.child_evals.len() != count
            || current.sumcheck.round_msgs.len() != n
            || current
                .sumcheck
                .round_msgs
                .iter()
                .any(|message| message.len() != 3)
        {
            return Err(invalid("virt tree: malformed layer"));
        }
    }
    bind_context(groups, a, d, n, transcript, label);
    let nu = transcript.challenge_ks(&sublabel(label, b"/nu"), nterms);
    let mut claim = K::ZERO;
    let mut term_index = 0;
    for group in groups {
        for term in &group.terms {
            claim += nu[term_index] * term.value;
            term_index += 1;
        }
    }
    let mut point = Vec::new();
    let mut lambda: Vec<K> = Vec::new();
    for (layer, current) in proof.layers.iter().enumerate() {
        let children_per_group = 1usize << (layer + 1);
        let ll = layer_label(label, layer);
        let (next_point, expected) = sumcheck_verify(
            claim,
            n,
            3,
            &current.sumcheck,
            transcript,
            &sublabel(&ll, b"/sc"),
        )?;
        let actual = if layer == 0 {
            let mut total = U256::ZERO;
            let mut term_index = 0;
            for (gi, group) in groups.iter().enumerate() {
                let mut coefficient = U256::ZERO;
                for term in &group.terms {
                    coefficient += nu[term_index].clmul(eq_eval(&term.cyc, &next_point));
                    term_index += 1;
                }
                total += coefficient
                    .reduce()
                    .clmul(current.child_evals[2 * gi] * current.child_evals[2 * gi + 1]);
            }
            total.reduce()
        } else {
            let mut total = U256::ZERO;
            for (node, &weight) in lambda.iter().enumerate() {
                total +=
                    weight.clmul(current.child_evals[2 * node] * current.child_evals[2 * node + 1]);
            }
            eq_eval(&point, &next_point) * total.reduce()
        };
        if expected != actual {
            return Err(invalid("virt tree: layer final check failed"));
        }
        transcript.absorb_ks(&sublabel(&ll, b"/children"), &current.child_evals);
        point = next_point;
        if layer + 1 < height {
            lambda =
                transcript.challenge_ks(&sublabel(&ll, b"/lambda-next"), current.child_evals.len());
            claim = lambda
                .iter()
                .zip(&current.child_evals)
                .fold(U256::ZERO, |sum, (&weight, &value)| {
                    sum + weight.clmul(value)
                })
                .reduce();
        }
        debug_assert_eq!(current.child_evals.len(), groups.len() * children_per_group);
    }
    Ok(virt_claims(
        groups,
        d,
        &point,
        &proof.layers[height - 1].child_evals,
        a,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::VirtTerm;
    use om_sumcheck::{mle_eval, mle_eval_bits};

    fn random(seed: &mut u64) -> K {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        K::from_u128((*seed as u128) | ((seed.rotate_left(23) as u128) << 64))
    }

    fn group(comps: &AddrComps, a: usize, d: usize, n: usize, id: usize, rk: Vec<K>) -> VirtGroup {
        let chunks: Vec<Vec<K>> = split_point(&rk, a, d).iter().map(|r| eq_table(r)).collect();
        let product: Vec<K> = (0..1usize << n)
            .map(|j| (0..d).fold(K::ONE, |value, i| value * comps.row_eval(i, j, &chunks[i])))
            .collect();
        let mut seed = 0x3243f6a8885a308du64 ^ id as u64;
        VirtGroup {
            names: (0..d).map(|i| format!("g{id}/component{i}")).collect(),
            r_k: rk,
            terms: (0..2)
                .map(|_| {
                    let cyc: Vec<K> = (0..n).map(|_| random(&mut seed)).collect();
                    VirtTerm {
                        value: mle_eval(&product, &cyc),
                        cyc,
                    }
                })
                .collect(),
        }
    }

    fn fixture(d: usize, n: usize) -> (Vec<AddrComps>, Vec<VirtGroup>, usize) {
        let a = 2 * d;
        let mut seed = 0x13198a2e03707344;
        let mask = if a == 32 { u32::MAX } else { (1u32 << a) - 1 };
        let addrs = (0..1usize << n)
            .map(|_| {
                random(&mut seed);
                seed as u32 & mask
            })
            .collect();
        let first = AddrComps::from_addrs(addrs, a, d);
        // The same polynomials in a different physical representation must
        // alias safely. The third group intentionally has invalid one-hot rows.
        let mut second = first.clone();
        second.bits_mut();
        let mut third = second.clone();
        for table in third.bits_mut() {
            table[0] = !table[0];
            table[1] = !table[1];
        }
        let comps = vec![first, second, third];
        let rk: Vec<K> = (0..a).map(|_| random(&mut seed)).collect();
        let groups = comps
            .iter()
            .enumerate()
            .map(|(id, comp)| {
                let mut point = rk.clone();
                if id == 2 {
                    point[0] += K::ONE;
                }
                group(comp, a, d, n, id, point)
            })
            .collect();
        (comps, groups, a)
    }

    fn check_fixture(d: usize, n: usize) {
        let (comps, groups, a) = fixture(d, n);
        let refs: Vec<&AddrComps> = comps.iter().collect();
        let mut pt = Transcript::new_default(b"tree-core");
        let (proof, claims) = virt_tree_prove(&groups, &refs, a, d, n, &mut pt, b"v");
        let mut vt = Transcript::new_default(b"tree-core");
        let verified = virt_tree_verify(&groups, a, d, n, proof.as_ref(), &mut vt, b"v").unwrap();
        assert_eq!(claims, verified, "d={d}, n={n}");
        assert_eq!(pt.challenge_k(b"after"), vt.challenge_k(b"after"));
        assert_eq!(proof.is_some(), d != 1);
        if let Some(proof) = proof {
            assert_eq!(proof.layers.len(), d.ilog2() as usize);
            for (layer, proof) in proof.layers.iter().enumerate() {
                assert_eq!(proof.child_evals.len(), groups.len() * (1 << (layer + 1)));
                assert_eq!(proof.sumcheck.round_msgs.len(), n);
                assert!(proof.sumcheck.round_msgs.iter().all(|msg| msg.len() == 3));
            }
        }
        for claim in claims {
            let (gi, component) = groups
                .iter()
                .enumerate()
                .find_map(|(gi, group)| {
                    group
                        .names
                        .iter()
                        .position(|name| *name == claim.poly)
                        .map(|i| (gi, i))
                })
                .unwrap();
            assert_eq!(
                claim.value,
                mle_eval_bits(&comps[gi].table(component), &claim.point),
                "direct component MLE: d={d}, n={n}, group={gi}, component={component}"
            );
        }
    }

    #[test]
    fn tree_leaf_claims_match_direct_mles_with_aliases_and_arbitrary_bits() {
        for d in [1, 2, 4, 8, 16] {
            for n in [0, 1, 3] {
                check_fixture(d, n);
            }
        }
    }

    #[test]
    fn parallel_tree_leaf_claims_match_direct_mles() {
        check_fixture(4, 14);
    }

    #[test]
    fn tree_rejects_tampering_and_malformed_shapes_without_panics() {
        let (comps, groups, a) = fixture(8, 3);
        let refs: Vec<&AddrComps> = comps.iter().collect();
        let (proof, _) = virt_tree_prove(
            &groups,
            &refs,
            a,
            8,
            3,
            &mut Transcript::new_default(b"tree-core"),
            b"v",
        );
        let proof = proof.unwrap();
        let rejects = |proof: &VirtTreeProof| {
            assert!(
                virt_tree_verify(
                    &groups,
                    a,
                    8,
                    3,
                    Some(proof),
                    &mut Transcript::new_default(b"tree-core"),
                    b"v"
                )
                .is_err()
            );
        };
        for layer in 0..proof.layers.len() {
            for child in 0..proof.layers[layer].child_evals.len() {
                let mut changed = proof.clone();
                changed.layers[layer].child_evals[child] += K::ONE;
                rejects(&changed);
            }
            for round in 0..3 {
                for field in 0..3 {
                    let mut changed = proof.clone();
                    changed.layers[layer].sumcheck.round_msgs[round][field] += K::ONE;
                    rejects(&changed);
                }
            }
            let mut changed = proof.clone();
            changed.layers[layer].child_evals.pop();
            rejects(&changed);
            let mut changed = proof.clone();
            changed.layers[layer].sumcheck.round_msgs[0].pop();
            rejects(&changed);
            let mut changed = proof.clone();
            changed.layers[layer].sumcheck.round_msgs.pop();
            rejects(&changed);
        }
        let mut changed = proof.clone();
        changed.layers.pop();
        rejects(&changed);
        let mut changed = proof.clone();
        changed.layers.swap(0, 1);
        rejects(&changed);
        for (a, d, n) in [
            (a, 0, 3),
            (a, 3, 3),
            (a + 1, 8, 3),
            (a, 8, usize::BITS as usize),
        ] {
            assert!(
                virt_tree_verify(
                    &groups,
                    a,
                    d,
                    n,
                    Some(&proof),
                    &mut Transcript::new_default(b"tree-core"),
                    b"v"
                )
                .is_err()
            );
        }
        let mut changed = groups.clone();
        changed[0].terms[0].value += K::ONE;
        assert!(
            virt_tree_verify(
                &changed,
                a,
                8,
                3,
                Some(&proof),
                &mut Transcript::new_default(b"tree-core"),
                b"v"
            )
            .is_err()
        );
    }

    #[test]
    fn off_cube_internal_claim_uses_mle_of_pointwise_product() {
        let d = 4;
        let a = 4;
        let n = 1;
        let comps = AddrComps::from_addrs(vec![0, 15], a, d);
        let rk: Vec<K> = (2..6).map(K::t).collect();
        let z = K::t(7);
        let leaves: Vec<Vec<K>> = rk.iter().map(|&r| vec![K::ONE + r, r]).collect();
        let pointwise = vec![leaves[0][0] * leaves[1][0], leaves[0][1] * leaves[1][1]];
        assert_ne!(
            mle_eval(&pointwise, &[z]),
            mle_eval(&leaves[0], &[z]) * mle_eval(&leaves[1], &[z])
        );
        let g = group(&comps, a, d, n, 0, rk);
        let mut pt = Transcript::new_default(b"off-cube-tree");
        let (proof, claims) = virt_tree_prove(&[g.clone()], &[&comps], a, d, n, &mut pt, b"v");
        let mut vt = Transcript::new_default(b"off-cube-tree");
        assert_eq!(
            virt_tree_verify(&[g], a, d, n, proof.as_ref(), &mut vt, b"v").unwrap(),
            claims
        );
        for (component, claim) in claims.iter().enumerate() {
            assert_eq!(
                claim.value,
                mle_eval_bits(&comps.table(component), &claim.point)
            );
        }
    }
}
