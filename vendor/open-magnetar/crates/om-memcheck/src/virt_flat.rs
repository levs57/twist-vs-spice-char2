//! Flat virtualization with incoming-claim reduction and a shared cycle Eq.
//!
//! The first quadratic sumcheck moves every virtual selector claim to one
//! cycle point. Fresh batching coefficients then bind those intermediate
//! values to the product of the committed address chunks. The product
//! sumcheck keeps its equality factor transparent and its selectors compact.

use om_field::{K, U256};
use om_sumcheck::{SumcheckProof, VerificationError, eq_eval, sumcheck_verify};
use om_transcript::{Transcript, profiling::Span};

use crate::onehot::{
    AddrComps, EvalClaim, VirtGroup, components_equal, split_point, virt_claims, virt_direct,
};
use crate::sublabel;
use crate::virt_claim_reduction::{
    VirtClaimReductionProof, virt_claim_reduction_prove, virt_claim_reduction_verify,
};
use crate::virt_selectors::{SelectorColumn, SelectorState};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtFlatProof {
    pub reduction: VirtClaimReductionProof,
    pub sc: SumcheckProof,
    /// Original group-major committed component evaluations.
    pub evals: Vec<K>,
}

fn shape(groups: &[VirtGroup], a: usize, d: usize, n: usize) -> Result<usize, VerificationError> {
    if d == 0 || a == 0 || a > 32 || !a.is_multiple_of(d) || n >= usize::BITS as usize {
        return Err(VerificationError::new(
            "flat virtualization: invalid dimensions",
        ));
    }
    let mut terms = 0usize;
    for group in groups {
        if group.names.len() != d
            || group.r_k.len() != a
            || group.terms.iter().any(|term| term.cyc.len() != n)
        {
            return Err(VerificationError::new(
                "flat virtualization: malformed group",
            ));
        }
        terms = terms
            .checked_add(group.terms.len())
            .ok_or_else(|| VerificationError::new("flat virtualization: too many claims"))?;
    }
    Ok(terms)
}

fn bind_context(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    tr: &mut Transcript,
    label: &[u8],
) {
    tr.absorb_bytes(&sublabel(label, b"/protocol"), b"flat/common-cycle/v1");
    let dimensions: Vec<u8> = [a, d, n, groups.len()]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
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

pub fn virt_flat_prove(
    groups: &[VirtGroup],
    comps: &[&AddrComps],
    a: usize,
    d: usize,
    n: usize,
    tr: &mut Transcript,
    label: &[u8],
) -> (Option<VirtFlatProof>, Vec<EvalClaim>) {
    let terms = shape(groups, a, d, n).expect("flat virtualization: group shape");
    assert_eq!(
        groups.len(),
        comps.len(),
        "flat virtualization: component groups"
    );
    assert!(
        comps
            .iter()
            .all(|c| crate::virt_tree::component_shape(c, a, d, n)),
        "flat virtualization: malformed components"
    );
    bind_context(groups, a, d, n, tr, label);
    if d == 1 {
        return (None, virt_direct(groups));
    }
    let nu = tr.challenge_ks(&sublabel(label, b"/claim-weights"), terms);
    let (reduction, point) = {
        let _span = Span::new("virt.flat.claims");
        virt_claim_reduction_prove(
            groups,
            comps,
            a,
            d,
            n,
            &nu,
            tr,
            &sublabel(label, b"/reduce"),
        )
    };
    // Sample only after the reduction's original-order terminal values have
    // been absorbed. Their address points may differ, but their cycle point
    // is now shared by every product summand.
    let lambda = tr.challenge_ks(&sublabel(label, b"/product-weights"), groups.len());
    let initial = lambda
        .iter()
        .zip(&reduction.evals)
        .fold(U256::ZERO, |sum, (&weight, &value)| {
            sum + weight.clmul(value)
        })
        .reduce();
    let mut representatives = Vec::new();
    let mut original_to_unique = Vec::with_capacity(groups.len());
    let mut weights = Vec::new();
    for (g, &weight) in lambda.iter().enumerate() {
        let existing = representatives.iter().position(|&other: &usize| {
            groups[g].r_k == groups[other].r_k && components_equal(comps[g], comps[other])
        });
        let index = existing.unwrap_or_else(|| {
            representatives.push(g);
            weights.push(K::ZERO);
            representatives.len() - 1
        });
        weights[index] += weight;
        original_to_unique.push(index);
    }
    let points: Vec<Vec<Vec<K>>> = representatives
        .iter()
        .map(|&g| split_point(&groups[g].r_k, a, d))
        .collect();
    let columns = representatives
        .iter()
        .enumerate()
        .flat_map(|(unique, &g)| {
            points[unique]
                .iter()
                .enumerate()
                .map(move |(chunk, point)| SelectorColumn {
                    component: comps[g],
                    chunk,
                    point,
                })
        })
        .collect();
    let mut selectors = SelectorState::new(n, columns);
    let (sc, cycle, mles) = {
        let _span = Span::new("virt.flat.product");
        crate::virt_product::prove(
            &point,
            &mut selectors,
            &weights,
            d,
            initial,
            tr,
            &sublabel(label, b"/product"),
        )
    };
    let evals: Vec<K> = original_to_unique
        .iter()
        .flat_map(|&g| mles[g * d..(g + 1) * d].iter().copied())
        .collect();
    tr.absorb_ks(&sublabel(label, b"/evals"), &evals);
    let claims = virt_claims(groups, d, &cycle, &evals, a);
    (
        Some(VirtFlatProof {
            reduction,
            sc,
            evals,
        }),
        claims,
    )
}

pub fn virt_flat_verify(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    proof: Option<&VirtFlatProof>,
    tr: &mut Transcript,
    label: &[u8],
) -> Result<Vec<EvalClaim>, VerificationError> {
    let terms = shape(groups, a, d, n)?;
    if d == 1 {
        if proof.is_some() {
            return Err(VerificationError::new(
                "flat virtualization: unexpected direct proof",
            ));
        }
        bind_context(groups, a, d, n, tr, label);
        return Ok(virt_direct(groups));
    }
    let proof =
        proof.ok_or_else(|| VerificationError::new("flat virtualization: missing proof"))?;
    let width = groups
        .len()
        .checked_mul(d)
        .ok_or_else(|| VerificationError::new("flat virtualization: evaluation overflow"))?;
    if proof.evals.len() != width
        || proof.reduction.evals.len() != groups.len()
        || proof.sc.round_msgs.len() != n
        || proof.sc.round_msgs.iter().any(|msg| msg.len() != d + 1)
        || proof.reduction.sc.round_msgs.len() != n
        || proof
            .reduction
            .sc
            .round_msgs
            .iter()
            .any(|msg| msg.len() != 2)
    {
        return Err(VerificationError::new(
            "flat virtualization: malformed proof",
        ));
    }
    bind_context(groups, a, d, n, tr, label);
    let nu = tr.challenge_ks(&sublabel(label, b"/claim-weights"), terms);
    let point = virt_claim_reduction_verify(
        groups,
        a,
        d,
        n,
        &nu,
        &proof.reduction,
        tr,
        &sublabel(label, b"/reduce"),
    )?;
    let lambda = tr.challenge_ks(&sublabel(label, b"/product-weights"), groups.len());
    let initial = lambda
        .iter()
        .zip(&proof.reduction.evals)
        .fold(U256::ZERO, |sum, (&weight, &value)| {
            sum + weight.clmul(value)
        })
        .reduce();
    let (cycle, expected) = sumcheck_verify(
        initial,
        n,
        d + 1,
        &proof.sc,
        tr,
        &sublabel(label, b"/product"),
    )?;
    let mut sum = U256::ZERO;
    for (g, &weight) in lambda.iter().enumerate() {
        let product = proof.evals[g * d..(g + 1) * d]
            .iter()
            .copied()
            .fold(K::ONE, |product, value| product * value);
        sum += weight.clmul(product);
    }
    if eq_eval(&point, &cycle) * sum.reduce() != expected {
        return Err(VerificationError::new(
            "flat virtualization: product check failed",
        ));
    }
    tr.absorb_ks(&sublabel(label, b"/evals"), &proof.evals);
    Ok(virt_claims(groups, d, &cycle, &proof.evals, a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::{VirtTerm, full_table};
    use om_sumcheck::mle_eval_bits;

    fn point(seed: u128, len: usize) -> Vec<K> {
        (0..len)
            .map(|i| K::from_u128(seed.wrapping_mul(0x9e37_79b9 + i as u128)))
            .collect()
    }

    // Evaluate the full address table independently of both optimized stages.
    // Arbitrary bit rows must remain exact: the one-hot protocol, rather than
    // virtualization, is responsible for rejecting their invalid encoding.
    #[test]
    fn composition_matches_dense_openings_for_aliases_and_arbitrary_bit_rows() {
        for (a, d, n) in [(4, 2, 0), (6, 3, 1), (8, 4, 5), (8, 8, 4)] {
            for invalid in [false, true] {
                let t = 1usize << n;
                let first = AddrComps::from_addrs(
                    (0..t).map(|j| ((j * 17 + 3) % (1 << a)) as u32).collect(),
                    a,
                    d,
                );
                let mut second = AddrComps::from_addrs(
                    (0..t).map(|j| ((j * 11 + 9) % (1 << a)) as u32).collect(),
                    a,
                    d,
                );
                if invalid {
                    let tables = second.bits_mut();
                    for (i, table) in tables.iter_mut().enumerate() {
                        for (j, bit) in table.iter_mut().enumerate() {
                            if (j + i) % 5 == 0 {
                                *bit = !*bit;
                            }
                        }
                    }
                }
                let components = [first, second];
                let full: Vec<_> = components
                    .iter()
                    .map(|c| full_table(&(0..d).map(|i| c.table(i)).collect::<Vec<_>>(), a, d, t))
                    .collect();
                let sources = [0, 0, 0, 1];
                let groups: Vec<_> = sources
                    .iter()
                    .enumerate()
                    .map(|(g, &source)| {
                        let r_k = point(if g < 2 { 17 } else { 23 + g as u128 }, a);
                        let terms = (0..if g == 1 { 1 } else { 3 })
                            .map(|j| {
                                let cyc = if j == 0 {
                                    vec![K::ZERO; n]
                                } else {
                                    point(53 + j as u128, n)
                                };
                                let mut p = r_k.clone();
                                p.extend_from_slice(&cyc);
                                VirtTerm {
                                    cyc,
                                    value: mle_eval_bits(&full[source], &p),
                                }
                            })
                            .collect();
                        VirtGroup {
                            names: (0..d).map(|i| format!("component/{source}/{i}")).collect(),
                            r_k,
                            terms,
                        }
                    })
                    .collect();
                let comps: Vec<_> = sources.iter().map(|&i| &components[i]).collect();
                let mut prover = Transcript::new_default(b"flat-composition-oracle");
                let (proof, claims) =
                    virt_flat_prove(&groups, &comps, a, d, n, &mut prover, b"virt");
                let mut verifier = Transcript::new_default(b"flat-composition-oracle");
                assert_eq!(
                    virt_flat_verify(&groups, a, d, n, proof.as_ref(), &mut verifier, b"virt")
                        .unwrap(),
                    claims
                );
                assert_eq!(prover.challenge_k(b"after"), verifier.challenge_k(b"after"));
                assert_eq!(claims.len(), groups.len() * d);
                for (g, &source) in sources.iter().enumerate() {
                    for i in 0..d {
                        let claim = &claims[g * d + i];
                        assert_eq!(
                            claim.value,
                            mle_eval_bits(&components[source].table(i), &claim.point)
                        );
                    }
                }
                let mut bad = groups.clone();
                bad[0].terms[0].value += K::ONE;
                let mut prover = Transcript::new_default(b"bad-flat-claim");
                let (proof, _) = virt_flat_prove(&bad, &comps, a, d, n, &mut prover, b"virt");
                let mut verifier = Transcript::new_default(b"bad-flat-claim");
                assert!(
                    virt_flat_verify(&bad, a, d, n, proof.as_ref(), &mut verifier, b"virt")
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn empty_groups_and_zero_cycle_variables_compose() {
        for n in [0, 1, 5] {
            let mut prover = Transcript::new_default(b"empty-flat");
            let (proof, claims) = virt_flat_prove(&[], &[], 4, 2, n, &mut prover, b"virt");
            let mut verifier = Transcript::new_default(b"empty-flat");
            assert_eq!(
                virt_flat_verify(&[], 4, 2, n, proof.as_ref(), &mut verifier, b"virt").unwrap(),
                claims
            );
            assert!(claims.is_empty());
            assert_eq!(prover.challenge_k(b"after"), verifier.challenge_k(b"after"));
        }
    }
}
