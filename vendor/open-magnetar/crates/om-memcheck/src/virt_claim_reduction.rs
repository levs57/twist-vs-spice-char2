//! Reduce arbitrary virtual-address claims to one common cycle point.
//!
//! This adapts Jolt's two-pass, square-root-space RAM claim reduction to the
//! low-variable-first GF128 conventions used here. For each group, `H(j)` is
//! the pointwise product of its address-component contractions at the group's
//! fixed address point. Prefix rounds use `P(x) Q(x)`, where
//! `Q(x) = sum_y eq(c_hi,y) H(x,y)`. One further source pass constructs
//! `H'(y) = sum_x eq(s_lo,x) H(x,y)` for the suffix rounds. Neither pass
//! materializes a full-cycle field table.
//!
//! The returned evaluations are claims about *uncommitted virtual* address
//! polynomials. The caller must reduce them to authenticated component openings.
//! Batching weights must be sampled after the original claims are bound; this
//! module binds the complete statement again, but does not sample those weights.
//!
//! Algorithm reference: a16z/jolt at e21affaeaf0d2c0a4630b958a883a4ac877acbfd,
//! crates/jolt-kernels/src/optimized/ram_ra_claim_reduction.rs (MIT/Apache-2.0).

use om_field::{K, U256, clmul_pairs};
use om_sumcheck::{
    SumcheckProof, VerificationError, bind_low_in_place, eq_eval, eq_table, interpolate_eval,
    product_sum, sumcheck_init, sumcheck_verify,
};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::onehot::{AddrComps, PAR_THRESHOLD, VirtGroup, components_equal};
use crate::sublabel;

/// Degree-two sum-check plus one full-address virtual evaluation per input group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtClaimReductionProof {
    pub sc: SumcheckProof,
    /// Original group order, including aliases and groups with no nonzero weight.
    pub evals: Vec<K>,
}

fn invalid(message: &str) -> VerificationError {
    VerificationError::new(message)
}

fn validate(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    weights: &[K],
) -> Result<(), VerificationError> {
    if d == 0 || a < d || a > 32 || !a.is_multiple_of(d) || n >= usize::BITS as usize {
        return Err(invalid("virtual claim reduction: invalid dimensions"));
    }
    let mut count = 0usize;
    for group in groups {
        if group.names.len() != d
            || group.r_k.len() != a
            || group.terms.iter().any(|term| term.cyc.len() != n)
        {
            return Err(invalid("virtual claim reduction: malformed group"));
        }
        count = count
            .checked_add(group.terms.len())
            .ok_or_else(|| invalid("virtual claim reduction: too many terms"))?;
    }
    if weights.len() != count {
        return Err(invalid("virtual claim reduction: wrong weight count"));
    }
    Ok(())
}

fn component_shape(comps: &AddrComps, a: usize, d: usize, n: usize) -> bool {
    let cycles = 1usize << n;
    match comps {
        AddrComps::Addrs {
            addrs,
            a: ca,
            d: cd,
        } => {
            *ca == a
                && *cd == d
                && addrs.len() == cycles
                && addrs
                    .iter()
                    .all(|&address| u64::from(address) < (1u64 << a))
        }
        AddrComps::Bits { tables, ac } => {
            *ac == a / d
                && tables.len() == d
                && 1usize
                    .checked_shl(*ac as u32)
                    .and_then(|width| width.checked_mul(cycles))
                    .is_some_and(|len| tables.iter().all(|table| table.len() == len))
        }
    }
}

fn bind_context(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    weights: &[K],
    tr: &mut Transcript,
    label: &[u8],
) {
    tr.absorb_bytes(
        &sublabel(label, b"/protocol"),
        b"virtual-claim-reduction/v1",
    );
    let mut dimensions = Vec::with_capacity(32);
    for dimension in [a, d, n, groups.len()] {
        dimensions.extend_from_slice(&(dimension as u64).to_le_bytes());
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
    tr.absorb_ks(&sublabel(label, b"/weights"), weights);
}

/// A bounded lookup for the full address contraction. Normal A<=16 cases use
/// one gather; wider addresses use two square-root tables and one product.
enum AddressEq {
    Full(Vec<K>),
    Split { lo: Vec<K>, hi: Vec<K>, bits: usize },
}

impl AddressEq {
    fn new(point: &[K]) -> Self {
        const FULL_TABLE_BITS: usize = 16; // At most 1 MiB per distinct address point.
        if point.len() <= FULL_TABLE_BITS {
            Self::Full(eq_table(point))
        } else {
            let bits = point.len() / 2;
            Self::Split {
                lo: eq_table(&point[..bits]),
                hi: eq_table(&point[bits..]),
                bits,
            }
        }
    }

    #[inline]
    fn at(&self, address: usize) -> K {
        match self {
            Self::Full(table) => table[address],
            Self::Split { lo, hi, bits } => lo[address & (lo.len() - 1)] * hi[address >> bits],
        }
    }
}

enum Rows<'a> {
    Addrs {
        addresses: &'a [u32],
        eq: AddressEq,
    },
    Bits {
        comps: &'a AddrComps,
        chunk_eqs: Vec<Vec<K>>,
    },
}

impl<'a> Rows<'a> {
    fn new(comps: &'a AddrComps, point: &[K], d: usize) -> Self {
        match comps {
            AddrComps::Addrs { addrs, .. } => Self::Addrs {
                addresses: addrs,
                eq: AddressEq::new(point),
            },
            AddrComps::Bits { .. } => Self::Bits {
                comps,
                chunk_eqs: point.chunks_exact(point.len() / d).map(eq_table).collect(),
            },
        }
    }

    #[inline]
    fn at(&self, cycle: usize) -> K {
        match self {
            Self::Addrs { addresses, eq } => eq.at(addresses[cycle] as usize),
            Self::Bits { comps, chunk_eqs } => chunk_eqs
                .iter()
                .enumerate()
                .fold(K::ONE, |product, (i, eq)| {
                    product * comps.row_eval(i, cycle, eq)
                }),
        }
    }
}

struct CycleEq<'a> {
    point: &'a [K],
    lo: Vec<K>,
    hi: Vec<K>,
}

struct Term {
    source: usize,
    cycle: usize,
    coefficient: K,
}

struct ProductTables {
    left: Vec<K>,
    right: Vec<K>,
}

fn products_sum(products: &[ProductTables]) -> K {
    products.iter().fold(K::ZERO, |sum, product| {
        sum + product_sum(&product.left, &product.right)
    })
}

/// The current quadratic's endpoint and leading coefficient suffice because
/// g(1)=claim+g(0). Each product contributes two unreduced carryless products.
fn round_evals(products: &[ProductTables], claim: K) -> [K; 3] {
    let mut endpoint = U256::ZERO;
    let mut leading = U256::ZERO;
    for product in products {
        for (a, b) in product
            .left
            .chunks_exact(2)
            .zip(product.right.chunks_exact(2))
        {
            let [low, high] = clmul_pairs([a[0], a[0] + a[1]], [b[0], b[0] + b[1]]);
            endpoint += low;
            leading += high;
        }
    }
    let g0 = endpoint.reduce();
    let g2 = g0 + claim.mul_t(2) + leading.reduce().mul_t(6);
    [g0, claim + g0, g2]
}

fn bind_products(products: &mut [ProductTables], r: K) {
    for product in products {
        bind_low_in_place(&mut product.left, r);
        bind_low_in_place(&mut product.right, r);
    }
}

/// First source pass. Worker-local wide accumulators reduce once per prefix
/// coordinate, not once per input row. All terms on this H share its row gather.
fn prefix_q(
    rows: &Rows<'_>,
    term_cycles: &[usize],
    cycles: &[CycleEq<'_>],
    prefix_bits: usize,
    suffix_size: usize,
) -> Vec<Vec<K>> {
    let width = 1usize << prefix_bits;
    let empty = || vec![vec![U256::ZERO; width]; term_cycles.len()];
    let fill = |mut sums: Vec<Vec<U256>>, y: usize| {
        for x in 0..width {
            let value = rows.at(x + (y << prefix_bits));
            let mut terms = term_cycles.iter().enumerate();
            while let Some((first, &cycle)) = terms.next() {
                if let Some((second, &other)) = terms.next() {
                    let [a, b] =
                        clmul_pairs([value, value], [cycles[cycle].hi[y], cycles[other].hi[y]]);
                    sums[first][x] += a;
                    sums[second][x] += b;
                } else {
                    sums[first][x] += value.clmul(cycles[cycle].hi[y]);
                }
            }
        }
        sums
    };
    let wide = if width.saturating_mul(suffix_size) >= PAR_THRESHOLD && suffix_size >= 16 {
        (0..suffix_size)
            .into_par_iter()
            .with_min_len(16)
            .fold(empty, fill)
            .reduce(empty, |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                }
                left
            })
    } else {
        (0..suffix_size).fold(empty(), fill)
    };
    wide.into_iter()
        .map(|table| table.into_iter().map(U256::reduce).collect())
        .collect()
}

/// Second source pass. Each output owns an independent suffix coordinate.
fn suffix_h(rows: &Rows<'_>, prefix: &[K], prefix_bits: usize, size: usize) -> Vec<K> {
    let evaluate = |y: usize| {
        let mut sum = U256::ZERO;
        for (x, &weight) in prefix.iter().enumerate() {
            sum += weight.clmul(rows.at(x + (y << prefix_bits)));
        }
        sum.reduce()
    };
    if size.saturating_mul(prefix.len()) >= PAR_THRESHOLD && size >= 16 {
        (0..size).into_par_iter().map(evaluate).collect()
    } else {
        (0..size).map(evaluate).collect()
    }
}

fn supplied_claim(groups: &[VirtGroup], weights: &[K]) -> K {
    let mut accumulator = U256::ZERO;
    for (term, &weight) in groups.iter().flat_map(|group| &group.terms).zip(weights) {
        accumulator += weight.clmul(term.value);
    }
    accumulator.reduce()
}

/// Prove a common-cycle reduction. Input shapes are prover preconditions.
///
/// `weights` follow original group-major, then term order and must be sampled
/// after those claims are bound. Bad claimed values do not panic: the prover
/// uses the actual sum, and verification against the supplied claims rejects.
#[allow(clippy::too_many_arguments)]
pub fn virt_claim_reduction_prove(
    groups: &[VirtGroup],
    comps: &[&AddrComps],
    a: usize,
    d: usize,
    n: usize,
    weights: &[K],
    transcript: &mut Transcript,
    label: &[u8],
) -> (VirtClaimReductionProof, Vec<K>) {
    let _span = Span::new("virt.claim_reduction");
    validate(groups, a, d, n, weights).expect("virtual claim reduction: input shapes");
    assert_eq!(
        groups.len(),
        comps.len(),
        "virtual claim reduction: sources"
    );
    assert!(comps.iter().all(|comps| component_shape(comps, a, d, n)));
    bind_context(groups, a, d, n, weights, transcript, label);

    let prefix_bits = n / 2;
    let suffix_size = 1usize << (n - prefix_bits);
    let mut representatives: Vec<usize> = Vec::new();
    let mut source_map = Vec::with_capacity(groups.len());
    for (gi, group) in groups.iter().enumerate() {
        let source = representatives.iter().position(|&other| {
            group.r_k == groups[other].r_k && components_equal(comps[gi], comps[other])
        });
        source_map.push(source.unwrap_or_else(|| {
            representatives.push(gi);
            representatives.len() - 1
        }));
    }
    let rows: Vec<_> = representatives
        .iter()
        .map(|&gi| Rows::new(comps[gi], &groups[gi].r_k, d))
        .collect();
    let mut cycle_eqs: Vec<CycleEq<'_>> = Vec::new();
    let mut terms: Vec<Term> = Vec::new();
    let mut weight_index = 0;
    for (gi, group) in groups.iter().enumerate() {
        for term in &group.terms {
            let coefficient = weights[weight_index];
            weight_index += 1;
            if coefficient == K::ZERO {
                continue;
            }
            let cycle = cycle_eqs
                .iter()
                .position(|eq| eq.point == term.cyc)
                .unwrap_or_else(|| {
                    cycle_eqs.push(CycleEq {
                        point: &term.cyc,
                        lo: eq_table(&term.cyc[..prefix_bits]),
                        hi: eq_table(&term.cyc[prefix_bits..]),
                    });
                    cycle_eqs.len() - 1
                });
            if let Some(existing) = terms
                .iter_mut()
                .find(|existing| existing.source == source_map[gi] && existing.cycle == cycle)
            {
                existing.coefficient += coefficient;
            } else {
                terms.push(Term {
                    source: source_map[gi],
                    cycle,
                    coefficient,
                });
            }
        }
    }
    terms.retain(|term| term.coefficient != K::ZERO);
    let mut prefix_products: Vec<_> = if prefix_bits == 0 {
        Vec::new()
    } else {
        terms
            .iter()
            .map(|term| ProductTables {
                left: cycle_eqs[term.cycle]
                    .lo
                    .iter()
                    .map(|&v| term.coefficient * v)
                    .collect(),
                right: Vec::new(),
            })
            .collect()
    };
    if prefix_bits != 0 {
        for (source, rows) in rows.iter().enumerate() {
            let indices: Vec<_> = terms
                .iter()
                .enumerate()
                .filter_map(|(i, term)| (term.source == source).then_some(i))
                .collect();
            if indices.is_empty() {
                continue;
            }
            let cycles: Vec<_> = indices.iter().map(|&i| terms[i].cycle).collect();
            for (&i, q) in indices.iter().zip(prefix_q(
                rows,
                &cycles,
                &cycle_eqs,
                prefix_bits,
                suffix_size,
            )) {
                prefix_products[i].right = q;
            }
        }
    }

    let sc_label = sublabel(label, b"/sc");
    let msg_label = sublabel(&sc_label, b"/msg");
    let chal_label = sublabel(&sc_label, b"/chal");
    let mut point = Vec::with_capacity(n);
    let mut messages = Vec::with_capacity(n);
    let mut current = if prefix_bits != 0 {
        products_sum(&prefix_products)
    } else {
        K::ZERO
    };
    // With no prefix the suffix source pass supplies the initial sum instead.
    if prefix_bits != 0 {
        sumcheck_init(transcript, &sc_label, current, n, 2);
        for _ in 0..prefix_bits {
            let values = round_evals(&prefix_products, current);
            let message = vec![values[0], values[2]];
            transcript.absorb_ks(&msg_label, &message);
            let challenge = transcript.challenge_k(&chal_label);
            messages.push(message);
            point.push(challenge);
            current = interpolate_eval(&values, challenge);
            bind_products(&mut prefix_products, challenge);
        }
    }
    drop(prefix_products);

    let prefix = eq_table(&point);
    let scales: Vec<_> = terms
        .iter()
        .map(|term| term.coefficient * eq_eval(&cycle_eqs[term.cycle].point[..prefix_bits], &point))
        .collect();
    let mut suffix_products: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(source, rows)| {
            let active: Vec<_> = terms
                .iter()
                .enumerate()
                .filter_map(|(i, term)| (term.source == source).then_some((i, term.cycle)))
                .collect();
            let left = (0..suffix_size)
                .map(|y| {
                    let mut sum = U256::ZERO;
                    for &(i, cycle) in &active {
                        sum += scales[i].clmul(cycle_eqs[cycle].hi[y]);
                    }
                    sum.reduce()
                })
                .collect();
            ProductTables {
                left,
                right: suffix_h(rows, &prefix, prefix_bits, suffix_size),
            }
        })
        .collect();
    if prefix_bits == 0 {
        current = products_sum(&suffix_products);
        sumcheck_init(transcript, &sc_label, current, n, 2);
    } else {
        debug_assert_eq!(current, products_sum(&suffix_products));
    }
    for _ in prefix_bits..n {
        let values = round_evals(&suffix_products, current);
        let message = vec![values[0], values[2]];
        transcript.absorb_ks(&msg_label, &message);
        let challenge = transcript.challenge_k(&chal_label);
        messages.push(message);
        point.push(challenge);
        current = interpolate_eval(&values, challenge);
        bind_products(&mut suffix_products, challenge);
    }
    let evals = source_map
        .iter()
        .map(|&source| suffix_products[source].right[0])
        .collect::<Vec<_>>();
    transcript.absorb_ks(&sublabel(label, b"/evals"), &evals);
    (
        VirtClaimReductionProof {
            sc: SumcheckProof {
                round_msgs: messages,
            },
            evals,
        },
        point,
    )
}

/// Verify the reduction and return its common cycle point. The caller must
/// authenticate `proof.evals[g]` as the virtual address value for every group.
/// Every public/proof shape is checked before transcript interaction.
#[allow(clippy::too_many_arguments)]
pub fn virt_claim_reduction_verify(
    groups: &[VirtGroup],
    a: usize,
    d: usize,
    n: usize,
    weights: &[K],
    proof: &VirtClaimReductionProof,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<Vec<K>, VerificationError> {
    validate(groups, a, d, n, weights)?;
    if proof.evals.len() != groups.len()
        || proof.sc.round_msgs.len() != n
        || proof.sc.round_msgs.iter().any(|message| message.len() != 2)
    {
        return Err(invalid("virtual claim reduction: malformed proof"));
    }
    bind_context(groups, a, d, n, weights, transcript, label);
    let (point, expected) = sumcheck_verify(
        supplied_claim(groups, weights),
        n,
        2,
        &proof.sc,
        transcript,
        &sublabel(label, b"/sc"),
    )?;
    transcript.absorb_ks(&sublabel(label, b"/evals"), &proof.evals);
    let mut index = 0;
    let mut final_value = U256::ZERO;
    for (group, &evaluation) in groups.iter().zip(&proof.evals) {
        let mut scale = U256::ZERO;
        for term in &group.terms {
            scale += weights[index].clmul(eq_eval(&term.cyc, &point));
            index += 1;
        }
        final_value += scale.reduce().clmul(evaluation);
    }
    if expected != final_value.reduce() {
        return Err(invalid("virtual claim reduction: terminal relation failed"));
    }
    Ok(point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::{VirtTerm, bits_of, comp_tables};
    use om_sumcheck::{combine_sum, mle_eval, sumcheck_prove};
    use rayon::ThreadPoolBuilder;

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random(state: &mut u64) -> K {
        K::from_u128(u128::from(next(state)) | (u128::from(next(state)) << 64))
    }

    /// Independent dense oracle: direct equality on all address bits for Addrs,
    /// and literal sums over the arbitrary bit rows for Bits. No Rows/AddressEq
    /// lookup, source deduplication, or prefix/suffix decomposition is reused.
    fn dense_h(group: &VirtGroup, comps: &AddrComps, d: usize, n: usize) -> Vec<K> {
        match comps {
            AddrComps::Addrs { addrs, a, .. } => addrs
                .iter()
                .map(|&address| eq_eval(&group.r_k, &bits_of(u64::from(address), *a)))
                .collect(),
            AddrComps::Bits { tables, ac } => {
                let chunk_eqs: Vec<_> = group.r_k.chunks_exact(*ac).map(eq_table).collect();
                let width = 1usize << ac;
                (0..1usize << n)
                    .map(|j| {
                        let mut product = K::ONE;
                        for i in 0..d {
                            let mut contraction = K::ZERO;
                            for u in 0..width {
                                if tables[i][u + width * j] {
                                    contraction += chunk_eqs[i][u];
                                }
                            }
                            product *= contraction;
                        }
                        product
                    })
                    .collect()
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_oracle(
        groups: &[VirtGroup],
        comps: &[&AddrComps],
        a: usize,
        d: usize,
        n: usize,
        weights: &[K],
        transcript: &mut Transcript,
        label: &[u8],
    ) -> (VirtClaimReductionProof, Vec<K>) {
        bind_context(groups, a, d, n, weights, transcript, label);
        let hs: Vec<_> = groups
            .iter()
            .zip(comps)
            .map(|(g, c)| dense_h(g, c, d, n))
            .collect();
        let mut tables = hs.clone();
        let mut owners = Vec::new();
        for (gi, group) in groups.iter().enumerate() {
            for term in &group.terms {
                owners.push(gi);
                tables.push(eq_table(&term.cyc));
            }
        }
        if tables.is_empty() {
            tables.push(vec![K::ZERO; 1usize << n]);
        }
        let combine = |values: &[K]| {
            owners
                .iter()
                .zip(weights)
                .enumerate()
                .fold(K::ZERO, |sum, (i, (&gi, &weight))| {
                    sum + weight * values[gi] * values[groups.len() + i]
                })
        };
        let actual = combine_sum(&tables, &combine);
        let (sc, point, _) = sumcheck_prove(
            actual,
            tables,
            &combine,
            2,
            transcript,
            &sublabel(label, b"/sc"),
        );
        let evals: Vec<_> = hs.iter().map(|h| mle_eval(h, &point)).collect();
        transcript.absorb_ks(&sublabel(label, b"/evals"), &evals);
        (VirtClaimReductionProof { sc, evals }, point)
    }

    fn fixture(
        a: usize,
        d: usize,
        n: usize,
        boolean: bool,
    ) -> (Vec<VirtGroup>, Vec<AddrComps>, Vec<K>) {
        let mut state = 0x7ad9_042d_13b5_8561;
        let addresses: Vec<_> = (0..1usize << n)
            .map(|_| next(&mut state) as usize & ((1usize << a) - 1))
            .collect();
        let independent: Vec<_> = (0..1usize << n)
            .map(|_| next(&mut state) as usize & ((1usize << a) - 1))
            .collect();
        let original = AddrComps::from_usizes(&addresses, a, d);
        let mut invalid = comp_tables(&addresses, a, d);
        invalid[0][0] = !invalid[0][0];
        if invalid[0].len() > 1 {
            invalid[0][1] = true;
        }
        let comps = vec![
            original.clone(),
            AddrComps::from_usizes(&independent, a, d),
            original.clone(),
            AddrComps::Bits {
                tables: invalid,
                ac: a / d,
            },
            original,
        ];
        let address_point: Vec<_> = (0..a)
            .map(|_| {
                if boolean {
                    K::from_bit(next(&mut state) & 1 != 0)
                } else {
                    random(&mut state)
                }
            })
            .collect();
        let other_address: Vec<_> = (0..a).map(|_| random(&mut state)).collect();
        let cycle: Vec<_> = (0..n)
            .map(|_| {
                if boolean {
                    K::from_bit(next(&mut state) & 1 != 0)
                } else {
                    random(&mut state)
                }
            })
            .collect();
        let other_cycle: Vec<_> = (0..n).map(|_| random(&mut state)).collect();
        let mut weights = Vec::new();
        let groups = comps
            .iter()
            .enumerate()
            .map(|(gi, component)| {
                let mut group = VirtGroup {
                    names: (0..d).map(|i| format!("group{gi}/component{i}")).collect(),
                    r_k: if gi == 2 {
                        other_address.clone()
                    } else {
                        address_point.clone()
                    },
                    terms: Vec::new(),
                };
                let h = dense_h(&group, component, d, n);
                group.terms = [cycle.clone(), other_cycle.clone(), cycle.clone()]
                    .into_iter()
                    .map(|cyc| {
                        weights.push(random(&mut state));
                        VirtTerm {
                            value: mle_eval(&h, &cyc),
                            cyc,
                        }
                    })
                    .collect();
                group
            })
            .collect();
        (groups, comps, weights)
    }

    fn compare(
        groups: &[VirtGroup],
        comps: &[&AddrComps],
        a: usize,
        d: usize,
        n: usize,
        weights: &[K],
    ) {
        let label = b"reduction-oracle";
        let mut actual_tr = Transcript::new_default(b"common-cycle-test");
        let mut expected_tr = Transcript::new_default(b"common-cycle-test");
        let mut verifier_tr = Transcript::new_default(b"common-cycle-test");
        let actual =
            virt_claim_reduction_prove(groups, comps, a, d, n, weights, &mut actual_tr, label);
        let expected = dense_oracle(groups, comps, a, d, n, weights, &mut expected_tr, label);
        assert_eq!(actual, expected, "a={a}, d={d}, n={n}");
        assert_eq!(
            virt_claim_reduction_verify(
                groups,
                a,
                d,
                n,
                weights,
                &actual.0,
                &mut verifier_tr,
                label
            )
            .unwrap(),
            actual.1
        );
        let after = actual_tr.challenge_k(b"after");
        assert_eq!(after, expected_tr.challenge_k(b"after"));
        assert_eq!(after, verifier_tr.challenge_k(b"after"));
    }

    #[test]
    fn sqrt_reduction_matches_dense_original_terms_and_challenges() {
        for (a, d) in [(2, 1), (4, 2), (6, 3), (8, 4)] {
            for n in [0, 1, 2, 3, 6] {
                for boolean in [false, true] {
                    let (groups, components, weights) = fixture(a, d, n, boolean);
                    let comps: Vec<_> = components.iter().collect();
                    compare(&groups, &comps, a, d, n, &weights);
                }
            }
        }
    }

    #[test]
    fn parallel_passes_match_dense_in_one_and_four_worker_pools() {
        let (groups, components, weights) = fixture(4, 2, 14, false);
        let comps: Vec<_> = components.iter().collect();
        for threads in [1, 4] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    compare(&groups, &comps, 4, 2, 14, &weights);
                });
        }
    }

    #[test]
    fn prefix_transition_preserves_boolean_and_random_binding_points() {
        for n in [2, 5, 6] {
            let (groups, components, _) = fixture(4, 2, n, false);
            let bits = n / 2;
            let width = 1usize << bits;
            let size = 1usize << (n - bits);
            let mut state = 71;
            let random_point: Vec<_> = (0..bits).map(|_| random(&mut state)).collect();
            for gi in [0, 3] {
                // Compact Addrs and genuinely arbitrary Bits.
                let rows = Rows::new(&components[gi], &groups[gi].r_k, 2);
                let h = dense_h(&groups[gi], &components[gi], 2, n);
                let point = &groups[gi].terms[0].cyc;
                let cycle = CycleEq {
                    point,
                    lo: eq_table(&point[..bits]),
                    hi: eq_table(&point[bits..]),
                };
                let qs = prefix_q(&rows, &[0], std::slice::from_ref(&cycle), bits, size);
                for bound in [
                    vec![K::ZERO; bits],
                    vec![K::ONE; bits],
                    random_point.clone(),
                ] {
                    let h_prime = suffix_h(&rows, &eq_table(&bound), bits, size);
                    for (y, &actual) in h_prime.iter().enumerate() {
                        assert_eq!(actual, mle_eval(&h[y * width..(y + 1) * width], &bound));
                    }
                    assert_eq!(mle_eval(&qs[0], &bound), product_sum(&h_prime, &cycle.hi));
                    let prefix_value = eq_eval(&point[..bits], &bound);
                    let transitioned_claim = prefix_value * product_sum(&h_prime, &cycle.hi);
                    assert_eq!(
                        transitioned_claim,
                        mle_eval(&cycle.lo, &bound) * mle_eval(&qs[0], &bound)
                    );
                }
            }
        }
    }

    #[test]
    fn split_address_tables_cover_wide_points_without_dense_address_domain() {
        let a = 32;
        let d = 4;
        let n = 3;
        let addresses = vec![0, u32::MAX, 1, 0x10_8001, 0x8000_0000, 17, 0xdead_beef, 7];
        let comps = AddrComps::from_addrs(addresses, a, d);
        let mut state = 19;
        let mut group = VirtGroup {
            names: (0..d).map(|i| format!("wide/{i}")).collect(),
            r_k: (0..a).map(|_| random(&mut state)).collect(),
            terms: Vec::new(),
        };
        let h = dense_h(&group, &comps, d, n);
        for point in [
            vec![K::ZERO; n],
            vec![K::ONE; n],
            (0..n).map(|_| random(&mut state)).collect(),
        ] {
            group.terms.push(VirtTerm {
                value: mle_eval(&h, &point),
                cyc: point,
            });
        }
        compare(&[group], &[&comps], a, d, n, &[K::ONE, K::t(3), K::t(9)]);
    }

    #[test]
    fn empty_terms_groups_zero_weights_and_duplicate_cancellation() {
        for n in [0, 1, 5] {
            compare(&[], &[], 4, 2, n, &[]);
            let (mut groups, components, mut weights) = fixture(4, 2, n, false);
            let comps: Vec<_> = components.iter().collect();
            weights.fill(K::ZERO);
            compare(&groups, &comps, 4, 2, n, &weights);
            for (gi, group) in groups.iter_mut().enumerate() {
                group.terms[1] = group.terms[0].clone();
                group.terms[2] = group.terms[0].clone();
                weights[3 * gi] = K::t(7);
                weights[3 * gi + 1] = K::t(7);
                weights[3 * gi + 2] = K::ZERO;
            }
            compare(&groups, &comps, 4, 2, n, &weights);
            for group in &mut groups {
                group.terms.clear();
            }
            compare(&groups, &comps, 4, 2, n, &[]);
        }
    }

    #[test]
    fn bad_claims_produce_a_proof_but_verification_rejects() {
        for n in [0, 1, 5] {
            let (mut groups, components, weights) = fixture(4, 2, n, false);
            groups[0].terms[0].value += K::ONE;
            let comps: Vec<_> = components.iter().collect();
            let mut tr = Transcript::new_default(b"bad-claim");
            let (proof, _) =
                virt_claim_reduction_prove(&groups, &comps, 4, 2, n, &weights, &mut tr, b"cr");
            let mut verify_tr = Transcript::new_default(b"bad-claim");
            assert!(
                virt_claim_reduction_verify(
                    &groups,
                    4,
                    2,
                    n,
                    &weights,
                    &proof,
                    &mut verify_tr,
                    b"cr"
                )
                .is_err()
            );
        }
    }

    #[test]
    fn messages_evaluations_and_statement_replays_reject() {
        let n = 5;
        let (groups, components, weights) = fixture(4, 2, n, false);
        let comps: Vec<_> = components.iter().collect();
        let mut tr = Transcript::new_default(b"tamper");
        let (proof, _) =
            virt_claim_reduction_prove(&groups, &comps, 4, 2, n, &weights, &mut tr, b"cr");
        let rejects = |groups: &[VirtGroup], weights: &[K], proof: &VirtClaimReductionProof| {
            let mut tr = Transcript::new_default(b"tamper");
            assert!(
                virt_claim_reduction_verify(groups, 4, 2, n, weights, proof, &mut tr, b"cr")
                    .is_err()
            );
        };
        for round in 0..n {
            for coefficient in 0..2 {
                let mut changed = proof.clone();
                changed.sc.round_msgs[round][coefficient] += K::ONE;
                rejects(&groups, &weights, &changed);
            }
        }
        for gi in 0..groups.len() {
            let mut changed = proof.clone();
            changed.evals[gi] += K::ONE;
            rejects(&groups, &weights, &changed);
        }
        for mutator in 0..4 {
            let mut changed = groups.clone();
            match mutator {
                0 => changed[0].r_k[0] += K::ONE,
                1 => changed[0].terms[0].cyc[0] += K::ONE,
                2 => changed[0].names[0].push_str("-other"),
                _ => changed.swap(0, 1),
            }
            rejects(&changed, &weights, &proof);
        }
        let mut changed = weights.clone();
        changed[0] += K::ONE;
        rejects(&groups, &changed, &proof);
    }

    #[test]
    fn malformed_shapes_reject_before_transcript_use() {
        let (groups, components, weights) = fixture(4, 2, 3, false);
        let comps: Vec<_> = components.iter().collect();
        let mut tr = Transcript::new_default(b"shape");
        let (proof, _) =
            virt_claim_reduction_prove(&groups, &comps, 4, 2, 3, &weights, &mut tr, b"cr");
        let check =
            |groups: &[VirtGroup], a, d, n, weights: &[K], proof: &VirtClaimReductionProof| {
                let mut tr = Transcript::new_default(b"shape");
                let mut untouched = tr.clone();
                assert!(
                    virt_claim_reduction_verify(groups, a, d, n, weights, proof, &mut tr, b"cr")
                        .is_err()
                );
                assert_eq!(tr.challenge_k(b"after"), untouched.challenge_k(b"after"));
            };
        for mutation in 0..6 {
            let mut changed = proof.clone();
            match mutation {
                0 => {
                    changed.sc.round_msgs.pop();
                }
                1 => changed.sc.round_msgs.push(vec![K::ZERO; 2]),
                2 => {
                    changed.sc.round_msgs[1].pop();
                }
                3 => changed.sc.round_msgs[1].push(K::ZERO),
                4 => {
                    changed.evals.pop();
                }
                _ => changed.evals.push(K::ZERO),
            }
            check(&groups, 4, 2, 3, &weights, &changed);
        }
        check(&groups, 4, 0, 3, &weights, &proof);
        check(&groups, 4, 3, 3, &weights, &proof);
        check(&groups, 33, 1, 3, &weights, &proof);
        check(&groups, 4, 2, usize::BITS as usize, &weights, &proof);
        check(&groups, 4, 2, 3, &weights[..weights.len() - 1], &proof);
        for mutation in 0..3 {
            let mut changed = groups.clone();
            match mutation {
                0 => {
                    changed[0].names.pop();
                }
                1 => {
                    changed[0].r_k.pop();
                }
                _ => {
                    changed[0].terms[0].cyc.pop();
                }
            }
            check(&changed, 4, 2, 3, &weights, &proof);
        }
    }
}
