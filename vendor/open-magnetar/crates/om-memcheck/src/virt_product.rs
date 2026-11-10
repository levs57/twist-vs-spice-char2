//! Flat, equality-factored selector product sum-check over GF(2^128).
//!
//! The product tree here is a prover-local polynomial multiplication tree in
//! each cycle pair. It introduces no intermediate protocol claims. Coefficient
//! Karatsuba uses characteristic-two XORs, never integer-grid differences.
//!
//! Strategy reference: a16z/jolt, revision
//! `e21affaeaf0d2c0a4630b958a883a4ac877acbfd`,
//! `crates/jolt-kernels/src/optimized/{ram_ra_virtualization.rs,support.rs}`:
//! <https://github.com/a16z/jolt/blob/e21affaeaf0d2c0a4630b958a883a4ac877acbfd/crates/jolt-kernels/src/optimized/support.rs#L144-L288>.
//! Jolt copyright (c) a16z and Microsoft Corporation; its MIT notice is retained
//! in `vendor/open-magnetar/LICENSE-JOLT-MIT`. The coefficient arithmetic here
//! is a GF128 adaptation, not Jolt's prime-field finite-difference formulas.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_sumcheck::{SumcheckProof, sumcheck_init};
use om_transcript::Transcript;

use crate::eq_factor::{
    batch_inverse_one_plus, eval_coefficients, reconstruct_cofactor, scale_low_weights,
};
use crate::lazy::SplitEq;
use crate::sublabel;
use crate::virt_selectors::SelectorState;

#[derive(Clone, Copy)]
enum NodeKind {
    Leaf(usize),
    Product(usize, usize),
}

struct Node {
    kind: NodeKind,
    offset: usize,
    capacity: usize,
}

/// A cached small-point linear transform. Consecutive selected powers reuse
/// the same mul-by-X chain (e.g. t_31 needs four shifts, not 1+2+3+4).
struct SmallPoint {
    shifts: Vec<usize>,
}

impl SmallPoint {
    fn new(index: usize) -> Self {
        let mut shifts = Vec::new();
        let mut previous = 0;
        for bit in 0..usize::BITS as usize {
            if index >> bit & 1 != 0 {
                shifts.push(bit - previous);
                previous = bit;
            }
        }
        Self { shifts }
    }
    #[inline]
    fn multiply(&self, mut value: K) -> K {
        let mut out = K::ZERO;
        for &shifts in &self.shifts {
            for _ in 0..shifts {
                value = value.mul_x();
            }
            out += value;
        }
        out
    }
}

struct Plan {
    nodes: Vec<Node>,
    arena: usize,
    field_work: usize,
    raw_work: usize,
    wire: Vec<SmallPoint>,
    degree: usize,
}

impl Plan {
    fn new(degree: usize) -> Self {
        fn build(start: usize, count: usize, nodes: &mut Vec<Node>, arena: &mut usize) -> usize {
            let kind = if count == 1 {
                NodeKind::Leaf(start)
            } else {
                let half = count / 2;
                let left = build(start, half, nodes, arena);
                let right = build(start + half, count - half, nodes, arena);
                NodeKind::Product(left, right)
            };
            let id = nodes.len();
            nodes.push(Node {
                kind,
                offset: *arena,
                capacity: count + 1,
            });
            *arena += count + 1;
            id
        }
        let mut nodes = Vec::new();
        let mut arena = 0;
        build(0, degree, &mut nodes, &mut arena);
        // Variable-degree pruning can change the balanced lengths seen by
        // Karatsuba. These conservative linear workspaces cover every call.
        Self {
            nodes,
            arena,
            field_work: 8 * (degree + 1),
            raw_work: 16 * (degree + 1),
            wire: (0..degree + 2).map(SmallPoint::new).collect(),
            degree,
        }
    }
    fn evaluations(&self, coefficients: &[K], z: K, prefix: K) -> Vec<K> {
        if prefix == K::ZERO {
            return vec![K::ZERO; self.degree + 2];
        }
        let zero_factor = prefix * (K::ONE + z);
        self.wire
            .iter()
            .map(|point| {
                let h = coefficients
                    .iter()
                    .rev()
                    .fold(K::ZERO, |sum, &c| point.multiply(sum) + c);
                (zero_factor + point.multiply(prefix)) * h
            })
            .collect()
    }
}

fn plan(degree: usize) -> Arc<Plan> {
    static CACHE: OnceLock<RwLock<BTreeMap<usize, Arc<Plan>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(BTreeMap::new()));
    if let Some(plan) = cache
        .read()
        .expect("product plan cache")
        .get(&degree)
        .cloned()
    {
        return plan;
    }
    cache
        .write()
        .expect("product plan cache")
        .entry(degree)
        .or_insert_with(|| Arc::new(Plan::new(degree)))
        .clone()
}

/// Polynomial convolution with unreduced coefficient outputs. Splitting in
/// the coefficient index is ordinary Karatsuba; all additions are XOR, so its
/// identity holds already for the 256-bit raw carryless products.
fn multiply(a: &[K], b: &[K], out: &mut [U256], fields: &mut [K], raw: &mut [U256]) {
    out.fill(U256::ZERO);
    if a.len() == 2 && b.len() == 2 {
        let ends = clmul_pairs([a[0], a[1]], [b[0], b[1]]);
        out[0] = ends[0];
        out[2] = ends[1];
        out[1] = (a[0] + a[1]).clmul(b[0] + b[1]) + ends[0] + ends[1];
        return;
    }
    if a.len().min(b.len()) <= 2 {
        for (i, &a) in a.iter().enumerate() {
            let mut pairs = b.chunks_exact(2);
            for (j, b) in pairs.by_ref().enumerate() {
                let products = clmul_pairs([a; 2], [b[0], b[1]]);
                out[i + 2 * j] += products[0];
                out[i + 2 * j + 1] += products[1];
            }
        }
        // The scalar tail is separate to keep the packed loop's indexing simple.
        if b.len() % 2 != 0 {
            for (i, &a) in a.iter().enumerate() {
                out[i + b.len() - 1] += a.clmul(b[b.len() - 1]);
            }
        }
        return;
    }
    let cut = a.len().max(b.len()) / 2;
    let (a0, a1) = a.split_at(cut.min(a.len()));
    let (b0, b1) = b.split_at(cut.min(b.len()));
    // Empty high halves occur only with a strongly unbalanced trimmed subtree.
    if a1.is_empty() || b1.is_empty() {
        for (i, &a) in a.iter().enumerate() {
            for (j, &b) in b.iter().enumerate() {
                out[i + j] += a.clmul(b);
            }
        }
        return;
    }
    let (sa_len, sb_len) = (a0.len().max(a1.len()), b0.len().max(b1.len()));
    let (sums, work_fields) = fields.split_at_mut(sa_len + sb_len);
    let (sa, sb) = sums.split_at_mut(sa_len);
    for (i, slot) in sa.iter_mut().enumerate() {
        *slot = a0.get(i).copied().unwrap_or(K::ZERO) + a1.get(i).copied().unwrap_or(K::ZERO);
    }
    for (i, slot) in sb.iter_mut().enumerate() {
        *slot = b0.get(i).copied().unwrap_or(K::ZERO) + b1.get(i).copied().unwrap_or(K::ZERO);
    }
    let (l0, l2, l1) = (
        a0.len() + b0.len() - 1,
        a1.len() + b1.len() - 1,
        sa_len + sb_len - 1,
    );
    let (products, work_raw) = raw.split_at_mut(l0 + l1 + l2);
    let (p0, rest) = products.split_at_mut(l0);
    let (p1, p2) = rest.split_at_mut(l1);
    multiply(a0, b0, p0, work_fields, work_raw);
    multiply(a1, b1, p2, work_fields, work_raw);
    multiply(sa, sb, p1, work_fields, work_raw);
    for (i, &p) in p0.iter().enumerate() {
        out[i] += p;
        p1[i] += p;
    }
    for (i, &p) in p2.iter().enumerate() {
        out[2 * cut + i] += p;
        p1[i] += p;
    }
    for (i, &p) in p1.iter().enumerate() {
        if cut + i < out.len() {
            out[cut + i] += p;
        }
    }
}

struct Round {
    eq: SplitEq,
    weights: Vec<K>,
    group_weights: Vec<K>,
    zero_endpoint: bool,
    point: Vec<K>,
}

impl Round {
    fn new(point: &[K], weights: &[K], zero_endpoint: bool) -> Self {
        let eq = SplitEq::new(point);
        let group_weights = weights.to_vec();
        let weights = scale_low_weights(weights, &eq.lo);
        Self {
            eq,
            weights,
            group_weights,
            zero_endpoint,
            point: point.to_vec(),
        }
    }
}

struct Scratch {
    arena: Vec<K>,
    lengths: Vec<usize>,
    coefficients: Vec<U256>,
    fields: Vec<K>,
    raw: Vec<U256>,
    high: Option<usize>,
    block: Vec<U256>,
    total: Vec<U256>,
}

impl Scratch {
    fn new(plan: &Plan) -> Self {
        Self {
            arena: vec![K::ZERO; plan.arena],
            lengths: vec![0; plan.nodes.len()],
            coefficients: vec![U256::ZERO; plan.degree + 1],
            fields: vec![K::ZERO; plan.field_work],
            raw: vec![U256::ZERO; plan.raw_work],
            high: None,
            block: vec![U256::ZERO; plan.degree + 1],
            total: vec![U256::ZERO; plan.degree + 1],
        }
    }
    fn product(&mut self, plan: &Plan, left: &[K], right: &[K], weight: K) -> usize {
        for (id, node) in plan.nodes.iter().enumerate() {
            let len = match node.kind {
                NodeKind::Leaf(i) => {
                    let mut line = [left[i], left[i] + right[i]];
                    if i == 0 {
                        line = mul_pairs([weight; 2], line);
                    }
                    self.arena[node.offset..node.offset + 2].copy_from_slice(&line);
                    if line[1] != K::ZERO {
                        2
                    } else {
                        usize::from(line[0] != K::ZERO)
                    }
                }
                NodeKind::Product(l, r) => {
                    let (ll, rl) = (self.lengths[l], self.lengths[r]);
                    if ll == 0 || rl == 0 {
                        0
                    } else {
                        let len = ll + rl - 1;
                        let a = &self.arena[plan.nodes[l].offset..plan.nodes[l].offset + ll];
                        let b = &self.arena[plan.nodes[r].offset..plan.nodes[r].offset + rl];
                        multiply(
                            a,
                            b,
                            &mut self.coefficients[..len],
                            &mut self.fields,
                            &mut self.raw,
                        );
                        if id + 1 != plan.nodes.len() {
                            for (out, &value) in self.arena[node.offset..node.offset + len]
                                .iter_mut()
                                .zip(&self.coefficients)
                            {
                                *out = value.reduce();
                            }
                        }
                        len
                    }
                }
            };
            debug_assert!(len <= node.capacity);
            self.lengths[id] = len;
        }
        let len = *self.lengths.last().expect("product leaves");
        if plan.degree == 1 {
            let offset = plan.nodes[0].offset;
            for (out, &value) in self.coefficients[..len]
                .iter_mut()
                .zip(&self.arena[offset..offset + len])
            {
                *out = U256::from_k(value);
            }
        }
        len
    }
    fn flush(&mut self, round: &Round) {
        if let Some(high) = self.high.take() {
            let weight = round.eq.hi[high];
            for i in 0..self.block.len() {
                if (i == 0 && !round.zero_endpoint) || (i == 1 && round.zero_endpoint) {
                    continue;
                }
                self.total[i] += weight.clmul(self.block[i].reduce());
                self.block[i] = U256::ZERO;
            }
        }
    }
    fn pair(
        &mut self,
        plan: &Plan,
        round: &Round,
        groups: usize,
        pair: usize,
        left: &[K],
        right: &[K],
    ) {
        let high = pair >> round.eq.h;
        if self.high != Some(high) {
            self.flush(round);
            self.high = Some(high);
        }
        let low = pair & ((1usize << round.eq.h) - 1);
        for group in 0..groups {
            let weight = round.weights[low * groups + group];
            if weight == K::ZERO {
                continue;
            }
            let range = group * plan.degree..(group + 1) * plan.degree;
            let len = self.product(plan, &left[range.clone()], &right[range], weight);
            if len == 0 {
                continue;
            }
            if round.zero_endpoint {
                self.block[0] += self.coefficients[0];
            } else {
                self.block[1] += self.coefficients[..len]
                    .iter()
                    .copied()
                    .fold(U256::ZERO, |sum, c| sum + c);
            }
            for i in 2..len {
                self.block[i] += self.coefficients[i];
            }
        }
    }
    fn merge(mut self, mut other: Self, round: &Round) -> Self {
        self.flush(round);
        other.flush(round);
        for (a, b) in self.total.iter_mut().zip(other.total) {
            *a += b;
        }
        self
    }
    fn finish(mut self, round: &Round) -> Vec<K> {
        self.flush(round);
        self.total.into_iter().map(U256::reduce).collect()
    }
}

fn statistics(
    plan: &Plan,
    selectors: &mut SelectorState<'_>,
    round: &Round,
    groups: usize,
    bind: Option<K>,
) -> Vec<K> {
    let init = || Scratch::new(plan);
    let visit = |scratch: &mut Scratch, pair: usize, left: &[K], right: &[K]| {
        scratch.pair(plan, round, groups, pair, left, right)
    };
    let merge = |a: Scratch, b: Scratch| a.merge(b, round);
    let scratch = if let Some(r) = bind {
        selectors.bind_and_visit_pairs(r, init, visit, merge)
    } else {
        selectors.visit_pairs(init, visit, merge)
    };
    let mut stats = scratch.finish(round);
    let pairs = selectors.len() / 2;
    let prefix_pairs = selectors.prefix_len().div_ceil(2).min(pairs);
    if prefix_pairs < pairs {
        if let Some(tail) = selectors.tail_values() {
            let value = (0..groups).fold(K::ZERO, |sum, g| {
                sum + tail[g * plan.degree..(g + 1) * plan.degree]
                    .iter()
                    .copied()
                    .fold(round.group_weights[g], |product, value| product * value)
            });
            let mass = K::ONE + eq_prefix_mass(&round.point, prefix_pairs);
            stats[usize::from(!round.zero_endpoint)] += value * mass;
        }
    }
    stats
}

/// Sum Eq(point,j) for j<end, using high-bit prefix decisions; lower-bit
/// suffix masses equal one. This works also at Boolean field points.
fn eq_prefix_mass(point: &[K], end: usize) -> K {
    if end == 0 {
        return K::ZERO;
    }
    if end >= 1usize << point.len() {
        return K::ONE;
    }
    let (mut mass, mut prefix) = (K::ZERO, K::ONE);
    for bit in (0..point.len()).rev() {
        if end >> bit & 1 != 0 {
            mass += prefix * (K::ONE + point[bit]);
            prefix *= point[bit];
        } else {
            prefix *= K::ONE + point[bit];
        }
    }
    mass
}

/// Product columns are group-major. `initial_claim` comes from this prover's
/// preceding claim-reduction terminal evaluations, never an unchecked user sum.
pub(crate) fn prove(
    point: &[K],
    selectors: &mut SelectorState<'_>,
    weights: &[K],
    d: usize,
    initial_claim: K,
    tr: &mut Transcript,
    label: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<K>) {
    prove_sampled(
        point,
        selectors,
        weights,
        d,
        initial_claim,
        tr,
        label,
        |tr, label| tr.challenge_k(label),
    )
}

fn prove_sampled(
    point: &[K],
    selectors: &mut SelectorState<'_>,
    weights: &[K],
    d: usize,
    initial_claim: K,
    tr: &mut Transcript,
    label: &[u8],
    mut sample: impl FnMut(&mut Transcript, &[u8]) -> K,
) -> (SumcheckProof, Vec<K>, Vec<K>) {
    assert!(
        d > 0 && point.len() < usize::BITS as usize,
        "virtual product dimensions"
    );
    assert_eq!(
        selectors.columns(),
        weights.len().checked_mul(d).expect("product width")
    );
    assert_eq!(selectors.len(), 1usize << point.len());
    let n = point.len();
    sumcheck_init(tr, label, initial_claim, n, d + 1);
    if n == 0 {
        let values = selectors.final_values();
        let actual = weights.iter().enumerate().fold(K::ZERO, |sum, (g, &w)| {
            sum + values[g * d..(g + 1) * d]
                .iter()
                .copied()
                .fold(w, |product, value| product * value)
        });
        assert_eq!(actual, initial_claim, "product initial claim");
        return (
            SumcheckProof {
                round_msgs: Vec::new(),
            },
            Vec::new(),
            values,
        );
    }
    let plan = plan(d);
    let inverses = batch_inverse_one_plus(point);
    let first = Round::new(&point[1..], weights, point[0] == K::ONE);
    let stats = statistics(&plan, selectors, &first, weights.len(), None);
    let mut coefficients = reconstruct_cofactor(&stats, point[0], Some(initial_claim), inverses[0]);
    let (msg_label, chal_label) = (sublabel(label, b"/msg"), sublabel(label, b"/chal"));
    let mut prefix = K::ONE;
    let mut current = initial_claim;
    let mut messages = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    for i in 0..n {
        let evaluations = plan.evaluations(&coefficients, point[i], prefix);
        debug_assert_eq!(evaluations[0] + evaluations[1], current);
        let mut message = vec![evaluations[0]];
        message.extend_from_slice(&evaluations[2..]);
        tr.absorb_ks(&msg_label, &message);
        let r = sample(tr, &chal_label);
        messages.push(message);
        challenges.push(r);
        let normalized = eval_coefficients(&coefficients, r);
        prefix *= K::ONE + point[i] + r;
        current = prefix * normalized;
        if i + 1 < n && prefix != K::ZERO {
            let round = Round::new(&point[i + 2..], weights, point[i + 1] == K::ONE);
            let stats = statistics(&plan, selectors, &round, weights.len(), Some(r));
            coefficients =
                reconstruct_cofactor(&stats, point[i + 1], Some(normalized), inverses[i + 1]);
        } else {
            selectors.bind(r);
            coefficients.fill(K::ZERO);
        }
    }
    (
        SumcheckProof {
            round_msgs: messages,
        },
        challenges,
        selectors.final_values(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::AddrComps;
    use crate::virt_selectors::SelectorColumn;
    use om_sumcheck::{
        combine_sum, eq_eval, eq_table, interpolate_eval, mle::bind_low_in_place, mle_eval,
        sumcheck_prove, sumcheck_round_evals, sumcheck_verify,
    };

    fn random(seed: &mut u64) -> K {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        K::from_u128((*seed as u128) | ((seed.rotate_left(29) as u128) << 64))
    }

    #[test]
    fn raw_coefficient_karatsuba_matches_schoolbook_in_characteristic_two() {
        let mut seed = 0x452821e638d01377;
        for la in 1..=33 {
            for lb in 1..=33 {
                let a: Vec<K> = (0..la).map(|_| random(&mut seed)).collect();
                let b: Vec<K> = (0..lb).map(|_| random(&mut seed)).collect();
                let mut expected = vec![U256::ZERO; la + lb - 1];
                for (i, &a) in a.iter().enumerate() {
                    for (j, &b) in b.iter().enumerate() {
                        expected[i + j] += a.clmul(b);
                    }
                }
                let mut actual = vec![U256::ZERO; expected.len()];
                multiply(
                    &a,
                    &b,
                    &mut actual,
                    &mut vec![K::ZERO; 8 * (la + lb)],
                    &mut vec![U256::ZERO; 16 * (la + lb)],
                );
                assert_eq!(actual, expected, "shape {la}x{lb}");
            }
        }
    }

    #[test]
    fn product_tree_reuses_coefficients_and_prunes_constant_zero_lines() {
        let mut seed = 0x082efa98ec4e6c89;
        for d in [1, 2, 3, 4, 8, 16, 32] {
            let p = plan(d);
            for pattern in 0..4 {
                let mut left: Vec<K> = (0..d).map(|_| random(&mut seed)).collect();
                let mut right: Vec<K> = (0..d).map(|_| random(&mut seed)).collect();
                if pattern == 1 {
                    right.clone_from(&left);
                }
                if pattern == 2 {
                    left[d / 2] = K::ZERO;
                    right[d / 2] = K::ZERO;
                }
                if pattern == 3 {
                    for i in (0..d).step_by(2) {
                        right[i] = left[i];
                    }
                }
                let weight = random(&mut seed);
                let mut scratch = Scratch::new(&p);
                let len = scratch.product(&p, &left, &right, weight);
                let coefficients: Vec<K> = scratch.coefficients[..len]
                    .iter()
                    .map(|&u| u.reduce())
                    .collect();
                for t in [K::ZERO, K::ONE, K::t(2), K::t(31), random(&mut seed)] {
                    let expected = (0..d).fold(weight, |product, i| {
                        product * (left[i] + t * (left[i] + right[i]))
                    });
                    assert_eq!(
                        eval_coefficients(&coefficients, t),
                        expected,
                        "d={d} pattern={pattern}"
                    );
                }
            }
        }
    }

    #[test]
    fn cached_small_point_and_prefix_mass_are_exact() {
        let mut seed = 0x9216d5d98979fb1b;
        for i in 0..130 {
            let point = SmallPoint::new(i);
            for _ in 0..3 {
                let value = random(&mut seed);
                assert_eq!(point.multiply(value), value * K::t(i as u128));
            }
        }
        for n in 0..=7 {
            for boolean in [false, true] {
                let point: Vec<K> = (0..n)
                    .map(|i| {
                        if boolean {
                            K::from_bit(i % 2 == 0)
                        } else {
                            random(&mut seed)
                        }
                    })
                    .collect();
                let eq = eq_table(&point);
                for end in 0..=eq.len() {
                    assert_eq!(
                        eq_prefix_mass(&point, end),
                        eq[..end].iter().copied().fold(K::ZERO, |sum, v| sum + v)
                    );
                }
            }
        }
    }

    fn compare(
        d: usize,
        n: usize,
        groups: usize,
        pattern: usize,
        bits: bool,
        boolean_point: bool,
        zero_weight: bool,
        forced: Option<Vec<K>>,
    ) {
        let ac = if d <= 4 { 4 } else { 1 };
        let a = d * ac;
        let cycles = 1usize << n;
        let mask = if a == 32 { u32::MAX } else { (1u32 << a) - 1 };
        let mut seed = 0x3f84d5b5b5470917;
        let comps: Vec<AddrComps> = (0..groups)
            .map(|g| {
                let addresses: Vec<u32> = (0..cycles)
                    .map(|j| match pattern {
                        0 => 3 & mask,
                        1 => (j as u32 * 13 + 17 * g as u32 + 1) & mask,
                        _ if j >= cycles / 2 + usize::from(cycles > 1) => 0,
                        _ => ((random(&mut seed).value() as u32) & mask).max(1),
                    })
                    .collect();
                let mut comp = AddrComps::from_addrs(addresses, a, d);
                if bits {
                    for table in comp.bits_mut() {
                        table[0] = !table[0];
                    }
                }
                comp
            })
            .collect();
        let points: Vec<Vec<K>> = (0..groups)
            .map(|_| (0..a).map(|_| random(&mut seed)).collect())
            .collect();
        let columns: Vec<SelectorColumn<'_>> = comps
            .iter()
            .enumerate()
            .flat_map(|(g, comp)| {
                let points = &points[g];
                (0..d).map(move |i| SelectorColumn {
                    component: comp,
                    chunk: i,
                    point: &points[i * ac..(i + 1) * ac],
                })
            })
            .collect();
        let mut tables = Vec::new();
        for c in &columns {
            let eq = eq_table(c.point);
            tables.push(
                (0..cycles)
                    .map(|j| c.component.row_eval(c.chunk, j, &eq))
                    .collect::<Vec<K>>(),
            );
        }
        let weights: Vec<K> = (0..groups)
            .map(|g| {
                if zero_weight || (pattern == 2 && g == 0) {
                    K::ZERO
                } else if g == 0 {
                    K::ONE
                } else {
                    random(&mut seed)
                }
            })
            .collect();
        let point: Vec<K> = (0..n)
            .map(|i| {
                if boolean_point {
                    K::from_bit(i % 2 == 0)
                } else {
                    random(&mut seed)
                }
            })
            .collect();
        let mut dense = vec![eq_table(&point)];
        dense.extend(tables.clone());
        let combine = |values: &[K]| {
            values[0]
                * (0..groups).fold(K::ZERO, |sum, g| {
                    sum + values[1 + g * d..1 + (g + 1) * d]
                        .iter()
                        .copied()
                        .fold(weights[g], |prod, v| prod * v)
                })
        };
        let claim = combine_sum(&dense, &combine);
        let mut reference_tr = Transcript::new_default(b"flat-product-oracle");
        let expected = if let Some(forced) = &forced {
            assert_eq!(forced.len(), n);
            sumcheck_init(&mut reference_tr, b"sc", claim, n, d + 1);
            let mut current = claim;
            let mut messages = Vec::new();
            for &r in forced {
                let ev = sumcheck_round_evals(&dense, &combine, d + 1);
                assert_eq!(ev[0] + ev[1], current);
                let mut message = vec![ev[0]];
                message.extend_from_slice(&ev[2..]);
                reference_tr.absorb_ks(b"sc/msg", &message);
                reference_tr.challenge_k(b"sc/chal");
                messages.push(message);
                current = interpolate_eval(&ev, r);
                for table in &mut dense {
                    bind_low_in_place(table, r);
                }
            }
            (
                SumcheckProof {
                    round_msgs: messages,
                },
                forced.clone(),
                dense.iter().map(|t| t[0]).collect::<Vec<K>>(),
            )
        } else {
            sumcheck_prove(claim, dense, &combine, d + 1, &mut reference_tr, b"sc")
        };
        let mut actual_tr = Transcript::new_default(b"flat-product-oracle");
        let mut selectors = SelectorState::new(n, columns);
        let actual = if let Some(forced) = &forced {
            let mut pos = 0;
            prove_sampled(
                &point,
                &mut selectors,
                &weights,
                d,
                claim,
                &mut actual_tr,
                b"sc",
                |tr, label| {
                    tr.challenge_k(label);
                    let r = forced[pos];
                    pos += 1;
                    r
                },
            )
        } else {
            prove(
                &point,
                &mut selectors,
                &weights,
                d,
                claim,
                &mut actual_tr,
                b"sc",
            )
        };
        assert_eq!(
            actual.0, expected.0,
            "messages d={d} n={n} pattern={pattern} bits={bits}"
        );
        assert_eq!(actual.1, expected.1);
        assert_eq!(actual.2, expected.2[1..]);
        assert_eq!(
            actual_tr.challenge_k(b"after"),
            reference_tr.challenge_k(b"after")
        );
        for (column, &value) in tables.iter().zip(&actual.2) {
            assert_eq!(mle_eval(column, &actual.1), value);
        }
        if forced.is_none() {
            let mut verify_tr = Transcript::new_default(b"flat-product-oracle");
            let (s, expected) =
                sumcheck_verify(claim, n, d + 1, &actual.0, &mut verify_tr, b"sc").unwrap();
            let value = (0..groups).fold(K::ZERO, |sum, g| {
                sum + actual.2[g * d..(g + 1) * d]
                    .iter()
                    .copied()
                    .fold(weights[g], |prod, v| prod * v)
            });
            assert_eq!(expected, eq_eval(&point, &s) * value);
        }
    }

    #[test]
    fn flat_product_matches_dense_messages_for_shapes_bits_and_lazy_tails() {
        for d in [2, 3, 4, 8, 16, 32] {
            for n in [0, 1, 5] {
                for pattern in 0..3 {
                    compare(d, n, 2, pattern, false, false, false, None);
                    compare(d, n, 2, pattern, true, true, false, None);
                }
                compare(d, n, 2, 2, false, true, true, None);
            }
        }
        for d in [1, 3, 16] {
            for n in [0, 1, 5] {
                compare(d, n, 0, 0, false, true, false, None);
            }
        }
    }

    #[test]
    fn flat_product_handles_zero_one_and_annihilated_equality_challenges() {
        for d in [2, 3, 4, 8, 16, 32] {
            for r in [K::ZERO, K::ONE, K::t(2), K::t(4)] {
                // Boolean z_0=1 and r_0=0 deliberately annihilate Eq prefix.
                compare(d, 5, 2, 2, false, true, false, Some(vec![r; 5]));
                compare(d, 1, 2, 1, true, true, false, Some(vec![r]));
            }
        }
    }

    #[test]
    fn parallel_flat_product_matches_dense_with_lazy_dense_transition() {
        compare(8, 14, 2, 1, false, false, false, None);
        compare(4, 14, 2, 2, true, false, false, None);
    }

    #[test]
    fn product_is_formed_before_off_cube_cycle_evaluation() {
        let a = [K::ONE + K::t(2), K::t(2)];
        let b = [K::ONE + K::t(3), K::t(3)];
        let product = [a[0] * b[0], a[1] * b[1]];
        let z = [K::t(7)];
        assert_ne!(mle_eval(&product, &z), mle_eval(&a, &z) * mle_eval(&b, &z));
    }
}
