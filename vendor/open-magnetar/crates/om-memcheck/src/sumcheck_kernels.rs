//! Prover-only round kernels. Their outputs use the original finite-point wire format.

use om_field::{K, U256};
use rayon::prelude::*;

use crate::lazy::RowFn;
use crate::onehot::PAR_THRESHOLD;

#[derive(Clone, Copy)]
pub(crate) enum Relation<'a> {
    #[cfg(test)]
    Generic {
        combine: &'a (dyn Fn(&[K]) -> K + Sync),
        degree: usize,
    },
    Mul2,
    #[cfg(test)]
    OneHot {
        weights: &'a [K],
        c: K,
    },
    Products {
        groups: usize,
        components: usize,
    },
    #[cfg(test)]
    WeightedProducts {
        weights: &'a [K],
    },
    /// Address tail of Twist RW: eq and Inc are already scalars. The
    /// polynomial is quadratic, while its original wire declares degree 3.
    RwAddress {
        weights: &'a [K],
        inc: K,
    },
}

impl Relation<'_> {
    pub(crate) fn degree(self) -> usize {
        match self {
            #[cfg(test)]
            Self::Generic { degree, .. } => degree,
            Self::Mul2 => 2,
            #[cfg(test)]
            Self::OneHot { .. } => 4,
            #[cfg(test)]
            Self::WeightedProducts { .. } => 3,
            Self::RwAddress { .. } => 3,
            Self::Products { components, .. } => components + 1,
        }
    }

    pub(crate) fn value(self, values: &[K]) -> K {
        match self {
            #[cfg(test)]
            Self::Generic { combine, .. } => combine(values),
            Self::Mul2 => values[0] * values[1],
            #[cfg(test)]
            Self::OneHot { weights, c } => {
                let mut sum = K::ZERO;
                for (i, &weight) in weights.iter().enumerate() {
                    let a = values[1 + 2 * i];
                    sum += weight * (a.square() * a + c * values[2 + 2 * i]);
                }
                values[0] * sum
            }
            Self::Products { groups, components } => {
                let mut sum = K::ZERO;
                for group in 0..groups {
                    let mut product = values[group];
                    for component in 0..components {
                        product *= values[groups + group * components + component];
                    }
                    sum += product;
                }
                sum
            }
            #[cfg(test)]
            Self::WeightedProducts { weights } => {
                let mut sum = U256::ZERO;
                for (node, &weight) in weights.iter().enumerate() {
                    sum += weight.clmul(values[1 + 2 * node] * values[2 + 2 * node]);
                }
                values[0] * sum.reduce()
            }
            Self::RwAddress { weights, inc } => {
                let np = weights.len();
                let mut sum = U256::ZERO;
                for (x, &weight) in weights.iter().enumerate() {
                    let other =
                        values[0] + values[1 + np + x] + if x + 1 == np { inc } else { K::ZERO };
                    sum += (weight * values[1 + x]).clmul(other);
                }
                sum.reduce()
            }
        }
    }

    /// Accumulate one pair. OneHot stores polynomial coefficients; Mul2 stores
    /// g(0), optional g(1), and its quadratic coefficient. Other modes store
    /// finite-point evaluations. Reduction is deferred across pairs.
    #[inline]
    fn accumulate(
        self,
        left: &[K],
        right: &[K],
        acc: &mut [U256],
        _scratch: &mut [K],
        known_claim: bool,
    ) {
        match self {
            #[cfg(test)]
            Self::OneHot { weights, c } => {
                let mut core = [U256::ZERO; 4];
                for (i, &weight) in weights.iter().enumerate() {
                    let a = left[1 + 2 * i];
                    let delta = a + right[1 + 2 * i];
                    let b = left[2 + 2 * i];
                    let b_delta = b + right[2 + 2 * i];
                    let (square, delta_square) = (a.square(), delta.square());
                    core[0] += weight.clmul(square * a + c * b);
                    core[1] += weight.clmul(square * delta + c * b_delta);
                    core[2] += weight.clmul(a * delta_square);
                    core[3] += weight.clmul(delta_square * delta);
                }
                let [f0, f1, f2, f3] = core.map(U256::reduce);
                let (e, delta) = (left[0], left[0] + right[0]);
                acc[0] += e.clmul(f0);
                if !known_claim {
                    acc[1] += e.clmul(f1) + delta.clmul(f0);
                }
                acc[2] += e.clmul(f2) + delta.clmul(f1);
                acc[3] += e.clmul(f3) + delta.clmul(f2);
                acc[4] += delta.clmul(f3);
            }
            Self::Mul2 => {
                acc[0] += left[0].clmul(left[1]);
                if !known_claim {
                    acc[1] += right[0].clmul(right[1]);
                }
                acc[2] += (left[0] + right[0]).clmul(left[1] + right[1]);
            }
            Self::Products {
                groups,
                components: 2,
            } => {
                // For a cubic, g(X+1)+g(X)=claim+c3*(X²+X).
                // Accumulate g(0), g(2), c3 and recover g(3) globally.
                // This replaces the second finite-point extrapolation with
                // endpoint differences and retains the original wire values.
                for group in 0..groups {
                    let cols = [group, groups + 2 * group, groups + 2 * group + 1];
                    let a = cols.map(|i| left[i]);
                    let delta = cols.map(|i| left[i] + right[i]);
                    let at_two = core::array::from_fn::<_, 3, _>(|i| a[i] + delta[i].mul_x());
                    let low = om_field::mul_pairs([a[0], at_two[0]], [a[1], at_two[1]]);
                    let products = om_field::clmul_pairs(low, [a[2], at_two[2]]);
                    acc[0] += products[0];
                    acc[2] += products[1];
                    if !known_claim {
                        let high = om_field::mul_pairs(
                            [right[cols[0]], delta[0]],
                            [right[cols[1]], delta[1]],
                        );
                        let products = om_field::clmul_pairs(high, [right[cols[2]], delta[2]]);
                        acc[1] += products[0];
                        acc[3] += products[1];
                    } else {
                        acc[3] += (delta[0] * delta[1]).clmul(delta[2]);
                    }
                }
            }
            Self::Products { groups, components } => {
                // Independent finite-point evaluations share packed CLMUL
                // instructions. Their individual values and wire order remain
                // unchanged; the known endpoint is never scanned again.
                let at = |column: usize, point: usize| match point {
                    0 => left[column],
                    1 => right[column],
                    _ => left[column] + (left[column] + right[column]).mul_t(point as u128),
                };
                let mut point = 0;
                while point < acc.len() {
                    if known_claim && point == 1 {
                        point += 1;
                        continue;
                    }
                    let mut next = point + 1;
                    if known_claim && next == 1 {
                        next += 1;
                    }
                    if next < acc.len() {
                        for group in 0..groups {
                            let mut products = [at(group, point), at(group, next)];
                            for component in 0..components - 1 {
                                let column = groups + group * components + component;
                                products = om_field::mul_pairs(
                                    products,
                                    [at(column, point), at(column, next)],
                                );
                            }
                            let last = groups + group * components + components - 1;
                            let products =
                                om_field::clmul_pairs(products, [at(last, point), at(last, next)]);
                            acc[point] += products[0];
                            acc[next] += products[1];
                        }
                    } else {
                        for group in 0..groups {
                            let mut product = at(group, point);
                            for component in 0..components - 1 {
                                product *= at(groups + group * components + component, point);
                            }
                            acc[point] += product
                                .clmul(at(groups + group * components + components - 1, point));
                        }
                    }
                    point = next + 1;
                }
            }
            #[cfg(test)]
            Self::WeightedProducts { weights } => {
                let mut core = [U256::ZERO; 3];
                for (node, &weight) in weights.iter().enumerate() {
                    let (l0, l1) = (left[1 + 2 * node], right[1 + 2 * node]);
                    let (r0, r1) = (left[2 + 2 * node], right[2 + 2 * node]);
                    let p0 = l0 * r0;
                    let p1 = l1 * r1;
                    let p2 = (l0 + l1) * (r0 + r1);
                    core[0] += weight.clmul(p0);
                    core[1] += weight.clmul(p0 + p1 + p2);
                    core[2] += weight.clmul(p2);
                }
                let [f0, f1, f2] = core.map(U256::reduce);
                let (e, delta) = (left[0], left[0] + right[0]);
                acc[0] += e.clmul(f0);
                if !known_claim {
                    acc[1] += e.clmul(f1) + delta.clmul(f0);
                }
                acc[2] += e.clmul(f2) + delta.clmul(f1);
                acc[3] += delta.clmul(f2);
            }
            Self::RwAddress { weights, inc } => {
                let np = weights.len();
                for (x, &weight) in weights.iter().enumerate() {
                    let [a0, a1] = om_field::mul_pairs([weight; 2], [left[1 + x], right[1 + x]]);
                    let b0 = left[0] + left[1 + np + x] + if x + 1 == np { inc } else { K::ZERO };
                    let b1 = right[0] + right[1 + np + x] + if x + 1 == np { inc } else { K::ZERO };
                    let products = om_field::clmul_pairs([a0, a0 + a1], [b0, b0 + b1]);
                    acc[0] += products[0];
                    acc[2] += products[1];
                    if !known_claim {
                        acc[1] += a1.clmul(b1);
                    }
                }
            }
            #[cfg(test)]
            Self::Generic { combine, .. } => {
                for (point, out) in acc.iter_mut().enumerate() {
                    if known_claim && point == 1 {
                        continue;
                    }
                    let value = match point {
                        0 => combine(left),
                        1 => combine(right),
                        _ => {
                            for ((out, &a), &b) in _scratch.iter_mut().zip(left).zip(right) {
                                *out = a + (a + b).mul_t(point as u128);
                            }
                            combine(_scratch)
                        }
                    };
                    *out += U256::from_k(value);
                }
            }
        }
    }

    fn finish(self, accumulated: Vec<U256>, claim: Option<K>) -> Vec<K> {
        let mut values: Vec<K> = accumulated.into_iter().map(U256::reduce).collect();
        match self {
            Self::Products { components: 2, .. } => {
                let sum = claim.unwrap_or(values[0] + values[1]);
                values[1] = sum + values[0];
                values[3] = values[2] + sum + values[3].mul_t(6);
                values
            }
            Self::RwAddress { .. } => {
                let constant = values[0];
                let quadratic = values[2];
                let sum = claim.unwrap_or(constant + values[1]);
                let linear = sum + quadratic;
                vec![
                    constant,
                    constant + sum,
                    constant + linear.mul_t(2) + quadratic.mul_t(4),
                    constant + linear.mul_t(3) + quadratic.mul_t(5),
                ]
            }
            #[cfg(test)]
            Self::OneHot { .. } | Self::WeightedProducts { .. } => {
                if let Some(claim) = claim {
                    values[1] = values[2..]
                        .iter()
                        .copied()
                        .fold(claim, |sum, value| sum + value);
                }
                let coefficients = values;
                (0..coefficients.len())
                    .map(|point| {
                        coefficients
                            .iter()
                            .rev()
                            .fold(K::ZERO, |value, &coefficient| {
                                value.mul_t(point as u128) + coefficient
                            })
                    })
                    .collect()
            }
            Self::Mul2 => {
                let s0 = values[0];
                let s1 = claim.map_or(values[1], |claim| claim + s0);
                let delta = values[2];
                vec![
                    s0,
                    s1,
                    s0 + (s0 + s1 + delta).mul_x() + delta.mul_x().mul_x(),
                ]
            }
            _ => {
                if let Some(claim) = claim {
                    values[1] = claim + values[0];
                }
                values
            }
        }
    }
}

fn add_acc(mut left: Vec<U256>, right: Vec<U256>) -> Vec<U256> {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

/// Initial message from the original row oracle, without full-length tables.
pub(crate) fn first_round(
    row: &RowFn<'_>,
    width: usize,
    n: usize,
    relation: Relation<'_>,
) -> Vec<K> {
    let half = 1usize << (n - 1);
    let fresh = || {
        (
            vec![U256::ZERO; relation.degree() + 1],
            vec![K::ZERO; width],
            vec![K::ZERO; width],
            vec![K::ZERO; width],
        )
    };
    let pair =
        |(mut acc, mut left, mut right, mut scratch): (Vec<U256>, Vec<K>, Vec<K>, Vec<K>),
         index: usize| {
            row(2 * index, &mut left);
            row(2 * index + 1, &mut right);
            relation.accumulate(&left, &right, &mut acc, &mut scratch, false);
            (acc, left, right, scratch)
        };
    let accumulated = if 1usize << n >= PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .with_min_len(1024)
            .fold(fresh, pair)
            .map(|(acc, ..)| acc)
            .reduce(|| vec![U256::ZERO; relation.degree() + 1], add_acc)
    } else {
        (0..half).fold(fresh(), pair).0
    };
    relation.finish(accumulated, None)
}

/// A round on existing dense tables, with the running claim supplied by the
/// preceding sparse phase. The table callback is statically dispatched.
pub(crate) fn dense_round(tables: &[Vec<K>], relation: Relation<'_>, claim: K) -> Vec<K> {
    let half = tables[0].len() / 2;
    let width = tables.len();
    let fresh = || {
        (
            vec![U256::ZERO; relation.degree() + 1],
            vec![K::ZERO; width],
            vec![K::ZERO; width],
        )
    };
    let pair = |(mut acc, mut left, mut right): (Vec<U256>, Vec<K>, Vec<K>), index: usize| {
        for ((l, r), column) in left.iter_mut().zip(&mut right).zip(tables) {
            *l = column[2 * index];
            *r = column[2 * index + 1];
        }
        relation.accumulate(&left, &right, &mut acc, &mut [], true);
        (acc, left, right)
    };
    let accumulated = if 2 * half >= PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .with_min_len(1024)
            .fold(fresh, pair)
            .map(|(acc, ..)| acc)
            .reduce(|| vec![U256::ZERO; relation.degree() + 1], add_acc)
    } else {
        (0..half).fold(fresh(), pair).0
    };
    relation.finish(accumulated, Some(claim))
}

fn prepare(bound: &mut Vec<Vec<K>>, width: usize, size: usize) {
    bound.resize_with(width, Vec::new);
    for column in bound {
        if column.len() > size {
            column.truncate(size);
        } else if column.len() < size && size >= PAR_THRESHOLD / 2 {
            // Initialize fresh pages on the workers that will fill the table,
            // rather than serially touching the whole destination first.
            (0..size)
                .into_par_iter()
                .map(|_| K::ZERO)
                .collect_into_vec(column);
        } else {
            column.resize(size, K::ZERO);
        }
    }
}

/// Fill output row pairs and emit their round message while they are resident.
/// `fill` binds each source row pair into a destination row; it can read either
/// the source oracle (first binding) or the current dense tables (later rounds).
fn fill_and_round<F: Fn(usize, &mut [K], &mut [K]) + Sync>(
    bound: &mut Vec<Vec<K>>,
    width: usize,
    size: usize,
    parallel: bool,
    relation: Relation<'_>,
    claim: K,
    fill: &F,
) -> Vec<K> {
    prepare(bound, width, size);
    if size == 1 {
        let (mut values, mut scratch) = (vec![K::ZERO; width], vec![K::ZERO; width]);
        fill(0, &mut values, &mut scratch);
        for (column, value) in bound.iter_mut().zip(values) {
            column[0] = value;
        }
        return Vec::new();
    }
    const CHUNK: usize = 1024;
    let mut pieces: Vec<(usize, Vec<&mut [K]>)> = (0..size.div_ceil(CHUNK))
        .map(|chunk| (chunk * CHUNK, Vec::with_capacity(width)))
        .collect();
    for column in bound {
        for (piece, values) in pieces.iter_mut().zip(column.chunks_mut(CHUNK)) {
            piece.1.push(values);
        }
    }
    type Worker = (Vec<U256>, Vec<K>, Vec<K>, Vec<K>);
    let fresh = || {
        (
            vec![U256::ZERO; relation.degree() + 1],
            vec![K::ZERO; width],
            vec![K::ZERO; width],
            vec![K::ZERO; width],
        )
    };
    let process = |(mut acc, mut left, mut right, mut scratch): Worker,
                   (base, mut columns): (usize, Vec<&mut [K]>)| {
        for pair in 0..columns[0].len() / 2 {
            fill(base + 2 * pair, &mut left, &mut scratch);
            fill(base + 2 * pair + 1, &mut right, &mut scratch);
            for (i, column) in columns.iter_mut().enumerate() {
                column[2 * pair] = left[i];
                column[2 * pair + 1] = right[i];
            }
            relation.accumulate(&left, &right, &mut acc, &mut scratch, true);
        }
        (acc, left, right, scratch)
    };
    let accumulated = if parallel {
        pieces
            .into_par_iter()
            .fold(fresh, process)
            .map(|(acc, ..)| acc)
            .reduce(|| vec![U256::ZERO; relation.degree() + 1], add_acc)
    } else {
        pieces.into_iter().fold(fresh(), process).0
    };
    relation.finish(accumulated, Some(claim))
}

/// Preserve the original memory-lean strategy: regenerate original rows once
/// after the first challenge, but fuse materialization with the second message.
pub(crate) fn materialize_and_round(
    row: &RowFn<'_>,
    width: usize,
    n: usize,
    challenge: K,
    relation: Relation<'_>,
    claim: K,
) -> (Vec<Vec<K>>, Vec<K>) {
    let size = 1usize << (n - 1);
    let mut tables = Vec::new();
    let fill = |index: usize, out: &mut [K], scratch: &mut [K]| {
        row(2 * index, out);
        row(2 * index + 1, scratch);
        om_field::fold_slices(out, scratch, challenge);
    };
    let next = fill_and_round(
        &mut tables,
        width,
        size,
        1usize << n >= PAR_THRESHOLD,
        relation,
        claim,
        &fill,
    );
    (tables, next)
}

pub(crate) fn bind_and_round_into(
    tables: &[Vec<K>],
    bound: &mut Vec<Vec<K>>,
    challenge: K,
    relation: Relation<'_>,
    claim: K,
) -> Vec<K> {
    bind_and_round_with_affine_into(tables, bound, challenge, relation, claim, &[])
}

/// The final columns may be transparent affine polynomials. Their sibling
/// differences are constants, so the caller supplies challenge*delta once
/// per column/round, rather than multiplying that same delta in every row.
pub(crate) fn bind_and_round_with_affine_into(
    tables: &[Vec<K>],
    bound: &mut Vec<Vec<K>>,
    challenge: K,
    relation: Relation<'_>,
    claim: K,
    affine_offsets: &[K],
) -> Vec<K> {
    let size = tables[0].len() / 2;
    let non_affine = tables.len() - affine_offsets.len();
    let fill = |index: usize, out: &mut [K], scratch: &mut [K]| {
        for ((out, other), column) in out[..non_affine]
            .iter_mut()
            .zip(scratch.iter_mut())
            .zip(tables)
        {
            *out = column[2 * index];
            *other = column[2 * index + 1];
        }
        om_field::fold_slices(&mut out[..non_affine], &scratch[..non_affine], challenge);
        for ((out, column), &offset) in out[non_affine..]
            .iter_mut()
            .zip(&tables[non_affine..])
            .zip(affine_offsets)
        {
            *out = column[2 * index] + offset;
        }
    };
    fill_and_round(
        bound,
        tables.len(),
        size,
        tables[0].len() >= PAR_THRESHOLD,
        relation,
        claim,
        &fill,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{
        sumcheck_prove_rows, sumcheck_prove_rows_mul2, sumcheck_prove_rows_onehot,
        sumcheck_prove_rows_products, sumcheck_prove_rows_weighted_products,
    };
    use om_transcript::Transcript;

    fn compare(n: usize, width: usize, relation: Relation<'_>) {
        let mut seed = 0x243f6a8885a308d3u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            K::from_u128((seed as u128) | ((seed.rotate_left(19) as u128) << 64))
        };
        let tables: Vec<Vec<K>> = (0..width)
            .map(|_| (0..1usize << n).map(|_| random()).collect())
            .collect();
        let row = |index: usize, out: &mut [K]| {
            for (out, column) in out.iter_mut().zip(&tables) {
                *out = column[index];
            }
        };
        let combine = |values: &[K]| relation.value(values);
        let claim = om_sumcheck::combine_sum(&tables, &combine);
        let mut reference_tr = Transcript::new_default(b"kernel-reference");
        let expected = om_sumcheck::sumcheck_prove(
            claim,
            tables.clone(),
            &combine,
            relation.degree(),
            &mut reference_tr,
            b"sc",
        );
        let mut actual_tr = Transcript::new_default(b"kernel-reference");
        let actual = match relation {
            Relation::OneHot { weights, c } => {
                sumcheck_prove_rows_onehot(n, &row, weights, c, &mut actual_tr, b"sc")
            }
            Relation::Products { groups, components } => {
                sumcheck_prove_rows_products(n, &row, groups, components, &mut actual_tr, b"sc")
            }
            Relation::WeightedProducts { weights } => {
                sumcheck_prove_rows_weighted_products(n, &row, weights, &mut actual_tr, b"sc")
            }
            Relation::Mul2 => sumcheck_prove_rows_mul2(n, &row, &mut actual_tr, b"sc"),
            Relation::RwAddress { .. } => {
                crate::lazy::prove_rows(n, width, &row, relation, &mut actual_tr, b"sc")
            }
            Relation::Generic { combine, degree } => {
                sumcheck_prove_rows(n, width, &row, combine, degree, &mut actual_tr, b"sc")
            }
        };
        assert_eq!(actual.0, claim);
        assert_eq!(
            (actual.1, actual.2, actual.3),
            expected,
            "n={n}, width={width}"
        );
        assert_eq!(
            actual_tr.challenge_k(b"after"),
            reference_tr.challenge_k(b"after")
        );
    }

    #[test]
    fn specialized_rows_match_dense_proof_and_transcript() {
        let weights = [K::ONE, K::ZERO, K::t(17), K::t(29)];
        for n in [0, 1, 2, 5] {
            compare(
                n,
                9,
                Relation::OneHot {
                    weights: &weights,
                    c: K::t(71),
                },
            );
            compare(n, 2, Relation::Mul2);
            compare(
                n,
                9,
                Relation::RwAddress {
                    weights: &weights,
                    inc: K::t(113),
                },
            );
            compare(n, 9, Relation::WeightedProducts { weights: &weights });
            for components in [1, 2, 4, 8, 9, 16] {
                compare(
                    n,
                    2 * (components + 1),
                    Relation::Products {
                        groups: 2,
                        components,
                    },
                );
            }
            let combine = |v: &[K]| v[0] * v[1] + v[0].square() + v[0];
            compare(
                n,
                2,
                Relation::Generic {
                    combine: &combine,
                    degree: 2,
                },
            );
        }
    }

    #[test]
    fn parallel_specialized_fusion_matches_dense() {
        let weights = [K::t(7), K::t(19)];
        compare(
            14,
            5,
            Relation::OneHot {
                weights: &weights,
                c: K::t(37),
            },
        );
        compare(
            14,
            4,
            Relation::Products {
                groups: 1,
                components: 3,
            },
        );
    }
}
