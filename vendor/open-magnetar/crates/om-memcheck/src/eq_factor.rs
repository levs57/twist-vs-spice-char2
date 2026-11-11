//! Equality-factored sum-check, retaining the original finite-point messages.
//!
//! After binding `r[0..k]`, equality is `prefix * eq(z[k..], ·)`. A round is
//! therefore `G(X) = prefix * (1 + z[k] + X) * H(X)`. We accumulate only the
//! cofactor under suffix equality weights. Once the claim is known, its omitted
//! endpoint is recovered from the normalized claim, without dividing `prefix`.

use om_field::{K, U256};
use om_sumcheck::{SumcheckProof, sumcheck_init};
use om_transcript::Transcript;
use rayon::prelude::*;

use crate::{lazy::SplitEq, onehot::PAR_THRESHOLD, sublabel};

/// Which cofactor endpoints are needed. High coefficients `h2..hD` are always
/// accumulated; slot 0 is `H(0)`, slot 1 is `H(1)`.
#[derive(Clone, Copy)]
pub enum Endpoint {
    Both,
    Zero,
    One,
    Neither,
}

impl Endpoint {
    #[inline(always)]
    pub(crate) fn zero(self) -> bool {
        matches!(self, Self::Both | Self::Zero)
    }
    #[inline(always)]
    pub(crate) fn one(self) -> bool {
        matches!(self, Self::Both | Self::One)
    }
}

/// Statically dispatched arithmetic for an equality-weighted relation.
pub trait Cofactor: Sync {
    const DEGREE: usize;
    type Prepared: Sync;
    fn width(&self) -> usize;
    fn value(&self, row: &[K]) -> K;
    /// A structural witness guarantee, never an inference from sampled rows.
    /// The default keeps the initial actual-claim calculation for arbitrary rows.
    fn initial_endpoints_zero(&self) -> bool {
        false
    }
    /// Optional round-local precomputation, e.g. lambda times each low Eq weight.
    fn prepare(&self, low_eq: &[K]) -> Self::Prepared;
    /// Accumulate Eq-low-weighted endpoint/high-coefficient statistics as
    /// unreduced products. High Eq factors are applied once per block outside.
    fn accumulate(
        &self,
        prepared: &Self::Prepared,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    );
}

/// Reuse each constant-times-low-Eq product throughout all high Eq blocks.
pub fn scale_low_weights(weights: &[K], low_eq: &[K]) -> Vec<K> {
    low_eq
        .iter()
        .flat_map(|&eq| weights.iter().map(move |&weight| weight * eq))
        .collect()
}

/// Batch-invert nonzero `1+z`; entries for `z=1` are zero sentinels. At most
/// one inversion is needed for all rounds, and no zero value is ever inverted.
#[doc(hidden)]
pub fn batch_inverse_one_plus(point: &[K]) -> Vec<K> {
    let mut prefixes = Vec::with_capacity(point.len());
    let mut product = K::ONE;
    for &z in point {
        prefixes.push(product);
        if z != K::ONE {
            product *= K::ONE + z;
        }
    }
    let mut inverse = if product == K::ONE {
        K::ONE
    } else {
        product.inverse()
    };
    let mut out = vec![K::ZERO; point.len()];
    for i in (0..point.len()).rev() {
        if point[i] != K::ONE {
            out[i] = inverse * prefixes[i];
            inverse *= K::ONE + point[i];
        }
    }
    out
}

/// Recover cofactor coefficients from `[H(0), H(1), h2, ..., hD]`. With a
/// normalized claim, one endpoint can be omitted. `z=1` uses the other endpoint
/// explicitly, so the zero sentinel is never used as an inverse.
#[doc(hidden)]
pub fn reconstruct_cofactor(
    stats: &[K],
    z: K,
    normalized_claim: Option<K>,
    inverse_one_plus: K,
) -> Vec<K> {
    assert!(stats.len() >= 2);
    let (h0, at_one) = match normalized_claim {
        None => (stats[0], stats[1]),
        Some(claim) if z == K::ONE => (stats[0], claim),
        Some(claim) => ((claim + z * stats[1]) * inverse_one_plus, stats[1]),
    };
    let mut coefficients = stats.to_vec();
    coefficients[0] = h0;
    coefficients[1] = coefficients[2..]
        .iter()
        .copied()
        .fold(h0 + at_one, |sum, c| sum + c);
    coefficients
}

#[inline]
#[doc(hidden)]
pub fn eval_coefficients(coefficients: &[K], r: K) -> K {
    coefficients
        .iter()
        .rev()
        .fold(K::ZERO, |sum, &c| sum * r + c)
}

/// Original degree-(D+1) wire evaluation points, obtained only after the large
/// table scan. Multiplication by small points uses shifts/XORs.
#[doc(hidden)]
pub fn factored_evaluations(coefficients: &[K], z: K, prefix: K) -> Vec<K> {
    (0..=coefficients.len())
        .map(|point| {
            let h = coefficients
                .iter()
                .rev()
                .fold(K::ZERO, |sum, &c| sum.mul_t(point as u128) + c);
            prefix * (K::ONE + z + K::t(point as u128)) * h
        })
        .collect()
}

struct Round<C: Cofactor> {
    eq: SplitEq,
    prepared: C::Prepared,
    endpoint: Endpoint,
}

impl<C: Cofactor> Round<C> {
    fn new(cofactor: &C, suffix: &[K], endpoint: Endpoint) -> Self {
        let eq = SplitEq::new(suffix);
        let prepared = cofactor.prepare(&eq.lo);
        Self {
            eq,
            prepared,
            endpoint,
        }
    }
}

/// A worker's high-Eq block accumulator. Neither a node nor a cycle pair causes
/// a modular reduction here; reductions occur when the high block changes.
struct Blocks<'a> {
    eq: &'a SplitEq,
    endpoint: Endpoint,
    high: usize,
    block: Vec<U256>,
    total: Vec<U256>,
}

impl<'a> Blocks<'a> {
    fn new(eq: &'a SplitEq, count: usize, first_pair: usize, endpoint: Endpoint) -> Self {
        Self {
            eq,
            endpoint,
            high: first_pair >> eq.h,
            block: vec![U256::ZERO; count],
            total: vec![U256::ZERO; count],
        }
    }
    #[inline]
    fn advance(&mut self, pair: usize) -> usize {
        let high = pair >> self.eq.h;
        if high != self.high {
            self.flush();
            self.high = high;
        }
        pair & ((1usize << self.eq.h) - 1)
    }
    fn flush(&mut self) {
        let weight = self.eq.hi[self.high];
        for (index, (total, block)) in self.total.iter_mut().zip(&mut self.block).enumerate() {
            if (index == 0 && !self.endpoint.zero()) || (index == 1 && !self.endpoint.one()) {
                continue;
            }
            *total += weight.clmul(block.reduce());
            *block = U256::ZERO;
        }
    }
    fn finish(mut self) -> Vec<U256> {
        self.flush();
        self.total
    }
}

fn add(mut left: Vec<U256>, right: Vec<U256>) -> Vec<U256> {
    for (a, b) in left.iter_mut().zip(right) {
        *a += b;
    }
    left
}

fn first_statistics<'a, C: Cofactor, F: Fn(usize, &mut [K]) + Sync>(
    cofactor: &C,
    row: &F,
    n: usize,
    round: &'a Round<C>,
) -> Vec<K> {
    const PAIRS: usize = 1024;
    let count = 1usize << (n - 1);
    let fresh = || {
        (
            vec![K::ZERO; cofactor.width()],
            vec![K::ZERO; cofactor.width()],
            Blocks::new(&round.eq, C::DEGREE + 1, 0, round.endpoint),
        )
    };
    let process = |(mut left, mut right, mut sums): (Vec<K>, Vec<K>, Blocks<'a>), chunk: usize| {
        let first = chunk * PAIRS;
        for pair in first..(first + PAIRS).min(count) {
            row(2 * pair, &mut left);
            row(2 * pair + 1, &mut right);
            let low = sums.advance(pair);
            cofactor.accumulate(
                &round.prepared,
                low,
                &left,
                &right,
                round.endpoint,
                &mut sums.block,
            );
        }
        (left, right, sums)
    };
    let chunks = count.div_ceil(PAIRS);
    let total = if 1usize << n >= PAR_THRESHOLD {
        (0..chunks)
            .into_par_iter()
            .fold(fresh, process)
            .map(|(_, _, sums)| sums.finish())
            .reduce(|| vec![U256::ZERO; C::DEGREE + 1], add)
    } else {
        (0..chunks).fold(fresh(), process).2.finish()
    };
    total.into_iter().map(U256::reduce).collect()
}

fn prepare(bound: &mut Vec<Vec<K>>, width: usize, size: usize) {
    bound.resize_with(width, Vec::new);
    for column in bound {
        if column.len() > size {
            column.truncate(size);
        } else if column.len() < size && size >= PAR_THRESHOLD / 2 {
            (0..size)
                .into_par_iter()
                .map(|_| K::ZERO)
                .collect_into_vec(column);
        } else {
            column.resize(size, K::ZERO);
        }
    }
}

/// Fill each column tile using packed field folds, then consume the output
/// while it is resident in cache. Scratch is reused across each worker's tiles.
fn fold_and_statistics<'a, C, F>(
    cofactor: &C,
    bound: &mut Vec<Vec<K>>,
    size: usize,
    parallel: bool,
    fill: F,
    next: Option<&'a Round<C>>,
) -> Vec<K>
where
    C: Cofactor,
    F: Fn(usize, &mut [&mut [K]], &mut [K], &mut [K]) + Sync,
{
    const ROWS: usize = 1024;
    prepare(bound, cofactor.width(), size);
    let mut pieces: Vec<(usize, Vec<&mut [K]>)> = (0..size.div_ceil(ROWS))
        .map(|chunk| (chunk * ROWS, Vec::with_capacity(cofactor.width())))
        .collect();
    for column in bound {
        for (piece, values) in pieces.iter_mut().zip(column.chunks_mut(ROWS)) {
            piece.1.push(values);
        }
    }
    let fresh = || {
        (
            vec![K::ZERO; cofactor.width()],
            vec![K::ZERO; cofactor.width()],
            next.map(|next| Blocks::new(&next.eq, C::DEGREE + 1, 0, next.endpoint)),
        )
    };
    let process = |(mut left, mut right, mut sums): (Vec<K>, Vec<K>, Option<Blocks<'a>>),
                   (base, mut columns): (usize, Vec<&mut [K]>)| {
        let rows = (size - base).min(ROWS);
        fill(base, &mut columns, &mut left, &mut right);
        if let (Some(next), Some(sums)) = (next, &mut sums) {
            for pair in 0..rows / 2 {
                for (i, column) in columns.iter().enumerate() {
                    left[i] = column[2 * pair];
                    right[i] = column[2 * pair + 1];
                }
                let low = sums.advance(base / 2 + pair);
                cofactor.accumulate(
                    &next.prepared,
                    low,
                    &left,
                    &right,
                    next.endpoint,
                    &mut sums.block,
                );
            }
        }
        (left, right, sums)
    };
    let total = if parallel {
        pieces
            .into_par_iter()
            .fold(fresh, process)
            .map(|(_, _, sums)| {
                sums.map_or_else(|| vec![U256::ZERO; C::DEGREE + 1], Blocks::finish)
            })
            .reduce(|| vec![U256::ZERO; C::DEGREE + 1], add)
    } else {
        pieces
            .into_iter()
            .fold(fresh(), process)
            .2
            .map_or_else(|| vec![U256::ZERO; C::DEGREE + 1], Blocks::finish)
    };
    total.into_iter().map(U256::reduce).collect()
}

/// Prove `sum_j eq(point,j) * cofactor(row(j))`. Rows exclude the equality
/// column. Full-size source rows are regenerated once after round 0 to retain
/// the existing half-size field-table memory strategy.
pub fn prove<C: Cofactor, F: Fn(usize, &mut [K]) + Sync>(
    point: &[K],
    cofactor: &C,
    row: F,
    tr: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_with_initial_claim(point, cofactor, row, None, tr, label)
}

/// Prove with an optional incoming claim, omitting one first-round endpoint.
/// `Some` trusts that claim for endpoint recovery; the caller must verify the
/// resulting sumcheck and authenticate every terminal row value. Use [`prove`]
/// when the actual initial sum must be calculated from the supplied rows.
pub fn prove_with_initial_claim<C: Cofactor, F: Fn(usize, &mut [K]) + Sync>(
    point: &[K],
    cofactor: &C,
    row: F,
    initial_claim: Option<K>,
    tr: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_sampled(
        point,
        cofactor,
        row,
        initial_claim,
        tr,
        label,
        |tr, label| tr.challenge_k(label),
    )
}

fn prove_sampled<C: Cofactor, F: Fn(usize, &mut [K]) + Sync>(
    point: &[K],
    cofactor: &C,
    row: F,
    initial_claim: Option<K>,
    tr: &mut Transcript,
    label: &[u8],
    mut sample: impl FnMut(&mut Transcript, &[u8]) -> K,
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    let n = point.len();
    if n == 0 {
        let mut values = vec![K::ZERO; cofactor.width()];
        row(0, &mut values);
        let claim = cofactor.value(&values);
        sumcheck_init(tr, label, claim, 0, C::DEGREE + 1);
        return (
            claim,
            SumcheckProof {
                round_msgs: Vec::new(),
            },
            Vec::new(),
            values,
        );
    }
    // Default round zero has explicit or structural-zero endpoints and needs
    // no inverse. An internally known initial claim can omit one endpoint;
    // include its nonzero denominator in the same batch inversion.
    let inverses = if initial_claim.is_some() && !cofactor.initial_endpoints_zero() {
        batch_inverse_one_plus(point)
    } else {
        let mut values = vec![K::ZERO];
        values.extend(batch_inverse_one_plus(&point[1..]));
        values
    };
    let initial_endpoint = if cofactor.initial_endpoints_zero() {
        Endpoint::Neither
    } else if initial_claim.is_some() {
        if point[0] == K::ONE {
            Endpoint::Zero
        } else {
            Endpoint::One
        }
    } else {
        Endpoint::Both
    };
    let first = Round::new(cofactor, &point[1..], initial_endpoint);
    let stats = first_statistics(cofactor, &row, n, &first);
    let mut coefficients = reconstruct_cofactor(&stats, point[0], initial_claim, inverses[0]);
    drop(first);
    let initial = factored_evaluations(&coefficients, point[0], K::ONE);
    let claim = initial[0] + initial[1];
    sumcheck_init(tr, label, claim, n, C::DEGREE + 1);
    let (msg_label, chal_label) = (sublabel(label, b"/msg"), sublabel(label, b"/chal"));
    let mut prefix = K::ONE;
    let mut current = claim;
    let mut tables: Vec<Vec<K>> = Vec::new();
    let mut scratch = Vec::new();
    let mut messages = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    for round in 0..n {
        let evaluations = factored_evaluations(&coefficients, point[round], prefix);
        debug_assert_eq!(evaluations[0] + evaluations[1], current);
        let mut message = vec![evaluations[0]];
        message.extend_from_slice(&evaluations[2..]);
        tr.absorb_ks(&msg_label, &message);
        let challenge = sample(tr, &chal_label);
        messages.push(message);
        challenges.push(challenge);
        let normalized_next = eval_coefficients(&coefficients, challenge);
        prefix *= K::ONE + point[round] + challenge;
        current = prefix * normalized_next;
        let next = if round + 1 < n && prefix != K::ZERO {
            let endpoint = if point[round + 1] == K::ONE {
                Endpoint::Zero
            } else {
                Endpoint::One
            };
            Some(Round::new(cofactor, &point[round + 2..], endpoint))
        } else {
            None
        };
        let size = 1usize << (n - round - 1);
        let stats = if round == 0 {
            let fill = |base: usize, columns: &mut [&mut [K]], left: &mut [K], right: &mut [K]| {
                let rows = (size - base).min(1024);
                for j in 0..rows {
                    row(2 * (base + j), left);
                    row(2 * (base + j) + 1, right);
                    om_field::fold_slices(left, right, challenge);
                    for (column, &value) in columns.iter_mut().zip(left.iter()) {
                        column[j] = value;
                    }
                }
            };
            fold_and_statistics(
                cofactor,
                &mut tables,
                size,
                1usize << n >= PAR_THRESHOLD,
                fill,
                next.as_ref(),
            )
        } else {
            let fill = |base: usize, columns: &mut [&mut [K]], _: &mut [K], _: &mut [K]| {
                for (out, source) in columns.iter_mut().zip(&tables) {
                    om_field::fold_pairs(&source[2 * base..2 * (base + out.len())], challenge, out);
                }
            };
            let stats = fold_and_statistics(
                cofactor,
                &mut scratch,
                size,
                2 * size >= PAR_THRESHOLD,
                fill,
                next.as_ref(),
            );
            std::mem::swap(&mut tables, &mut scratch);
            stats
        };
        if round + 1 < n {
            coefficients = if prefix == K::ZERO {
                vec![K::ZERO; C::DEGREE + 1]
            } else {
                reconstruct_cofactor(
                    &stats,
                    point[round + 1],
                    Some(normalized_next),
                    inverses[round + 1],
                )
            };
        }
    }
    let mles = tables.iter().map(|column| column[0]).collect();
    (
        claim,
        SumcheckProof {
            round_msgs: messages,
        },
        challenges,
        mles,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use om_sumcheck::{
        combine_sum, eq_table, interpolate_eval, mle::bind_low_in_place, sumcheck_prove,
        sumcheck_round_evals,
    };

    pub(crate) fn random(seed: &mut u64) -> K {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        K::from_u128((*seed as u128) | ((seed.rotate_left(23) as u128) << 64))
    }

    /// Independent generic dense oracle, including a test-only challenge source
    /// so zero equality prefixes and every small field challenge can be tested.
    pub(crate) fn compare<C: Cofactor>(cofactor: &C, point: &[K], forced: Option<&[K]>) {
        compare_mode(cofactor, point, forced, false);
    }

    pub(crate) fn compare_known<C: Cofactor>(cofactor: &C, point: &[K], forced: Option<&[K]>) {
        compare_mode(cofactor, point, forced, true);
    }

    fn compare_mode<C: Cofactor>(cofactor: &C, point: &[K], forced: Option<&[K]>, known: bool) {
        let n = point.len();
        let mut seed = 0x452821e638d01377;
        let columns: Vec<Vec<K>> = (0..cofactor.width())
            .map(|_| (0..1usize << n).map(|_| random(&mut seed)).collect())
            .collect();
        compare_columns_mode(cofactor, point, &columns, forced, known);
    }

    pub(crate) fn compare_columns<C: Cofactor>(
        cofactor: &C,
        point: &[K],
        columns: &[Vec<K>],
        forced: Option<&[K]>,
    ) {
        compare_columns_mode(cofactor, point, columns, forced, false);
    }

    fn compare_columns_mode<C: Cofactor>(
        cofactor: &C,
        point: &[K],
        columns: &[Vec<K>],
        forced: Option<&[K]>,
        known: bool,
    ) {
        let n = point.len();
        assert_eq!(columns.len(), cofactor.width());
        assert!(columns.iter().all(|column| column.len() == 1usize << n));
        let row = |j: usize, out: &mut [K]| {
            for (out, column) in out.iter_mut().zip(columns) {
                *out = column[j];
            }
        };
        let mut dense = vec![eq_table(point)];
        dense.extend_from_slice(columns);
        let combine = |row: &[K]| row[0] * cofactor.value(&row[1..]);
        let claim = combine_sum(&dense, &combine);
        let mut reference_tr = Transcript::new_default(b"factored-eq-oracle");
        let expected = if let Some(forced) = forced {
            assert_eq!(forced.len(), n);
            sumcheck_init(&mut reference_tr, b"sc", claim, n, C::DEGREE + 1);
            let mut current = claim;
            let mut messages = Vec::new();
            for &challenge in forced {
                let evaluations = sumcheck_round_evals(&dense, &combine, C::DEGREE + 1);
                assert_eq!(evaluations[0] + evaluations[1], current);
                let mut message = vec![evaluations[0]];
                message.extend_from_slice(&evaluations[2..]);
                reference_tr.absorb_ks(b"sc/msg", &message);
                reference_tr.challenge_k(b"sc/chal");
                current = interpolate_eval(&evaluations, challenge);
                messages.push(message);
                for column in &mut dense {
                    bind_low_in_place(column, challenge);
                }
            }
            (
                SumcheckProof {
                    round_msgs: messages,
                },
                forced.to_vec(),
                dense.iter().map(|c| c[0]).collect::<Vec<K>>(),
            )
        } else {
            sumcheck_prove(
                claim,
                dense,
                &combine,
                C::DEGREE + 1,
                &mut reference_tr,
                b"sc",
            )
        };
        let mut actual_tr = Transcript::new_default(b"factored-eq-oracle");
        let actual = if let Some(forced) = forced {
            let mut position = 0;
            prove_sampled(
                point,
                cofactor,
                row,
                known.then_some(claim),
                &mut actual_tr,
                b"sc",
                |tr, label| {
                    tr.challenge_k(label);
                    let value = forced[position];
                    position += 1;
                    value
                },
            )
        } else if known {
            prove_with_initial_claim(point, cofactor, row, Some(claim), &mut actual_tr, b"sc")
        } else {
            prove(point, cofactor, row, &mut actual_tr, b"sc")
        };
        assert_eq!(actual.0, claim);
        assert_eq!(actual.1, expected.0, "wire messages, n={n}");
        assert_eq!(actual.2, expected.1, "challenges, n={n}");
        assert_eq!(actual.3, expected.2[1..], "terminal MLEs, n={n}");
        assert_eq!(
            actual_tr.challenge_k(b"after"),
            reference_tr.challenge_k(b"after")
        );
    }

    #[test]
    fn recovery_handles_boolean_points_and_zero_prefixes_without_zero_inverse() {
        let point = [K::ZERO, K::ONE, K::t(2), K::t(5), K::ONE, K::t(17)];
        let inverses = batch_inverse_one_plus(&point);
        for (&z, &inverse) in point.iter().zip(&inverses) {
            if z == K::ONE {
                assert_eq!(inverse, K::ZERO);
            } else {
                assert_eq!((K::ONE + z) * inverse, K::ONE);
            }
            for degree in [2, 3] {
                let coefficients: Vec<K> =
                    (0..=degree).map(|i| K::t((13 + i * 9) as u128)).collect();
                let mut stats = coefficients.clone();
                stats[1] = coefficients
                    .iter()
                    .copied()
                    .fold(K::ZERO, |sum, value| sum + value);
                let claim = (K::ONE + z) * coefficients[0] + z * stats[1];
                assert_eq!(reconstruct_cofactor(&stats, z, None, inverse), coefficients);
                if z == K::ONE {
                    stats[1] = K::ZERO;
                } else {
                    stats[0] = K::ZERO;
                }
                assert_eq!(
                    reconstruct_cofactor(&stats, z, Some(claim), inverse),
                    coefficients
                );
                assert!(
                    factored_evaluations(&coefficients, z, K::ZERO)
                        .iter()
                        .all(|&value| value == K::ZERO)
                );
            }
        }
    }
}
