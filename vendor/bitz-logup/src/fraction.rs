// Copyright (c) 2026 World Foundation. MIT; see ../LICENSE.
// Derived from BitZ cdfb9b73b5857f4dc83761866e475186c6a75ab3,
// crates/gkr/src/logup_cut/fraction.rs. Arithmetic and scheduling retained.
use crate::{Error, Wide256, bind, challenge, write};
use om_field::K as F128;
use om_transcript::Transcript;
use rayon::prelude::*;

use crate::product::{ProductRelationProof, prove_product_relation_into, verify_product_relation};

const DOMAIN: &[u8] = b"bitz/logup-cut/rational/v1";
const PARALLEL_THRESHOLD: usize = 1 << 13;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractionClaim {
    pub point: Vec<F128>,
    pub num: F128,
    pub den: F128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RationalClaims {
    pub left: FractionClaim,
    pub right: FractionClaim,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionLayerProof {
    pub ac_eval: F128,
    pub product: ProductRelationProof,
    pub numerators: [F128; 2],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FractionTreeProof {
    pub layers: Vec<FractionLayerProof>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalProof {
    pub roots: [F128; 4],
    pub left: FractionTreeProof,
    pub right: FractionTreeProof,
}

#[derive(Clone, Debug, Default)]
pub struct FractionTreeWitness {
    nums: Vec<Vec<F128>>,
    dens: Vec<Vec<F128>>,
    ac: Vec<Vec<F128>>,
    num_padding: Vec<F128>,
    den_padding: Vec<F128>,
    ac_padding: Vec<F128>,
    products: Vec<F128>,
    suffix: Vec<F128>,
    eq_low: Vec<F128>,
    eq_high: Vec<F128>,
}

impl FractionTreeWitness {
    pub fn rebuild(
        &mut self,
        numerator: &[F128],
        denominator: &[F128],
        dimension: usize,
        numerator_padding: F128,
        denominator_padding: F128,
    ) -> Result<(), Error> {
        let full_len = 1usize
            .checked_shl(dimension as u32)
            .filter(|_| dimension < usize::BITS as usize)
            .ok_or(Error::InvalidShape)?;
        if numerator.is_empty()
            || numerator.len() > full_len
            || numerator.len() != denominator.len()
        {
            return Err(Error::InvalidShape);
        }
        if denominator_padding == F128::ZERO || denominator.iter().any(|value| *value == F128::ZERO)
        {
            return Err(Error::ZeroDenominator);
        }

        self.nums.resize_with(dimension + 1, Vec::new);
        self.nums.truncate(dimension + 1);
        self.dens.resize_with(dimension + 1, Vec::new);
        self.dens.truncate(dimension + 1);
        self.ac.resize_with(dimension, Vec::new);
        self.ac.truncate(dimension);
        self.num_padding.resize(dimension + 1, F128::ZERO);
        self.den_padding.resize(dimension + 1, F128::ZERO);
        self.ac_padding.resize(dimension, F128::ZERO);
        self.products.resize(full_len, F128::ZERO);
        self.suffix
            .resize(1usize << dimension.saturating_sub(1), F128::ZERO);
        self.eq_low
            .reserve((1usize << (dimension / 2)).saturating_sub(self.eq_low.capacity()));
        self.eq_high
            .reserve((1usize << dimension.div_ceil(2)).saturating_sub(self.eq_high.capacity()));
        self.nums[0].clear();
        self.nums[0].extend_from_slice(numerator);
        self.dens[0].clear();
        self.dens[0].extend_from_slice(denominator);
        self.num_padding.fill(F128::ZERO);
        self.den_padding.fill(F128::ZERO);
        self.ac_padding.fill(F128::ZERO);
        self.num_padding[0] = numerator_padding;
        self.den_padding[0] = denominator_padding;

        for level in 0..dimension {
            let parent_len = self.nums[level].len().div_ceil(2);
            let num_padding = self.num_padding[level];
            let den_padding = self.den_padding[level];
            self.nums[level + 1].resize(parent_len, F128::ZERO);
            self.dens[level + 1].resize(parent_len, F128::ZERO);
            self.ac[level].resize(parent_len, F128::ZERO);
            let (child_nums, parent_nums) = self.nums.split_at_mut(level + 1);
            let (child_dens, parent_dens) = self.dens.split_at_mut(level + 1);
            let child_num = &child_nums[level];
            let child_den = &child_dens[level];
            let next_num = &mut parent_nums[0];
            let next_den = &mut parent_dens[0];
            let next_ac = &mut self.ac[level];
            let fill =
                |index: usize, next_num: &mut F128, next_den: &mut F128, next_ac: &mut F128| {
                    let a = child_num[2 * index];
                    let b = child_den[2 * index];
                    let c = child_num.get(2 * index + 1).copied().unwrap_or(num_padding);
                    let d = child_den.get(2 * index + 1).copied().unwrap_or(den_padding);
                    let ac = a * c;
                    let bd = b * d;
                    *next_num = (a + b) * (c + d) + ac + bd;
                    *next_den = bd;
                    *next_ac = ac;
                };
            if parent_len >= PARALLEL_THRESHOLD {
                next_num
                    .par_iter_mut()
                    .zip(next_den.par_iter_mut())
                    .zip(next_ac.par_iter_mut())
                    .enumerate()
                    .for_each(|(index, ((num, den), ac))| fill(index, num, den, ac));
            } else {
                next_num
                    .iter_mut()
                    .zip(next_den.iter_mut())
                    .zip(next_ac.iter_mut())
                    .enumerate()
                    .for_each(|(index, ((num, den), ac))| fill(index, num, den, ac));
            }
            self.ac_padding[level] = self.num_padding[level].square();
            self.num_padding[level + 1] = F128::ZERO;
            self.den_padding[level + 1] = self.den_padding[level].square();
        }

        debug_assert_eq!(self.nums[dimension].len(), 1);
        Ok(())
    }

    pub fn retained_bytes(&self) -> usize {
        (self
            .nums
            .iter()
            .map(|level| level.capacity())
            .sum::<usize>()
            .saturating_add(self.dens.iter().map(|level| level.capacity()).sum())
            .saturating_add(self.ac.iter().map(|level| level.capacity()).sum())
            .saturating_add(self.num_padding.capacity())
            .saturating_add(self.den_padding.capacity())
            .saturating_add(self.ac_padding.capacity())
            .saturating_add(self.products.capacity())
            .saturating_add(self.suffix.capacity())
            .saturating_add(self.eq_low.capacity())
            .saturating_add(self.eq_high.capacity()))
            * core::mem::size_of::<F128>()
    }

    pub fn dimension(&self) -> usize {
        self.ac.len()
    }

    pub fn root(&self) -> (F128, F128) {
        let level = self.dimension();
        (self.nums[level][0], self.dens[level][0])
    }

    fn prove(
        &mut self,
        transcript: &mut Transcript,
    ) -> Result<(FractionTreeProof, FractionClaim), Error> {
        let mut layers = Vec::with_capacity(self.dimension());
        let (mut claim_num, mut claim_den) = self.root();
        let mut claim_point = Vec::with_capacity(self.dimension());
        for level in (0..self.dimension()).rev() {
            let ac_eval = mle_eval(
                &self.ac[level],
                self.ac_padding[level],
                &claim_point,
                &mut self.eq_low,
                &mut self.eq_high,
            );
            write(transcript, &[ac_eval]);
            let gamma = challenge(transcript);
            if gamma == F128::ZERO {
                return Err(Error::ZeroChallenge);
            }

            let full_parent_len = 1usize << claim_point.len();
            let products = &mut self.products[..2 * full_parent_len];
            let (left_products, right_products) = products.split_at_mut(full_parent_len);
            let product_padding = self.den_padding[level] + gamma * self.num_padding[level];
            left_products.fill(product_padding);
            right_products.fill(product_padding);
            let fill = |index: usize, left: &mut F128, right: &mut F128| {
                let even = 2 * index;
                *left = self.dens[level][even] + gamma * self.nums[level][even];
                if even + 1 < self.nums[level].len() {
                    *right = self.dens[level][even + 1] + gamma * self.nums[level][even + 1];
                }
            };
            let active_parent_len = self.ac[level].len();
            if active_parent_len >= PARALLEL_THRESHOLD {
                left_products[..active_parent_len]
                    .par_iter_mut()
                    .zip(right_products[..active_parent_len].par_iter_mut())
                    .enumerate()
                    .for_each(|(index, (left, right))| fill(index, left, right));
            } else {
                left_products[..active_parent_len]
                    .iter_mut()
                    .zip(right_products[..active_parent_len].iter_mut())
                    .enumerate()
                    .for_each(|(index, (left, right))| fill(index, left, right));
            }
            let (product, product_point, [left, right]) =
                prove_product_relation_into(transcript, &claim_point, products, &mut self.suffix);

            let [a, c] = mle_eval_interleaved_pair(
                &self.nums[level],
                self.num_padding[level],
                &product_point,
                &mut self.eq_low,
                &mut self.eq_high,
            );
            write(transcript, &[a, c]);
            layers.push(FractionLayerProof {
                ac_eval,
                product,
                numerators: [a, c],
            });
            let b = left + gamma * a;
            let d = right + gamma * c;
            let selector = challenge(transcript);

            claim_point.clear();
            claim_point.push(selector);
            claim_point.extend_from_slice(&product_point);
            claim_num = a + selector * (a + c);
            claim_den = b + selector * (b + d);
            debug_assert_eq!(
                claim_num,
                mle_eval(
                    &self.nums[level],
                    self.num_padding[level],
                    &claim_point,
                    &mut self.eq_low,
                    &mut self.eq_high,
                ),
                "fraction numerator layer {level}"
            );
            debug_assert_eq!(
                claim_den,
                mle_eval(
                    &self.dens[level],
                    self.den_padding[level],
                    &claim_point,
                    &mut self.eq_low,
                    &mut self.eq_high,
                ),
                "fraction denominator layer {level}"
            );
        }
        Ok((
            FractionTreeProof { layers },
            FractionClaim {
                point: claim_point,
                num: claim_num,
                den: claim_den,
            },
        ))
    }
}

pub fn prove_rational(
    left: &mut FractionTreeWitness,
    right: &mut FractionTreeWitness,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<(RationalProof, RationalClaims), Error> {
    if left.nums.is_empty() || right.nums.is_empty() {
        return Err(Error::InvalidShape);
    }
    let left_root = left.root();
    let right_root = right.root();
    if left_root.1 == F128::ZERO || right_root.1 == F128::ZERO {
        return Err(Error::ZeroDenominator);
    }
    if left_root.0 * right_root.1 != right_root.0 * left_root.1 {
        return Err(Error::InvalidClaim);
    }
    bind(
        transcript,
        label,
        DOMAIN,
        &[left.dimension(), right.dimension()],
    );
    let roots = [left_root.0, left_root.1, right_root.0, right_root.1];
    write(transcript, &roots);
    let (left_proof, left_claim) = left.prove(transcript)?;
    let (right_proof, right_claim) = right.prove(transcript)?;
    Ok((
        RationalProof {
            roots,
            left: left_proof,
            right: right_proof,
        },
        RationalClaims {
            left: left_claim,
            right: right_claim,
        },
    ))
}

pub fn verify_rational(
    left_dimension: usize,
    right_dimension: usize,
    proof: &RationalProof,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<RationalClaims, Error> {
    if left_dimension >= usize::BITS as usize
        || right_dimension >= usize::BITS as usize
        || !tree_shape(&proof.left, left_dimension)
        || !tree_shape(&proof.right, right_dimension)
    {
        return Err(Error::MalformedProof);
    }
    bind(
        transcript,
        label,
        DOMAIN,
        &[left_dimension, right_dimension],
    );
    let [left_num, left_den, right_num, right_den] = proof.roots;
    write(transcript, &proof.roots);
    if left_den == F128::ZERO || right_den == F128::ZERO {
        return Err(Error::ZeroDenominator);
    }
    if left_num * right_den != right_num * left_den {
        return Err(Error::InvalidProof);
    }
    Ok(RationalClaims {
        left: verify_tree(transcript, left_dimension, left_num, left_den, &proof.left)?,
        right: verify_tree(
            transcript,
            right_dimension,
            right_num,
            right_den,
            &proof.right,
        )?,
    })
}

fn tree_shape(proof: &FractionTreeProof, dimension: usize) -> bool {
    proof.layers.len() == dimension
        && proof
            .layers
            .iter()
            .enumerate()
            .all(|(layer, proof)| proof.product.rounds.len() == layer)
}

fn verify_tree(
    transcript: &mut Transcript,
    dimension: usize,
    mut claim_num: F128,
    mut claim_den: F128,
    proof: &FractionTreeProof,
) -> Result<FractionClaim, Error> {
    let mut claim_point = Vec::with_capacity(dimension);
    for layer in &proof.layers {
        let ac_eval = layer.ac_eval;
        write(transcript, &[ac_eval]);
        let gamma = challenge(transcript);
        if gamma == F128::ZERO {
            return Err(Error::ZeroChallenge);
        }
        let folded = claim_den + gamma * claim_num + gamma.square() * ac_eval;
        let (product_point, [left, right]) =
            verify_product_relation(transcript, folded, &claim_point, &layer.product)?;
        let [a, c] = layer.numerators;
        write(transcript, &layer.numerators);
        let b = left + gamma * a;
        let d = right + gamma * c;
        let selector = challenge(transcript);
        claim_point.clear();
        claim_point.push(selector);
        claim_point.extend_from_slice(&product_point);
        claim_num = a + selector * (a + c);
        claim_den = b + selector * (b + d);
    }
    Ok(FractionClaim {
        point: claim_point,
        num: claim_num,
        den: claim_den,
    })
}

fn mle_eval(
    values: &[F128],
    padding: F128,
    point: &[F128],
    eq_low: &mut Vec<F128>,
    eq_high: &mut Vec<F128>,
) -> F128 {
    assert!(!values.is_empty() && values.len() <= 1usize << point.len());
    prepare_eq_sqrt(point, eq_low, eq_high);
    padding + dot_sqrt(values, padding, eq_low, eq_high)
}

fn mle_eval_interleaved_pair(
    values: &[F128],
    padding: F128,
    point: &[F128],
    eq_low: &mut Vec<F128>,
    eq_high: &mut Vec<F128>,
) -> [F128; 2] {
    assert!(!values.is_empty() && values.len() <= 2 * (1usize << point.len()));
    prepare_eq_sqrt(point, eq_low, eq_high);
    let low_len = eq_low.len();
    let pair_len = values.len().div_ceil(2);
    let high_len = pair_len.div_ceil(low_len);
    let block = |high: usize| {
        let pair_base = high * low_len;
        let block_len = (pair_len - pair_base).min(low_len);
        let mut even = Wide256::zero();
        let mut odd = Wide256::zero();
        for low in 0..block_len {
            let source = 2 * (pair_base + low);
            even += Wide256::mul(eq_low[low], values[source] + padding);
            if let Some(&value) = values.get(source + 1) {
                odd += Wide256::mul(eq_low[low], value + padding);
            }
        }
        [
            Wide256::mul(even.reduce(), eq_high[high]),
            Wide256::mul(odd.reduce(), eq_high[high]),
        ]
    };
    let sums = if pair_len >= PARALLEL_THRESHOLD && high_len > 1 {
        (0..high_len).into_par_iter().map(block).reduce(
            || [Wide256::zero(), Wide256::zero()],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        )
    } else {
        let mut sums = [Wide256::zero(), Wide256::zero()];
        for high in 0..high_len {
            let block = block(high);
            sums[0] += block[0];
            sums[1] += block[1];
        }
        sums
    };
    [padding + sums[0].reduce(), padding + sums[1].reduce()]
}

fn prepare_eq_sqrt(point: &[F128], low: &mut Vec<F128>, high: &mut Vec<F128>) {
    let split = point.len() / 2;
    fill_eq(&point[..split], low);
    fill_eq(&point[split..], high);
}

fn fill_eq(point: &[F128], out: &mut Vec<F128>) {
    out.clear();
    out.push(F128::ONE);
    for &coordinate in point {
        let len = out.len();
        out.resize(2 * len, F128::ZERO);
        let (low, high) = out.split_at_mut(len);
        for (low, high) in low.iter_mut().zip(high) {
            let value = *low;
            *low = value * (F128::ONE + coordinate);
            *high = value * coordinate;
        }
    }
}

fn dot_sqrt(values: &[F128], padding: F128, eq_low: &[F128], eq_high: &[F128]) -> F128 {
    let low_len = eq_low.len();
    let blocks = values.len().div_ceil(low_len);
    debug_assert!(blocks <= eq_high.len());
    let block = |high: usize| {
        let base = high * low_len;
        let values = &values[base..(base + low_len).min(values.len())];
        let mut inner = Wide256::zero();
        for (&value, &weight) in values.iter().zip(eq_low) {
            inner += Wide256::mul(value + padding, weight);
        }
        Wide256::mul(inner.reduce(), eq_high[high])
    };
    if values.len() >= PARALLEL_THRESHOLD && blocks > 1 {
        (0..blocks)
            .into_par_iter()
            .map(block)
            .reduce(Wide256::zero, |left, right| left + right)
            .reduce()
    } else {
        let mut value = Wide256::zero();
        for high in 0..blocks {
            value += block(high);
        }
        value.reduce()
    }
}
