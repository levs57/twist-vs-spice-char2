// Copyright (c) 2026 World Foundation. MIT; see ../LICENSE.
// Derived from BitZ cdfb9b73b5857f4dc83761866e475186c6a75ab3,
// crates/gkr/src/lib.rs:56-312,354-408. Donor arithmetic and scheduling retained.
use crate::{Error, Wide256, challenge, write};
use om_field::K as F128;
use om_transcript::Transcript;
use rayon::prelude::*;
use std::collections::VecDeque;
type Field = F128;
type Point = VecDeque<Field>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductRelationProof {
    /// One endpoint and the quadratic cofactor's leading coefficient.
    pub rounds: Vec<[F128; 2]>,
    pub evaluations: [F128; 2],
}

fn prove_product_relation_inner(
    ps: &mut Transcript,
    storage: &mut [Field],
    mut point: Point,
    wnext: &mut [Field],
) -> (ProductRelationProof, Point, [Field; 2]) {
    let mut round_messages = Vec::with_capacity(point.len());
    let suffix_table = SuffixTable::new(storage, &point);
    let mut factor = Field::ONE;

    let mid = wnext.len() / 2;
    // Tree is encoded in LSB order
    let (mut mle_l, mut mle_r) = wnext.split_at_mut(mid);

    let rounds = point.len();
    let mut next_point = VecDeque::with_capacity(rounds + 1);

    // `SuffixTable::layer(i)` has length `2^i`, so `h` is even on every round
    // except the last (`i == 0`). Peel that round out instead of checking
    // for it on every iteration.
    let last_z = point.pop_back();

    for (i, z) in (1..rounds).rev().zip(point) {
        let eq = suffix_table.layer(i);
        let h = mle_l.len() / 2;
        debug_assert_eq!(eq.len(), h);
        debug_assert_eq!(h % 2, 0);

        let (lo_l, hi_l) = mle_l.split_at_mut(h);
        let (lo_r, hi_r) = mle_r.split_at_mut(h);

        // `send_one` is constant for the whole round: at z = 0 the incoming
        // claim determines the value at zero, so this round sends the value
        // at one instead. Branch on it once here -- SEND_ONE below is a
        // const generic, so each instantiation gets only its own branch,
        // not a per-element check inside the hot fold.
        let send_one = z == Field::ZERO;
        let (sum_endpoint, sum_inf) = if send_one {
            reduce_sumcheck_round::<true>(lo_l, lo_r, hi_l, hi_r, eq)
        } else {
            reduce_sumcheck_round::<false>(lo_l, lo_r, hi_l, hi_r, eq)
        };

        let message = [factor * sum_endpoint.reduce(), factor * sum_inf.reduce()];
        write(ps, &message);
        round_messages.push(message);

        let r = challenge(ps);
        next_point.push_back(r);

        lo_l.par_iter_mut()
            .zip(lo_r.par_iter_mut())
            .zip(hi_l.par_iter())
            .zip(hi_r.par_iter())
            .with_min_len(PARALLEL_MIN_LANES)
            .for_each(|(((l_lo, r_lo), &l_hi), &r_hi)| {
                // Recalculating delta rather than writing it in the top half of the array in the previous loop to save on writes.
                // No benchmarking has been done to check the difference.
                let (d_l, d_r) = (l_hi - *l_lo, r_hi - *r_lo);

                // narrow multiply as the next round will multiply with this result
                *l_lo += r * d_l;
                *r_lo += r * d_r;
            });
        mle_l = &mut mle_l[..h];
        mle_r = &mut mle_r[..h];

        factor *= eq_factor(r, z);
    }

    if let Some(z) = last_z {
        // i == 0, h == 1: `SuffixTable::new` always seeds the base layer
        // (`eq[0]`) with `Field::ONE`, so the `eq[i] *` multiplication every
        // other round needs is the identity here -- skipped rather than
        // spent multiplying by one.
        debug_assert_eq!(mle_l.len(), 2);
        let (l_lo, l_hi) = (mle_l[0], mle_l[1]);
        let (r_lo, r_hi) = (mle_r[0], mle_r[1]);
        let (d_l, d_r) = (l_hi - l_lo, r_hi - r_lo);

        // Same `send_one` swap as the main loop above: at z = 0 the incoming
        // claim determines the value at zero, so send the value at one.
        let (l_endpoint, r_endpoint) = if z == Field::ZERO {
            (l_hi, r_hi)
        } else {
            (l_lo, r_lo)
        };
        let sum_endpoint = Wide256::mul(l_endpoint, r_endpoint);
        let sum_inf = Wide256::mul(d_l, d_r);

        let message = [factor * sum_endpoint.reduce(), factor * sum_inf.reduce()];
        write(ps, &message);
        round_messages.push(message);

        let r = challenge(ps);
        next_point.push_back(r);

        mle_l[0] = l_lo + r * d_l;
        mle_r[0] = r_lo + r * d_r;
        mle_l = &mut mle_l[..1];
        mle_r = &mut mle_r[..1];

        factor *= eq_factor(r, z);
    }

    let values = [mle_l[0], mle_r[0]];
    write(ps, &values);
    (
        ProductRelationProof {
            rounds: round_messages,
            evaluations: values,
        },
        next_point,
        values,
    )
}

/// Public point convention is low-variable-first; donor loops bind high first.
pub(crate) fn prove_product_relation_into(
    ps: &mut Transcript,
    point: &[F128],
    witness: &mut [F128],
    storage: &mut [F128],
) -> (ProductRelationProof, Vec<F128>, [F128; 2]) {
    assert_eq!(witness.len(), 2usize << point.len());
    let mut reversed = point.to_vec();
    reversed.reverse();
    let (proof, point, values) =
        prove_product_relation_inner(ps, storage, VecDeque::from(reversed), witness);
    let mut point = Vec::from(point);
    point.reverse();
    (proof, point, values)
}

pub(crate) fn reduce_sumcheck_round<const SEND_ONE: bool>(
    lo_l: &[Field],
    lo_r: &[Field],
    hi_l: &[Field],
    hi_r: &[Field],
    eq: &[Field],
) -> (Wide256, Wide256) {
    // chunking reduces variance in benchmarking
    lo_l.par_chunks_exact(2)
        .zip(lo_r.par_chunks_exact(2))
        .zip(hi_l.par_chunks_exact(2))
        .zip(hi_r.par_chunks_exact(2))
        .zip(eq.par_chunks_exact(2))
        // Chunked by 2, so the element-count threshold below is halved in
        // terms of chunks.
        .with_min_len(PARALLEL_MIN_LANES / 2)
        .fold(
            || (Wide256::zero(), Wide256::zero()),
            |(mut sum_endpoint, mut sum_inf), ((((l_lo, r_lo), l_hi), r_hi), e)| {
                let (d_l0, d_r0) = (l_hi[0] - l_lo[0], r_hi[0] - r_lo[0]);
                let (d_l1, d_r1) = (l_hi[1] - l_lo[1], r_hi[1] - r_lo[1]);

                let (l_e0, r_e0) = if SEND_ONE {
                    (l_hi[0], r_hi[0])
                } else {
                    (l_lo[0], r_lo[0])
                };
                let (l_e1, r_e1) = if SEND_ONE {
                    (l_hi[1], r_hi[1])
                } else {
                    (l_lo[1], r_lo[1])
                };

                let (a0, b0) = (e[0] * l_e0, e[0] * d_l0);
                let (a1, b1) = (e[1] * l_e1, e[1] * d_l1);

                sum_endpoint += Wide256::mul(a0, r_e0);
                sum_inf += Wide256::mul(b0, d_r0);
                sum_endpoint += Wide256::mul(a1, r_e1);
                sum_inf += Wide256::mul(b1, d_r1);

                (sum_endpoint, sum_inf)
            },
        )
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(a0, ainf), (b0, binf)| (a0 + b0, ainf + binf),
        )
}

/// `eq(r, z) = r*z + (1 - r)*(1 - z)`. Expanding gives
/// `1 + r + z + 2*r*z`, and in characteristic 2 `2*r*z = r*z + r*z = 0`, so
/// this is just `1 + r + z` -- no multiplication at all, and so nothing for
/// widemul to help with.
pub(crate) fn eq_factor(r: Field, z: Field) -> Field {
    Field::ONE + r + z
}

// TODO: Split suffix table
pub(crate) struct SuffixTable<'a> {
    storage: &'a [Field],
}

impl<'a> SuffixTable<'a> {
    pub(crate) fn layer(&self, i: usize) -> &[Field] {
        let start = (1 << i) - 1;
        let end = (1 << (i + 1)) - 1;
        &self.storage[start..end]
    }
}

impl<'a> SuffixTable<'a> {
    /// Allocates all directly as it is as much space as a double buffer approach would take.
    #[inline(never)]
    pub(crate) fn new(storage: &'a mut [Field], point: &Point) -> SuffixTable<'a> {
        // Establishes the whole table's space budget up front so LLVM can
        // prove the `low`/`hi` writes in the loop below are in bounds
        let needed = (1usize << point.len()).saturating_sub(1).max(1);
        assert!(storage.len() >= needed);

        storage[0] = F128::ONE;
        let (mut prev, mut remaining) = storage.split_at_mut(1);
        // The selector is the first entry of the point and we need to skip
        // that -- except when `point` is itself empty, in which case there
        // is no selector and `c` must stay empty too (`min(1)` keeps the
        // range start in-bounds there instead of panicking).
        let c = point.range(point.len().min(1)..);

        // Suffix table is in the reverse order of the point
        for &z in c.rev() {
            let size = prev.len() << 1;
            let (entry, next) = remaining.split_at_mut(size);
            let (low, hi) = entry.split_at_mut(entry.len() >> 1);

            for ((l, h), &e) in low.iter_mut().zip(hi.iter_mut()).zip(prev.iter()) {
                let tmp = z * e;
                // (1-z)*e, z*e
                (*l, *h) = (e - tmp, tmp)
            }

            prev = entry;
            remaining = next;
        }
        SuffixTable { storage }
    }
}

/// TODO replace this with a proper approach
///  Matches `poly`'s `PARALLEL_FOLD_THRESHOLD`, chosen for the same per-round
/// MLE-fold shape.
const PARALLEL_MIN_LANES: usize = 1 << 12;

fn verify_product_relation_inner(
    vs: &mut Transcript,
    mut claim: Field,
    point: Point,
    proof: &ProductRelationProof,
) -> Result<(Point, [Field; 2]), Error> {
    let mut prefix = Field::ONE;

    let mut next_point: Point = VecDeque::new();

    for (z, message) in point.into_iter().zip(&proof.rounds) {
        let [sum_endpoint, suminf] = *message;
        write(vs, message);
        // claim = (1 - z) * sum0 + z * sum1. At z = 0, sum0 is known
        // and the prover supplies sum1; otherwise recover sum1 by division.
        let (sum0, sum1) = if z == Field::ZERO {
            (claim, sum_endpoint)
        } else {
            let eqjsum0 = (Field::ONE - z) * sum_endpoint;
            (sum_endpoint, (claim - eqjsum0) / z)
        };

        let r = challenge(vs);
        next_point.push_back(r);
        let factor = eq_factor(r, z);

        // `claim = factor * (sum0 + r*(sum1 - sum0) + r*(r - 1)*suminf)`.
        let bracket = (sum1 - sum0) + (r - Field::ONE) * suminf;
        claim = factor * (sum0 + r * bracket);

        prefix *= factor;
    }

    let elem_lr = proof.evaluations;
    write(vs, &elem_lr);
    // Check if line polynomial hits same spot as sumcheck check
    if (prefix * elem_lr[0] * elem_lr[1]) != claim {
        Err(Error::InvalidProof)
    } else {
        Ok((next_point, elem_lr))
    }
}

pub(crate) fn verify_product_relation(
    vs: &mut Transcript,
    claim: F128,
    point: &[F128],
    proof: &ProductRelationProof,
) -> Result<(Vec<F128>, [F128; 2]), Error> {
    if proof.rounds.len() != point.len() {
        return Err(Error::MalformedProof);
    }
    let mut reversed = point.to_vec();
    reversed.reverse();
    let (point, values) =
        verify_product_relation_inner(vs, claim, VecDeque::from(reversed), proof)?;
    let mut point = Vec::from(point);
    point.reverse();
    Ok((point, values))
}
