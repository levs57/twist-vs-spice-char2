//! One Eq-factored sumcheck for the two cycle masks and public-initial mask.
//!
//! A smaller source domain is repeated along high variables, not zero-padded.
//! Its Eq weights therefore sum to one, including at Boolean challenge points.
//! Terminal scalars are conditional claims: the caller MUST authenticate the
//! three cycle sources at point[..n] and initial flag at point[..a] with PCS.

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_memcheck::eq_factor::{self, Cofactor, Endpoint};
use om_sumcheck::{SumcheckProof, eq_eval, eq_table, mle_eval_bits, sumcheck_verify};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskClaims {
    pub cycle: [K; 2],
    pub initial: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskProof {
    pub sc: SumcheckProof,
    pub value: K,
    pub is_write: K,
    pub next_is_write: K,
    pub initial_next_is_write: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceClaims {
    /// Full common point; the outer statement supplies each source dimension.
    pub point: Vec<K>,
    pub value: K,
    pub is_write: K,
    pub next_is_write: K,
    pub initial_next_is_write: K,
}

fn require(ok: bool, message: &str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
fn label(base: &[u8], suffix: &[u8]) -> Vec<u8> {
    [base, suffix].concat()
}

fn public_shape(initial: &[u64], gp_point: &[K]) -> Result<(usize, usize), Error> {
    require(
        initial.len().is_power_of_two(),
        "masked reduction: initial domain must be a nonzero power of two",
    )?;
    require(
        !gp_point.is_empty() && gp_point.len() <= 32,
        "masked reduction: invalid GP point dimension",
    )?;
    let a = initial.len().trailing_zeros() as usize;
    let l = gp_point.len() - 1;
    require(
        a <= l,
        "masked reduction: initial domain exceeds common point",
    )?;
    Ok((a, l))
}

fn setup(
    a: usize,
    l: usize,
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> [K; 3] {
    tr.absorb_bytes(
        &label(base, b"/protocol"),
        b"spice/masked-reduction/replicated-eq/v1",
    );
    tr.absorb_bytes(
        &label(base, b"/dimensions"),
        &[a as u64, l as u64, 64]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    tr.absorb_ks(&label(base, b"/gp-point"), gp_point);
    tr.absorb_ks(
        &label(base, b"/claims"),
        &[claims.cycle[0], claims.cycle[1], claims.initial],
    );
    // All three claimed values are fixed before any batching coefficient.
    std::array::from_fn(|_| tr.challenge_k(&label(base, b"/batch")))
}

fn batched_claim(claims: &MaskClaims, weights: [K; 3]) -> K {
    let values = [claims.cycle[0], claims.cycle[1], claims.initial];
    (0..3)
        .fold(U256::ZERO, |sum, i| sum + weights[i].clmul(values[i]))
        .reduce()
}

struct Products;
impl Cofactor for Products {
    const DEGREE: usize = 2;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        4
    }
    fn value(&self, row: &[K]) -> K {
        row[0] * row[1] + row[2] * row[3]
    }
    fn prepare(&self, low_eq: &[K]) -> Vec<K> {
        low_eq.to_vec()
    }
    fn accumulate(
        &self,
        prepared: &Vec<K>,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    ) {
        let mut add = |slot: usize, a: [K; 2], b: [K; 2]| {
            let scaled = mul_pairs(a, [prepared[low]; 2]);
            let products = clmul_pairs(scaled, b);
            out[slot] += products[0] + products[1];
        };
        if matches!(endpoint, Endpoint::Both | Endpoint::Zero) {
            add(0, [left[0], left[2]], [left[1], left[3]]);
        }
        if matches!(endpoint, Endpoint::Both | Endpoint::One) {
            add(1, [right[0], right[2]], [right[1], right[3]]);
        }
        add(
            2,
            [left[0] + right[0], left[2] + right[2]],
            [left[1] + right[1], left[3] + right[3]],
        );
    }
}

fn evaluate_flags(
    is_write: &[bool],
    next_is_write: &[bool],
    initial_next: &[bool],
    point: &[K],
    n: usize,
    a: usize,
) -> [K; 3] {
    // Share the cycle Eq table and bit scan. The public-initial flag has its
    // own smaller-domain point; existing bit-MLE code handles that domain.
    let eq = eq_table(&point[..n]);
    let add = |mut sum: [K; 2], ((&first, &next), &weight): ((&bool, &bool), &K)| {
        if first {
            sum[0] += weight;
        }
        if next {
            sum[1] += weight;
        }
        sum
    };
    let pair = if is_write.len() >= 1 << 14 {
        is_write
            .par_iter()
            .zip(next_is_write)
            .zip(eq.par_iter())
            .fold(|| [K::ZERO; 2], add)
            .reduce(|| [K::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    } else {
        is_write
            .iter()
            .zip(next_is_write)
            .zip(&eq)
            .fold([K::ZERO; 2], add)
    };
    [pair[0], pair[1], mle_eval_bits(initial_next, &point[..a])]
}

fn absorb_sources(tr: &mut Transcript, base: &[u8], proof: &MaskProof) {
    tr.absorb_ks(
        &label(base, b"/sources"),
        &[
            proof.value,
            proof.is_write,
            proof.next_is_write,
            proof.initial_next_is_write,
        ],
    );
}

/// Run only the sum-check, returning `(proof, point, weights, value,
/// folded masks)`. `value` is the word source at `point`. The caller sends it
/// and the flag sources later, absorbs them before any challenge that depends
/// on them, and the verifier then calls [`Deferred::check`].
#[allow(clippy::too_many_arguments)]
pub fn prove_sumcheck(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(SumcheckProof, Vec<K>, [K; 3], K, [K; 2]), Error> {
    let (a, l) = public_shape(initial_memory, gp_point)?;
    require(
        values.len().is_power_of_two(),
        "masked reduction: cycle domain must be a nonzero power of two",
    )?;
    let n = values.len().trailing_zeros() as usize;
    require(
        l == n.max(a),
        "masked reduction: GP point does not match source domains",
    )?;
    require(
        is_write.len() == values.len()
            && next_is_write.len() == values.len()
            && initial_next_is_write.len() == initial_memory.len(),
        "masked reduction: flag shape mismatch",
    )?;
    let weights = setup(a, l, gp_point, claims, tr, base);
    let initial = batched_claim(claims, weights);
    let row = |j: usize, out: &mut [K]| {
        let cycle = j & (values.len() - 1);
        let memory = j & (initial_memory.len() - 1);
        let mut mask = weights[0] + weights[1];
        if is_write[cycle] {
            mask += weights[0];
        }
        if next_is_write[cycle] {
            mask += weights[1];
        }
        out[0] = K::from_u128(values[cycle] as u128);
        out[1] = mask;
        out[2] = K::from_u128(initial_memory[memory] as u128);
        out[3] = if initial_next_is_write[memory] {
            K::ZERO
        } else {
            weights[2]
        };
    };
    let (actual, sc, point, finals) = {
        let _span = Span::new("spice.masked.sumcheck");
        eq_factor::prove_with_initial_claim(
            &gp_point[..l],
            &Products,
            row,
            Some(initial),
            tr,
            &label(base, b"/sc"),
        )
    };
    require(
        actual == initial,
        "masked reduction: incorrect zero-dimensional claim",
    )?;
    // finals[1] and finals[3] are the folded mask columns; the caller can
    // compare them with the flag sources as a prover-side self-check.
    Ok((sc, point, weights, finals[0], [finals[1], finals[3]]))
}

/// The masked-reduction terminal relation, awaiting the flag sources.
#[derive(Clone, Debug)]
pub struct Deferred {
    pub point: Vec<K>,
    weights: [K; 3],
    equality: K,
    expected: K,
    public_value: K,
}

impl Deferred {
    /// Check the terminal product relation given the sources at `point`.
    pub fn check(
        &self,
        value: K,
        is_write: K,
        next_is_write: K,
        initial_next_is_write: K,
    ) -> Result<(), Error> {
        let w = self.weights;
        let cofactor = value * (w[0] + w[1] + w[0] * is_write + w[1] * next_is_write)
            + self.public_value * w[2] * (K::ONE + initial_next_is_write);
        require(
            self.equality * cofactor == self.expected,
            "masked reduction: terminal product check failed",
        )
    }
}

/// Verifier half of [`prove_sumcheck`]. Initial memory is public and is
/// evaluated here; nothing after the sum-check is absorbed.
pub fn verify_sumcheck(
    initial_memory: &[u64],
    gp_point: &[K],
    claims: &MaskClaims,
    sc: &SumcheckProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<Deferred, Error> {
    let (a, l) = public_shape(initial_memory, gp_point)?;
    require(
        sc.round_msgs.len() == l && sc.round_msgs.iter().all(|m| m.len() == 3),
        "masked reduction: malformed sumcheck shape",
    )?;
    let weights = setup(a, l, gp_point, claims, tr, base);
    let initial = batched_claim(claims, weights);
    let (point, expected) = sumcheck_verify(initial, l, 3, sc, tr, &label(base, b"/sc"))?;
    Ok(Deferred {
        public_value: crate::word_eval(initial_memory, &point[..a], 6),
        equality: eq_eval(&gp_point[..l], &point),
        point,
        weights,
        expected,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prove(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(MaskProof, SourceClaims), Error> {
    let (sc, point, weights, value, masks) = prove_sumcheck(
        values,
        is_write,
        next_is_write,
        initial_memory,
        initial_next_is_write,
        gp_point,
        claims,
        tr,
        base,
    )?;
    let (a, _) = public_shape(initial_memory, gp_point)?;
    let n = values.len().trailing_zeros() as usize;
    let _span = Span::new("spice.masked.claims");
    let [first, next, initial_next] =
        evaluate_flags(is_write, next_is_write, initial_next_is_write, &point, n, a);
    require(
        masks[0] == weights[0] + weights[1] + weights[0] * first + weights[1] * next
            && masks[1] == weights[2] * (K::ONE + initial_next),
        "masked reduction: linear source binding failed",
    )?;
    let proof = MaskProof {
        sc,
        value,
        is_write: first,
        next_is_write: next,
        initial_next_is_write: initial_next,
    };
    let sources = SourceClaims {
        point,
        value: proof.value,
        is_write: first,
        next_is_write: next,
        initial_next_is_write: initial_next,
    };
    absorb_sources(tr, base, &proof);
    Ok((proof, sources))
}

/// The outer verifier MUST authenticate the returned values at its statement's
/// cycle/initial prefixes. No source dimensions or evaluation points are taken
/// from the prover. Initial memory is public and evaluated independently here.
pub fn verify(
    initial_memory: &[u64],
    gp_point: &[K],
    claims: &MaskClaims,
    proof: &MaskProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<SourceClaims, Error> {
    let deferred = verify_sumcheck(initial_memory, gp_point, claims, &proof.sc, tr, base)?;
    deferred.check(
        proof.value,
        proof.is_write,
        proof.next_is_write,
        proof.initial_next_is_write,
    )?;
    absorb_sources(tr, base, proof);
    Ok(SourceClaims {
        point: deferred.point,
        value: proof.value,
        is_write: proof.is_write,
        next_is_write: proof.next_is_write,
        initial_next_is_write: proof.initial_next_is_write,
    })
}

#[cfg(test)]
#[path = "masked_reduction_tests.rs"]
mod tests;
