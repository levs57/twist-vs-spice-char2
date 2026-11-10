// Copyright (c) 2026 World Foundation. MIT; see ../LICENSE.
// Derived from BitZ cdfb9b73b5857f4dc83761866e475186c6a75ab3,
// crates/gkr/src/logup_cut/inner_product.rs. Round arithmetic retained.
use crate::{Error, Wide256, bind, challenge, write};
use om_field::K as F128;
use om_transcript::Transcript;
use rayon::prelude::*;
const DOMAIN: &[u8] = b"bitz/logup-cut/inner-product/v1";
const PARALLEL_THRESHOLD: usize = 1 << 13;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalClaim {
    pub point: Vec<F128>,
    pub value: F128,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerProductProof {
    pub rounds: Vec<[F128; 2]>,
    pub value: F128,
}

pub fn prove_inner_product(
    committed: &[F128],
    public: &[F128],
    mut claim: F128,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<(InnerProductProof, EvalClaim), Error> {
    if committed.len() != public.len() || committed.is_empty() || !committed.len().is_power_of_two()
    {
        return Err(Error::InvalidShape);
    }
    bind(
        transcript,
        label,
        DOMAIN,
        &[committed.len().ilog2() as usize],
    );
    let mut committed = committed.to_vec();
    let mut public = public.to_vec();
    let mut len = committed.len();
    let mut point = Vec::with_capacity(len.ilog2() as usize);
    let mut rounds = Vec::with_capacity(point.capacity());
    while len > 1 {
        let [constant, linear] = round(&committed[..len], &public[..len]);
        write(transcript, &[constant, linear]);
        rounds.push([constant, linear]);
        let challenge = challenge(transcript);
        let quadratic = claim + linear;
        claim = constant + challenge * (linear + challenge * quadratic);
        fold(&mut committed[..len], challenge);
        fold(&mut public[..len], challenge);
        len >>= 1;
        point.push(challenge);
    }
    write(transcript, &[committed[0]]);
    if claim != committed[0] * public[0] {
        return Err(Error::InvalidClaim);
    }
    Ok((
        InnerProductProof {
            rounds,
            value: committed[0],
        },
        EvalClaim {
            point,
            value: committed[0],
        },
    ))
}

pub fn verify_inner_product(
    dimension: usize,
    mut claim: F128,
    public_eval: impl FnOnce(&[F128]) -> F128,
    proof: &InnerProductProof,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<EvalClaim, Error> {
    if dimension >= usize::BITS as usize || proof.rounds.len() != dimension {
        return Err(Error::MalformedProof);
    }
    bind(transcript, label, DOMAIN, &[dimension]);
    let mut point = Vec::with_capacity(dimension);
    for &[constant, linear] in &proof.rounds {
        write(transcript, &[constant, linear]);
        let quadratic = claim + linear;
        let challenge = challenge(transcript);
        claim = constant + challenge * (linear + challenge * quadratic);
        point.push(challenge);
    }
    write(transcript, &[proof.value]);
    if claim != proof.value * public_eval(&point) {
        return Err(Error::InvalidProof);
    }
    Ok(EvalClaim {
        point,
        value: proof.value,
    })
}

fn round(a: &[F128], b: &[F128]) -> [F128; 2] {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() & 1, 0);
    let pair = |(a, b): (&[F128], &[F128])| {
        let constant = Wide256::mul(a[0], b[0]);
        let linear = Wide256::mul(a[0], b[1]) + Wide256::mul(a[1], b[0]);
        [constant, linear]
    };
    let [constant, linear] = if a.len() >= PARALLEL_THRESHOLD {
        a.par_chunks_exact(2)
            .zip(b.par_chunks_exact(2))
            .map(pair)
            .reduce(
                || [Wide256::zero(), Wide256::zero()],
                |left, right| [left[0] + right[0], left[1] + right[1]],
            )
    } else {
        a.chunks_exact(2)
            .zip(b.chunks_exact(2))
            .map(pair)
            .fold([Wide256::zero(), Wide256::zero()], |left, right| {
                [left[0] + right[0], left[1] + right[1]]
            })
    };
    [constant.reduce(), linear.reduce()]
}

fn fold(values: &mut [F128], challenge: F128) {
    let half = values.len() / 2;
    for index in 0..half {
        let low = values[2 * index];
        values[index] = low + challenge * (low + values[2 * index + 1]);
    }
}
