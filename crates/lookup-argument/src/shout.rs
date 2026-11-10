//! Indexed lookup through BinaryShout and the current virtualization kernels.

use om_field::K;
use om_memcheck::onehot::{AddrComps, EvalClaim};
use om_memcheck::shout::{
    ShoutClaims, ShoutParams, ShoutProof, shout_prove_with_virtualization,
    shout_verify_with_virtualization,
};
use om_sumcheck::VerificationError;
use om_transcript::Transcript;

use crate::{Config, InputClaims};

fn claims(cycle: &[K], input: &InputClaims) -> ShoutClaims {
    ShoutClaims {
        r_c: cycle.to_vec(),
        cols: vec![0],
        rv: vec![input.read_value],
        bits: Some(input.address_bits.clone()),
    }
}

pub(crate) fn prove(
    config: &Config,
    params: &ShoutParams,
    table: &[u64],
    comps: &AddrComps,
    cycle: &[K],
    input: &InputClaims,
    tr: &mut Transcript,
) -> (ShoutProof, Vec<EvalClaim>) {
    let table = vec![
        table
            .iter()
            .map(|&word| K::from_u128(u128::from(word)))
            .collect(),
    ];
    shout_prove_with_virtualization(
        params,
        &table,
        comps,
        &[claims(cycle, input)],
        tr,
        b"lookup/shout",
        config.virtualization,
    )
}

pub(crate) fn verify(
    config: &Config,
    params: &ShoutParams,
    table: &[u64],
    cycle: &[K],
    input: &InputClaims,
    proof: &ShoutProof,
    tr: &mut Transcript,
) -> Result<Vec<EvalClaim>, VerificationError> {
    let table = vec![
        table
            .iter()
            .map(|&word| K::from_u128(u128::from(word)))
            .collect(),
    ];
    shout_verify_with_virtualization(
        params,
        &table,
        &[claims(cycle, input)],
        proof,
        tr,
        b"lookup/shout",
        config.virtualization,
    )
}
