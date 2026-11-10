//! Weighted Logup* pushforward, using the extracted BitZ fraction/IP kernels.
//!
//! For random postcommit cycle point r, M[k]=sum_{j:a[j]=k}Eq(r,j).
//! Commit M BEFORE tau. The rational identity checks its source indices;
//! <M,publicROM>=read_value checks the weighted read. Source address leaves,
//! both M evaluations, and read_value all reach the existing PCS.

use std::time::Instant;

use bitz_logup::{
    FractionTreeWitness, InnerProductProof, RationalClaims, RationalProof, prove_inner_product,
    prove_rational, verify_inner_product, verify_rational,
};
use om_field::K;
use om_memcheck::onehot::{AddrComps, EvalClaim};
use om_sumcheck::{PackedBits, eq_eval, eq_table, mle_eval};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::{
    AuthenticationProof, CommitmentTimings, Error, Schema, Statement, Summary, require, word_claim,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupProof {
    pub rational: RationalProof,
    pub inner_product: InnerProductProof,
    pub address_evals: Vec<K>,
    pub auxiliary: AuthenticationProof,
}

fn table_values(statement: &Statement) -> Vec<K> {
    statement
        .table
        .iter()
        .map(|&word| K::from_u128(word as u128))
        .collect()
}

fn pack_fields(values: &[K]) -> PackedBits {
    let words = values
        .iter()
        .flat_map(|value| [value.value() as u64, (value.value() >> 64) as u64])
        .collect();
    PackedBits::from_words(words, values.len() * 128)
}

fn source_claims(values: &[K], point: &[K]) -> Vec<EvalClaim> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| EvalClaim {
            poly: format!("input/address/{i}"),
            point: point.to_vec(),
            value,
        })
        .collect()
}

fn check_leaves(cycle: &[K], tau: K, claims: &RationalClaims, address: &[K]) -> Result<(), Error> {
    let source = address
        .iter()
        .enumerate()
        .fold(tau, |sum, (i, &v)| sum + K::from_u128(1u128 << i) * v);
    let table = claims
        .right
        .point
        .iter()
        .enumerate()
        .fold(tau, |sum, (i, &v)| sum + K::from_u128(1u128 << i) * v);
    require(
        claims.left.num == eq_eval(cycle, &claims.left.point),
        "logup source weight check failed",
    )?;
    require(claims.left.den == source, "logup source index check failed")?;
    require(
        claims.right.den == table,
        "logup public table index check failed",
    )
}

pub(crate) fn prove(
    statement: &Statement,
    comps: &AddrComps,
    cycle: &[K],
    read_value: K,
    tr: &mut Transcript,
    summary: &mut Summary,
    timings: &mut CommitmentTimings,
) -> Result<(LogupProof, Vec<EvalClaim>), Error> {
    let _span = Span::new("lookup.logup");
    let a = statement.config.address_bits;
    let weights = {
        let _span = Span::new("lookup.logup.pushforward");
        comps.ra_at(a, cycle)
    };
    let schema = Schema::auxiliary(statement);
    let g = {
        let _span = Span::new("lookup.logup.pack_aux");
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        schema.place(&mut g, "aux/pushforward", &pack_fields(&weights));
        g
    };
    tr.absorb_bytes(b"lookup/auxiliary", b"weighted-pushforward/v1");
    let (commitment, state) = {
        let started = Instant::now();
        let _span = Span::new("lookup.aux.commit");
        let result = schema.pcs.commit_packed(&g, tr);
        timings.auxiliary_commit = started.elapsed();
        result
    };
    // Field weights are fixed by the root before the denominator challenge.
    let tau = tr.challenge_k(b"lookup/logup/tau");
    let (mut left, mut right) = {
        let _span = Span::new("lookup.logup.fraction_witness");
        let source_num = eq_table(cycle);
        let addresses = comps.addrs();
        let source_den: Vec<K> = addresses
            .par_iter()
            .map(|&a| tau + K::from_u128(a as u128))
            .collect();
        let table_den: Vec<K> = (0..weights.len())
            .into_par_iter()
            .map(|k| tau + K::from_u128(k as u128))
            .collect();
        let mut left = FractionTreeWitness::default();
        let mut right = FractionTreeWitness::default();
        left.rebuild(
            &source_num,
            &source_den,
            statement.log_lookups,
            K::ZERO,
            K::ONE,
        )?;
        right.rebuild(&weights, &table_den, a, K::ZERO, K::ONE)?;
        (left, right)
    };
    let (rational, leaves) = {
        let _span = Span::new("lookup.logup.rational");
        prove_rational(&mut left, &mut right, tr, b"lookup/rational")?
    };
    drop(left);
    drop(right);
    let address_evals = {
        let _span = Span::new("lookup.logup.source_claims");
        comps.bit_evals(a, &leaves.left.point)
    };
    check_leaves(cycle, tau, &leaves, &address_evals)?;
    tr.absorb_ks(b"lookup/logup/address-evals", &address_evals);
    let (inner_product, ip) = {
        let _span = Span::new("lookup.logup.table_product");
        prove_inner_product(
            &weights,
            &table_values(statement),
            read_value,
            tr,
            b"lookup/table-product",
        )?
    };
    let aux_claims = vec![
        word_claim(&leaves.right.point, leaves.right.num, "aux/pushforward", 7),
        word_claim(&ip.point, ip.value, "aux/pushforward", 7),
    ];
    summary.auxiliary_bits = weights.len() * 128;
    summary.auxiliary_committed_bits = g.len();
    summary.auxiliary_claims = aux_claims.len();
    let (mosaic, opening, elapsed) = schema.open(&g, state, &aux_claims, tr, "lookup.aux.open")?;
    timings.auxiliary_open = elapsed;
    let claims = source_claims(&address_evals, &leaves.left.point);
    Ok((
        LogupProof {
            rational,
            inner_product,
            address_evals,
            auxiliary: AuthenticationProof {
                commitment,
                mosaic,
                opening,
            },
        },
        claims,
    ))
}

pub(crate) fn verify(
    statement: &Statement,
    cycle: &[K],
    read_value: K,
    proof: &LogupProof,
    tr: &mut Transcript,
) -> Result<Vec<EvalClaim>, Error> {
    let a = statement.config.address_bits;
    require(
        proof.address_evals.len() == a,
        "malformed logup address evaluations",
    )?;
    let schema = Schema::auxiliary(statement);
    schema.valid_commitment(&proof.auxiliary.commitment)?;
    tr.absorb_bytes(b"lookup/auxiliary", b"weighted-pushforward/v1");
    schema
        .pcs
        .absorb_commitment(&proof.auxiliary.commitment, tr);
    let tau = tr.challenge_k(b"lookup/logup/tau");
    let leaves = verify_rational(
        statement.log_lookups,
        a,
        &proof.rational,
        tr,
        b"lookup/rational",
    )?;
    check_leaves(cycle, tau, &leaves, &proof.address_evals)?;
    tr.absorb_ks(b"lookup/logup/address-evals", &proof.address_evals);
    let ip = verify_inner_product(
        a,
        read_value,
        |p| mle_eval(&table_values(statement), p),
        &proof.inner_product,
        tr,
        b"lookup/table-product",
    )?;
    let aux_claims = vec![
        word_claim(&leaves.right.point, leaves.right.num, "aux/pushforward", 7),
        word_claim(&ip.point, ip.value, "aux/pushforward", 7),
    ];
    schema.verify(
        &proof.auxiliary.commitment,
        &proof.auxiliary.mosaic,
        &proof.auxiliary.opening,
        &aux_claims,
        tr,
    )?;
    Ok(source_claims(&proof.address_evals, &leaves.left.point))
}
