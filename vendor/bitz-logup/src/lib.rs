//! Optimized weighted LogUp reductions from BitZ.
//!
//! Ported from BitZ commit cdfb9b73b5857f4dc83761866e475186c6a75ab3,
//! `crates/gkr/src/logup_cut/{fraction,inner_product}.rs` and the equality
//! product-relation kernels in `crates/gkr/src/lib.rs` (MIT; see LICENSE).
//! Field arithmetic is the existing OM GHASH GF128 backend. Public points
//! use low-variable-first order, matching OM multilinear tables.
//!
//! This crate only reduces rational and inner-product claims. The caller
//! must authenticate every returned nonpublic evaluation against its PCS.

mod fraction;
mod inner_product;
mod product;

pub use fraction::{
    FractionClaim, FractionLayerProof, FractionTreeProof, FractionTreeWitness, RationalClaims,
    RationalProof, prove_rational, verify_rational,
};
pub use inner_product::{EvalClaim, InnerProductProof, prove_inner_product, verify_inner_product};
pub use product::ProductRelationProof;

use om_field::{K, U256};
use om_transcript::Transcript;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidShape,
    ZeroDenominator,
    InvalidClaim,
    MalformedProof,
    InvalidProof,
    ZeroChallenge,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

fn bind(tr: &mut Transcript, label: &[u8], domain: &[u8], dimensions: &[usize]) {
    tr.absorb_bytes(b"bitz-logup/label", label);
    tr.absorb_bytes(b"bitz-logup/domain", domain);
    for &dimension in dimensions {
        tr.absorb_bytes(b"bitz-logup/dimension", &(dimension as u64).to_le_bytes());
    }
}

#[inline]
fn write(tr: &mut Transcript, values: &[K]) {
    tr.absorb_ks(b"bitz-logup/message", values);
}

#[inline]
fn challenge(tr: &mut Transcript) -> K {
    tr.challenge_k(b"bitz-logup/challenge")
}

// The donor's static Wide256 interface mapped onto OM's existing U256.
// Keeping this zero-cost wrapper makes arithmetic/scheduling diffs auditable.
#[derive(Clone, Copy)]
struct Wide256(U256);
impl Wide256 {
    #[inline(always)]
    fn zero() -> Self {
        Self(U256::ZERO)
    }
    #[inline(always)]
    fn mul(a: K, b: K) -> Self {
        Self(a.clmul(b))
    }
    #[inline(always)]
    fn reduce(self) -> K {
        self.0.reduce()
    }
}
impl core::ops::Add for Wide256 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl core::ops::AddAssign for Wide256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

#[cfg(test)]
mod tests;
